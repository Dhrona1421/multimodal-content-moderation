"""
Hackathon submission inference runner.

This file is intentionally strict about stdout formatting:
  [START] ...
  [STEP] ...
  [END] ...

Environment variables:
  API_BASE_URL
  MODEL_NAME
  HF_TOKEN
  LOCAL_IMAGE_NAME (used only when environments are launched from docker image)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError


def _safe_emit(line: str, *, err: bool = False) -> None:
    text = str(line)
    stream = sys.stderr if err else sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
        return
    except Exception:
        pass

    try:
        fd = 2 if err else 1
        os.write(fd, (text + "\n").encode("utf-8", errors="replace"))
    except Exception:
        pass


def _debug(message: str) -> None:
    _safe_emit(f"[DEBUG] {message}", err=True)

# ── Guarded optional imports ──────────────────────────────────────────────────
# openai is optional — falls back to rule-based when absent or unconfigured.
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore[assignment, misc]
    _OPENAI_AVAILABLE = False

# Environment class is resolved lazily to keep API startup fast and robust.
_ENV_CLASS: Optional[Any] = None
_ENV_SOURCE = "unresolved"
_ENV_IMPORT_ERROR: Optional[str] = None


# ── Runtime configuration ─────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        _debug(f"invalid {name}={raw!r}; using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        _debug(f"invalid {name}={raw!r}; using default {default}")
        return default


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

DEFAULT_TASK = os.getenv("OPENENV_TASK", "hard")
DEFAULT_BENCHMARK = os.getenv("OPENENV_BENCHMARK", "multimodal-content-moderation")
DEFAULT_MAX_STEPS = _env_int("MAX_STEPS", 12)
DEFAULT_SEED = _env_int("SEED", 42)
DEFAULT_TEMPERATURE = _env_float("TEMPERATURE", 0.2)
DEFAULT_MAX_TOKENS = _env_int("MAX_TOKENS", 180)
DEFAULT_SUCCESS_THRESHOLD = _env_float("SUCCESS_SCORE_THRESHOLD", 0.5)

ACTION_VALUES  = {"allow", "flag", "remove"}
JSON_BLOCK_RE  = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a content moderation agent. "
    "Given one observation, return exactly one JSON object with keys: "
    "action, confidence, reasoning. "
    "action must be one of allow, flag, remove. "
    "confidence must be a float from 0.0 to 1.0."
)


# ── OpenAI client (optional) ──────────────────────────────────────────────────
def _build_client() -> Optional[Any]:
    if not _OPENAI_AVAILABLE or not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as exc:
        _debug(f"OpenAI client init failed: {exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_action(action: Any) -> str:
    candidate = str(action).strip().lower()
    return candidate if candidate in ACTION_VALUES else "flag"


def _coerce_confidence(confidence: Any, default: float = 0.7) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def _extract_error(info: Dict[str, Any]) -> str:
    raw = info.get("last_action_error")
    if raw is None or raw == "":
        return "null"
    return _sanitize_text(raw)


def _build_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    history_tail = " | ".join(history[-3:]) if history else "none"
    return (
        f"step={obs.get('step', 1)}/{obs.get('max_steps', 12)}\n"
        f"difficulty={obs.get('difficulty', 'medium')}\n"
        f"image_tag={obs.get('image_tag', 'safe')}\n"
        f"user_type={obs.get('user_type', 'new')}\n"
        f"user_history={obs.get('user_history', 0.0)}\n"
        f"text={obs.get('text', '')}\n"
        f"recent_actions={history_tail}\n"
        "Return JSON only."
    )


# FastAPI endpoints for `uvicorn inference:app`.
class ResetRequest(BaseModel):
    task: str = Field(default=DEFAULT_TASK, pattern="^(easy|medium|hard)$")
    seed: int = Field(default=DEFAULT_SEED)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=64)

class ActionPayloadModel(BaseModel):
    action: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    agent_reasoning: Optional[Dict[str, Any]] = None


_API_LOCK = threading.Lock()
_API_ENV: Optional[Any] = None


def _resolve_env_class() -> Optional[Any]:
    global _ENV_CLASS, _ENV_SOURCE, _ENV_IMPORT_ERROR
    if _ENV_CLASS is not None:
        return _ENV_CLASS

    try:
        from openenv_env import OpenEnvModerationEnv as canonical_env  # type: ignore[import]
        _ENV_CLASS = canonical_env
        _ENV_SOURCE = "openenv_env"
        _ENV_IMPORT_ERROR = None
        return _ENV_CLASS
    except Exception as exc:  # noqa: BLE001
        _ENV_IMPORT_ERROR = str(exc)
        _debug(f"openenv_env import failed: {exc}")

    try:
        from env import ContentModerationEnv as fallback_env  # type: ignore[import]
        _ENV_CLASS = fallback_env
        _ENV_SOURCE = "env_fallback"
        return _ENV_CLASS
    except Exception as exc:  # noqa: BLE001
        detail = f"openenv_env={_ENV_IMPORT_ERROR!r}; env={exc!r}"
        _ENV_IMPORT_ERROR = detail
        _debug(f"env fallback import failed: {detail}")
        return None


def _create_env(task: str, seed: int, max_steps: int) -> Any:
    env_class = _resolve_env_class()
    if env_class is None:
        raise RuntimeError("environment unavailable")
    return env_class(task=task, seed=seed, max_steps=max_steps)


def _validate_action_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    validate = getattr(ActionPayloadModel, "model_validate", None)
    if callable(validate):
        model = validate(candidate)
        dump = getattr(model, "model_dump", None)
        if callable(dump):
            return dump(exclude_none=True)
    model = ActionPayloadModel.parse_obj(candidate)
    return model.dict(exclude_none=True)


def _require_openenv() -> None:
    if _resolve_env_class() is None:
        detail = _ENV_IMPORT_ERROR or "environment unavailable"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


def _normalize_step_payload(step_body: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(step_body or {})
    nested_action = payload.get("action")

    if isinstance(nested_action, dict):
        candidate: Dict[str, Any] = dict(nested_action)
        for key in ("confidence", "agent_reasoning"):
            if key in payload and key not in candidate:
                candidate[key] = payload[key]
    else:
        candidate = {
            key: payload[key]
            for key in ("action", "confidence", "agent_reasoning")
            if key in payload
        }

    if "action" not in candidate:
        raise ValueError("step payload must include 'action'")

    validated = _validate_action_payload(candidate)
    validated["action"] = _coerce_action(validated.get("action"))
    validated["confidence"] = _coerce_confidence(validated.get("confidence", 1.0), default=1.0)
    return validated


app = FastAPI(title="OpenEnv Inference API", version="3.0.0")


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ready",
        "health": "/health",
        "reset": "/reset",
        "step": "/step",
        "state": "/state",
    }


@app.get("/health")
@app.get("/healthz")
@app.get("/healthcheck")
@app.get("/livez")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
def ready() -> Dict[str, str]:
    if _resolve_env_class() is not None:
        return {"status": "ready", "source": _ENV_SOURCE}
    return {"status": "unready", "source": "none"}


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> Dict[str, Any]:
    _require_openenv()
    global _API_ENV
    with _API_LOCK:
        _API_ENV = _create_env(task=request.task, seed=request.seed, max_steps=request.max_steps)
        observation = _API_ENV.reset()
    return {"observation": observation, "reward": None, "done": False}


@app.get("/state")
@app.post("/state")
def state_endpoint() -> Dict[str, Any]:
    with _API_LOCK:
        if _API_ENV is None:
            return {}
        return _API_ENV.state()


@app.post("/step")
def step(step_body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    _require_openenv()
    try:
        action_payload = _normalize_step_payload(step_body)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    global _API_ENV
    with _API_LOCK:
        if _API_ENV is None:
            _API_ENV = _create_env(task=DEFAULT_TASK, seed=DEFAULT_SEED, max_steps=DEFAULT_MAX_STEPS)
            _API_ENV.reset()

        if _API_ENV.state() == {}:
            _API_ENV.reset()

        observation, reward, done, info = _API_ENV.step(action_payload)

    return {
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "info": dict(info),
    }


# ── Agents ────────────────────────────────────────────────────────────────────
def rule_based_agent(obs: Dict[str, Any]) -> Tuple[str, float, Dict[str, str]]:
    image_tag    = str(obs.get("image_tag", "safe"))
    user_type    = str(obs.get("user_type", "new"))
    user_history = float(obs.get("user_history", 0.0))

    if image_tag == "nudity":
        return "remove", 0.95, {"reasoning": "Nudity image tag is a direct violation."}
    if image_tag in {"violence", "drugs"}:
        return "flag",   0.80, {"reasoning": "Potentially harmful image content needs review."}
    if image_tag == "misleading":
        return "flag",   0.75, {"reasoning": "Possible misinformation requires review."}
    if user_type == "suspicious" and user_history >= 1.5:
        return "flag",   0.65, {"reasoning": "Suspicious history raises moderation risk."}
    return "allow", 0.85, {"reasoning": "No strong violation signals detected."}


def llm_agent(
    obs: Dict[str, Any],
    client: Optional[Any] = None,
    model_name: str = MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    history: Optional[List[str]] = None,
) -> Tuple[str, float, Dict[str, str]]:
    effective_client = client or DEFAULT_CLIENT
    if effective_client is None:
        return rule_based_agent(obs)

    user_prompt = _build_user_prompt(obs, history or [])
    try:
        completion = effective_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            timeout=10,
        )
    except Exception as exc:
        _debug(f"model request failed: {exc}")
        return rule_based_agent(obs)

    content = completion.choices[0].message.content
    if isinstance(content, list):
        text = "".join(
            str(p.get("text", "") if isinstance(p, dict) else p) for p in content
        ).strip()
    else:
        text = str(content or "").strip()

    parsed = _extract_json_block(text)
    if not parsed:
        return rule_based_agent(obs)

    action     = _coerce_action(parsed.get("action", "flag"))
    confidence = _coerce_confidence(parsed.get("confidence", 0.7))
    rv = parsed.get("reasoning", "model-decision")
    reasoning  = ({str(k): _sanitize_text(v) for k, v in rv.items()}
                  if isinstance(rv, dict)
                  else {"reasoning": _sanitize_text(rv)})
    return action, confidence, reasoning


# ── Scoring ───────────────────────────────────────────────────────────────────
def _compute_success(
    rewards: List[float],
    last_info: Dict[str, Any],
    max_steps: int,
    success_threshold: float,
) -> bool:
    if "episode_score" in last_info:
        try:
            return float(last_info["episode_score"]) >= success_threshold
        except (TypeError, ValueError):
            pass
    if max_steps <= 0 or not rewards:
        return False
    avg = sum(rewards) / len(rewards)
    return float(max(0.0, min(1.0, avg))) >= success_threshold


# ── Logging ───────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    _safe_emit(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    _safe_emit(
        f"[STEP] step={step} action={_sanitize_text(action)} reward={reward:.2f} "
        f"done={_to_bool_str(done)} error={'null' if error == 'null' else _sanitize_text(error)}"
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    _safe_emit(
        f"[END] success={_to_bool_str(success)} steps={steps} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}"
    )


def _safe_close(env: Any) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:
            _debug(f"env.close() failed: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict OpenEnv inference runner")
    parser.add_argument("--task",              choices=["easy", "medium", "hard"], default=DEFAULT_TASK)
    parser.add_argument("--benchmark",         default=DEFAULT_BENCHMARK)
    parser.add_argument("--seed",              type=int,   default=DEFAULT_SEED)
    parser.add_argument("--max-steps",         type=int,   default=DEFAULT_MAX_STEPS)
    parser.add_argument("--agent",             choices=["llm", "rule-based"], default="rule-based")
    parser.add_argument("--rule-based",        action="store_true")
    parser.add_argument("--temperature",       type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens",        type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--verbose",           action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        _debug(f"ignoring unknown args: {' '.join(unknown)}")
    return args


# ── Main inference loop ───────────────────────────────────────────────────────
def main() -> None:
    rewards: List[float]      = []
    history: List[str]        = []
    last_info: Dict[str, Any] = {}
    steps_taken = 0
    success     = False
    env         = None

    args = argparse.Namespace(
        task=DEFAULT_TASK,
        benchmark=DEFAULT_BENCHMARK,
        seed=DEFAULT_SEED,
        max_steps=DEFAULT_MAX_STEPS,
        agent="rule-based",
        rule_based=True,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        success_threshold=DEFAULT_SUCCESS_THRESHOLD,
        verbose=False,
    )

    try:
        parsed_args = _parse_args()
        args = parsed_args
    except BaseException as exc:  # noqa: BLE001
        _debug(f"argument parsing failed: {exc}")

    if getattr(args, "rule_based", False):
        args.agent = "rule-based"

    log_start(task=args.task, env=args.benchmark, model=MODEL_NAME)

    try:
        # Abort cleanly if no supported environment class can be loaded.
        if _resolve_env_class() is None:
            _debug("OpenEnvModerationEnv unavailable")
            return

        env = _create_env(task=args.task, max_steps=args.max_steps, seed=args.seed)
        observation = env.reset()

        for step in range(1, args.max_steps + 1):
            try:
                current_state = env.state() or observation
            except Exception as exc:
                _debug(f"env.state() failed at step {step}: {exc}")
                current_state = observation

            try:
                if args.agent == "llm":
                    action, confidence, reasoning = llm_agent(
                        current_state,
                        client=DEFAULT_CLIENT,
                        model_name=MODEL_NAME,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        history=history,
                    )
                else:
                    action, confidence, reasoning = rule_based_agent(current_state)
            except Exception as exc:
                _debug(f"agent failed at step {step}: {exc}")
                action, confidence, reasoning = rule_based_agent(current_state)

            payload = {
                "action":          _coerce_action(action),
                "confidence":      _coerce_confidence(confidence),
                "agent_reasoning": reasoning,
            }

            try:
                next_obs, reward, done, info = env.step(payload)
            except Exception as exc:
                _debug(f"env.step() failed at step {step}: {exc}")
                log_step(step=step, action=payload["action"], reward=0.0, done=True, error=str(exc))
                steps_taken = step
                break

            reward_val = float(reward)
            log_step(step=step, action=payload["action"], reward=reward_val,
                     done=bool(done), error=_extract_error(info))

            rewards.append(reward_val)
            steps_taken  = step
            last_info    = dict(info)
            observation  = next_obs
            history.append(payload["action"])

            if done:
                break

        success = _compute_success(
            rewards=rewards,
            last_info=last_info,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
        )

    except BaseException as exc:  # noqa: BLE001
        _debug(f"inference loop failed: {exc}")
    finally:
        if env is not None:
            _safe_close(env)
        # Always emit [END] — the validator requires it.
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Backward-compatible shim ──────────────────────────────────────────────────
def run_inference(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible import path for local validation tooling."""
    try:
        from inference_eval import run_inference as _run_inference  # type: ignore[import]
        return _run_inference(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        _debug(f"run_inference shim failed: {exc}")
        return {"success": False, "error": str(exc)}


# ── Entry point ───────────────────────────────────────────────────────────────
# IMPORTANT: `python inference.py` runs the inference loop.
# The FastAPI `app` object above is for `uvicorn inference:app`.
def _run_cli() -> None:
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        _debug(f"fatal inference error: {exc}")
        _safe_emit("[START] task=unknown env=unknown model=unknown")
        _safe_emit("[END] success=false steps=0 rewards=")
    finally:
        raise SystemExit(0)


if __name__ == "__main__":
    _run_cli()
