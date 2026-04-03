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
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from openenv_env import OpenEnvModerationEnv


# Required by hackathon prompt (LOCAL_IMAGE_NAME is optional for this local env path).
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", os.getenv("IMAGE_NAME", ""))
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

DEFAULT_TASK = os.getenv("OPENENV_TASK", "hard")
DEFAULT_BENCHMARK = os.getenv("OPENENV_BENCHMARK", "multimodal-content-moderation")
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))
DEFAULT_SEED = int(os.getenv("SEED", "42"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
DEFAULT_SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))

ACTION_VALUES = {"allow", "flag", "remove"}
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a content moderation agent. "
    "Given one observation, return exactly one JSON object with keys: "
    "action, confidence, reasoning. "
    "action must be one of allow, flag, remove. "
    "confidence must be a float from 0.0 to 1.0."
)


def _build_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


DEFAULT_CLIENT = _build_client()
LLM_AVAILABLE = DEFAULT_CLIENT is not None


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
    if candidate in ACTION_VALUES:
        return candidate
    return "flag"


def _coerce_confidence(confidence: Any, default: float = 0.7) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _extract_error(info: Dict[str, Any]) -> str:
    raw = info.get("last_action_error")
    if raw is None or raw == "":
        return "null"
    return _sanitize_text(raw)


def _build_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    history_tail = " | ".join(history[-3:]) if history else "none"
    text = obs.get("text", "")
    image_tag = obs.get("image_tag", "safe")
    user_type = obs.get("user_type", "new")
    difficulty = obs.get("difficulty", "medium")
    user_history = obs.get("user_history", 0.0)
    step = obs.get("step", 1)
    max_steps = obs.get("max_steps", 12)
    return (
        f"step={step}/{max_steps}\n"
        f"difficulty={difficulty}\n"
        f"image_tag={image_tag}\n"
        f"user_type={user_type}\n"
        f"user_history={user_history}\n"
        f"text={text}\n"
        f"recent_actions={history_tail}\n"
        "Return JSON only."
    )


def rule_based_agent(obs: Dict[str, Any]) -> Tuple[str, float, Dict[str, str]]:
    image_tag = str(obs.get("image_tag", "safe"))
    user_type = str(obs.get("user_type", "new"))
    user_history = float(obs.get("user_history", 0.0))

    if image_tag == "nudity":
        return "remove", 0.95, {"reasoning": "Nudity image tag is a direct violation."}
    if image_tag in {"violence", "drugs"}:
        return "flag", 0.80, {"reasoning": "Potentially harmful image content needs review."}
    if image_tag == "misleading":
        return "flag", 0.75, {"reasoning": "Possible misinformation requires review."}
    if user_type == "suspicious" and user_history >= 1.5:
        return "flag", 0.65, {"reasoning": "Suspicious history raises moderation risk."}
    return "allow", 0.85, {"reasoning": "No strong violation signals detected."}


def llm_agent(
    obs: Dict[str, Any],
    client: Optional[OpenAI] = None,
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
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] model request failed: {exc}", file=sys.stderr, flush=True)
        return rule_based_agent(obs)

    content = completion.choices[0].message.content
    if isinstance(content, list):
        merged_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                merged_parts.append(str(part.get("text", "")))
            else:
                merged_parts.append(str(part))
        text = "".join(merged_parts).strip()
    else:
        text = str(content or "").strip()

    parsed = _extract_json_block(text)
    if not parsed:
        return rule_based_agent(obs)

    action = _coerce_action(parsed.get("action", "flag"))
    confidence = _coerce_confidence(parsed.get("confidence", 0.7))
    reasoning_val = parsed.get("reasoning", "model-decision")
    if isinstance(reasoning_val, dict):
        reasoning = {str(k): _sanitize_text(v) for k, v in reasoning_val.items()}
    else:
        reasoning = {"reasoning": _sanitize_text(reasoning_val)}
    return action, confidence, reasoning


def _compute_success(
    rewards: List[float],
    last_info: Dict[str, Any],
    max_steps: int,
    success_threshold: float,
) -> bool:
    if "episode_score" in last_info:
        try:
            score = float(last_info["episode_score"])
            return score >= success_threshold
        except (TypeError, ValueError):
            pass

    if max_steps <= 0:
        return False
    total = float(sum(rewards))
    max_reward = max_steps * 1.1
    min_reward = max_steps * -1.5
    if max_reward == min_reward:
        return False
    score = (total - min_reward) / (max_reward - min_reward)
    score = max(0.0, min(1.0, score))
    return score >= success_threshold


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    action_str = _sanitize_text(action)
    error_str = "null" if error == "null" else _sanitize_text(error)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={_to_bool_str(done)} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_to_bool_str(success)} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _safe_close(env: Any) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[DEBUG] env.close() failed: {exc}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict OpenEnv inference runner")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default=DEFAULT_TASK)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--agent", choices=["llm", "rule-based"], default="llm")
    parser.add_argument("--rule-based", action="store_true", help="Compatibility alias for --agent rule-based")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--verbose", action="store_true", help="Accepted for backward compatibility; ignored")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rule_based:
        args.agent = "rule-based"

    env = OpenEnvModerationEnv(task=args.task, max_steps=args.max_steps, seed=args.seed)
    rewards: List[float] = []
    history: List[str] = []
    last_info: Dict[str, Any] = {}
    steps_taken = 0
    success = False

    log_start(task=args.task, env=args.benchmark, model=MODEL_NAME)

    try:
        observation = env.reset()
        for step in range(1, args.max_steps + 1):
            current_state = env.state() or observation
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

            payload = {
                "action": _coerce_action(action),
                "confidence": _coerce_confidence(confidence),
                "agent_reasoning": reasoning,
            }

            next_obs, reward, done, info = env.step(payload)
            reward_val = float(reward)
            error_val = _extract_error(info)
            log_step(
                step=step,
                action=payload["action"],
                reward=reward_val,
                done=bool(done),
                error=error_val,
            )

            rewards.append(reward_val)
            steps_taken = step
            last_info = dict(info)
            observation = next_obs
            history.append(payload["action"])

            if done:
                break

        success = _compute_success(
            rewards=rewards,
            last_info=last_info,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] inference loop failed: {exc}", file=sys.stderr, flush=True)
    finally:
        _safe_close(env)
        log_end(success=success, steps=steps_taken, rewards=rewards)


def run_inference(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Backward-compatible import path for local validation tooling.
    The full multi-agent evaluation runner now lives in inference_eval.py.
    """
    from inference_eval import run_inference as _run_inference

    return _run_inference(*args, **kwargs)


if __name__ == "__main__":
    main()
