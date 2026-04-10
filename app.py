"""
app.py — Advanced Gradio demo for the Multimodal Content Moderation Environment v2.

Tabs:
  🎮 Play            — Human makes decisions step-by-step, sees reward + explanation
  🤖 Auto-Pilot      — Agent runs full episode, detailed step breakdown
  📈 Training        — Run PPO training live, plot reward curves + confusion matrix
  📊 Dataset         — Browse all 41 posts with filters and analytics
  🏆 Leaderboard     — Compare PPO vs rule-based vs LLM with radar chart
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

_IMPORT_API_ONLY = os.getenv("OPENENV_API_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
if _IMPORT_API_ONLY:
    gr = None
    _GRADIO_IMPORT_ERROR: Optional[Exception] = None
else:
    try:
        import gradio as gr  # type: ignore[assignment]
        _GRADIO_IMPORT_ERROR = None
    except Exception as exc:  # noqa: BLE001
        gr = None  # type: ignore[assignment]
        _GRADIO_IMPORT_ERROR = exc

from env      import ContentModerationEnv
from features import ACTIONS
from grader   import ModerationGrader, _confusion_matrix, _classification_report
from inference import llm_agent, rule_based_agent, LLM_AVAILABLE
from schemas  import ActionModel, ObservationModel
from tasks    import TASKS, make_task

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "allow":   "#22c55e",
    "flag":    "#f59e0b",
    "remove":  "#ef4444",
    "easy":    "#86efac",
    "medium":  "#fde68a",
    "hard":    "#fca5a5",
    "safe":    "#d1fae5",
    "nudity":  "#fee2e2",
    "violence":"#fecaca",
    "drugs":   "#e0e7ff",
    "misleading":"#fef9c3",
}
TAG_ICON  = {"safe":"✅","nudity":"🔞","violence":"⚔️","drugs":"💊","misleading":"⚠️"}
USER_ICON = {"new":"🆕","trusted":"⭐","suspicious":"🚨"}
DIFF_ICON = {"easy":"🟢","medium":"🟡","hard":"🔴"}

PPO_CHECKPOINT_CANDIDATES = (
    "ppo_checkpoint_best",
    "ppo_final",
    "ppo_checkpoint_final",
)

ENV_NAME = "multimodal-content-moderation"
ENV_TITLE = "Multimodal Content Moderation"
ENV_DESCRIPTION = (
    "A real-world OpenEnv environment for multimodal social media content moderation. "
    "Agents review text, image safety tags, user trust signals, and session history to "
    "decide whether to allow, flag, or remove a post while calibrating confidence for "
    "human escalation."
)
UI_THEME = gr.themes.Soft(primary_hue="blue") if gr is not None else None
UI_CSS = """
  .gradio-container { max-width:900px!important; margin:auto }
  footer { display:none!important }
"""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_pretrained_ppo_agent():
    from network import ActorCriticNetwork
    from train import make_ppo_agent

    last_error: Exception | None = None
    for checkpoint in PPO_CHECKPOINT_CANDIDATES:
        if not Path(f"{checkpoint}.npz").exists():
            continue
        net = ActorCriticNetwork()
        try:
            net.load(checkpoint)
            return make_ppo_agent(net)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    expected = ", ".join(f"{name}.npz" for name in PPO_CHECKPOINT_CANDIDATES)
    if last_error is not None:
        raise RuntimeError(f"failed to load PPO checkpoint ({last_error})") from last_error
    raise FileNotFoundError(f"no PPO checkpoint found; expected one of: {expected}")


def _grader_agent(agent_fn):
    """
    Adapt any agent output to grader-compatible `(action, confidence)`.
    Accepts:
      - dict with action/confidence
      - tuple/list with at least 2 items
      - raw action string
    """
    def _wrapped(obs: Dict[str, Any]):
        raw = agent_fn(obs)
        if isinstance(raw, dict):
            return str(raw.get("action", "flag")), float(raw.get("confidence", 1.0))
        if isinstance(raw, (list, tuple)):
            if len(raw) >= 2:
                return str(raw[0]), float(raw[1])
            if len(raw) == 1:
                return str(raw[0]), 1.0
        return str(raw), 1.0

    return _wrapped


class ResetRequest(BaseModel):
    task: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    seed: int = 42
    max_steps: int = Field(default=12, ge=1, le=64)
    env_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ResetResponseModel(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepResponseModel(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class CloseSessionRequest(BaseModel):
    env_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


def _env_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        return max(min_value, value)
    return value


class EnvService:
    DEFAULT_ENV_ID = "default"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._envs: Dict[str, ContentModerationEnv] = {}
        self._session_config: Dict[str, Dict[str, Any]] = {}
        self._last_touched: Dict[str, float] = {}
        self._session_idle_timeout_s = _env_int("OPENENV_SESSION_IDLE_TIMEOUT_S", 1800, min_value=0)
        self._max_sessions = _env_int("OPENENV_MAX_SESSIONS", 256, min_value=1)
        self._default_seed = _env_int("OPENENV_DEFAULT_SEED", 42)
        self._default_max_steps = _env_int("OPENENV_DEFAULT_MAX_STEPS", 12, min_value=1)
        default_task = os.getenv("OPENENV_DEFAULT_TASK", "medium").strip().lower()
        self._default_task = default_task if default_task in {"easy", "medium", "hard"} else "medium"
        self._ready = False
        self._ready_error: Optional[str] = None
        self._probe_readiness()

    def _probe_readiness(self) -> None:
        try:
            probe_env = ContentModerationEnv(task="easy", seed=0, max_steps=1)
            probe_obs = probe_env.reset()
            ObservationModel.model_validate(probe_obs)
            self._ready = True
            self._ready_error = None
        except Exception as exc:  # noqa: BLE001
            self._ready = False
            self._ready_error = str(exc)

    def _resolve_env_id(self, env_id: Optional[str]) -> str:
        if env_id is None:
            return self.DEFAULT_ENV_ID
        env_id = str(env_id).strip()
        return env_id or self.DEFAULT_ENV_ID

    def _touch_locked(self, session_id: str, now: Optional[float] = None) -> None:
        self._last_touched[session_id] = time.monotonic() if now is None else now

    def _close_locked(self, session_id: str) -> bool:
        env = self._envs.pop(session_id, None)
        self._session_config.pop(session_id, None)
        self._last_touched.pop(session_id, None)
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        return env is not None

    def _evict_oldest_locked(self) -> None:
        if not self._last_touched:
            return
        oldest_id = min(self._last_touched, key=self._last_touched.get)
        self._close_locked(oldest_id)

    def _prune_locked(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if self._session_idle_timeout_s > 0:
            stale_ids = [
                sid
                for sid, touched in self._last_touched.items()
                if (now - touched) > float(self._session_idle_timeout_s)
            ]
            for sid in stale_ids:
                self._close_locked(sid)

        while len(self._envs) > self._max_sessions:
            self._evict_oldest_locked()

    def _build_env(self, task: str, seed: int, max_steps: int) -> ContentModerationEnv:
        return ContentModerationEnv(task=task, seed=seed, max_steps=max_steps)

    def _create_or_replace_session_locked(
        self,
        session_id: str,
        task: str,
        seed: int,
        max_steps: int,
    ) -> ContentModerationEnv:
        if session_id not in self._envs and len(self._envs) >= self._max_sessions:
            self._evict_oldest_locked()
        else:
            self._close_locked(session_id)

        env = self._build_env(task=task, seed=seed, max_steps=max_steps)
        self._envs[session_id] = env
        self._session_config[session_id] = {
            "task": task,
            "seed": seed,
            "max_steps": max_steps,
        }
        self._touch_locked(session_id)
        return env

    def _ensure_session_locked(self, session_id: str) -> Tuple[ContentModerationEnv, bool]:
        env = self._envs.get(session_id)
        if env is not None:
            return env, False

        if len(self._envs) >= self._max_sessions:
            self._evict_oldest_locked()
        env = self._build_env(
            task=self._default_task,
            seed=self._default_seed,
            max_steps=self._default_max_steps,
        )
        self._envs[session_id] = env
        self._session_config[session_id] = {
            "task": self._default_task,
            "seed": self._default_seed,
            "max_steps": self._default_max_steps,
        }
        self._touch_locked(session_id)
        return env, True

    def reset(
        self,
        task: str = "medium",
        seed: int = 42,
        max_steps: int = 12,
        env_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            session_id = self._resolve_env_id(env_id)
            env = self._create_or_replace_session_locked(
                session_id=session_id,
                task=task,
                seed=seed,
                max_steps=max_steps,
            )
            self._touch_locked(session_id)
            return session_id, env.reset()

    def state(self, env_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session_id = self._resolve_env_id(env_id)
            env = self._envs.get(session_id)
            if env is None:
                return {}
            self._touch_locked(session_id)
            return env.state()

    def step(self, action_payload: Dict[str, Any], env_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            self._prune_locked()
            session_id = self._resolve_env_id(env_id)
            env, is_new = self._ensure_session_locked(session_id)
            if is_new or env.done or not env.episode_posts:
                env.reset()

            observation, reward, done, info = env.step(action_payload)
            self._touch_locked(session_id)
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }

    def close(self, env_id: Optional[str] = None) -> bool:
        with self._lock:
            self._prune_locked()
            session_id = self._resolve_env_id(env_id)
            return self._close_locked(session_id)

    def active_sessions(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._envs)

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def readiness_error(self) -> Optional[str]:
        with self._lock:
            return self._ready_error

    def session_idle_timeout_s(self) -> int:
        with self._lock:
            return self._session_idle_timeout_s


ENV_SERVICE = EnvService()

# ─────────────────────────────────────────────────────────────────────────────
# HTML components
# ─────────────────────────────────────────────────────────────────────────────

def _badge(label: str, colour: str, text_colour: str = "#1f2937") -> str:
    return (f'<span style="background:{colour};color:{text_colour};padding:3px 10px;'
            f'border-radius:20px;font-size:0.82em;font-weight:600">{label}</span>')


def _post_card(obs: Dict) -> str:
    diff = obs.get("difficulty", "medium")
    tag  = obs.get("image_tag",  "safe")
    usr  = obs.get("user_type",  "new")
    return f"""
<div style="border:1px solid #d1d5db;border-radius:14px;padding:22px;
            background:#fafafa;font-family:sans-serif;max-width:760px">
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:14px;flex-wrap:wrap;gap:8px">
    <span style="font-weight:700;font-size:1.1em">Post #{obs.get("post_id","?")}</span>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      {_badge(f'{DIFF_ICON.get(diff,"")} {diff.upper()}', C.get(diff,"#e5e7eb"))}
      {_badge(f'{TAG_ICON.get(tag,"")} {tag}', C.get(tag,"#f3f4f6"))}
      {_badge(f'{USER_ICON.get(usr,"")} {usr}', "#f3f4f6")}
    </div>
    <span style="color:#9ca3af;font-size:0.9em">
      Step {obs.get("step","?")} / {obs.get("max_steps",12)}
    </span>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
              padding:16px;margin-bottom:12px;font-size:0.97em;line-height:1.6">
    {obs.get("text","")}
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;font-size:0.84em;color:#4b5563">
    <span>🕓 History: <strong>{obs.get("user_history",0):.2f}</strong></span>
    <span>📊 {obs.get("session_stats",{})}</span>
  </div>
</div>"""


def _result_card(info: Dict) -> str:
    correct   = info.get("correct_action", "?")
    agent_act = info.get("agent_action",   "?")
    reward    = info.get("reward",         0.0)
    escalated = info.get("escalated",      False)
    is_right  = info.get("is_correct",     False)
    reasoning = info.get("agent_reasoning", {}) or {}
    
    icon = "✅" if is_right else ("↗️" if escalated else "❌")
    label = "Correct!" if is_right else ("Escalated to human" if escalated else "Wrong")
    ac = C.get(agent_act, "#gray")
    cc = C.get(correct,   "#gray")
    reward_pct = int(max(0, min(100, (reward + 1.5) / 2.6 * 100)))
    bar_col = "#22c55e" if reward > 0 else ("#f59e0b" if reward == 0 else "#ef4444")
    
    reasoning_html = ""
    if reasoning:
        reasoning_html = "<div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82em'>"
        for k, v in reasoning.items():
            k_fmt = k.replace("_", " ").title()
            reasoning_html += f"<div style='background:#f1f5f9;padding:6px 10px;border-radius:6px'><strong>{k_fmt}:</strong> {v}</div>"
        reasoning_html += "</div>"

    return f"""
<div style="border:1px solid #d1d5db;border-radius:14px;padding:18px;
            background:#f8fafc;max-width:760px;margin-top:10px;font-family:sans-serif">
  <div style="font-size:1.2em;font-weight:700;margin-bottom:10px">{icon} {label}</div>
  <div style="display:flex;gap:24px;margin-bottom:12px;flex-wrap:wrap">
    <span>Your action: <strong style="color:{ac}">{agent_act.upper()}</strong></span>
    <span>Correct: <strong style="color:{cc}">{correct.upper()}</strong></span>
    <span>Reward: <strong>{reward:+.3f}</strong></span>
    <span>Confidence: {info.get("confidence",1.0):.2f}</span>
  </div>
  <div style="background:#e5e7eb;border-radius:20px;height:8px;margin-bottom:10px">
    <div style="background:{bar_col};width:{reward_pct}%;height:100%;
                border-radius:20px;transition:width 0.3s"></div>
  </div>
  {reasoning_html}
  <div style="background:#fff;border-left:4px solid {cc};padding:10px 14px;
              border-radius:0 8px 8px 0;font-size:0.91em;color:#374151;margin-top:10px">
    <strong>📋 Ground Truth Why:</strong> {info.get("reason","")}
  </div>
</div>"""


def _progress_widget(steps: int, score: float, stats: Dict) -> str:
    if steps == 0:
        return "<p style='color:#9ca3af;font-family:sans-serif'>Start a game to see progress.</p>"
    pct = int(max(0, min(1, (score + 1.5) / 2.6)) * 100)
    col = "#22c55e" if score > 0.4 else ("#f59e0b" if score > 0 else "#ef4444")
    return f"""
<div style="font-family:sans-serif;max-width:760px">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;
              font-size:0.9em;color:#374151">
    <span>Steps: <strong>{steps}/{stats.get("max_steps", 12) if stats else 12}</strong></span>
    <span>Score: <strong>{score:+.3f}</strong></span>
    <span>✓ {stats.get("correct",0)}  ✗ {stats.get("wrong",0)}</span>
  </div>
  <div style="background:#e5e7eb;border-radius:20px;height:14px;overflow:hidden">
    <div style="background:{col};width:{pct}%;height:100%;border-radius:20px"></div>
  </div>
</div>"""


def _episode_summary_html(history: List[Dict], score: float) -> str:
    correct = sum(1 for h in history if h.get("is_correct"))
    col = "#22c55e" if score >= 0.7 else ("#f59e0b" if score >= 0.4 else "#ef4444")
    rows = ""
    for i, h in enumerate(history, 1):
        icon = "✅" if h["is_correct"] else ("↗️" if h.get("escalated") else "❌")
        ac = C.get(h["agent_action"], "#gray")
        cc = C.get(h["correct_action"], "#gray")
        rows += (
            f"<tr style='border-bottom:1px solid #f3f4f6'>"
            f"<td style='padding:7px 10px'>{i}</td>"
            f"<td style='padding:7px 10px'>{icon}</td>"
            f"<td style='padding:7px 10px;color:{ac}'><strong>{h['agent_action'].upper()}</strong></td>"
            f"<td style='padding:7px 10px;color:{cc}'>{h['correct_action'].upper()}</td>"
            f"<td style='padding:7px 10px'>{h['reward']:+.3f}</td>"
            f"<td style='padding:7px 10px'>{h['confidence']:.2f}</td>"
            f"<td style='padding:7px 10px'>{DIFF_ICON.get(h['difficulty'],'')} {h['difficulty']}</td>"
            f"</tr>"
        )
    return f"""
<div style="max-width:760px;font-family:sans-serif">
  <div style="background:{col};color:#fff;border-radius:14px;padding:18px 24px;
              margin-bottom:16px;text-align:center">
    <div style="font-size:2.2em;font-weight:700">{score:.1%}</div>
    <div style="font-size:1em">Episode Score &nbsp;|&nbsp; {correct}/{len(history)} correct</div>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.88em">
    <thead><tr style="background:#f3f4f6;font-weight:600">
      <th style="padding:9px 10px;text-align:left">#</th>
      <th style="padding:9px 10px;text-align:left"></th>
      <th style="padding:9px 10px;text-align:left">Your action</th>
      <th style="padding:9px 10px;text-align:left">Correct</th>
      <th style="padding:9px 10px;text-align:left">Reward</th>
      <th style="padding:9px 10px;text-align:left">Conf.</th>
      <th style="padding:9px 10px;text-align:left">Diff.</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _hero_panel() -> str:
    return """
<section class="hero-shell">
  <div class="hero-copy">
    <div class="eyebrow">OpenEnv Submission · Real-world Trust & Safety</div>
    <h1>Multimodal Content Moderation</h1>
    <p class="hero-text">
      A judge-ready environment for training and evaluating agents on the real moderation tradeoff:
      act quickly, avoid dangerous misses, and escalate honestly when uncertain.
    </p>
    <div class="hero-pills">
      <span>3 tasks: easy to hard</span>
      <span>Typed environment contract</span>
      <span>Deterministic baseline</span>
      <span>Docker + HF Space ready</span>
    </div>
  </div>
  <div class="hero-scoreboard">
    <div class="metric-card">
      <div class="metric-label">Real Task</div>
      <div class="metric-value">Content moderation</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Action Space</div>
      <div class="metric-value">Allow · Flag · Remove</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Core Challenge</div>
      <div class="metric-value">Cross-modal conflicts + calibrated escalation</div>
    </div>
  </div>
</section>"""


def _judge_brief() -> str:
    return """
<div class="judge-grid">
  <div class="judge-card">
    <div class="judge-kicker">Start Here</div>
    <h3>1. Auto-Pilot</h3>
    <p>Run the hard task and inspect immediate benchmark behavior with step-level rewards.</p>
  </div>
  <div class="judge-card">
    <div class="judge-kicker">Compare</div>
    <h3>2. Leaderboard</h3>
    <p>See aggregate score, hard-task macro-F1, and calibration quality across agents.</p>
  </div>
  <div class="judge-card">
    <div class="judge-kicker">Inspect</div>
    <h3>3. Play</h3>
    <p>Step through a post manually to understand why the reward shaping reflects real moderation tradeoffs.</p>
  </div>
  <div class="judge-card accent">
    <div class="judge-kicker">Why Hard</div>
    <h3>Signals Conflict</h3>
    <p>Safe-looking text can pair with harmful imagery, and trusted users can still spread misinformation.</p>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Play
# ─────────────────────────────────────────────────────────────────────────────

def start_game(task: str, state: Dict):
    env = ContentModerationEnv(task=task, max_steps=12, seed=42)
    obs = env.reset()
    state.update({"env": env, "obs": obs, "history": [],
                  "total_reward": 0.0, "steps": 0})
    return (
        _post_card(obs), "",
        gr.update(interactive=True), gr.update(interactive=True),
        gr.update(interactive=True), state,
        _progress_widget(0, 0.0, {}),
    )


def take_action(action: str, confidence: float, state: Dict):
    env: ContentModerationEnv = state["env"]
    obs, reward, done, info = env.step(
        {"action": action, "confidence": confidence / 100}
    )
    state["total_reward"] += reward
    state["steps"]        += 1
    state["history"].append(info)

    result_html = _result_card(info)
    stats       = info.get("session_stats", {})
    score       = sum(r for r in env.episode_rewards) / max(len(env.episode_rewards), 1)

    if done:
        final_score = env.compute_score()
        return (
            _episode_summary_html(state["history"], final_score),
            result_html,
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), state,
            _progress_widget(state["steps"], final_score, stats),
        )

    state["obs"] = obs
    return (
        _post_card(obs), result_html,
        gr.update(interactive=True), gr.update(interactive=True),
        gr.update(interactive=True), state,
        _progress_widget(state["steps"], score, stats),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Auto-Pilot
# ─────────────────────────────────────────────────────────────────────────────

def run_autopilot(task: str, agent_name: str) -> str:
    agent_fn    = rule_based_agent
    agent_label = "Rule-Based Agent"
    if agent_name == "LLM Agent" and LLM_AVAILABLE:
        agent_fn    = llm_agent
        agent_label = f"LLM ({os.environ.get('MODEL_NAME','gpt-4o-mini')})"
    elif agent_name == "PPO Agent":
        try:
            agent_fn    = _load_pretrained_ppo_agent()
            agent_label = "PPO Agent (trained)"
        except Exception as e:
            agent_fn    = rule_based_agent
            agent_label = f"Rule-Based (PPO load failed: {e})"

    env = make_task(task)
    obs = env.reset()
    history = []
    
    for _ in range(env.max_steps):
        res = agent_fn(obs)
        
        # Safely unpack agent response (handle variable-length tuples)
        if isinstance(res, (tuple, list)):
            act = res[0] if len(res) > 0 else "flag"
            conf = res[1] if len(res) > 1 else 1.0
            reasoning = res[2] if len(res) > 2 else {}
        else:
            act = res
            conf = 1.0
            reasoning = {}
        
        obs, reward, done, info = env.step({
            "action": act,
            "confidence": conf,
            "agent_reasoning": reasoning
        })
        history.append(info)
        if done: break

    # Detailed metrics
    correct_count = sum(1 for h in history if h["is_correct"])
    steps = len(history)
    acc = correct_count / max(steps, 1)  # Protect against division by zero
    
    # Step rows
    rows = ""
    for r in history:
        icon = "✅" if r["is_correct"] else ("↗️" if r["escalated"] else "❌")
        ac   = C.get(r["agent_action"],   "#gray")
        cc   = C.get(r["correct_action"], "#gray")
        
        # Format reasoning as compact text
        reason_txt = ""
        if r.get("agent_reasoning"):
            reason_txt = "<br>".join([f"<b>{k.split('_')[0].title()}:</b> {v}" for k,v in r["agent_reasoning"].items()])

        rows += (
            f"<tr style='border-bottom:1px solid #f3f4f6'>"
            f"<td style='padding:8px'>{r['post_id']}</td>"
            f"<td style='padding:8px'>{icon}</td>"
            f"<td style='padding:8px;font-size:0.83em;max-width:320px'>{reason_txt if reason_txt else r['reason']}</td>"
            f"<td style='padding:8px;color:{ac}'><strong>{r['agent_action'].upper()}</strong></td>"
            f"<td style='padding:8px;color:{cc}'>{r['correct_action'].upper()}</td>"
            f"<td style='padding:8px'>{r['reward']:+.3f}</td>"
            f"<td style='padding:8px'>{r['confidence']:.2f}</td>"
            f"</tr>"
        )

    score = env.compute_score()
    col   = "#22c55e" if score >= 0.75 else ("#f59e0b" if score >= 0.5 else "#ef4444")
    return f"""
<div style="font-family:sans-serif;max-width:860px">
  <div style="background:{col};color:#fff;border-radius:12px;padding:14px 22px;
              margin-bottom:16px;display:flex;justify-content:space-between;align-items:center">
    <div><strong>{agent_label}</strong> &nbsp;·&nbsp; Task: {task.upper()}</div>
    <div style="font-size:1.7em;font-weight:700">{score:.1%}</div>
  </div>
  <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;font-size:0.9em">
    <span>✓ Correct: {correct_count}/{steps}</span>
    <span>📊 Accuracy: {acc:.0%}</span>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.86em;margin-top:12px">
    <thead><tr style="background:#f3f4f6;font-weight:600">
      <th style="padding:8px;text-align:left">#</th>
      <th style="padding:8px;text-align:left"></th>
      <th style="padding:8px;text-align:left">Agent Logic / Reasoning</th>
      <th style="padding:8px;text-align:left">Agent</th>
      <th style="padding:8px;text-align:left">Correct</th>
      <th style="padding:8px;text-align:left">Reward</th>
      <th style="padding:8px;text-align:left">Conf.</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Training dashboard
# ─────────────────────────────────────────────────────────────────────────────

def run_training(n_updates: int, task: str) -> tuple:
    """Run PPO training and return (log_html, plot_data_json)."""
    from train import train, PPOConfig, make_ppo_agent
    from grader import ModerationGrader

    cfg = PPOConfig()
    cfg.n_steps = 32

    log_lines: List[str] = []
    scores:    Dict[str, List] = {"update": [], "easy": [], "medium": [], "hard": [], "aggregate": []}

    import io, contextlib, sys as _sys

    # Capture printed output from train()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net = train(
            task=task if task != "curriculum" else None,
            n_updates=n_updates,
            eval_interval=max(n_updates // 5, 5),
            checkpoint_path="ppo_checkpoint",
            log_path="training_log.csv",
            seed=42,
            cfg=cfg,
            verbose=True,
        )
    log_text = buf.getvalue()

    # Parse scores from log CSV if available
    import csv, os
    if os.path.exists("training_log.csv"):
        with open("training_log.csv") as f:
            for row in csv.DictReader(f):
                scores["update"].append(int(row.get("update", 0)))
                for t in ["easy", "medium", "hard"]:
                    k = f"score_{t}"
                    if k in row:
                        scores[t].append(float(row[k]))
                if "agg_score" in row:
                    scores["aggregate"].append(float(row["agg_score"]))

    # Final eval
    grader   = ModerationGrader(seed=42)
    agent_fn = make_ppo_agent(net)
    report   = grader.grade_all_tasks(_grader_agent(agent_fn))

    agg      = report["aggregate_score"]
    rb_report= grader.grade_all_tasks(_grader_agent(rule_based_agent))
    rb_agg   = rb_report["aggregate_score"]
    score_cards = []
    for t in ["easy", "medium", "hard"]:
        score_cards.append(
            f"""
    <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center">
      <div style="font-size:0.8em;color:#6b7280;margin-bottom:4px">{t.upper()}</div>
      <div style="font-size:1.4em;font-weight:700">{report["tasks"][t]["score"]:.1%}</div>
      <div style="font-size:0.8em;color:#9ca3af">vs baseline {rb_report["tasks"][t]["score"]:.1%}</div>
    </div>"""
        )
    score_cards_html = "".join(score_cards)

    result_html = f"""
<div style="font-family:sans-serif;max-width:760px">
  <div style="background:#1e40af;color:#fff;border-radius:12px;padding:16px 22px;
              margin-bottom:14px;display:flex;justify-content:space-between">
    <div><strong>PPO Training Complete</strong></div>
    <div style="font-size:1.5em;font-weight:700">{agg:.1%} aggregate</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px">
    {score_cards_html}
  </div>
  <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:14px;
              font-size:0.9em;color:#166534">
    Δ vs rule-based baseline:
    easy {report["tasks"]["easy"]["score"]-rb_report["tasks"]["easy"]["score"]:+.4f} &nbsp;|&nbsp;
    medium {report["tasks"]["medium"]["score"]-rb_report["tasks"]["medium"]["score"]:+.4f} &nbsp;|&nbsp;
    hard {report["tasks"]["hard"]["score"]-rb_report["tasks"]["hard"]["score"]:+.4f} &nbsp;|&nbsp;
    <strong>aggregate {agg-rb_agg:+.4f}</strong>
  </div>
  <details style="margin-top:12px"><summary style="cursor:pointer;font-weight:600">
    Training log (last 40 lines)</summary>
    <pre style="background:#1e1e1e;color:#d4d4d4;padding:14px;border-radius:8px;
                font-size:0.78em;overflow-x:auto;max-height:300px;overflow-y:auto">{"".join(log_text.splitlines(True)[-40:])}</pre>
  </details>
</div>"""

    return result_html, json.dumps(scores)


def plot_training(scores_json: str) -> Any:
    """Build a Gradio plot from the scores JSON."""
    try:
        scores = json.loads(scores_json)
        if not scores.get("update"):
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        updates = scores["update"]
        colours = {"easy": "#22c55e", "medium": "#f59e0b",
                   "hard": "#ef4444", "aggregate": "#6366f1"}
        for key, col in colours.items():
            vals = scores.get(key, [])
            if vals:
                lw = 2.5 if key == "aggregate" else 1.5
                ax.plot(updates[:len(vals)], vals, label=key, color=col,
                        linewidth=lw, marker="o", markersize=3)
        ax.set_xlabel("PPO Update")
        ax.set_ylabel("Score (0–1)")
        ax.set_title("Training Progress")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Dataset explorer
# ─────────────────────────────────────────────────────────────────────────────

def explore_dataset(difficulty: str, image_tag: str, user_type: str,
                    correct_action: str) -> str:
    with open("moderation_dataset.json", encoding="utf-8") as f:
        posts = json.load(f)

    if difficulty    != "All": posts = [p for p in posts if p["difficulty"]    == difficulty]
    if image_tag     != "All": posts = [p for p in posts if p["image_tag"]     == image_tag]
    if user_type     != "All": posts = [p for p in posts if p["user_type"]     == user_type]
    if correct_action!= "All": posts = [p for p in posts if p["correct_action"]== correct_action]

    if not posts:
        return "<p style='color:#9ca3af;font-family:sans-serif'>No posts match these filters.</p>"

    cards = ""
    for p in posts:
        ac   = C.get(p["correct_action"], "#gray")
        dc   = C.get(p["difficulty"], "#e5e7eb")
        tc   = C.get(p["image_tag"], "#f3f4f6")
        cards += f"""
<div style="border:1px solid #e5e7eb;border-radius:12px;padding:16px;
            margin-bottom:10px;background:#fff;font-family:sans-serif">
  <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center">
    <span style="font-weight:700;color:#374151">#{p['id']}</span>
    {_badge(f'{DIFF_ICON.get(p["difficulty"],"")} {p["difficulty"]}', dc)}
    {_badge(f'{TAG_ICON.get(p["image_tag"],"")} {p["image_tag"]}', tc)}
    {_badge(f'{USER_ICON.get(p["user_type"],"")} {p["user_type"]}', "#f3f4f6")}
    <span style="margin-left:auto">
      {_badge(p["correct_action"].upper(), ac, "#fff")}
    </span>
  </div>
  <div style="margin-bottom:8px;line-height:1.55;font-size:0.94em">{p["text"]}</div>
  <div style="color:#6b7280;font-size:0.86em;border-top:1px solid #f3f4f6;padding-top:8px">
    📋 {p["reason"]}
  </div>
</div>"""

    return f"""<div style="max-width:800px">
<p style="color:#6b7280;font-family:sans-serif;margin-bottom:12px">
  <strong>{len(posts)}</strong> posts matching filters
</p>{cards}</div>"""


def dataset_analytics() -> str:
    with open("moderation_dataset.json", encoding="utf-8") as f:
        posts = json.load(f)

    from collections import Counter

    def _bar(counts: Dict, width: int = 18) -> str:
        total = sum(counts.values())
        rows  = ""
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / max(total, 1)
            b   = "█" * int(pct * width) + "░" * (width - int(pct * width))
            rows += (f"<tr><td style='padding:4px 10px;color:#374151'>{label}</td>"
                     f"<td style='padding:4px 10px;font-family:monospace;color:#6366f1'>{b}</td>"
                     f"<td style='padding:4px 6px;color:#374151'>{count} ({pct:.0%})</td></tr>")
        return f"<table style='font-size:0.85em'>{rows}</table>"

    sections = {
        "Correct Action":    Counter(p["correct_action"] for p in posts),
        "Image Tag":         Counter(p["image_tag"]      for p in posts),
        "User Type":         Counter(p["user_type"]      for p in posts),
        "Difficulty":        Counter(p["difficulty"]     for p in posts),
    }

    html = "<div style='font-family:sans-serif;max-width:760px'>"
    for title, counts in sections.items():
        html += f"<h4 style='color:#374151;margin-bottom:4px'>{title}</h4>"
        html += _bar(dict(counts))
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def run_leaderboard() -> tuple:
    grader    = ModerationGrader(seed=42)
    agents    = {"Rule-Based": _grader_agent(rule_based_agent)}

    # Try loading PPO checkpoint
    try:
        agents["PPO Agent"] = _grader_agent(_load_pretrained_ppo_agent())
    except Exception:
        pass

    if LLM_AVAILABLE:
        agents[f"LLM ({os.environ.get('MODEL_NAME','gpt-4o-mini')})"] = _grader_agent(llm_agent)

    import random as _r
    _rng = _r.Random(7)
    agents["Random"] = lambda obs: (_rng.choice(["allow","flag","remove"]), 0.5)

    results: Dict[str, Dict] = {}
    for name, fn in agents.items():
        report = grader.grade_all_tasks(fn)
        results[name] = {
            "aggregate":  report["aggregate_score"],
            **{t: report["tasks"][t]["score"] for t in ["easy","medium","hard"]},
            **{f"f1_{t}": report["tasks"][t]["classification"]["macro_f1"] for t in ["easy","medium","hard"]},
            **{f"ece_{t}": report["tasks"][t]["ece"] for t in ["easy","medium","hard"]},
        }

    # Leaderboard table
    sorted_agents = sorted(results.items(), key=lambda x: -x[1]["aggregate"])
    rows = ""
    for rank, (name, r) in enumerate(sorted_agents, 1):
        medal = {1:"🥇",2:"🥈",3:"🥉"}.get(rank,"  ")
        rows += (
            f"<tr style='border-bottom:1px solid #f3f4f6'>"
            f"<td style='padding:10px'>{medal} {rank}</td>"
            f"<td style='padding:10px;font-weight:600'>{name}</td>"
            f"<td style='padding:10px;font-weight:700;color:#6366f1'>{r['aggregate']:.4f}</td>"
            f"<td style='padding:10px'>{r['easy']:.4f}</td>"
            f"<td style='padding:10px'>{r['medium']:.4f}</td>"
            f"<td style='padding:10px'>{r['hard']:.4f}</td>"
            f"<td style='padding:10px'>{r.get('f1_hard',0):.4f}</td>"
            f"<td style='padding:10px'>{r.get('ece_hard',0):.4f}</td>"
            f"</tr>"
        )

    table_html = f"""
<div style="font-family:sans-serif;max-width:860px">
  <table style="width:100%;border-collapse:collapse;font-size:0.9em">
    <thead><tr style="background:#f3f4f6;font-weight:600">
      <th style="padding:10px;text-align:left">Rank</th>
      <th style="padding:10px;text-align:left">Agent</th>
      <th style="padding:10px;text-align:left">Aggregate ↓</th>
      <th style="padding:10px;text-align:left">Easy</th>
      <th style="padding:10px;text-align:left">Medium</th>
      <th style="padding:10px;text-align:left">Hard</th>
      <th style="padding:10px;text-align:left">Hard F1</th>
      <th style="padding:10px;text-align:left">Hard ECE</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""

    # Radar chart data
    radar_data = json.dumps({
        name: {
            "Easy Score": r["easy"],
            "Medium Score": r["medium"],
            "Hard Score": r["hard"],
            "Macro F1 (Hard)": r.get("f1_hard", 0),
            "Calibration (1-ECE)": 1 - r.get("ece_hard", 0),
        }
        for name, r in results.items()
    })
    return table_html, radar_data


def plot_radar(radar_json: str) -> Any:
    try:
        data = json.loads(radar_json)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        categories = ["Easy Score","Medium Score","Hard Score",
                      "Macro F1 (Hard)","Calibration (1-ECE)"]
        N    = len(categories)
        angles = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection":"polar"})
        colours = ["#6366f1","#22c55e","#f59e0b","#ef4444"]
        for (name, vals), col in zip(data.items(), colours):
            v = [vals.get(c, 0) for c in categories] + [vals.get(categories[0], 0)]
            ax.plot(angles, v, "o-", linewidth=2, label=name, color=col)
            ax.fill(angles, v, alpha=0.07, color=col)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25","0.50","0.75","1.0"], size=7)
        ax.grid(color="#e5e7eb")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax.set_title("Agent Performance Radar", pad=20, fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Content Moderation Environment v2",
    ) as demo:


        gr.HTML("""
<div style="text-align:center;font-family:sans-serif;padding:24px 0 8px">
  <div style="font-size:2.2em">🛡️</div>
  <h1 style="margin:6px 0;font-size:1.7em;font-weight:800">
    Multimodal Content Moderation</h1>
  <p style="color:#6b7280;margin:0;font-size:0.95em">
    OpenEnv RL Environment v2 &nbsp;·&nbsp;
    Meta × Hugging Face × PyTorch Hackathon
  </p>
  <p style="color:#9ca3af;margin:6px 0 0;font-size:0.72em">
    created by Dhrona
  </p>
</div>""")

        with gr.Tabs():

            # ── Tab 1: Play ───────────────────────────────────────────────────
            with gr.Tab("🎮 Play"):
                gr.Markdown("Make real moderation decisions. Confidence < 45 → escalate to human review.")
                with gr.Row():
                    t_dd    = gr.Dropdown(["easy","medium","hard"], value="medium", label="Difficulty")
                    s_btn   = gr.Button("▶ Start Game", variant="primary")
                prog_html  = gr.HTML(_progress_widget(0, 0.0, {}))
                post_html  = gr.HTML(
                    "<div style='color:#9ca3af;font-family:sans-serif;padding:24px'>Press Start Game →</div>"
                )
                res_html   = gr.HTML("")
                with gr.Row():
                    a_btn = gr.Button("✅ Allow",  variant="secondary", interactive=False)
                    f_btn = gr.Button("⚠️ Flag",   variant="secondary", interactive=False)
                    r_btn = gr.Button("🗑 Remove", variant="secondary", interactive=False)
                conf_sl = gr.Slider(0, 100, value=80, step=1,
                                    label="Confidence % (< 45 → human escalation)")
                g_state = gr.State({})

                s_btn.click(start_game, [t_dd, g_state],
                            [post_html, res_html, a_btn, f_btn, r_btn, g_state, prog_html])
                for btn, act in [(a_btn,"allow"),(f_btn,"flag"),(r_btn,"remove")]:
                    btn.click(lambda c,s,a=act: take_action(a,c,s),
                              [conf_sl, g_state],
                              [post_html, res_html, a_btn, f_btn, r_btn, g_state, prog_html])

            # ── Tab 2: Auto-Pilot ─────────────────────────────────────────────
            with gr.Tab("🤖 Auto-Pilot"):
                gr.Markdown("Watch an AI agent moderate posts. Includes confusion matrix & per-class F1.")
                with gr.Row():
                    ap_task  = gr.Dropdown(["easy","medium","hard"], value="hard", label="Task")
                    ap_agent = gr.Dropdown(
                        ["Rule-Based Agent","PPO Agent","LLM Agent"],
                        value="Rule-Based Agent", label="Agent"
                    )
                    ap_run   = gr.Button("▶ Run", variant="primary")
                ap_out = gr.HTML(
                    "<div style='color:#9ca3af;font-family:sans-serif;padding:18px'>Press Run →</div>"
                )
                ap_run.click(run_autopilot, [ap_task, ap_agent], [ap_out])

            # ── Tab 3: Training ───────────────────────────────────────────────
            with gr.Tab("📈 Training"):
                gr.Markdown("Train a PPO agent live. Chart shows score progression across all tasks.")
                with gr.Row():
                    tr_updates = gr.Slider(20, 300, value=60, step=10, label="PPO Updates")
                    tr_task    = gr.Dropdown(
                        ["curriculum","easy","medium","hard"],
                        value="curriculum", label="Training task"
                    )
                    tr_btn     = gr.Button("🚀 Train", variant="primary")
                tr_result  = gr.HTML("")
                tr_scores  = gr.State("{}")
                tr_plot    = gr.Plot(label="Training Curve")

                def train_and_plot(n, t):
                    result_html, scores_json = run_training(int(n), t)
                    fig = plot_training(scores_json)
                    return result_html, scores_json, fig

                tr_btn.click(
                    train_and_plot, [tr_updates, tr_task],
                    [tr_result, tr_scores, tr_plot]
                )

            # ── Tab 4: Dataset ────────────────────────────────────────────────
            with gr.Tab("📊 Dataset"):
                with gr.Tabs():
                    with gr.Tab("Explorer"):
                        gr.Markdown("Filter and browse all 41 posts with ground-truth labels.")
                        with gr.Row():
                            f_diff = gr.Dropdown(["All","easy","medium","hard"], value="All", label="Difficulty")
                            f_tag  = gr.Dropdown(["All","safe","nudity","violence","drugs","misleading"],
                                                 value="All", label="Image Tag")
                            f_user = gr.Dropdown(["All","new","trusted","suspicious"], value="All", label="User Type")
                            f_act  = gr.Dropdown(["All","allow","flag","remove"], value="All", label="Action")
                            f_btn  = gr.Button("🔍 Filter", variant="primary")
                        ds_out = gr.HTML()
                        f_btn.click(explore_dataset, [f_diff, f_tag, f_user, f_act], [ds_out])
                        demo.load(lambda: explore_dataset("All","All","All","All"), outputs=[ds_out])

                    with gr.Tab("Analytics"):
                        gr.Markdown("Distribution of labels across the dataset.")
                        an_btn = gr.Button("📊 Show Analytics", variant="primary")
                        an_out = gr.HTML()
                        an_btn.click(dataset_analytics, outputs=[an_out])

            # ── Tab 5: Leaderboard ────────────────────────────────────────────
            with gr.Tab("🏆 Leaderboard"):
                gr.Markdown("Compare all available agents. Run PPO training first to populate the PPO row.")
                lb_btn    = gr.Button("🏆 Run Leaderboard", variant="primary")
                lb_table  = gr.HTML("")
                lb_data   = gr.State("{}")
                lb_radar  = gr.Plot(label="Radar Chart")

                def leaderboard_and_radar():
                    table_html, radar_json = run_leaderboard()
                    fig = plot_radar(radar_json)
                    return table_html, radar_json, fig

                lb_btn.click(leaderboard_and_radar, outputs=[lb_table, lb_data, lb_radar])

            # ── Tab 6: About ──────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
## Multimodal Content Moderation Environment v2

**OpenEnv-compliant RL environment** — Meta × Hugging Face × PyTorch Hackathon.

### Architecture
| Component | Detail |
|-----------|--------|
| Feature extractor | 64-dim multimodal vector (image + user + text TF-IDF + cross-modal interactions) |
| Policy network | Deep MLP: 64→128→64→32 + Actor/Critic heads, LayerNorm, Dropout |
| Training algorithm | PPO-Clip with GAE(λ=0.95), entropy bonus, KL early stop, Adam + cosine LR |
| Reward | Multi-objective: accuracy + severity weighting + calibration bonus |

### Novel Features
- **Confidence-gated escalation** — models human review queue for uncertain cases
- **Severity-weighted rewards** — allowing nudity/violence penalised harder than misinfo
- **Cross-modal interaction features** — detects conflicts between image tag and text
- **Expected Calibration Error (ECE)** — second scoring dimension beyond accuracy
- **Fairness gap metric** — measures accuracy disparity across user trust types
- **Vectorised environment** — batch rollout collection for PPO

### Dataset
41 posts · 14 easy · 13 medium · 14 hard — including adversarial edge cases:
Arabic-language prayer, nurse harm-reduction content, coding drug sales as food emoji.

*Meta × Hugging Face × PyTorch OpenEnv Hackathon*
""")

    return demo


def _normalize_step_payload(step_body: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    payload = dict(step_body or {})
    body_env_id = payload.pop("env_id", None)

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
        raise ValueError("Step payload must include an 'action' field.")

    normalized = ActionModel.model_validate(candidate).model_dump(exclude_none=True)
    return normalized, body_env_id


def create_app(api_only: Optional[bool] = None) -> FastAPI:
    global gr, UI_THEME

    if api_only is None:
        api_only = _truthy_env("OPENENV_API_ONLY") if os.getenv("OPENENV_API_ONLY", "").strip() else False

    if not api_only and gr is None:
        try:
            import gradio as _gr  # type: ignore[import]
            gr = _gr  # type: ignore[assignment]
            UI_THEME = gr.themes.Soft(primary_hue="blue")
        except Exception:
            api_only = True

    ui = None if api_only else build_ui()
    ui_enabled = bool((not api_only) and (ui is not None))
    api = FastAPI(title=f"{ENV_TITLE} OpenEnv API", version="3.0.0")

    @api.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": ENV_NAME,
            "status": "ready",
            "docs": "/docs",
            "health": "/health",
            "healthz": "/healthz",
            "ready": "/ready",
            "ui": None if api_only else "/ui",
            "api_mode": "only" if api_only else "api+ui",
        }

    @api.get("/health")
    @api.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "name": ENV_NAME,
            "version": api.version,
            "ready": ENV_SERVICE.is_ready(),
            "active_sessions": ENV_SERVICE.active_sessions(),
        }

    @api.get("/ready")
    def readiness() -> JSONResponse:
        ready = ENV_SERVICE.is_ready()
        payload: Dict[str, Any] = {
            "status": "ready" if ready else "unready",
            "name": ENV_NAME,
            "version": api.version,
            "active_sessions": ENV_SERVICE.active_sessions(),
        }
        if not ready:
            payload["error"] = ENV_SERVICE.readiness_error() or "environment bootstrap failed"
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

    @api.get("/metadata")
    def metadata() -> Dict[str, Any]:
        return {
            "name": ENV_NAME,
            "title": ENV_TITLE,
            "description": ENV_DESCRIPTION,
            "version": api.version,
            "license": "MIT",
            "mode": "simulation",
            "tags": ["openenv", "content-moderation", "multimodal", "safety"],
            "tasks": list(TASKS.keys()),
            "ui_enabled": ui_enabled,
            "llm_available": LLM_AVAILABLE,
            "session_transport": {
                "header": "X-Env-Id",
                "body_field": "env_id",
                "query_field": "env_id",
                "close_endpoint": "/close",
                "default": EnvService.DEFAULT_ENV_ID,
                "idle_timeout_seconds": ENV_SERVICE.session_idle_timeout_s(),
            },
        }

    @api.get("/schema")
    def schema() -> Dict[str, Any]:
        action_schema = ActionModel.model_json_schema()
        flat_props = dict(action_schema.get("properties", {}))
        return {
            "action": action_schema,
            "step_request": {
                "type": "object",
                "description": (
                    "Accepts canonical OpenEnv form {action: {...}} and flat form "
                    "{action: 'flag', confidence: 0.8}. Optional env_id may be sent in body, "
                    "query param, or X-Env-Id header."
                ),
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": action_schema,
                            "env_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        },
                    },
                    {
                        "type": "object",
                        "required": action_schema.get("required", ["action"]),
                        "properties": {
                            **flat_props,
                            "env_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        },
                    },
                ],
            },
            "observation": ObservationModel.model_json_schema(),
            "state": {
                "type": "object",
                "description": (
                    "Current observation without advancing, or an empty object when "
                    "the environment is not initialized or the episode is complete."
                ),
                "oneOf": [
                    ObservationModel.model_json_schema(),
                    {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ],
            },
        }

    @api.post("/mcp")
    def mcp(_: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "result": {
                "name": ENV_NAME,
                "status": "ready",
                "mode": "simulation",
            },
        }

    @api.post("/reset", response_model=ResetResponseModel)
    def reset_environment(
        response: Response,
        request: ResetRequest = Body(default_factory=ResetRequest),
        x_env_id: Optional[str] = Header(default=None, alias="X-Env-Id"),
    ) -> ResetResponseModel:
        session_id = request.env_id or x_env_id
        if session_id and session_id.strip().lower() in {"new", "auto"}:
            session_id = f"session-{uuid.uuid4().hex[:12]}"
        env_id, observation = ENV_SERVICE.reset(
            task=request.task,
            seed=request.seed,
            max_steps=request.max_steps,
            env_id=session_id,
        )
        response.headers["X-Env-Id"] = env_id
        return ResetResponseModel(observation=observation, reward=None, done=False)

    @api.get("/state")
    @api.post("/state")
    def current_state(
        env_id: Optional[str] = Query(default=None),
        x_env_id: Optional[str] = Header(default=None, alias="X-Env-Id"),
    ) -> Dict[str, Any]:
        return ENV_SERVICE.state(env_id=env_id or x_env_id)

    @api.post("/close")
    @api.delete("/close")
    def close_environment(
        request: CloseSessionRequest = Body(default_factory=CloseSessionRequest),
        env_id: Optional[str] = Query(default=None),
        x_env_id: Optional[str] = Header(default=None, alias="X-Env-Id"),
    ) -> Dict[str, Any]:
        resolved_env_id = request.env_id or env_id or x_env_id
        closed = ENV_SERVICE.close(env_id=resolved_env_id)
        return {
            "env_id": (resolved_env_id or EnvService.DEFAULT_ENV_ID),
            "closed": closed,
        }

    @api.post("/step", response_model=StepResponseModel)
    def step_environment(
        response: Response,
        action: Dict[str, Any] = Body(default_factory=dict),
        env_id: Optional[str] = Query(default=None),
        x_env_id: Optional[str] = Header(default=None, alias="X-Env-Id"),
    ) -> StepResponseModel:
        try:
            payload, body_env_id = _normalize_step_payload(action)
            resolved_env_id = body_env_id or env_id or x_env_id
            if resolved_env_id and str(resolved_env_id).strip().lower() in {"new", "auto"}:
                resolved_env_id = f"session-{uuid.uuid4().hex[:12]}"
            result = ENV_SERVICE.step(payload, env_id=resolved_env_id)
            resolved_header_env_id = (
                str(resolved_env_id).strip() if resolved_env_id else EnvService.DEFAULT_ENV_ID
            )
            response.headers["X-Env-Id"] = resolved_header_env_id or EnvService.DEFAULT_ENV_ID
            return StepResponseModel(
                observation=result["observation"],
                reward=result["reward"],
                done=result["done"],
            )
        except ValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if api_only or gr is None or ui is None:
        return api
    return gr.mount_gradio_app(api, ui, path="/ui", theme=UI_THEME, css=UI_CSS)


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
