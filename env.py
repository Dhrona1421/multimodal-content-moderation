"""
env.py — Advanced Multimodal Content Moderation Environment (v2).

OpenEnv-compliant RL environment with:
  • Multi-objective reward decomposition  (accuracy + calibration + severity)
  • Confidence-gated human escalation
  • Contextual action masking
  • Rolling per-user-type history with recency decay
  • Severity-weighted penalties (false negative on CSAM/violence > spam)
  • Full explainability dict per step
  • Vectorised reset/step for batch training
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from features import FEATURE_DIM, ACTIONS, extract_features
from schemas import (
    ActionModel,
    ObservationModel,
    RewardModel,
    SessionStatsModel,
    StepInfoModel,
)

# ─────────────────────────────────────────────────────────────────────────────
# Reward configuration
# ─────────────────────────────────────────────────────────────────────────────

# Base reward matrix (correct_action, agent_action) → reward
BASE_REWARD: Dict[Tuple[str, str], float] = {
    ("allow",  "allow"):  +1.00,
    ("allow",  "flag"):   -0.50,
    ("allow",  "remove"): -1.00,
    ("flag",   "flag"):   +1.00,
    ("flag",   "allow"):  -1.00,
    ("flag",   "remove"): +0.50,
    ("remove", "remove"): +1.00,
    ("remove", "flag"):   +0.50,
    ("remove", "allow"):  -1.00,
}

# Severity multipliers applied when agent allows harmful content
SEVERITY: Dict[str, float] = {
    "nudity":      1.5,
    "violence":    1.4,
    "drugs":       1.3,
    "misleading":  1.2,
    "safe":        1.0,
}

# Confidence calibration reward bonus/penalty
# Agent gets partial credit for honest uncertainty on hard cases
CONF_THRESHOLD = 0.45   # below → escalate to human

HIGH_RISK_IMAGES = {"nudity", "drugs", "violence"}


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class ContentModerationEnv:
    """
    Advanced OpenEnv-compliant content moderation environment.

    Observation dict keys
    ─────────────────────
    post_id       int    — unique identifier
    text          str    — post caption
    image_tag     str    — safe | nudity | violence | drugs | misleading
    user_type     str    — new | trusted | suspicious
    difficulty    str    — easy | medium | hard
    step          int    — 1-indexed current step
    max_steps     int    — configurable per episode (default 12)
    user_history  int    — rolling weighted flags on this user_type
    session_stats dict   — {correct, wrong, flagged, removed} so far
    features      ndarray— pre-computed 64-dim feature vector

    Action space:  allow | flag | remove
    Reward range:  [-1.5, +1.1]  (severity-adjusted, clipped)
    """

    ACTIONS    = ACTIONS
    N_ACTIONS  = len(ACTIONS)
    ObservationModel = ObservationModel
    ActionModel = ActionModel
    RewardModel = RewardModel
    StepInfoModel = StepInfoModel
    API_VERSION = "3.0.0"

    def __init__(
        self,
        dataset_path: str  = "moderation_dataset.json",
        task:         str  = "medium",
        max_steps:    int  = 12,
        seed:         Optional[int] = None,
        severity_scale: float = 0.3,   # how much severity multiplies penalty
        calib_weight:   float = 0.15,  # reward weight for calibration
    ):
        self.dataset_path   = dataset_path
        self.task           = task
        self.max_steps      = max_steps
        self.seed           = seed
        self.severity_scale = severity_scale
        self.calib_weight   = calib_weight

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self._load_dataset()
        self._build_task_pool()

        # episode state
        self.current_step:    int                  = 0
        self.episode_posts:   List[Dict]           = []
        self.episode_rewards: List[float]          = []
        self.episode_info:    List[Dict]           = []
        self.user_history:    Dict[str, deque]     = {}
        self.session_stats:   Dict[str, int]       = {}
        self.done:            bool                 = True

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_dataset(self) -> None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, self.dataset_path)
        with open(path, encoding="utf-8") as fh:
            self.full_dataset: List[Dict] = json.load(fh)

    def _build_task_pool(self) -> None:
        diff_map = {
            "easy":   {"easy"},
            "medium": {"medium"},
            "hard":   {"hard"},
        }
        allowed = diff_map.get(self.task, {"easy", "medium", "hard"})
        self.task_pool = [p for p in self.full_dataset if p["difficulty"] in allowed]

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Reset episode and return first observation."""
        self.current_step  = 0
        self.episode_rewards = []
        self.episode_info    = []
        self.user_history    = {}      # user_type → deque of recent actions
        self.session_stats   = {"correct": 0, "wrong": 0, "flagged": 0, "removed": 0}
        self.done            = False

        pool = list(self.task_pool)
        if len(pool) >= self.max_steps:
            selected_posts = self._rng.sample(pool, self.max_steps)
        else:
            mult = (self.max_steps // len(pool)) + 1
            selected_posts = (pool * mult)[:self.max_steps]
            self._rng.shuffle(selected_posts)

        # Build deterministic per-episode text variants so the environment is
        # not a pure static replay while remaining reproducible by seed.
        self.episode_posts = [
            self._materialize_post_variant(post, slot_idx)
            for slot_idx, post in enumerate(selected_posts)
        ]

        return self._build_obs(self.current_step)

    def step(
        self,
        action:     Union[str, Dict[str, Any], ActionModel],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one moderation decision.

        Args:
            action: canonical OpenEnv action payload, for example
                    {"action": "flag", "confidence": 0.8, "agent_reasoning": {...}}.
                    Legacy string actions are still accepted for backwards
                    compatibility.

        Returns:
            (next_obs, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode done — call reset() first.")
        action_model = self._coerce_action(action)
        action = action_model.action
        confidence = action_model.confidence
        agent_reasoning = action_model.agent_reasoning

        post       = self.episode_posts[self.current_step]
        correct    = post["correct_action"]
        image_tag  = post["image_tag"]
        user_type  = post["user_type"]
        difficulty = post["difficulty"]

        # ── Escalation path (low confidence) ─────────────────────────────────
        escalated = confidence < CONF_THRESHOLD
        if escalated:
            reward = self._escalation_reward(difficulty)
        else:
            reward = self._base_reward(action, correct, image_tag, user_type,
                                       confidence, difficulty)

        reward = RewardModel(value=float(np.clip(reward, -1.5, 1.1))).value

        # ── Session bookkeeping ───────────────────────────────────────────────
        is_correct = (action == correct) and not escalated
        self.session_stats["correct" if is_correct else "wrong"] += 1
        if action == "flag":
            self.session_stats["flagged"] += 1
        elif action == "remove":
            self.session_stats["removed"] += 1

        # rolling history (last 5 decisions per user_type, recency-decayed)
        if user_type not in self.user_history:
            self.user_history[user_type] = deque(maxlen=5)
        self.user_history[user_type].append(1 if action in ("flag", "remove") else 0)

        self.episode_rewards.append(reward)

        # ── Build info dict ───────────────────────────────────────────────────
        info: Dict[str, Any] = {
            "post_id":        post["id"],
            "correct_action": correct,
            "agent_action":   action,
            "confidence":     confidence,
            "agent_reasoning": agent_reasoning,
            "escalated":      escalated,
            "reward":         reward,
            "is_correct":     is_correct,
            "reason":         post["reason"],
            "variant_applied": bool(post.get("variant_applied", False)),
            "image_tag":      image_tag,
            "user_type":      user_type,
            "difficulty":     difficulty,
            "user_history_val": self._weighted_history(user_type),
            "session_stats":  dict(self.session_stats),
        }

        self.episode_info.append(info)
        self.current_step += 1
        self.done = (self.current_step >= self.max_steps)

        if self.done:
            info["episode_reward"] = round(sum(self.episode_rewards), 4)
            info["episode_score"]  = self.compute_score()
            info["episode_metrics"] = self._compute_episode_metrics()
            # Return current obs even on done for spec-strictness
            next_obs = self._build_obs(self.current_step - 1)
        else:
            next_obs = self._build_obs(self.current_step)

        validated_info = StepInfoModel(**info).model_dump()
        return next_obs, reward, self.done, validated_info

    def state(self) -> Dict[str, Any]:
        """Return current observation without advancing state (OpenEnv spec)."""
        if self.done:
            return {}
        return self._build_obs(self.current_step)

    # ── Reward computation ────────────────────────────────────────────────────

    def _base_reward(
        self,
        action:     str,
        correct:    str,
        image_tag:  str,
        user_type:  str,
        confidence: float,
        difficulty: str,
    ) -> float:
        # 1. Base accuracy reward
        r = BASE_REWARD[(correct, action)]

        # 2. Severity penalty amplifier on false negatives (allow harmful)
        if action == "allow" and correct != "allow":
            sev  = SEVERITY.get(image_tag, 1.0)
            r   -= self.severity_scale * (sev - 1.0)

        # 3. Contextual modifier
        r += self._context_modifier(action, correct, image_tag, user_type)

        # 4. Calibration bonus: high confidence on correct = small bonus
        if action == correct and confidence >= 0.7:
            r += self.calib_weight * min((confidence - 0.7) / 0.3, 1.0) * 0.1

        # 5. Over-removal penalty on trusted users with safe content
        if user_type == "trusted" and action == "remove" and correct == "allow":
            r -= 0.1

        return r

    def _escalation_reward(self, difficulty: str) -> float:
        """Partial reward for honest human escalation."""
        return {
            "easy":   -0.10,   # shouldn't need human for easy cases
            "medium": +0.20,   # acceptable to escalate medium
            "hard":   +0.30,   # smart to escalate hard adversarial cases
        }.get(difficulty, +0.15)

    def _context_modifier(
        self, action: str, correct: str,
        image_tag: str, user_type: str,
    ) -> float:
        mod = 0.0
        if action == correct:
            if user_type == "suspicious" and action in ("flag", "remove"):
                mod += 0.05
            if user_type == "trusted" and action == "allow":
                mod += 0.05
        else:
            if image_tag in HIGH_RISK_IMAGES and action == "allow":
                mod -= 0.05
            if user_type == "trusted" and action == "remove" and correct == "allow":
                mod -= 0.05
        return mod

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, idx: int) -> Dict[str, Any]:
        post = self.episode_posts[idx]
        wh   = self._weighted_history(post["user_type"])
        obs  = {
            "post_id":      post["id"],
            "text":         post["text"],
            "image_tag":    post["image_tag"],
            "user_type":    post["user_type"],
            "difficulty":   post["difficulty"],
            "step":         idx + 1,
            "max_steps":    self.max_steps,
            "user_history": wh,
            "session_stats": dict(self.session_stats),
        }
        obs["features"] = extract_features(obs).astype(np.float32).tolist()
        obs["session_stats"] = SessionStatsModel(**obs["session_stats"])
        return ObservationModel(**obs).model_dump()

    def _materialize_post_variant(self, post: Dict[str, Any], slot_idx: int) -> Dict[str, Any]:
        """
        Generate deterministic variants by difficulty tier.
        Easy remains close to canonical dataset examples.
        """
        variant = dict(post)
        difficulty = post.get("difficulty", self.task)
        original_text = str(post.get("text", ""))
        text = original_text

        if difficulty == "medium":
            if self._rng.random() < 0.35:
                text = self._inject_contextual_distractor(text)
            if self._rng.random() < 0.25:
                text = self._light_surface_noise(text)

        elif difficulty == "hard":
            if self._rng.random() < 0.65:
                text = self._inject_conflicting_clause(text, post.get("correct_action", "flag"))
            if self._rng.random() < 0.45:
                text = self._light_surface_noise(text)
            if self._rng.random() < 0.30:
                text = self._inject_hashtag_noise(text)

        variant["text"] = text
        if text != original_text:
            variant["reason"] = f"{post.get('reason', '')} [variant={slot_idx + 1}]"
            variant["variant_applied"] = True
        else:
            variant["variant_applied"] = False
        return variant

    def _inject_contextual_distractor(self, text: str) -> str:
        suffixes = [
            "Context note: this post is circulating in local groups.",
            "FYI this topic is currently trending in related communities.",
            "Shared for discussion; some details may be incomplete.",
        ]
        return f"{text} {self._rng.choice(suffixes)}"

    def _inject_conflicting_clause(self, text: str, correct_action: str) -> str:
        if correct_action == "allow":
            clauses = [
                "Without context, this could still be misread by viewers.",
                "Surface-level signals may look riskier than the underlying intent.",
            ]
        elif correct_action == "remove":
            clauses = [
                "Framed as humor, but still violating platform policy.",
                "Packaged as awareness language despite clearly harmful cues.",
            ]
        else:
            clauses = [
                "Signals conflict and likely require human review.",
                "Intent appears ambiguous across text and context.",
            ]
        return f"{text} {self._rng.choice(clauses)}"

    def _inject_hashtag_noise(self, text: str) -> str:
        tags = ["#context", "#viral", "#fyi", "#trending", "#awareness", "#justsaying"]
        n = 1 if self._rng.random() < 0.7 else 2
        chosen = " ".join(self._rng.sample(tags, n))
        return f"{text} {chosen}"

    def _light_surface_noise(self, text: str) -> str:
        # Conservative perturbations: keep semantics while reducing overfitting
        # to exact punctuation/casing.
        noisy = text
        noisy = re.sub(r"\s{2,}", " ", noisy)
        noisy = re.sub(r"\.\.\.+", "...", noisy)
        noisy = re.sub(r"!!!+", "!!", noisy)
        if self._rng.random() < 0.35:
            noisy = noisy.replace(" and ", " & ", 1)
        if self._rng.random() < 0.25:
            noisy = noisy.replace(" for ", " 4 ", 1)
        return noisy

    def _coerce_action(
        self,
        action: Union[str, Dict[str, Any], ActionModel],
    ) -> ActionModel:
        if isinstance(action, ActionModel):
            return action
        if isinstance(action, dict):
            payload = dict(action)
            payload["confidence"] = float(np.clip(payload.get("confidence", 1.0), 0.0, 1.0))
            return ActionModel(**payload)
        return ActionModel(action=action, confidence=1.0)

    def _weighted_history(self, user_type: str) -> float:
        """Recency-decayed flag count for user_type (0..5 range → normalised)."""
        if user_type not in self.user_history:
            return 0.0
        flags  = list(self.user_history[user_type])
        total  = sum(f * (0.8 ** i) for i, f in enumerate(reversed(flags)))
        return round(min(total, 5.0), 3)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def compute_score(self) -> float:
        """Normalise episode reward to [0.0, 1.0]."""
        if not self.episode_rewards:
            return 0.0
        total    = sum(self.episode_rewards)
        max_r    = self.max_steps * 1.1
        min_r    = self.max_steps * -1.5
        score    = (total - min_r) / (max_r - min_r)
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _compute_episode_metrics(self) -> Dict[str, Any]:
        """Detailed end-of-episode metrics."""
        from collections import Counter
        correct_count = sum(1 for i in self.episode_info if i["is_correct"])
        actions       = [i["agent_action"]   for i in self.episode_info]
        corrects      = [i["correct_action"] for i in self.episode_info]
        confidences   = [i["confidence"]     for i in self.episode_info]
        escalated     = sum(1 for i in self.episode_info if i["escalated"])

        # Per-difficulty accuracy
        diff_groups: Dict[str, List[bool]] = {}
        for info in self.episode_info:
            d = info["difficulty"]
            diff_groups.setdefault(d, []).append(info["is_correct"])

        diff_acc = {d: sum(v) / len(v) for d, v in diff_groups.items()}

        # Confusion matrix values
        tp_flag   = sum(1 for a, c in zip(actions, corrects) if a == "flag"   == c)
        tp_remove = sum(1 for a, c in zip(actions, corrects) if a == "remove" == c)
        tp_allow  = sum(1 for a, c in zip(actions, corrects) if a == "allow"  == c)
        false_neg = sum(1 for a, c in zip(actions, corrects) if a == "allow"  and c != "allow")

        return {
            "accuracy":          round(correct_count / max(len(self.episode_info), 1), 4),
            "correct":           correct_count,
            "escalated":         escalated,
            "false_negatives":   false_neg,
            "tp_per_action":     {"allow": tp_allow, "flag": tp_flag, "remove": tp_remove},
            "action_dist":       dict(Counter(actions)),
            "diff_accuracy":     diff_acc,
            "mean_confidence":   round(float(np.mean(confidences)), 4),
            "episode_score":     self.compute_score(),
        }

    # ── Properties & rendering ────────────────────────────────────────────────

    @property
    def action_space(self) -> List[str]:
        return list(self.ACTIONS)

    @property
    def action_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(self.ACTIONS),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0,
                    "description": (
                        "Agent certainty. Values below "
                        f"{CONF_THRESHOLD:.2f} trigger human escalation."
                    ),
                },
            },
        }

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "post_id":      "int",
            "text":         "str",
            "image_tag":    f"Categorical({', '.join(['safe','nudity','violence','drugs','misleading'])})",
            "user_type":    f"Categorical({', '.join(['new','trusted','suspicious'])})",
            "difficulty":   f"Categorical({', '.join(['easy','medium','hard'])})",
            "step":         f"int [1, {self.max_steps}]",
            "max_steps":    f"int = {self.max_steps}",
            "user_history": "float [0, 5]",
            "session_stats":"object {correct, wrong, flagged, removed}",
            "features":     f"ndarray ({FEATURE_DIM},) ∈ [0,1]",
        }

    @property
    def reward_space(self) -> Dict[str, float]:
        return {
            "min": -1.5,
            "max": 1.1,
        }

    @property
    def contract(self) -> Dict[str, Any]:
        return {
            "api_version": self.API_VERSION,
            "reset_returns": "ObservationModel",
            "state_returns": "ObservationModel | {}",
            "step_accepts": "ActionModel-compatible payload",
            "step_returns": "(ObservationModel | {}, reward, done, StepInfoModel)",
            "action_schema": self.action_schema,
            "observation_space": self.observation_space,
            "reward_space": self.reward_space,
            "done_condition": f"current_step >= {self.max_steps}",
        }

    def render(self) -> str:
        if self.done:
            return "[Episode complete]"
        obs = self._build_obs(self.current_step)
        sep = "═" * 68
        stats = self.session_stats
        return (
            f"\n{sep}\n"
            f"  Step {obs['step']:>2}/{obs['max_steps']}  │  "
            f"Post #{obs['post_id']:>3}  │  "
            f"Difficulty: {obs['difficulty'].upper():<6}  │  "
            f"✓{stats['correct']} ✗{stats['wrong']}\n"
            f"{sep}\n"
            f"  TEXT  : {obs['text'][:105]}{'…' if len(obs['text'])>105 else ''}\n"
            f"  IMAGE : [{obs['image_tag'].upper():<11}]  "
            f"USER: [{obs['user_type'].upper():<10}]  "
            f"HIST: {obs['user_history']:.2f}\n"
            f"{sep}\n"
            f"  Actions: allow | flag | remove   "
            f"(confidence < 0.45 → human escalation)\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised environment wrapper for batch training
# ─────────────────────────────────────────────────────────────────────────────

class VecContentModerationEnv:
    """
    Run N independent ContentModerationEnv instances in lockstep.
    Enables batched rollout collection for PPO.
    """

    def __init__(self, n_envs: int, **kwargs):
        self.n_envs = n_envs
        base_seed = kwargs.pop("seed", 42)
        self.envs   = [ContentModerationEnv(**kwargs, seed=base_seed + i)
                       for i in range(n_envs)]
        self.obs_list: List[Dict] = []

    def reset(self) -> List[Dict]:
        self.obs_list = [env.reset() for env in self.envs]
        return self.obs_list

    def step(
        self,
        actions:     List[Union[str, Dict[str, Any], ActionModel]],
        confidences: Optional[List[float]] = None,
    ) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        if confidences is None:
            confidences = [1.0] * self.n_envs

        results = [
            env.step(
                a if isinstance(a, (dict, ActionModel)) else {"action": a, "confidence": c}
            )
            for env, a, c in zip(self.envs, actions, confidences)
        ]

        new_obs = []
        for i, (obs, r, done, info) in enumerate(results):
            if done:
                obs = self.envs[i].reset()
            new_obs.append(obs)

        self.obs_list = new_obs
        obs_list    = [r[0] for r in results]
        rewards     = [r[1] for r in results]
        dones       = [r[2] for r in results]
        infos       = [r[3] for r in results]
        return new_obs, rewards, dones, infos


if __name__ == "__main__":
    env  = ContentModerationEnv(task="hard", seed=0)
    obs  = env.reset()
    print(env.render())
    print(f"Feature dim: {len(obs['features'])}")
    print(f"Obs space:   {env.observation_space}")
    for step in range(env.max_steps):
        action = random.choice(["allow", "flag", "remove"])
        obs, r, done, info = env.step({"action": action, "confidence": 0.7})
        print(f"  step={step+1}  action={action:<6}  reward={r:+.3f}  correct={info['correct_action']}")
        if done:
            print(f"\nEpisode score: {info['episode_score']:.4f}")
            print(f"Metrics: {info['episode_metrics']}")
            break
