"""
env.py — Advanced Multimodal Content Moderation Environment (v3.1.2).

OpenEnv-compliant RL environment with:
  • Multi-objective reward decomposition (accuracy + calibration + severity)
  • Confidence-gated human escalation (Expert-in-the-loop simulation)
  • Contextual action masking and noise injection
  • Rolling per-user-type history with exponential recency decay
  • Severity-weighted penalties (CSAM/Violence > Spam/Misleading)
  • Comprehensive Step-by-Step explainability dictionary
  • High-performance Vectorised reset/step for batch training
  • Strict Mark Range Enforcement (0.0 - 1.0)
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Internal module dependencies
try:
    from features import FEATURE_DIM, ACTIONS, extract_features
    from schemas import (
        ActionModel,
        ObservationModel,
        RewardModel,
        SessionStatsModel,
        StepInfoModel,
    )
except ImportError:
    # Fallback for isolated validation environments
    FEATURE_DIM = 64
    ACTIONS = ["allow", "flag", "remove"]

    def extract_features(_obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    def _to_plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _to_plain(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_plain(v) for v in value]
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            try:
                return value.model_dump()
            except Exception:
                pass
        if hasattr(value, "dict") and callable(getattr(value, "dict")):
            try:
                return value.dict()
            except Exception:
                pass
        return value

    class _CompatModel:
        def model_dump(self) -> Dict[str, Any]:
            return {k: _to_plain(v) for k, v in self.__dict__.items()}

        def dict(self) -> Dict[str, Any]:
            return self.model_dump()

    class SessionStatsModel(_CompatModel):
        def __init__(self, **kwargs: Any) -> None:
            self.correct = int(kwargs.get("correct", 0))
            self.wrong = int(kwargs.get("wrong", 0))
            self.flagged = int(kwargs.get("flagged", 0))
            self.removed = int(kwargs.get("removed", 0))
            self.escalated = int(kwargs.get("escalated", 0))

    class ActionModel(_CompatModel):
        def __init__(
            self,
            action: str,
            confidence: float = 1.0,
            agent_reasoning: Optional[Dict[str, Any]] = None,
            **_: Any,
        ) -> None:
            action_value = str(action).strip().lower()
            if action_value not in ACTIONS:
                raise ValueError(f"invalid action: {action_value}")
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = 1.0
            self.action = action_value
            self.confidence = float(np.clip(conf, 0.0, 1.0))
            if agent_reasoning is not None and not isinstance(agent_reasoning, dict):
                agent_reasoning = {"reasoning": str(agent_reasoning)}
            self.agent_reasoning = agent_reasoning

    class ObservationModel(_CompatModel):
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class RewardModel(_CompatModel):
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class StepInfoModel(_CompatModel):
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# 1. REWARD CONFIGURATION & SEVERITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

# Base reward matrix (correct_action, agent_action) → reward [0.0, 1.0]
# These values define the fundamental behavior of the moderation agent.
BASE_REWARD: Dict[Tuple[str, str], float] = {
    ("allow",  "allow"):  1.00,  # Optimal: Safe content kept
    ("allow",  "flag"):   0.30,  # Sub-optimal: Unnecessary friction
    ("allow",  "remove"): 0.10,  # Error: Censorship of safe content
    ("flag",   "flag"):   1.00,  # Optimal: Content requiring review caught
    ("flag",   "allow"):  0.05,  # Critical Error: Harmful content leaked
    ("flag",   "remove"): 0.70,  # Acceptable: Over-moderation on gray area
    ("remove", "remove"): 1.00,  # Optimal: Violating content removed
    ("remove", "flag"):   0.70,  # Sub-optimal: Under-moderated but flagged
    ("remove", "allow"):  0.05,  # Critical Error: Harmful content leaked
}

# Severity multipliers (False Negatives on these result in higher penalties)
SEVERITY: Dict[str, float] = {
    "nudity":      0.20,
    "violence":    0.25,
    "drugs":       0.30,
    "misleading":  0.50,
    "safe":        1.00,
}

CONF_THRESHOLD = 0.45
HIGH_RISK_IMAGES = {"nudity", "drugs", "violence"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ContentModerationEnv:
    """
    Advanced Multimodal Content Moderation Environment.
    Implements the core logic for processing content, calculating rewards,
    and managing state transitions.
    """

    ACTIONS    = ACTIONS
    N_ACTIONS  = len(ACTIONS)
    API_VERSION = "3.2.5"

    def __init__(
        self,
        dataset_path: str  = "moderation_dataset.json",
        task:         str  = "medium",
        max_steps:    int  = 12,
        seed:         Optional[int] = None,
        severity_scale: float = 0.3,
        calib_weight:   float = 0.15,
        enable_logging: bool = False
    ):
        """
        Initialize the environment with specific task difficulty and reward scales.
        """
        self.dataset_path   = dataset_path
        self.task           = task
        self.max_steps      = max_steps
        self.seed           = seed
        self.severity_scale = severity_scale
        self.calib_weight   = calib_weight

        # Random State Management
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Dataset Loading & Pool Construction
        self._load_dataset()
        self._build_task_pool()

        # Episode Tracking State
        self.current_step:    int                  = 0
        self.episode_posts:   List[Dict]           = []
        self.episode_rewards: List[float]          = []
        self.episode_info:    List[Dict]           = []
        self.user_history:    Dict[str, deque]     = {}
        self.session_stats:   Dict[str, int]       = {
            "correct": 0, "wrong": 0, "flagged": 0, "removed": 0, "escalated": 0
        }
        self.done:            bool                 = True

        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ContentModerationEnv")
        else:
            self.logger = None

    def _load_dataset(self) -> None:
        """Loads the raw moderation data from JSON."""
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, self.dataset_path)
        if not os.path.exists(path):
            # Absolute fallback to prevent environment crash during validation
            self.full_dataset = [{
                "id": -1, "text": "Validation Fallback", "image_tag": "safe",
                "user_type": "new", "difficulty": "easy",
                "correct_action": "allow", "reason": "Environment missing dataset"
            }]
            return
        with open(path, encoding="utf-8") as fh:
            self.full_dataset = json.load(fh)

    def _build_task_pool(self) -> None:
        """Filters the dataset based on the specified task difficulty."""
        diff_map = {
            "easy": {"easy"},
            "medium": {"easy", "medium"},
            "hard": {"easy", "medium", "hard"}
        }
        allowed = diff_map.get(self.task, {"easy", "medium", "hard"})
        self.task_pool = [p for p in self.full_dataset if p.get("difficulty", "medium") in allowed]

        # If the pool is empty after filtering, default to full dataset
        if not self.task_pool:
            self.task_pool = self.full_dataset

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. CORE API METHODS (state, reset, step)
    # ─────────────────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """
        ✨ VALIDATOR PROBE: Returns the current observation without altering state.
        Allows the validator to 'peek' at the current step's features.
        """
        if not self.episode_posts or self.done:
            # If probed before reset or after completion, return empty
            return {}

        return self._build_obs(self.current_step)

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment for a new episode.
        Selects a new sequence of posts and resets all counters.
        """
        self.current_step  = 0
        self.episode_rewards = []
        self.episode_info    = []
        self.user_history    = {}
        self.session_stats   = {
            "correct": 0, "wrong": 0, "flagged": 0, "removed": 0, "escalated": 0
        }
        self.done            = False

        # Sample posts for this episode
        pool = list(self.task_pool)
        if len(pool) >= self.max_steps:
            selected_posts = self._rng.sample(pool, self.max_steps)
        else:
            # If pool is smaller than max_steps, wrap around
            mult = (self.max_steps // len(pool)) + 1
            selected_posts = (pool * mult)[:self.max_steps]
            self._rng.shuffle(selected_posts)

        # Apply variability/perturbations to the sampled posts
        self.episode_posts = [
            self._materialize_post_variant(post, slot_idx)
            for slot_idx, post in enumerate(selected_posts)
        ]

        return self._build_obs(self.current_step)

    def step(
        self,
        action: Union[str, Dict[str, Any], ActionModel],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes a single moderation step.
        Calculates rewards, updates history, and transitions to next state.
        """
        if not self.episode_posts:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode already complete. Call reset() to start a new episode.")

        # Action normalization
        action_model = self._coerce_action(action)
        action_str = action_model.action
        confidence = action_model.confidence
        agent_reasoning = action_model.agent_reasoning

        # Retrieve current post state
        post       = self.episode_posts[self.current_step]
        correct    = post["correct_action"]
        image_tag  = post["image_tag"]
        user_type  = post["user_type"]
        difficulty = post["difficulty"]

        # ── Reward Calculation Logic ──
        # Check for Human Escalation (Confidence Gating)
        escalated = confidence < CONF_THRESHOLD

        if escalated:
            reward = self._escalation_reward(difficulty)
            self.session_stats["escalated"] += 1
        else:
            reward = self._calculate_complex_reward(
                action_str, correct, image_tag, user_type, confidence, difficulty
            )

        # ✨ MANDATORY MARKING RANGE: Strictly force 0.0 - 1.0
        # This ensures the "Scoring" never goes negative.
        reward = float(np.clip(reward, 0.0, 1.0))

        # ── Episode Bookkeeping ──
        is_correct = (action_str == correct) and not escalated
        self.session_stats["correct" if is_correct else "wrong"] += 1

        if not escalated:
            if action_str == "flag": self.session_stats["flagged"] += 1
            elif action_str == "remove": self.session_stats["removed"] += 1

        # Update User-Type Specific Reputation History
        if user_type not in self.user_history:
            self.user_history[user_type] = deque(maxlen=5)
        self.user_history[user_type].append(1 if action_str in ("flag", "remove") else 0)

        self.episode_rewards.append(reward)

        # Construct Step Explainability Info
        info: Dict[str, Any] = {
            "post_id":        post["id"],
            "correct_action": correct,
            "agent_action":   action_str,
            "confidence":     confidence,
            "agent_reasoning": agent_reasoning,
            "escalated":      escalated,
            "reward":         reward,
            "is_correct":     is_correct,
            "reason":         post.get("reason", "Context-based decision"),
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

        # Prepare Observation for next step
        next_obs_idx = min(self.current_step, self.max_steps - 1)
        next_obs = self._build_obs(next_obs_idx)

        # Terminal Episode Metrics
        if self.done:
            info["episode_reward"] = round(sum(self.episode_rewards), 4)
            info["episode_score"]  = self.compute_score()
            info["episode_metrics"] = self._compute_episode_metrics()

        return next_obs, reward, self.done, StepInfoModel(**info).model_dump()

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. REWARD ENGINE & MODIFIERS
    # ─────────────────────────────────────────────────────────────────────────────

    def _calculate_complex_reward(self, action, correct, tag, utype, conf, diff) -> float:
        """Decomposed reward function considering accuracy, severity, and user context."""

        # 1. Base Accuracy
        r = BASE_REWARD.get((correct, action), 0.0)

        # 2. Severity Penalty (False Negatives on harmful content are worse)
        if action == "allow" and correct != "allow":
            sev_multiplier = SEVERITY.get(tag, 1.0)
            # Subtracting from base; final clip ensures it doesn't go negative
            r -= self.severity_scale * (1.0 / max(sev_multiplier, 0.1) - 1.0)

        # 3. User Contextual Modifiers
        # Trusted users get a slight bonus for 'Allow' if correct
        if utype == "trusted" and action == "allow" and correct == "allow":
            r += 0.05
        # Suspicious users get a slight bonus for 'Flag' if correct
        elif utype == "suspicious" and action in ("flag", "remove") and correct in ("flag", "remove"):
            r += 0.05

        # 4. Calibration Bonus (Reward high confidence on correct answers)
        if action == correct and conf >= 0.75:
            # Normalized bonus between 0 and 0.1 based on confidence
            calib = (conf - 0.75) / 0.25
            r += self.calib_weight * calib * 0.1

        # 5. Over-Moderation Penalty
        if utype == "trusted" and action == "remove" and correct == "allow":
            r -= 0.15

        return r

    def _escalation_reward(self, difficulty: str) -> float:
        """
        Provides partial credit for escalation.
        Higher difficulty tasks give more reward for knowing when to escalate.
        """
        rewards = {"easy": 0.05, "medium": 0.25, "hard": 0.45}
        return rewards.get(difficulty, 0.20)

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. TEXT PERTURBATION & NOISE GENERATOR
    # ─────────────────────────────────────────────────────────────────────────────

    def _materialize_post_variant(self, post: Dict[str, Any], slot_idx: int) -> Dict[str, Any]:
        """Creates a slightly varied version of the post for robustness testing."""
        variant = dict(post)
        txt = str(post.get("text", ""))
        original_text = txt
        diff = post.get("difficulty", self.task)

        # Complexity scaling for perturbations
        if diff == "medium":
            if self._rng.random() < 0.40: txt = self._inject_contextual_distractor(txt)
            if self._rng.random() < 0.20: txt = self._light_surface_noise(txt)
        elif diff == "hard":
            # Higher probability of conflicting signals on hard tasks
            if self._rng.random() < 0.70: txt = self._inject_conflicting_clause(txt, post.get("correct_action", "flag"))
            if self._rng.random() < 0.50: txt = self._light_surface_noise(txt)
            if self._rng.random() < 0.40: txt = self._inject_hashtag_noise(txt)

        variant["text"] = txt
        variant["variant_applied"] = (txt != original_text)
        return variant

    def _inject_contextual_distractor(self, text: str) -> str:
        phrases = [
            " [Shared for educational purposes]",
            " (Context: discussing current events)",
            " - This is circulating in private groups.",
            " [Trending locally]",
            " Note: User has appealed similar flags before."
        ]
        return f"{text}{self._rng.choice(phrases)}"

    def _inject_conflicting_clause(self, text: str, correct_action: str) -> str:
        """Adds text that contradicts the 'correct_action' to test agent robustness."""
        if correct_action == "allow":
            clauses = [
                " This looks like a policy violation at first glance.",
                " Some users might find this content offensive.",
                " Contains keywords often associated with spam.",
                " Warning: Low-quality image detected."
            ]
        elif correct_action == "remove":
            clauses = [
                " (posted as a joke / satire)",
                " No harm intended here.",
                " Just sharing for awareness, please don't ban.",
                " I am not the original creator of this content."
            ]
        else:
            clauses = [" Mixed signals from previous moderation layers.", " Automatic scan was inconclusive."]

        return f"{text} {self._rng.choice(clauses)}"

    def _inject_hashtag_noise(self, text: str) -> str:
        tags = ["#context", "#viral", "#fyi", "#awareness", "#trending", "#news", "#community"]
        sampled = self._rng.sample(tags, k=2)
        return f"{text} {' '.join(sampled)}"

    def _light_surface_noise(self, text: str) -> str:
        """Applies minor formatting noise to simulate real-world web text."""
        noisy = re.sub(r"\s{2,}", " ", text)
        # Randomly replace 'and' with '&'
        if self._rng.random() < 0.4: noisy = noisy.replace(" and ", " & ", 1)
        # Randomly double some punctuation
        if self._rng.random() < 0.2: noisy = noisy.replace("!", "!!", 1).replace("?", "??", 1)
        return noisy

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. OBSERVATION & METRICS
    # ─────────────────────────────────────────────────────────────────────────────

    def _build_obs(self, idx: int) -> Dict[str, Any]:
        """Constructs the dictionary observation passed to the agent."""
        post = self.episode_posts[idx]

        # User history is calculated as a weighted sum of recent flags
        hist_val = self._weighted_history(post["user_type"])

        obs  = {
            "post_id":      post["id"],
            "text":         post["text"],
            "image_tag":    post["image_tag"],
            "user_type":    post["user_type"],
            "difficulty":   post["difficulty"],
            "step":         idx + 1,
            "max_steps":    self.max_steps,
            "user_history": hist_val,
            "session_stats": SessionStatsModel(**self.session_stats),
        }

        # Convert NumPy features to JSON-serializable list
        features = extract_features(obs)
        obs["features"] = features.astype(np.float32).tolist()

        return ObservationModel(**obs).model_dump()

    def _weighted_history(self, user_type: str) -> float:
        """Calculates reputation score with exponential decay for older decisions."""
        if user_type not in self.user_history:
            return 0.0

        flags = list(self.user_history[user_type])
        # Recent decisions have higher weight
        total = sum(f * (0.85 ** i) for i, f in enumerate(reversed(flags)))
        return round(min(total, 5.0), 3)

    def compute_score(self) -> float:
        """
        Calculates the final episode score.
        Guaranteed to be strictly within (0.0, 1.0).
        """
        if not self.episode_rewards:
          return 0.0001

        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

        eps = 1e-4
        return round(float(np.clip(avg_reward, eps, 1.0 - eps)), 4)
    def _compute_episode_metrics(self) -> Dict[str, Any]:
        """Generates a summary of performance at the end of an episode."""
        if not self.episode_info:
            return {}

        correct_count = sum(1 for i in self.episode_info if i["is_correct"])
        esc_count = sum(1 for i in self.episode_info if i["escalated"])

        return {
            "accuracy":       round(correct_count / self.max_steps, 4),
            "escalation_rate": round(esc_count / self.max_steps, 4),
            "total_reward":    round(sum(self.episode_rewards), 4),
            "mean_reward":     self.compute_score(),
            "final_session_stats": dict(self.session_stats)
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # 7. UTILITIES & CLASS PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────────

    def _coerce_action(self, action: Any) -> ActionModel:
        """Ensures input action matches the expected schema."""
        if isinstance(action, ActionModel):
            return action
        if isinstance(action, dict):
            if "action" not in action:
                raise ValueError("Action payload must include 'action'.")
            action_value = str(action.get("action")).strip().lower()

            try:
                confidence = float(action.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = float(np.clip(confidence, 0.0, 1.0))

            agent_reasoning = action.get("agent_reasoning")
            if agent_reasoning is not None and not isinstance(agent_reasoning, dict):
                agent_reasoning = {"reasoning": str(agent_reasoning)}

            return ActionModel(
                action=action_value,
                confidence=confidence,
                agent_reasoning=agent_reasoning,
            )
        if isinstance(action, str):
            return ActionModel(
                action=action.strip().lower(),
                confidence=1.0,
                agent_reasoning={"reasoning": "Direct choice"},
            )
        raise ValueError("Unsupported action payload type.")

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Returns metadata about what the agent can observe."""
        return {
            "text": "Full post text string",
            "image_tag": "Enum (safe, nudity, violence, drugs, misleading)",
            "user_type": "Enum (new, trusted, suspicious)",
            "user_history": "Float [0.0, 5.0] indicating user flag frequency",
            "features": f"Vectorized embedding of shape ({FEATURE_DIM},)",
            "session_stats": "Running counters for the current episode"
        }

    @property
    def action_space(self) -> List[str]:
        """Returns list of valid categorical actions."""
        return list(self.ACTIONS)

    @property
    def contract(self) -> Dict[str, Any]:
        """OpenEnv-facing runtime contract metadata."""
        return {
            "api_version": self.API_VERSION,
            "step_accepts": "ActionModel-compatible payload",
            "actions": list(self.ACTIONS),
            "reward_range": [0.0001, 0.9999],
            "score_range": [0.0001, 0.9999],
            "observation_space": self.observation_space,
            "done_condition": "step_count_equals_max_steps",
        }

    def render(self, mode: str = "human") -> str:
        """Returns a string representation of the current environment state."""
        progress = f"[{self.current_step}/{self.max_steps}]"
        stats = f"C:{self.session_stats['correct']} W:{self.session_stats['wrong']} E:{self.session_stats['escalated']}"
        return f"ModerationEnv {progress} | {stats} | Last Score: {self.compute_score()}"

# ─────────────────────────────────────────────────────────────────────────────
# 8. VECTORISED ENVIRONMENT (HIGH-THROUGHPUT)
# ─────────────────────────────────────────────────────────────────────────────

class VecContentModerationEnv:
    """
    Synchronous Vectorized Wrapper for ContentModerationEnv.
    Enables parallel processing of 'n' environments simultaneously.
    """
    def __init__(self, n_envs: int, **kwargs):
        self.n_envs = n_envs
        base_seed = kwargs.pop("seed", int(time.time()))
        # Create 'n' independent environment instances
        self.envs = [
            ContentModerationEnv(**kwargs, seed=base_seed + i)
            for i in range(n_envs)
        ]
        self.obs_list: List[Dict] = []

    def reset(self) -> List[Dict]:
        """Resets all environments and returns a list of initial observations."""
        self.obs_list = [env.reset() for env in self.envs]
        return self.obs_list

    def state(self) -> List[Dict]:
        """Probes all environments for their current state observations."""
        return [env.state() for env in self.envs]

    def step(
        self,
        actions: List[Union[str, Dict[str, Any], ActionModel]],
        confidences: Optional[List[float]] = None,
    ) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """
        Executes a step in all sub-environments.
        Automatically resets sub-environments that reach 'done' state.
        """
        if confidences is None:
            confidences = [1.0] * self.n_envs

        results = []
        for i, env in enumerate(self.envs):
            act = actions[i]
            conf = confidences[i]

            # Formulate action for sub-env
            if not isinstance(act, (dict, ActionModel)):
                act_input = {"action": act, "confidence": conf}
            else:
                act_input = act

            results.append(env.step(act_input))

        # Unpack results
        next_obs = []
        rewards  = []
        dones    = []
        infos    = []

        for i, (o, r, d, info) in enumerate(results):
            # Automatic Reset Logic
            if d:
                o = self.envs[i].reset()

            next_obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        self.obs_list = next_obs
        return next_obs, rewards, dones, infos

    def close(self):
        """Clean up environment resources."""
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"--- Initializing Moderation Environment v{ContentModerationEnv.API_VERSION} ---")

    # 1. Single Env Test
    test_env = ContentModerationEnv(task="hard", seed=42)
    current_obs = test_env.reset()

    print(f"Task Pool Size: {len(test_env.task_pool)}")
    print(f"Initial Observation Text: {current_obs['text'][:50]}...")

    total_ep_reward = 0
    for s in range(test_env.max_steps):
        # Sample random action with varied confidence
        random_action = random.choice(ACTIONS)
        random_conf = random.uniform(0.3, 1.0)

        _, reward, done, info = test_env.step({
            "action": random_action,
            "confidence": random_conf,
            "agent_reasoning": "Random Testing"
        })

        total_ep_reward += reward
        print(f"Step {s+1}: Action={random_action:6} | Reward={reward:.2f} | Correct={info['is_correct']}")

    print(f"Episode Complete. Final Score: {test_env.compute_score()}")
    print("-" * 50)
