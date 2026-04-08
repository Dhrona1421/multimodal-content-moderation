"""
Strict OpenEnv adapter exposing the canonical environment API:
  - reset()
  - step(action)
  - state()

This wrapper keeps the task logic in ContentModerationEnv while presenting
the exact method surface expected by hackathon validators and reviewers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from env import ContentModerationEnv
from schemas import ActionModel, ObservationModel, RewardModel, StepInfoModel


class OpenEnvModerationEnv:
    """Canonical OpenEnv interface wrapper for content moderation."""

    ObservationModel = ObservationModel
    ActionModel = ActionModel
    RewardModel = RewardModel
    StepInfoModel = StepInfoModel

    def __init__(
        self,
        dataset_path: str = "moderation_dataset.json",
        task: str = "medium",
        max_steps: int = 12,
        seed: Optional[int] = None,
        severity_scale: float = 0.3,
        calib_weight: float = 0.15,
    ) -> None:
        self._env = ContentModerationEnv(
            dataset_path=dataset_path,
            task=task,
            max_steps=max_steps,
            seed=seed,
            severity_scale=severity_scale,
            calib_weight=calib_weight,
        )

    def reset(self) -> Dict[str, Any]:
        """Reset episode and return initial observation."""
        return self._env.reset()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""
        return self._env.step(action)

    def state(self) -> Dict[str, Any]:
        """Return current observation without advancing state."""
        return self._env.state()

    @property
    def max_steps(self) -> int:
        return self._env.max_steps

    @property
    def contract(self) -> Dict[str, Any]:
        return self._env.contract
