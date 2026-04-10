"""
Typed public models for the content moderation OpenEnv interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActionLiteral    = Literal["allow", "flag", "remove"]
ImageTagLiteral  = Literal["safe", "nudity", "violence", "drugs", "misleading"]
UserTypeLiteral  = Literal["new", "trusted", "suspicious"]
DifficultyLiteral = Literal["easy", "medium", "hard"]

# Strict open-interval bounds — validator rejects 0.0 and 1.0 exactly
_SCORE_MIN = 0.0001
_SCORE_MAX = 0.9999


class SessionStatsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct:   int
    wrong:     int
    flagged:   int
    removed:   int
    escalated: int


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_id:       int
    text:          str
    image_tag:     ImageTagLiteral
    user_type:     UserTypeLiteral
    difficulty:    DifficultyLiteral
    step:          int   = Field(ge=1)
    max_steps:     int   = Field(ge=1)
    user_history:  float = Field(ge=0.0, le=5.0)
    session_stats: SessionStatsModel
    features:      List[float]


class ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action:         ActionLiteral
    confidence:     float                    = Field(default=1.0, ge=0.0, le=1.0)
    agent_reasoning: Optional[Dict[str, str]] = None


class RewardModel(BaseModel):
    """
    Step-level reward.  Clamped to strictly open interval (0.0001, 0.9999)
    so the validator never sees exactly 0.0 or 1.0.
    """
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=_SCORE_MIN, le=_SCORE_MAX)

    @field_validator("value", mode="before")
    @classmethod
    def clamp_value(cls, v: float) -> float:
        import numpy as np
        return float(np.clip(float(v), _SCORE_MIN, _SCORE_MAX))


class StepInfoModel(BaseModel):
    """
    Per-step info dict.  episode_score is clamped strictly inside (0, 1).
    reward is also clamped to the same open interval.
    """
    model_config = ConfigDict(extra="allow")

    post_id:          int
    correct_action:   ActionLiteral
    agent_action:     ActionLiteral
    confidence:       float = Field(ge=0.0, le=1.0)
    agent_reasoning:  Optional[Dict[str, str]] = None
    escalated:        bool
    reward:           float = Field(ge=_SCORE_MIN, le=_SCORE_MAX)
    is_correct:       bool
    reason:           str
    image_tag:        ImageTagLiteral
    user_type:        UserTypeLiteral
    difficulty:       DifficultyLiteral
    user_history_val: float = Field(ge=0.0, le=5.0)
    session_stats:    SessionStatsModel

    episode_reward:  Optional[float] = None
    # Strict open interval — 0.0 and 1.0 are rejected by the validator
    episode_score:   Optional[float] = Field(default=None, ge=_SCORE_MIN, le=_SCORE_MAX)
    episode_metrics: Optional[Dict[str, Any]] = None

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        import numpy as np
        return float(np.clip(float(v), _SCORE_MIN, _SCORE_MAX))

    @field_validator("episode_score", mode="before")
    @classmethod
    def clamp_episode_score(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        import numpy as np
        return float(np.clip(float(v), _SCORE_MIN, _SCORE_MAX))
