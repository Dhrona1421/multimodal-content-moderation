"""
Typed public models for the content moderation OpenEnv interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ActionLiteral = Literal["allow", "flag", "remove"]
ImageTagLiteral = Literal["safe", "nudity", "violence", "drugs", "misleading"]
UserTypeLiteral = Literal["new", "trusted", "suspicious"]
DifficultyLiteral = Literal["easy", "medium", "hard"]


class SessionStatsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: int
    wrong: int
    flagged: int
    removed: int


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_id: int
    text: str
    image_tag: ImageTagLiteral
    user_type: UserTypeLiteral
    difficulty: DifficultyLiteral
    step: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    user_history: float = Field(ge=0.0, le=5.0)
    session_stats: SessionStatsModel
    features: List[float]


class ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionLiteral
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RewardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=-1.5, le=1.1)


class StepInfoModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    post_id: int
    correct_action: ActionLiteral
    agent_action: ActionLiteral
    confidence: float = Field(ge=0.0, le=1.0)
    escalated: bool
    reward: float = Field(ge=-1.5, le=1.1)
    is_correct: bool
    reason: str
    image_tag: ImageTagLiteral
    user_type: UserTypeLiteral
    difficulty: DifficultyLiteral
    user_history_val: float = Field(ge=0.0, le=5.0)
    session_stats: SessionStatsModel
    episode_reward: Optional[float] = None
    episode_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    episode_metrics: Optional[Dict[str, Any]] = None
