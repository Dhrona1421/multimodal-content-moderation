"""
Multimodal Content Moderation Environment — v2
================================================
OpenEnv-compliant RL environment for social media content moderation.

Quick start
-----------
>>> from env import ContentModerationEnv
>>> from tasks import make_task
>>> env = make_task("hard")
>>> obs = env.reset()
>>> obs, reward, done, info = env.step({"action": "flag", "confidence": 0.8})

Advanced (PPO training)
-----------------------
>>> from train import train, make_ppo_agent
>>> net = train(n_updates=200)
>>> agent = make_ppo_agent(net)
"""
from env      import ContentModerationEnv, VecContentModerationEnv
from openenv_env import OpenEnvModerationEnv
from tasks    import make_task, list_tasks, describe_task, TASKS
from grader   import ModerationGrader
from features import extract_features, FEATURE_DIM, ACTIONS
from network  import ActorCriticNetwork, Adam
from schemas  import (
    ActionModel,
    ObservationModel,
    RewardModel,
    SessionStatsModel,
    StepInfoModel,
)

__version__ = "3.0.0"
__all__ = [
    "ContentModerationEnv",
    "OpenEnvModerationEnv",
    "VecContentModerationEnv",
    "make_task",
    "list_tasks",
    "describe_task",
    "TASKS",
    "ModerationGrader",
    "ActionModel",
    "ObservationModel",
    "RewardModel",
    "SessionStatsModel",
    "StepInfoModel",
    "extract_features",
    "FEATURE_DIM",
    "ACTIONS",
    "ActorCriticNetwork",
    "Adam",
]
