"""
Local pre-submission smoke checks for the content moderation environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from env import ContentModerationEnv
from inference import run_inference
from schemas import ActionModel, ObservationModel, RewardModel, StepInfoModel
from tasks import TASKS


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_environment_contract() -> None:
    env = ContentModerationEnv(task="easy", seed=42)
    obs = env.reset()
    ObservationModel(**obs)
    _assert(obs["step"] == 1, "reset() must return the first observation")
    _assert(len(obs["features"]) == 64, "observation features must be 64-dimensional")
    _assert(env.state() == obs, "state() must expose the current observation without advancing")
    _assert(env.contract["step_accepts"] == "ActionModel-compatible payload",
            "env.contract must advertise the canonical action payload")

    next_obs, reward, done, info = env.step({"action": "flag", "confidence": 0.8})
    RewardModel(value=reward)
    StepInfoModel(**info)
    _assert(info["agent_action"] == "flag", "step() must execute the provided action payload")
    if not done:
        ObservationModel(**next_obs)
        _assert(next_obs["step"] == 2, "step() must advance the environment by one step")
        _assert(env.state() == next_obs, "state() must track the latest non-terminal observation")

    while not done:
        next_obs, reward, done, info = env.step({"action": "allow", "confidence": 0.9})

    # Terminal step returns the last obs for spec-strictness
    ObservationModel(**next_obs)
    _assert(next_obs["step"] == 12, "terminal step() must return the last observation")


def check_tasks() -> None:
    _assert(len(TASKS) >= 3, "at least 3 tasks are required")
    reports = run_inference(force_rule_based=True, verbose=False)
    for task_name, result in reports["Rule-Based"]["tasks"].items():
        score = result["score"]
        _assert(0.0 <= score <= 1.0, f"{task_name} score must be in [0.0, 1.0]")


def check_files() -> None:
    required = [
        "Dockerfile",
        "README.md",
        "openenv.yaml",
        "inference.py",
        "schemas.py",
    ]
    for name in required:
        _assert(Path(name).exists(), f"missing required file: {name}")


def main() -> None:
    check_files()
    check_environment_contract()
    check_tasks()
    if Path("results.json").exists():
        data = json.loads(Path("results.json").read_text(encoding="utf-8"))
        _assert("agents" in data, "results.json must contain agent results")
    print("submission validation smoke checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
