"""
tasks.py — Task registry and factory for the Content Moderation Environment.
"""
from __future__ import annotations
from typing import Any, Dict
from env import ContentModerationEnv

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "Easy Moderation",
        "description": (
            "Obvious cases only. All modality signals align. "
            "Strong keyword-matching baselines score well here."
        ),
        "difficulty":              "easy",
        "max_steps":               12,
        "reward_range":            [0.0, 1.0],
        "dataset_subset":          "12 sampled from 14 easy posts",
        "expected_baseline_score": 0.88,
    },
    "medium": {
        "name": "Intermediate Moderation",
        "description": (
            "Contextual reasoning required. Coded hate speech, "
            "health misinformation, trust-level signals matter."
        ),
        "difficulty":              "medium",
        "max_steps":               12,
        "reward_range":            [0.0, 1.0],
        "dataset_subset":          "12 sampled from 13 medium posts",
        "expected_baseline_score": 0.68,
    },
    "hard": {
        "name": "Expert Moderation",
        "description": (
            "Adversarial edge cases with conflicting signals: "
            "safe text + harmful image, trusted users spreading misinfo, "
            "suspicious users with innocent content."
        ),
        "difficulty":              "hard",
        "max_steps":               12,
        "reward_range":            [0.0, 1.0],
        "dataset_subset":          "12 sampled from 14 hard posts",
        "expected_baseline_score": 0.52,
    },
}


def make_task(
    task_name:    str,
    dataset_path: str = "moderation_dataset.json",
    seed:         int = 42,
) -> ContentModerationEnv:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
    cfg = TASKS[task_name]
    return ContentModerationEnv(
        dataset_path=dataset_path,
        task=cfg["difficulty"],
        max_steps=cfg["max_steps"],
        seed=seed,
    )


def list_tasks() -> Dict[str, Dict[str, Any]]:
    return {name: dict(cfg) for name, cfg in TASKS.items()}


def describe_task(task_name: str) -> str:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'.")
    cfg = TASKS[task_name]
    return (
        f"Task        : {cfg['name']}\n"
        f"Difficulty  : {cfg['difficulty'].upper()}\n"
        f"Max Steps   : {cfg['max_steps']}\n"
        f"Dataset     : {cfg['dataset_subset']}\n"
        f"Reward      : {cfg['reward_range']}\n"
        f"Description : {cfg['description']}\n"
    )


if __name__ == "__main__":
    for name in TASKS:
        print(describe_task(name))
        print()
