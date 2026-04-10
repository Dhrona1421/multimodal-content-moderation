"""
grader.py — Advanced grading engine for the Content Moderation Environment.

Computes:
  • Per-task scores (0–1) and aggregate
  • Confusion matrix (3×3) across allow/flag/remove
  • Per-class precision, recall, F1
  • Macro-F1 and weighted-F1
  • Per-category breakdowns (by image_tag, user_type, difficulty)
  • Expected Calibration Error (ECE)
  • False-negative rate on high-risk content
  • Fairness gap across user types
  • Human-readable report + machine-readable dict
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from env   import ContentModerationEnv
from tasks import TASKS, make_task

ACTIONS     = ["allow", "flag", "remove"]
ACTION_IDX  = {a: i for i, a in enumerate(ACTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# Core grader
# ─────────────────────────────────────────────────────────────────────────────

class ModerationGrader:

    def __init__(
        self,
        dataset_path: str = "moderation_dataset.json",
        seed:         int = 42,
    ):
        self.dataset_path = dataset_path
        self.seed         = seed

    # ── Public API ────────────────────────────────────────────────────────────

   def grade_all_tasks(self, agent_fn: Callable) -> Dict[str, Any]:
    task_results: Dict[str, Dict] = {}

    for task_name in TASKS:
        env = make_task(task_name, self.dataset_path, self.seed)
        task_results[task_name] = self._run_task(env, task_name, agent_fn)

    aggregate = round(
        sum(t["score"] for t in task_results.values()) / len(task_results), 4
    )

    # ✅ STRICT CLAMP (INSIDE FUNCTION)
    aggregate = min(max(float(aggregate), 0.0001), 0.9999)

    return {
        "aggregate_score": aggregate,
        "tasks":           task_results,
        "summary":         self._build_summary(task_results, aggregate),
    }

    def grade_single_task(
        self,
        task_name: str,
        agent_fn:  Callable,
    ) -> Dict[str, Any]:
        env = make_task(task_name, self.dataset_path, self.seed)
        return self._run_task(env, task_name, agent_fn)

    # ── Episode runner ────────────────────────────────────────────────────────

    def _run_task(
        self,
        env:       ContentModerationEnv,
        task_name: str,
        agent_fn:  Callable,
    ) -> Dict[str, Any]:
        obs            = env.reset()
        step_results:  List[Dict] = []
        total_reward   = 0.0
        conf_pairs:    List[Tuple[float, float]] = []   # (confidence, is_correct)
        y_true:        List[int]  = []
        y_pred:        List[int]  = []

        for step_idx in range(env.max_steps):
            if not obs:
                break
            raw = agent_fn(obs)
            if isinstance(raw, dict):
                action_payload = raw
                confidence = float(raw.get("confidence", 1.0))
            elif isinstance(raw, (list, tuple)) and len(raw) == 2:
                action_payload = {"action": str(raw[0]), "confidence": float(raw[1])}
                confidence = float(raw[1])
            else:
                action_payload = {"action": str(raw), "confidence": 1.0}
                confidence = 1.0

            obs, reward, done, info = env.step(action_payload)
            total_reward += reward
            conf_pairs.append((confidence, float(info["is_correct"])))
            # Use safe lookup for action indices; -1 signals invalid action
            true_idx = ACTION_IDX.get(info["correct_action"], -1)
            pred_idx = ACTION_IDX.get(info["agent_action"], -1)
            if true_idx == -1 or pred_idx == -1:
                print(f"⚠ Warning: Invalid action in step {step_idx}: "
                      f"correct={info['correct_action']}, agent={info['agent_action']}")
            y_true.append(max(true_idx, 0))  # Fallback to 0 (allow) if invalid
            y_pred.append(max(pred_idx, 0))

            step_results.append({
                "step":           step_idx + 1,
                "post_id":        info["post_id"],
                "agent_action":   info["agent_action"],
                "correct_action": info["correct_action"],
                "confidence":     round(confidence, 3),
                "escalated":      info["escalated"],
                "reward":         info["reward"],
                "is_correct":     info["is_correct"],
                "difficulty":     info["difficulty"],
                "image_tag":      info["image_tag"],
                "user_type":      info["user_type"],
                "reason":         info["reason"],
            })
            if done:
                break

        n          = len(step_results)
        score      = env.compute_score()
        score      = min(max(float(score), 0.0001), 0.9999)
        correct    = sum(1 for r in step_results if r["is_correct"])
        cm         = _confusion_matrix(y_true, y_pred, len(ACTIONS))
        clf_report = _classification_report(cm)
        ece        = _expected_calibration_error(conf_pairs)
        fnr        = _false_negative_rate(step_results)
        breakdowns = _per_category_accuracy(step_results)
        fairness   = _fairness_gap(breakdowns.get("user_type", {}))

        return {
            "task":             task_name,
            "score":            score,
            "total_reward":     round(total_reward, 4),
            "accuracy":         round(correct / max(n, 1), 4),
            "correct":          correct,
            "steps":            n,
            "confusion_matrix": cm.tolist(),
            "classification":   clf_report,
            "ece":              ece,
            "fnr_high_risk":    fnr,
            "breakdowns":       breakdowns,
            "fairness_gap":     fairness,
            "step_results":     step_results,
        }

    # ── Summary builder ───────────────────────────────────────────────────────

    def _build_summary(
        self, task_results: Dict[str, Dict], aggregate: float
    ) -> str:
        sep  = "═" * 78
        dash = "─" * 78
        lines = [
            sep,
            "  MULTIMODAL CONTENT MODERATION  —  GRADER REPORT",
            sep,
            f"  {'TASK':<10}  {'SCORE':>7}  {'ACC':>6}  {'MACRO-F1':>9}  "
            f"{'ECE':>6}  {'FNR':>6}  {'FAIR-Δ':>7}  {'CORRECT':>8}",
            dash,
        ]
        for tn, data in task_results.items():
            clf  = data["classification"]
            lines.append(
                f"  {tn.upper():<10}  "
                f"{data['score']:>7.4f}  "
                f"{data['accuracy']:>6.2%}  "
                f"{clf['macro_f1']:>9.4f}  "
                f"{data['ece']:>6.4f}  "
                f"{data['fnr_high_risk']:>6.4f}  "
                f"{data['fairness_gap']:>7.4f}  "
                f"{data['correct']:>3}/{data['steps']:<3}"
            )
        lines += [
            dash,
            f"  {'AGGREGATE':<10}  {aggregate:>7.4f}",
            sep,
        ]
        return "\n".join(lines)

    def print_report(
        self,
        report:  Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        print(report["summary"])
        if verbose:
            for task_name, data in report["tasks"].items():
                self._print_task_detail(task_name, data)

    def _print_task_detail(self, task_name: str, data: Dict) -> None:
        print(f"\n{'─'*78}")
        print(f"  [{task_name.upper()}] Detailed breakdown")
        print(f"{'─'*78}")

        # Confusion matrix
        cm    = np.array(data["confusion_matrix"])
        clf   = data["classification"]
        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        print(f"  {'':10s}  " + "  ".join(f"pred:{a:<6}" for a in ACTIONS))
        for i, action in enumerate(ACTIONS):
            row = "  ".join(f"{cm[i,j]:>12d}" for j in range(len(ACTIONS)))
            print(f"  true:{action:<6s}  {row}")

        # Per-class metrics
        print(f"\n  Per-class metrics:")
        print(f"  {'Action':<8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Sup':>5}")
        for a in ACTIONS:
            m = clf["per_class"][a]
            print(f"  {a:<8}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
                  f"{m['f1']:>6.3f}  {m['support']:>5d}")
        print(f"  {'macro':>8}  {'─':>6}  {'─':>6}  {clf['macro_f1']:>6.3f}")
        print(f"  {'weighted':>8}  {'─':>6}  {'─':>6}  {clf['weighted_f1']:>6.3f}")

        # Breakdowns
        bd = data["breakdowns"]
        if "difficulty" in bd:
            print(f"\n  Accuracy by difficulty:")
            for d, acc in bd["difficulty"].items():
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"    {d:<8}: {bar} {acc:.2%}")

        if "image_tag" in bd:
            print(f"\n  Accuracy by image tag:")
            for tag, acc in sorted(bd["image_tag"].items()):
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"    {tag:<12}: {bar} {acc:.2%}")

        if "user_type" in bd:
            print(f"\n  Accuracy by user type:")
            for ut, acc in bd["user_type"].items():
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"    {ut:<12}: {bar} {acc:.2%}")

        print(f"\n  ECE (calibration):        {data['ece']:.4f}  (↓ better)")
        print(f"  FNR on high-risk content: {data['fnr_high_risk']:.4f}  (↓ better)")
        print(f"  Fairness gap (user types):{data['fairness_gap']:.4f}  (↓ better)")

        # Step-by-step
        print(f"\n  Step-by-step:")
        for r in data["step_results"]:
            status = "✓" if r["is_correct"] else ("↗" if r["escalated"] else "✗")
            print(
                f"    {status} {r['step']:>2} │ "
                f"{r['agent_action']:>6} │ correct:{r['correct_action']:>6} │ "
                f"conf:{r['confidence']:.2f} │ r:{r['reward']:>+5.2f} │ "
                f"[{r['image_tag']:<11}][{r['user_type']:<10}] │ "
                f"{r['difficulty']:<6} │ {r['reason'][:50]}…"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true: List[int], y_pred: List[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _classification_report(cm: np.ndarray) -> Dict[str, Any]:
    n     = cm.shape[0]
    per_class: Dict[str, Dict] = {}
    precs, recs, f1s, sups = [], [], [], []

    for i, action in enumerate(ACTIONS):
        tp  = int(cm[i, i])
        fp  = int(cm[:, i].sum()) - tp
        fn  = int(cm[i, :].sum()) - tp
        sup = int(cm[i, :].sum())

        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[action] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   sup,
        }
        precs.append(prec); recs.append(rec); f1s.append(f1); sups.append(sup)

    macro_f1    = round(float(np.mean(f1s)), 4)
    total       = sum(sups)
    weighted_f1 = round(
        float(sum(f * s for f, s in zip(f1s, sups)) / max(total, 1)), 4
    )
    overall_acc = round(
        float(cm.diagonal().sum() / max(cm.sum(), 1)), 4
    )

    return {
        "per_class":    per_class,
        "macro_f1":     macro_f1,
        "weighted_f1":  weighted_f1,
        "overall_acc":  overall_acc,
    }


def _expected_calibration_error(
    pairs: List[Tuple[float, float]],
    n_bins: int = 10,
) -> float:
    """ECE: weighted average |avg_conf − avg_acc| per bin."""
    if not pairs:
        return 0.0
    bins: Dict[int, List] = {b: [] for b in range(n_bins)}
    for conf, correct in pairs:
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    n   = len(pairs)
    ece = 0.0
    for bp in bins.values():
        if not bp:
            continue
        avg_c = sum(p[0] for p in bp) / len(bp)
        avg_a = sum(p[1] for p in bp) / len(bp)
        ece  += (len(bp) / n) * abs(avg_c - avg_a)
    return round(ece, 4)


def _false_negative_rate(step_results: List[Dict]) -> float:
    """FNR on high-risk image content (nudity, violence, drugs)."""
    high_risk = [r for r in step_results
                 if r["image_tag"] in {"nudity", "violence", "drugs"}
                 and r["correct_action"] != "allow"]
    if not high_risk:
        return 0.0
    fn = sum(1 for r in high_risk if r["agent_action"] == "allow")
    return round(fn / len(high_risk), 4)


def _per_category_accuracy(step_results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Accuracy broken down by difficulty, image_tag, and user_type."""
    groups: Dict[str, Dict[str, List[bool]]] = {
        "difficulty": defaultdict(list),
        "image_tag":  defaultdict(list),
        "user_type":  defaultdict(list),
    }
    for r in step_results:
        groups["difficulty"][r["difficulty"]].append(r["is_correct"])
        groups["image_tag"] [r["image_tag"]].append( r["is_correct"])
        groups["user_type"] [r["user_type"]].append( r["is_correct"])

    return {
        cat: {k: round(sum(v) / len(v), 4) for k, v in grp.items()}
        for cat, grp in groups.items()
    }


def _fairness_gap(user_type_acc: Dict[str, float]) -> float:
    """Max accuracy gap across user types (0 = perfectly fair)."""
    if len(user_type_acc) < 2:
        return 0.0
    vals = list(user_type_acc.values())
    return round(max(vals) - min(vals), 4)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random as _r
    _rng = _r.Random(99)
    def random_agent(obs):
        return _rng.choice(["allow", "flag", "remove"]), 0.7

    grader = ModerationGrader()
    report = grader.grade_all_tasks(random_agent)
    grader.print_report(report, verbose="--verbose" in sys.argv)
