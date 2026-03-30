"""
train.py — Proximal Policy Optimisation (PPO) trainer.

Implements PPO-Clip with:
  • Generalised Advantage Estimation (GAE, λ=0.95)
  • Mini-batch stochastic updates (4 epochs × 4 mini-batches)
  • Entropy bonus for exploration
  • KL-divergence early stopping
  • Cosine annealing LR schedule
  • Curriculum learning: easy → medium → hard
  • CSV metrics logging + checkpoint management
  • Comparison against rule-based baseline

No external RL libraries. Pure NumPy + scikit-learn metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env      import ContentModerationEnv
from features import ACTIONS, FEATURE_DIM, extract_features
from network  import ActorCriticNetwork, Adam, CosineAnnealingScheduler
from tasks    import TASKS, make_task

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# PPO hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

class PPOConfig:
    # Rollout
    n_steps:        int   = 64        # steps per rollout collection
    n_envs:         int   = 1         # parallel envs (1 = sequential)

    # Training
    n_epochs:       int   = 4         # PPO epochs per rollout
    n_minibatches:  int   = 4         # mini-batches per epoch
    lr:             float = 3e-4      # initial learning rate
    weight_decay:   float = 1e-4

    # PPO
    clip_eps:       float = 0.2       # PPO clip ratio ε
    vf_coef:        float = 0.5       # value loss coefficient
    ent_coef:       float = 0.02      # entropy bonus coefficient
    max_grad_norm:  float = 0.5       # gradient clipping norm
    target_kl:      float = 0.02      # early stop if KL > this

    # GAE
    gamma:          float = 0.99      # discount factor
    gae_lambda:     float = 0.95      # GAE λ


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout of (features, actions, rewards, values, log_probs)."""

    def __init__(self, n_steps: int, feature_dim: int = FEATURE_DIM):
        self.n_steps     = n_steps
        self.feature_dim = feature_dim
        self.reset()

    def reset(self) -> None:
        self.features   = np.zeros((self.n_steps, self.feature_dim), dtype=np.float32)
        self.actions    = np.zeros( self.n_steps,                    dtype=np.int32)
        self.rewards    = np.zeros( self.n_steps,                    dtype=np.float32)
        self.values     = np.zeros( self.n_steps,                    dtype=np.float32)
        self.log_probs  = np.zeros( self.n_steps,                    dtype=np.float32)
        self.dones      = np.zeros( self.n_steps,                    dtype=np.float32)
        self.confidences= np.zeros( self.n_steps,                    dtype=np.float32)
        self._ptr       = 0

    def add(
        self,
        feat:      np.ndarray,
        action:    int,
        reward:    float,
        value:     float,
        log_prob:  float,
        done:      bool,
        confidence:float,
    ) -> None:
        i = self._ptr
        self.features[i]    = feat
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.values[i]      = value
        self.log_probs[i]   = log_prob
        self.dones[i]       = float(done)
        self.confidences[i] = confidence
        self._ptr           = (i + 1) % self.n_steps

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bootstrapped returns and GAE advantages.
        Returns (returns, advantages), both shape (n_steps,).
        """
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val          = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val          = self.values[t + 1]

            delta        = (self.rewards[t]
                            + gamma * next_val * next_non_terminal
                            - self.values[t])
            last_gae     = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_minibatches(
        self,
        n_minibatches: int,
        returns:       np.ndarray,
        advantages:    np.ndarray,
    ):
        """Yield shuffled mini-batches."""
        idx     = np.random.permutation(self.n_steps)
        bsize   = self.n_steps // n_minibatches
        for start in range(0, self.n_steps, bsize):
            b = idx[start:start + bsize]
            yield (
                self.features[b],
                self.actions[b],
                self.log_probs[b],
                returns[b],
                advantages[b],
            )


# ─────────────────────────────────────────────────────────────────────────────
# PPO backprop (NumPy analytic gradients)
# ─────────────────────────────────────────────────────────────────────────────

def _ppo_loss_and_grad(
    net:        ActorCriticNetwork,
    features:   np.ndarray,   # (B, D)
    actions:    np.ndarray,   # (B,)  int
    old_logp:   np.ndarray,   # (B,)
    returns:    np.ndarray,   # (B,)
    advantages: np.ndarray,   # (B,)
    clip_eps:   float,
    vf_coef:    float,
    ent_coef:   float,
) -> Tuple[float, float, float, float, Dict[str, np.ndarray]]:
    """
    Compute PPO losses and parameter gradients.

    Returns (policy_loss, value_loss, entropy, kl, grads_dict)
    """
    B = features.shape[0]

    # ── Forward ──────────────────────────────────────────────────────────────
    net.training = True
    probs, values, cache = net.forward(features)  # (B,3), (B,)

    # log-probabilities of taken actions
    eps       = 1e-8
    log_probs = np.log(probs + eps)
    taken_lp  = log_probs[np.arange(B), actions]      # (B,)

    # ratio r_t(θ) = exp(logπ_new - logπ_old)
    ratio     = np.exp(taken_lp - old_logp)           # (B,)

    # normalise advantages
    adv_norm  = (advantages - advantages.mean()) / (advantages.std() + eps)

    # clipped policy loss
    surr1     = ratio * adv_norm
    surr2     = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    policy_loss = -np.mean(np.minimum(surr1, surr2))

    # value loss (clipped)
    vf_loss   = np.mean((values - returns) ** 2)

    # entropy bonus
    entropy   = -np.mean(np.sum(probs * log_probs, axis=-1))

    # approximate KL for early stopping
    kl        = np.mean(old_logp - taken_lp)

    total_loss = policy_loss + vf_coef * vf_loss - ent_coef * entropy

    # ── Gradients via backprop ────────────────────────────────────────────────
    grads = _backprop(
        net, cache, features, probs, values, actions,
        adv_norm, ratio, clip_eps, returns, vf_coef, ent_coef, B
    )

    return float(policy_loss), float(vf_loss), float(entropy), float(kl), grads


def _backprop(
    net:       ActorCriticNetwork,
    cache:     Dict,
    features:  np.ndarray,
    probs:     np.ndarray,
    values:    np.ndarray,
    actions:   np.ndarray,
    adv_norm:  np.ndarray,
    ratio:     np.ndarray,
    clip_eps:  float,
    returns:   np.ndarray,
    vf_coef:   float,
    ent_coef:  float,
    B:         int,
) -> Dict[str, np.ndarray]:
    """
    Manual backprop through the actor-critic network.
    Returns gradient dict keyed by parameter name.
    """
    eps = 1e-8
    g   = {}   # gradients accumulator

    # ── Critic gradient (value loss) ─────────────────────────────────────────
    dVF     = 2.0 * (values - returns) * vf_coef / B    # (B,)
    dA3_v   = dVF[:, np.newaxis] * net.Wv               # (B, H3)
    g["Wv"] = dVF[:, np.newaxis].T @ cache["a3"] / B    # (1, H3)
    g["bv"] = dVF.mean(keepdims=True)

    # ── Actor gradient (PPO-clip policy loss) ────────────────────────────────
    # Gradient of -min(surr1, surr2) w.r.t log_prob_taken
    surr1    = ratio * adv_norm
    surr2    = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm
    clipped  = (ratio < (1 - clip_eps)) | (ratio > (1 + clip_eps))

    # d(-min)/d(taken_logp) — chain rule through ratio = exp(new - old)
    dsurr1   = -adv_norm * ratio
    dsurr2   = np.where(clipped, 0.0, -adv_norm * ratio)
    d_taken  = np.where(surr1 < surr2, dsurr1, dsurr2) / B   # (B,)

    # gradient of log_prob w.r.t logits → (B, 3)
    d_logits  = probs.copy()
    d_logits[np.arange(B), actions] -= 1.0
    d_logits *= d_taken[:, np.newaxis]

    # entropy gradient: d(-H)/d_probs = log(p) + 1
    d_ent    = -(np.log(probs + eps) + 1.0) * ent_coef / B   # (B, 3)
    d_logits += d_ent

    g["Wa"] = d_logits.T @ cache["a3"] / B   # (3, H3)
    g["ba"] = d_logits.mean(axis=0)

    # ── Shared trunk gradients ────────────────────────────────────────────────
    dA3   = d_logits @ net.Wa + dA3_v                          # (B, H3)
    dZ3   = dA3 * (cache["a3"] > 0).astype(np.float32)        # ReLU
    g["W3"] = dZ3.T @ cache["a2"] / B
    g["b3"] = dZ3.mean(axis=0)

    dA2   = dZ3 @ net.W3                                       # (B, H2)
    dA2  *= cache["mask2"] / (1 - net.dropout_rate + eps)
    dLN2  = dA2 * (cache["z2n"] > 0)                          # approx ReLU+LN
    g["W2"]  = dLN2.T @ cache["a1"] / B
    g["b2"]  = dLN2.mean(axis=0)
    g["g2"]  = (dLN2 * cache["z2n"]).mean(axis=0)
    g["b2n"] = dLN2.mean(axis=0)

    dA1   = dLN2 @ net.W2
    dA1  *= cache["mask1"] / (1 - net.dropout_rate + eps)
    dLN1  = dA1 * (cache["z1n"] > 0)
    g["W1"]  = dLN1.T @ features / B
    g["b1"]  = dLN1.mean(axis=0)
    g["g1"]  = (dLN1 * cache["z1n"]).mean(axis=0)
    g["b1n"] = dLN1.mean(axis=0)

    return g


def _clip_gradients(
    grads: Dict[str, np.ndarray], max_norm: float
) -> Tuple[Dict[str, np.ndarray], float]:
    """Clip gradient dictionary by global L2 norm."""
    total_norm = math.sqrt(sum(np.sum(g**2) for g in grads.values()))
    scale      = min(max_norm / (total_norm + 1e-8), 1.0)
    return {k: v * scale for k, v in grads.items()}, total_norm


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:

    def __init__(
        self,
        net:     ActorCriticNetwork,
        cfg:     PPOConfig = PPOConfig(),
    ):
        self.net    = net
        self.cfg    = cfg
        self.optim  = Adam(lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.buffer = RolloutBuffer(cfg.n_steps)
        self.metrics_log: List[Dict] = []

    # ── Rollout collection ────────────────────────────────────────────────────

    def collect_rollout(self, env: ContentModerationEnv) -> Dict[str, float]:
        """Collect n_steps transitions. Returns rollout statistics."""
        self.buffer.reset()
        self.net.training = False

        obs           = env.reset() if env.done else env.state()
        ep_rewards:   List[float] = []
        step_rewards: List[float] = []

        for _ in range(self.cfg.n_steps):
            feat  = np.asarray(obs["features"], dtype=np.float32)
            idx, conf, value = self.net.act(feat, greedy=False)
            action    = ACTIONS[idx]
            log_prob  = math.log(max(conf, 1e-8))

            next_obs, reward, done, info = env.step(
                {"action": action, "confidence": conf}
            )
            step_rewards.append(reward)

            self.buffer.add(feat, idx, reward, value, log_prob, done, conf)

            if done:
                ep_rewards.append(sum(env.episode_rewards))
                obs = env.reset()
            else:
                obs = next_obs

        # Bootstrap last value
        last_feat   = np.asarray(obs["features"], dtype=np.float32)
        _, _, last_val = self.net.act(last_feat, greedy=False)

        self.net.training = True
        return {
            "mean_reward":  float(np.mean(step_rewards)),
            "total_reward": float(np.sum(step_rewards)),
            "mean_ep_ret":  float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "last_value":   last_val,
        }

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self,
        rollout_stats: Dict[str, float],
        scheduler:     Optional[CosineAnnealingScheduler] = None,
    ) -> Dict[str, float]:
        """Run PPO epochs over the buffer. Return update statistics."""
        returns, advantages = self.buffer.compute_returns_and_advantages(
            rollout_stats["last_value"],
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )

        all_pl, all_vl, all_ent, all_kl, all_gnorm = [], [], [], [], []
        stop_early = False

        for epoch in range(self.cfg.n_epochs):
            if stop_early:
                break
            for batch in self.buffer.get_minibatches(
                self.cfg.n_minibatches, returns, advantages
            ):
                feats_b, acts_b, old_lp_b, rets_b, adv_b = batch

                pl, vl, ent, kl, grads = _ppo_loss_and_grad(
                    self.net, feats_b, acts_b, old_lp_b, rets_b, adv_b,
                    self.cfg.clip_eps, self.cfg.vf_coef, self.cfg.ent_coef,
                )
                all_pl.append(pl); all_vl.append(vl)
                all_ent.append(ent); all_kl.append(kl)

                # KL early stop
                if kl > self.cfg.target_kl:
                    stop_early = True
                    break

                grads, gnorm = _clip_gradients(grads, self.cfg.max_grad_norm)
                all_gnorm.append(gnorm)

                updated = self.optim.step(self.net.parameters(), grads)
                self.net.set_parameters(updated)

        if scheduler:
            scheduler.step()

        return {
            "policy_loss":  float(np.mean(all_pl)),
            "value_loss":   float(np.mean(all_vl)),
            "entropy":      float(np.mean(all_ent)),
            "mean_kl":      float(np.mean(all_kl)),
            "grad_norm":    float(np.mean(all_gnorm)) if all_gnorm else 0.0,
            "lr":           self.optim.get_lr(),
            "early_stop":   stop_early,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrappers
# ─────────────────────────────────────────────────────────────────────────────

def make_ppo_agent(net: ActorCriticNetwork, greedy: bool = True):
    """Return grader-compatible agent function from trained network."""
    def agent(obs: Dict) -> Tuple[str, float]:
        feat = obs.get("features")
        if feat is None:
            feat = extract_features(obs)
        feat = np.asarray(feat, dtype=np.float32)
        net.training = False
        idx, conf, _ = net.act(feat, greedy=greedy)
        return ACTIONS[idx], conf
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    task:            Optional[str] = None,
    n_updates:       int           = 200,
    eval_interval:   int           = 20,
    checkpoint_path: str           = "ppo_checkpoint",
    log_path:        str           = "training_log.csv",
    seed:            int           = SEED,
    cfg:             PPOConfig     = PPOConfig(),
    verbose:         bool          = True,
) -> ActorCriticNetwork:
    """
    Train a PPO agent on the content moderation environment.

    Args:
        task:            None → curriculum (easy→medium→hard), or specific task.
        n_updates:       PPO update iterations per curriculum stage.
        eval_interval:   Evaluate and checkpoint every N updates.
        checkpoint_path: Base path for saving .npz weights.
        log_path:        CSV file for training metrics.
        seed:            RNG seed.
        cfg:             PPO hyperparameter configuration.
        verbose:         Print progress.
    """
    from grader import ModerationGrader
    from inference import rule_based_agent

    net         = ActorCriticNetwork(seed=seed)
    trainer     = PPOTrainer(net, cfg)
    grader      = ModerationGrader(seed=seed)
    scheduler   = CosineAnnealingScheduler(
        trainer.optim, T_max=n_updates, lr_init=cfg.lr
    )

    best_score      = -1.0
    best_checkpoint = checkpoint_path + "_best"
    curriculum      = [task] if task else ["easy", "medium", "hard"]
    log_rows: List[Dict] = []

    if verbose:
        print(f"\n{'═'*72}")
        print(f"  PPO Training — Multimodal Content Moderation")
        print(f"  Network:    {net.param_count():,} parameters  (64→128→64→32 + A/C heads)")
        print(f"  Algorithm:  PPO-Clip (ε={cfg.clip_eps})  GAE(λ={cfg.gae_lambda})")
        print(f"  LR:         {cfg.lr:.0e}  (cosine annealing)")
        print(f"  Updates:    {n_updates} × {len(curriculum)} stages")
        print(f"{'═'*72}")

    for stage in curriculum:
        env = make_task(stage, seed=seed)

        if verbose:
            print(f"\n{'─'*72}")
            print(f"  STAGE: {stage.upper()}  |  {n_updates} updates × {cfg.n_steps} steps")
            print(f"{'─'*72}")

        stage_start = time.time()

        for update in range(1, n_updates + 1):
            rollout_stats = trainer.collect_rollout(env)
            update_stats  = trainer.update(rollout_stats, scheduler)

            if verbose and update % 10 == 0:
                print(
                    f"  [{stage.upper():>6}] "
                    f"upd={update:>4}/{n_updates}  "
                    f"r̄={rollout_stats['mean_reward']:>+6.3f}  "
                    f"pl={update_stats['policy_loss']:>7.4f}  "
                    f"vl={update_stats['value_loss']:>6.4f}  "
                    f"ent={update_stats['entropy']:>5.3f}  "
                    f"kl={update_stats['mean_kl']:>6.4f}  "
                    f"lr={update_stats['lr']:.2e}"
                    + ("  [KL stop]" if update_stats["early_stop"] else "")
                )

            # ── Periodic evaluation ───────────────────────────────────────────
            if update % eval_interval == 0:
                agent_fn = make_ppo_agent(net, greedy=True)
                report   = grader.grade_all_tasks(agent_fn)
                agg      = report["aggregate_score"]

                row = {
                    "stage":        stage,
                    "update":       update,
                    "agg_score":    agg,
                    "mean_reward":  rollout_stats["mean_reward"],
                    "policy_loss":  update_stats["policy_loss"],
                    "value_loss":   update_stats["value_loss"],
                    "entropy":      update_stats["entropy"],
                    "kl":           update_stats["mean_kl"],
                    "lr":           update_stats["lr"],
                    **{f"score_{t}": report["tasks"][t]["score"]
                       for t in report["tasks"]},
                    **{f"acc_{t}": report["tasks"][t]["accuracy"]
                       for t in report["tasks"]},
                }
                log_rows.append(row)

                if agg > best_score:
                    best_score = agg
                    net.save(best_checkpoint)
                    star = " ★ NEW BEST"
                else:
                    star = ""

                if verbose:
                    print(
                        f"\n  ── Eval @ update {update} ──────────────────────────\n"
                        f"     easy={report['tasks']['easy']['score']:.4f}  "
                        f"medium={report['tasks']['medium']['score']:.4f}  "
                        f"hard={report['tasks']['hard']['score']:.4f}  "
                        f"AGG={agg:.4f}{star}\n"
                    )

        if verbose:
            elapsed = time.time() - stage_start
            print(f"  Stage {stage.upper()} complete in {elapsed:.1f}s")

    # Save final weights
    net.save(checkpoint_path + "_final")
    _write_csv(log_path, log_rows)

    if verbose:
        print(f"\n{'═'*72}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best aggregate score : {best_score:.4f}")
        print(f"  Best checkpoint      : {best_checkpoint}.npz")
        print(f"  Metrics log          : {log_path}")
        print(f"{'═'*72}")

    return net


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Log] Metrics saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, seed: int = SEED) -> None:
    from grader import ModerationGrader
    from inference import rule_based_agent

    if not checkpoint_path.endswith(".npz"):
        checkpoint_path += ".npz"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    net = ActorCriticNetwork()
    net.load(checkpoint_path)
    grader = ModerationGrader(seed=seed)

    print("\n── PPO Agent (trained) ──────────────────────────────")
    ppo_report = grader.grade_all_tasks(make_ppo_agent(net))
    grader.print_report(ppo_report, verbose=True)

    print("\n── Rule-Based Baseline ─────────────────────────────")
    rb_report = grader.grade_all_tasks(rule_based_agent)
    grader.print_report(rb_report)

    print("\n── Delta (PPO − Baseline) ──────────────────────────")
    for t in ["easy", "medium", "hard"]:
        ppo_s = ppo_report["tasks"][t]["score"]
        rb_s  = rb_report["tasks"][t]["score"]
        Δ     = ppo_s - rb_s
        bar   = "▲" if Δ > 0 else ("▼" if Δ < 0 else "─")
        print(f"  {t:<8}: {ppo_s:.4f} vs {rb_s:.4f}  {bar} {abs(Δ):.4f}")
    Δagg = ppo_report["aggregate_score"] - rb_report["aggregate_score"]
    print(f"  {'TOTAL':<8}: {ppo_report['aggregate_score']:.4f} vs "
          f"{rb_report['aggregate_score']:.4f}  "
          f"{'▲' if Δagg>0 else '▼'} {abs(Δagg):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for content moderation")
    parser.add_argument("--task",       type=str, default=None,
                        choices=["easy","medium","hard"],
                        help="Train on one task (default: full curriculum)")
    parser.add_argument("--updates",    type=int, default=200,
                        help="PPO update iterations per stage (default: 200)")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--checkpoint", type=str, default="ppo_checkpoint")
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.checkpoint + "_best", seed=args.seed)
    else:
        net = train(
            task=args.task,
            n_updates=args.updates,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            verbose=not args.quiet,
        )
        evaluate(args.checkpoint + "_best", seed=args.seed)
