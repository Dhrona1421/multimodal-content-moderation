"""
network.py — Deep Actor-Critic Neural Network (pure NumPy, PyTorch-compatible API).

Architecture:
  Input (64) → FC(128) → LayerNorm → ReLU → Dropout
             → FC(64)  → LayerNorm → ReLU → Dropout
             → FC(32)  → ReLU
             ┌─────────────────┐
             │  Actor head     │  → FC(3)  → Softmax   (action probs)
             │  Critic head    │  → FC(1)              (state value)
             └─────────────────┘

This is an Actor-Critic used by PPO. The network is written in NumPy but
follows the exact PyTorch nn.Module idiom — porting is a direct translation.

Numerical stability:
  - He initialisation for ReLU layers
  - Gradient clipping externally applied
  - LayerNorm prevents internal covariate shift
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from features import FEATURE_DIM

# ── Constants ─────────────────────────────────────────────────────────────────

N_ACTIONS   = 3
HIDDEN_1    = 128
HIDDEN_2    = 64
HIDDEN_3    = 32
ACTIONS     = ["allow", "flag", "remove"]
ACTION_IDX  = {a: i for i, a in enumerate(ACTIONS)}


# ─────────────────────────────────────────────────────────────────────────────
# Layer primitives
# ─────────────────────────────────────────────────────────────────────────────

def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He (Kaiming) normal initialisation for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0, std, (fan_out, fan_in)).astype(np.float32)


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    """Layer normalisation over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var( axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# ActorCritic network
# ─────────────────────────────────────────────────────────────────────────────

class ActorCriticNetwork:
    """
    Deep Actor-Critic MLP with LayerNorm.

    Attributes (all trainable parameters):
        W1, b1 : shared layer 1  (FEATURE_DIM → HIDDEN_1)
        g1, b1n: LayerNorm 1 scale / bias
        W2, b2 : shared layer 2  (HIDDEN_1 → HIDDEN_2)
        g2, b2n: LayerNorm 2 scale / bias
        W3, b3 : shared layer 3  (HIDDEN_2 → HIDDEN_3)
        Wa, ba : actor head      (HIDDEN_3 → N_ACTIONS)
        Wv, bv : critic head     (HIDDEN_3 → 1)
    """

    def __init__(
        self,
        feature_dim:  int   = FEATURE_DIM,
        hidden_1:     int   = HIDDEN_1,
        hidden_2:     int   = HIDDEN_2,
        hidden_3:     int   = HIDDEN_3,
        n_actions:    int   = N_ACTIONS,
        dropout_rate: float = 0.10,
        seed:         int   = 42,
    ):
        self.feature_dim  = feature_dim
        self.hidden_1     = hidden_1
        self.hidden_2     = hidden_2
        self.hidden_3     = hidden_3
        self.n_actions    = n_actions
        self.dropout_rate = dropout_rate
        self._rng         = np.random.default_rng(seed)

        self._init_weights()
        self.training: bool = True   # controls dropout

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        rng = self._rng

        # Shared trunk
        self.W1  = _he_init(self.feature_dim,  self.hidden_1, rng)
        self.b1  = np.zeros(self.hidden_1, dtype=np.float32)
        self.g1  = np.ones( self.hidden_1, dtype=np.float32)   # LN scale
        self.b1n = np.zeros(self.hidden_1, dtype=np.float32)   # LN bias

        self.W2  = _he_init(self.hidden_1, self.hidden_2, rng)
        self.b2  = np.zeros(self.hidden_2, dtype=np.float32)
        self.g2  = np.ones( self.hidden_2, dtype=np.float32)
        self.b2n = np.zeros(self.hidden_2, dtype=np.float32)

        self.W3  = _he_init(self.hidden_2, self.hidden_3, rng)
        self.b3  = np.zeros(self.hidden_3, dtype=np.float32)

        # Actor head
        std      = 0.01
        self.Wa  = (rng.standard_normal((self.n_actions, self.hidden_3)) * std
                    ).astype(np.float32)
        self.ba  = np.zeros(self.n_actions, dtype=np.float32)

        # Critic head
        self.Wv  = (rng.standard_normal((1, self.hidden_3)) * std
                    ).astype(np.float32)
        self.bv  = np.zeros(1, dtype=np.float32)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass.

        Args:
            x: (batch, feature_dim) or (feature_dim,)

        Returns:
            probs:  (batch, n_actions) — action probabilities
            values: (batch,)           — state values
            cache:  intermediate activations for backprop
        """
        squeeze = (x.ndim == 1)
        if squeeze:
            x = x[np.newaxis, :]

        cache: Dict[str, np.ndarray] = {"x": x}

        # Layer 1
        z1       = x @ self.W1.T + self.b1
        z1n      = _layer_norm(z1, self.g1, self.b1n)
        a1       = relu(z1n)
        if self.training:
            mask1 = (self._rng.random(a1.shape) > self.dropout_rate).astype(np.float32)
            a1   *= mask1 / (1 - self.dropout_rate + 1e-8)
        else:
            mask1 = np.ones_like(a1)
        cache.update({"z1": z1, "z1n": z1n, "a1": a1, "mask1": mask1})

        # Layer 2
        z2       = a1 @ self.W2.T + self.b2
        z2n      = _layer_norm(z2, self.g2, self.b2n)
        a2       = relu(z2n)
        if self.training:
            mask2 = (self._rng.random(a2.shape) > self.dropout_rate).astype(np.float32)
            a2   *= mask2 / (1 - self.dropout_rate + 1e-8)
        else:
            mask2 = np.ones_like(a2)
        cache.update({"z2": z2, "z2n": z2n, "a2": a2, "mask2": mask2})

        # Layer 3 (shared trunk end)
        z3       = a2 @ self.W3.T + self.b3
        a3       = relu(z3)
        cache.update({"z3": z3, "a3": a3})

        # Actor head → action probabilities
        logits   = a3 @ self.Wa.T + self.ba
        probs    = softmax(logits)
        cache.update({"logits": logits, "probs": probs})

        # Critic head → state value
        values   = (a3 @ self.Wv.T + self.bv).squeeze(-1)
        cache["values"] = values

        if squeeze:
            return probs.squeeze(0), values.squeeze(), cache

        return probs, values, cache

    # ── Action sampling ───────────────────────────────────────────────────────

    def act(
        self,
        obs_feat: np.ndarray,
        greedy:   bool = False,
    ) -> Tuple[int, float, float]:
        """
        Sample or greedily select an action.

        Returns:
            action_idx, confidence (max prob), state_value
        """
        self.training = False
        probs, value, _ = self.forward(obs_feat)
        self.training = True

        if greedy:
            idx = int(np.argmax(probs))
        else:
            idx = int(self._rng.choice(len(probs), p=probs))

        return idx, float(probs[idx]), float(value)

    # ── Parameter management ──────────────────────────────────────────────────

    def parameters(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1,  "b1": self.b1,  "g1": self.g1,  "b1n": self.b1n,
            "W2": self.W2,  "b2": self.b2,  "g2": self.g2,  "b2n": self.b2n,
            "W3": self.W3,  "b3": self.b3,
            "Wa": self.Wa,  "ba": self.ba,
            "Wv": self.Wv,  "bv": self.bv,
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            setattr(self, k, v.copy())

    def save(self, path: str) -> None:
        np.savez(path, **self.parameters())
        print(f"  [Checkpoint] Saved → {path}.npz")

    def load(self, path: str) -> None:
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        self.set_parameters({k: data[k] for k in data.files})
        print(f"  [Checkpoint] Loaded ← {path}")

    def clone(self) -> "ActorCriticNetwork":
        net = ActorCriticNetwork(
            self.feature_dim, self.hidden_1, self.hidden_2,
            self.hidden_3, self.n_actions, self.dropout_rate
        )
        net.set_parameters(self.parameters())
        return net

    def param_count(self) -> int:
        return sum(v.size for v in self.parameters().values())


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser
# ─────────────────────────────────────────────────────────────────────────────

class Adam:
    """Adam optimiser operating on a flat parameter dict."""

    def __init__(
        self,
        lr:      float = 3e-4,
        beta1:   float = 0.9,
        beta2:   float = 0.999,
        eps:     float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.eps     = eps
        self.wd      = weight_decay
        self._m:  Dict[str, np.ndarray] = {}
        self._v:  Dict[str, np.ndarray] = {}
        self._t:  int = 0

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads:  Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply one Adam update step. Returns updated params."""
        self._t += 1
        updated = {}
        for k in params:
            if k not in self._m:
                self._m[k] = np.zeros_like(params[k])
                self._v[k] = np.zeros_like(params[k])

            g = grads.get(k, np.zeros_like(params[k]))
            # L2 weight decay
            g = g + self.wd * params[k]

            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * g**2

            m_hat = self._m[k] / (1 - self.beta1 ** self._t)
            v_hat = self._v[k] / (1 - self.beta2 ** self._t)

            updated[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return updated

    def get_lr(self) -> float:
        return self.lr

    def set_lr(self, lr: float) -> None:
        self.lr = lr


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

class CosineAnnealingScheduler:
    """Cosine annealing with warm restarts (SGDR-style)."""

    def __init__(
        self,
        optimizer: Adam,
        T_max:     int,
        eta_min:   float = 1e-6,
        lr_init:   float = 3e-4,
    ):
        self.opt     = optimizer
        self.T_max   = T_max
        self.eta_min = eta_min
        self.lr_init = lr_init
        self.step_n  = 0

    def step(self) -> float:
        self.step_n += 1
        lr = self.eta_min + 0.5 * (self.lr_init - self.eta_min) * (
            1 + math.cos(math.pi * self.step_n / self.T_max)
        )
        self.opt.set_lr(lr)
        return lr


if __name__ == "__main__":
    net   = ActorCriticNetwork()
    x     = np.random.randn(FEATURE_DIM).astype(np.float32)
    probs, value, cache = net.forward(x)
    print(f"Network params : {net.param_count():,}")
    print(f"Probs          : {probs.round(4)}  (sum={probs.sum():.6f})")
    print(f"Value          : {value:.4f}")
    print(f"Cache keys     : {list(cache.keys())}")
