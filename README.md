---
title: Multimodal Content Moderation
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - content-moderation
  - multimodal
---

# 🛡️ Multimodal Content Moderation Environment

> **OpenEnv RL Environment v2 · Meta × Hugging Face × PyTorch Hackathon**

A production-grade reinforcement learning environment that simulates **real-world social media content moderation** — the same class of problem Meta's Trust & Safety teams solve at billions-of-posts-per-day scale. An agent observes posts (text + image classification + user trust metadata) and calls the standard OpenEnv API with a single action payload like `{"action": "flag", "confidence": 0.78}`.

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Architecture Overview](#-architecture-overview)
3. [Environment Specification](#-environment-specification)
4. [Feature Extractor](#-feature-extractor-64-dim)
5. [Policy Network](#-policy-network)
6. [Training Algorithm](#-training-algorithm-ppo-clip)
7. [Reward System](#-reward-system)
8. [Novel Features](#-novel-features)
9. [Dataset](#-dataset-41-posts)
10. [Tasks](#-tasks)
11. [Metrics](#-metrics)
12. [Baseline Results](#-baseline-results)
13. [Quick Start](#-quick-start)
14. [Validation](#-validation)
15. [API Reference](#-api-reference)
16. [File Structure](#-file-structure)
17. [Deployment](#-deployment)

---

## 🎯 Problem Statement

Content moderation is one of the most consequential AI applications today:

- Platforms process **hundreds of millions of posts per day**
- Wrong decisions cause real harm — missed hate speech, undetected scams, false removal of legitimate content
- Human reviewers cannot scale; AI agents must make **calibrated decisions** and know *when to escalate*
- Moderation is inherently **multimodal** — text and image signals frequently conflict

This environment provides a reproducible RL sandbox for training and evaluating moderation agents across all of these challenges.

---

## 🏗️ Architecture Overview

```
Raw Post (text + image_tag + user_type + history)
          │
          ▼
┌─────────────────────────────────────────────────┐
│  features.py  —  64-dim Multimodal Extractor    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ one-hot  │ │ keyword  │ │ cross-modal      │ │
│  │ encodings│ │ TF scores│ │ interaction terms│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
          │ (64,) ∈ [0,1]
          ▼
┌─────────────────────────────────────────────────┐
│  network.py  —  Actor-Critic MLP (19,172 params)│
│  64→128(LN+Drop)→64(LN+Drop)→32                │
│       │ Actor head          │ Critic head       │
│  FC(3)→softmax → π(a|s)   FC(1) → V(s)         │
└─────────────────────────────────────────────────┘
          │ action, confidence, value
          ▼
┌─────────────────────────────────────────────────┐
│  env.py  —  ContentModerationEnv                │
│  Multi-objective reward · Severity weighting    │
│  Confidence-gated escalation · User history     │
└─────────────────────────────────────────────────┘
          │ reward ∈ [-1.5, +1.1]
          ▼
┌─────────────────────────────────────────────────┐
│  train.py  —  PPO-Clip + GAE                    │
│  Curriculum: easy → medium → hard               │
│  Adam + Cosine LR · KL early-stop               │
└─────────────────────────────────────────────────┘
```

---

## 🔁 Environment Specification

### Observation Space

| Field           | Type       | Values / Range                                               |
|-----------------|------------|--------------------------------------------------------------|
| `post_id`       | int        | 1–41                                                         |
| `text`          | str        | Post caption / body                                          |
| `image_tag`     | categorical | `safe` · `nudity` · `violence` · `drugs` · `misleading`    |
| `user_type`     | categorical | `new` · `trusted` · `suspicious`                            |
| `difficulty`    | categorical | `easy` · `medium` · `hard`                                  |
| `step`          | int        | 1 – `max_steps`                                             |
| `max_steps`     | int        | 12 (fixed)                                                   |
| `user_history`  | float      | 0.0–5.0 recency-decayed session flag count                  |
| `session_stats` | dict       | `{correct, wrong, flagged, removed}` running totals         |
| `features`      | ndarray    | **(64,) ∈ [0,1]** pre-computed multimodal feature vector    |

### Action Space

The canonical `step()` input is a single action object:

| Field        | Type   | Values / Range                     |
|--------------|--------|------------------------------------|
| `action`     | enum   | `allow` · `flag` · `remove`        |
| `confidence` | float  | `0.0`–`1.0` (default `1.0`)        |

### Episode

- **12 steps** per episode (posts sampled from task pool)
- `reset()` → first observation
- `step(action_payload)` → `(obs, reward, done, info)`
- `state()` → current observation without advancing (OpenEnv spec)

---

## 🧠 Feature Extractor (64-dim)

`features.py` converts a raw observation dict into a **64-dimensional float32 vector** where every value ∈ [0, 1].

| Dims    | Group                      | Description                                          |
|---------|----------------------------|------------------------------------------------------|
| 0–4     | Image tag one-hot          | 5 categories                                         |
| 5–7     | User type one-hot          | new / trusted / suspicious                           |
| 8–10    | Difficulty one-hot         | easy / medium / hard                                 |
| 11–19   | Keyword group TF scores    | 9 semantic buckets (spam, hate, violence, drugs, …)  |
| 20–24   | Bigram phrase signals      | 5 phrase-level detectors (credible threat, coded sales, …) |
| 25–30   | Surface text statistics    | length, ALL-CAPS ratio, emoji density, URLs, …       |
| 31–40   | **Cross-modal interactions** | 10 conflict terms — safe text + harmful image, trusted + misinfo, … |
| 41–45   | Session / history signals  | recency-decayed flags, repeat-offender binary        |
| 46–63   | Reserved (zero-padded)     | for future modalities                                |

The **cross-modal interaction terms** are the key innovation — they explicitly encode the conflicting signals that make hard-tier posts difficult:

```python
feat[31] = high_risk_img  * safe_score        # safe text + harmful image
feat[32] = trusted_user   * misinfo_score     # trusted user spreading misinfo
feat[33] = suspicious_usr * safe_score        # suspicious user + safe content
feat[38] = mislead_img    * trusted_user      # trusted + misleading image
feat[39] = high_risk_img  * suspicious_usr    # highest-risk combination
```

---

## 🧬 Policy Network

`network.py` implements a **Deep Actor-Critic MLP** in pure NumPy with a PyTorch-compatible API (direct port requires only replacing `@` with `torch.matmul`).

```
Input (64)
  → Linear(128) → LayerNorm → ReLU → Dropout(0.10)
  → Linear(64)  → LayerNorm → ReLU → Dropout(0.10)
  → Linear(32)  → ReLU
  ┌────────────────────────┐
  │ Actor head → Linear(3) → Softmax → π(a|s)  │
  │ Critic head → Linear(1)          → V(s)     │
  └────────────────────────┘

Parameters: 19,172  |  Init: He normal (actor/critic: σ=0.01)
```

**Design choices:**
- **LayerNorm** prevents internal covariate shift without batch statistics
- **Small actor/critic init** (σ=0.01) gives uniform initial action probs
- **Separate heads** on shared trunk = standard Actor-Critic architecture
- **Confidence output** = `max(π(a|s))` — directly interpretable

---

## 🚀 Training Algorithm (PPO-Clip)

`train.py` implements **Proximal Policy Optimisation with Clip** (Schulman et al. 2017) in pure NumPy — no external RL libraries.

### Algorithm

```
for stage in [easy, medium, hard]:             # curriculum
  for update in range(n_updates):
    collect n_steps transitions                 # rollout
    compute GAE advantages (λ=0.95, γ=0.99)
    for epoch in range(4):                      # PPO epochs
      for minibatch in shuffle(rollout):        # 4 mini-batches
        compute L_clip + c1·L_VF - c2·H        # PPO objective
        compute analytic gradients via backprop
        clip gradients (max_norm=0.5)
        Adam step (lr=3e-4, cosine annealed)
        if KL > 0.02: early stop epoch          # stability
```

### Hyperparameters

| Parameter      | Value  | Rationale                                  |
|----------------|--------|--------------------------------------------|
| `clip_eps`     | 0.2    | Standard PPO clip ratio                    |
| `gae_lambda`   | 0.95   | High λ = low bias, moderate variance       |
| `gamma`        | 0.99   | Near-1 for 8-step episodes                 |
| `n_epochs`     | 4      | Standard PPO                               |
| `n_minibatches`| 4      | 16 steps/batch from 64-step rollout        |
| `ent_coef`     | 0.02   | Encourages exploration across action space |
| `target_kl`    | 0.02   | Prevents catastrophic policy updates       |
| `lr`           | 3e-4   | Standard Adam LR                           |
| `weight_decay` | 1e-4   | L2 regularisation                          |

### Curriculum Learning

Training proceeds through three stages: **easy → medium → hard**. Each stage uses the dataset pool for that difficulty level and inherits the policy weights from the previous stage. This mirrors established curriculum learning practice and ensures the agent first learns obvious cases before facing adversarial edge cases.

---

## 💰 Reward System

### Multi-objective reward decomposition

```
reward = base_accuracy + severity_modifier + context_modifier + calibration_bonus
```

**Base accuracy matrix** (correct_action, agent_action):

|               | allow  | flag   | remove |
|---------------|--------|--------|--------|
| **allow**     | +1.00  | −0.50  | −1.00  |
| **flag**      | −1.00  | +1.00  | +0.50  |
| **remove**    | −1.00  | +0.50  | +1.00  |

**Severity amplifier** — applied when agent allows harmful content:

| Image Tag    | Severity | False-negative penalty |
|--------------|----------|------------------------|
| nudity       | 1.5×     | up to −1.45            |
| violence     | 1.4×     | up to −1.42            |
| drugs        | 1.3×     | up to −1.39            |
| misleading   | 1.2×     | up to −1.36            |

**Confidence-gated escalation** (confidence < 0.45 → human review):

| Difficulty | Escalation reward | Rationale |
|------------|-------------------|-----------|
| hard       | +0.30             | Smart to admit uncertainty on adversarial cases |
| medium     | +0.20             | Acceptable |
| easy       | −0.10             | Unnecessary — agent should be decisive |

**Reward range:** `[−1.5, +1.1]`  
**Score range:** `[0.0, 1.0]` (normalised over episode)

---

## ✨ Novel Features

### 1. Confidence-Gated Human Escalation
When an agent's `confidence < 0.45`, the action is treated as **"route to human review"** rather than a committed decision. This earns partial credit proportional to difficulty — mirrors how Meta's actual Trust & Safety pipeline handles low-confidence cases. No other OpenEnv submission models this mechanism.

### 2. Cross-Modal Interaction Features
The feature extractor explicitly computes **10 product terms** between image signals and text signals, directly encoding the conflicts that define hard-tier posts. A naive bag-of-words or one-hot approach misses these.

### 3. Severity-Weighted Reward
False negatives are not all equally bad. Allowing nudity content is penalised 1.5× harder than allowing misleading content — matching the real prioritisation hierarchy in Trust & Safety.

### 4. Multi-Metric Grading
Beyond accuracy, the grader computes:
- **ECE** (Expected Calibration Error) — is the agent's stated confidence reliable?
- **FNR on high-risk content** — the safety-critical miss rate
- **Fairness gap** — max accuracy disparity across `new/trusted/suspicious` users
- **Per-class F1** — does the agent know when to flag vs remove vs allow?

### 5. Vectorised Environment
`VecContentModerationEnv` runs N independent environments in lockstep for PPO batch rollout collection — standard in modern RL but unusual in OpenEnv submissions.

---

## 📊 Dataset (41 Posts)

Hand-crafted posts across 3 difficulty tiers designed to test reasoning, not keyword matching.

| Difficulty | Count | Design intent |
|------------|-------|---------------|
| Easy       | 14    | All signals align — blatant spam or clearly safe content |
| Medium     | 13    | Coded language, trust-level reasoning, health misinfo |
| Hard       | 14    | Intentionally conflicting signals (see edge cases below) |

### Hard-Tier Edge Cases

| Post | Conflict | Correct action |
|------|----------|----------------|
| #28 | Uplifting text + `nudity` image + trusted user | **remove** — image overrides everything |
| #29 | Historical Jefferson quote as political threat + trusted | **flag** — intent context matters |
| #30 | Baby's first steps + suspicious account | **allow** — suspicion ≠ guilt |
| #31 | ICU nurse explaining acetaminophen thresholds + `drugs` image | **allow** — medical education |
| #37 | "Fresh batch dropped 🍪🔥 DM if you know" + trusted | **flag** — coded trafficking |
| #38 | Mental health medication advocacy + `drugs` image | **allow** — destigmatisation |
| #40 | Agent actively debunking misinformation + `misleading` image | **allow** — content of the image ≠ stance |
| #41 | Arabic-language prayer + suspicious account | **allow** — multilingual fairness test |

### Distribution

| Dimension      | Breakdown                                          |
|----------------|----------------------------------------------------|
| Correct action | allow: 14 · flag: 18 · remove: 9                  |
| Image tag      | safe: 16 · misleading: 10 · drugs: 6 · violence: 6 · nudity: 3 |
| User type      | trusted: 16 · new: 13 · suspicious: 12            |

---

## 🧩 Tasks

```python
from tasks import make_task

env = make_task("easy")    # 14-post pool, obvious cases
env = make_task("medium")  # 27-post pool, reasoning required
env = make_task("hard")    # 41-post pool, full adversarial suite
```

---

## 📐 Metrics

The grader (`grader.py`) computes the following for every agent × task combination:

| Metric            | Description                                    | Direction |
|-------------------|------------------------------------------------|-----------|
| `score`           | Normalised episode reward (0–1), primary rank  | ↑         |
| `accuracy`        | Fraction of steps with correct action          | ↑         |
| `macro_f1`        | Unweighted mean F1 across allow/flag/remove    | ↑         |
| `weighted_f1`     | Support-weighted F1                            | ↑         |
| `ece`             | Expected Calibration Error                     | ↓         |
| `fnr_high_risk`   | False-negative rate on nudity/violence/drugs   | ↓         |
| `fairness_gap`    | Max accuracy gap across user trust types       | ↓         |
| `confusion_matrix`| 3×3 true × predicted action matrix            | —         |

---

## 📈 Baseline Results

All values below are verified against the bundled artifacts at `seed=42`, 12 steps per episode.

### Verified score (primary metric, 0–1)

| Agent             | Easy   | Medium | Hard   | **Aggregate** |
|-------------------|--------|--------|--------|---------------|
| PPO (`ppo_checkpoint_best.npz`) | 0.5128 | 0.7308 | 0.6667 | 0.6368 |
| **Rule-Based**    | **0.9426** | **0.8037** | **0.8520** | **0.8661** |

### Additional metrics (Rule-Based, Hard task)

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | 83.33% |
| Macro-F1        | 0.8778 |
| ECE             | 0.2649 |
| FNR (high-risk) | 0.0000 |
| Fairness gap    | 0.2857 |

Reproduce the verified numbers with:

```bash
python inference.py --rule-based --seed 42 --verbose
python train.py --eval-only --checkpoint ppo_checkpoint --seed 42
```

> **Why does rule-based beat PPO?** The reward function strongly penalizes confident false negatives on harmful content while still giving modest partial credit for cautious escalation. The shipped PPO checkpoint converges toward a conservative `flag`-heavy policy, which is safer than random but weaker than the hand-authored moderation heuristic. That is acceptable for this benchmark: the baseline is deterministic, reproducible, and the hard tier remains nontrivial for learned agents. External LLM scores are intentionally omitted from this fixed table because they depend on the provider, model, and token configuration.

---

## 🚀 Quick Start

### Local (no Docker)

```bash
git clone <your-repo-url>
cd multimodal-content-moderation
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Run rule-based agent (no API key needed)
python inference.py --rule-based --seed 42 --verbose

# Run with LLM agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key_here
python inference.py --seed 42 --verbose

# Single task
python inference.py --task hard --seed 42 --verbose

# Run local submission smoke checks
python validate_submission.py

# Run the official OpenEnv validator
openenv validate --verbose

# Run the OpenEnv server entrypoint directly
uv run server

# Train PPO from scratch (full curriculum)
python train.py --updates 200

# Evaluate saved checkpoint
python train.py --eval-only --checkpoint ppo_checkpoint

# Launch Gradio demo
python app.py
```

### Docker

```bash
# Build
docker build -t content-moderation-env .

# Run interactive demo (rule-based, no key needed)
docker run -p 7860:7860 content-moderation-env

# With LLM agent
docker run -e HF_TOKEN=your_key -p 7860:7860 content-moderation-env

# CLI evaluation only
docker run content-moderation-env python inference.py --rule-based --seed 42 --verbose

# Submission smoke checks
docker run content-moderation-env python validate_submission.py

# Train inside container
docker run content-moderation-env python train.py --updates 200
```

### Hugging Face Spaces

1. Create a new Space with the **Docker** SDK
2. Push this repository
3. Add the required runtime configuration:
   - variable `API_BASE_URL`
   - variable `MODEL_NAME`
   - secret `HF_TOKEN`
4. Restart the Space after saving variables/secrets
5. Verify the public Space responds with `200` on `POST /reset`
6. The Space launches `app.py` automatically on port 7860

---

## 🔌 Plug in Your Own Agent

The grader accepts any callable that returns either an OpenEnv action payload or a legacy `(action, confidence)` tuple:

```python
from grader import ModerationGrader
from features import extract_features

def my_agent(obs):
    # obs keys: post_id, text, image_tag, user_type, difficulty,
    #           step, max_steps, user_history, session_stats, features (64,)
    features = obs["features"]    # pre-computed 64-dim feature list
    action     = "flag"           # your logic here
    confidence = 0.80             # calibrated confidence [0, 1]
    return {"action": action, "confidence": confidence}

grader = ModerationGrader(seed=42)
report = grader.grade_all_tasks(my_agent)
grader.print_report(report, verbose=True)
print(f"Aggregate: {report['aggregate_score']:.4f}")
```

---

## 📚 API Reference

### `ContentModerationEnv`

```python
env = ContentModerationEnv(
    dataset_path   = "moderation_dataset.json",
    task           = "hard",        # easy | medium | hard
    max_steps      = 12,
    seed           = 42,
    severity_scale = 0.3,           # weight of severity penalty
    calib_weight   = 0.15,          # weight of calibration bonus
)

obs            = env.reset()
obs, r, done, info = env.step({"action": "flag", "confidence": 0.75})
obs            = env.state()        # OpenEnv spec: non-advancing read
score          = env.compute_score()
print(env.render())
```

### `VecContentModerationEnv`

```python
from env import VecContentModerationEnv

vec      = VecContentModerationEnv(n_envs=4, task="hard", seed=0)
obs_list = vec.reset()
obs_list, rewards, dones, infos = vec.step(
    actions=[
        {"action": "allow", "confidence": 0.9},
        {"action": "flag", "confidence": 0.7},
        {"action": "remove", "confidence": 0.95},
        {"action": "flag", "confidence": 0.6},
    ],
)
```

### `ActorCriticNetwork`

```python
from network import ActorCriticNetwork
from features import extract_features

net = ActorCriticNetwork()              # 19,172 parameters
net.load("ppo_final")                   # load checkpoint

feat = extract_features(obs)            # (64,) ndarray
probs, value, cache = net.forward(feat) # probs sums to 1.0
action_idx, conf, val = net.act(feat, greedy=True)

net.save("my_checkpoint")              # saves .npz file
```

### `ModerationGrader`

```python
from grader import ModerationGrader

grader = ModerationGrader(seed=42)

# Grade one task
result = grader.grade_single_task("hard", my_agent)
print(result["score"])               # 0–1
print(result["classification"])      # per-class precision/recall/F1
print(result["confusion_matrix"])    # 3×3 list
print(result["fnr_high_risk"])       # false-negative rate on harmful content
print(result["fairness_gap"])        # accuracy gap across user types

# Grade all tasks
report = grader.grade_all_tasks(my_agent)
grader.print_report(report, verbose=True)
print(report["aggregate_score"])
```

### `PPOTrainer`

```python
from train import PPOConfig, PPOTrainer, make_ppo_agent
from network import ActorCriticNetwork

cfg = PPOConfig()
cfg.n_steps   = 64
cfg.n_epochs  = 4
cfg.lr        = 3e-4

net     = ActorCriticNetwork()
trainer = PPOTrainer(net, cfg)
env     = make_task("hard")

# One update cycle
rollout_stats = trainer.collect_rollout(env)
update_stats  = trainer.update(rollout_stats)

# Wrap as grader-compatible agent
agent = make_ppo_agent(net, greedy=True)
```

---

## 📁 File Structure

```text
multimodal-content-moderation/
|-- moderation_dataset.json   # 41 posts with ground-truth labels and reasons
|-- features.py               # 64-dim multimodal feature extractor
|-- network.py                # Deep Actor-Critic MLP + Adam optimiser
|-- env.py                    # OpenEnv-compliant RL environment (+ VecEnv)
|-- tasks.py                  # Task registry and make_task() factory
|-- grader.py                 # Full grading engine with F1, ECE, FNR, fairness
|-- inference.py              # LLM (CoT) agent + rule-based agent + eval runner
|-- train.py                  # PPO-Clip trainer with GAE and curriculum learning
|-- app.py                    # 6-tab Gradio demo for Hugging Face Spaces
|-- server/
|   |-- __init__.py           # OpenEnv server package exports
|   `-- app.py                # OpenEnv-compatible server entry point (main)
|-- __init__.py               # Package init - public API exports
|-- pyproject.toml            # OpenEnv packaging metadata + `server` script
|-- uv.lock                   # Locked dependency resolution for uv/openenv
|-- openenv.yaml              # OpenEnv metadata specification
|-- requirements.txt          # Python dependencies
|-- Dockerfile                # Multi-stage production Docker build
|-- scripts/
|   `-- validate-submission.sh # End-to-end HF + Docker + openenv validator
|-- LICENSE                   # MIT license
|-- ppo_checkpoint_best.npz   # Bundled PPO checkpoint used by demo + evaluation
|-- ppo_checkpoint_final.npz  # Final PPO checkpoint after training
|-- ppo_final.npz             # Legacy bundled PPO checkpoint alias
|-- training_log.csv          # Training metrics CSV (generated by train.py)
`-- results.json              # Last evaluation results (generated by inference.py)
```

---

## 🚢 Deployment

### Hugging Face Spaces (recommended)

The `Dockerfile` is configured for HF Spaces:
- Exposes port 7860 (Gradio default)
- Serves validator-compatible HTTP endpoints: `POST /reset`, `POST /step`, `GET /state`
- Health-check validates environment integrity
- Bundles the trained PPO checkpoints used by the UI and CLI evaluation
- `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` enable the LLM agent path required by the hackathon
- Falls back to rule-based agent automatically if no token

### Environment Variables

| Variable      | Default                        | Purpose                              |
|---------------|--------------------------------|--------------------------------------|
| `HF_TOKEN`    | *(unset)*                      | API key for LLM agent                |
| `OPENAI_API_KEY` | *(unset)*                   | Fallback API key alias               |
| `MODEL_NAME`  | `gpt-4o-mini`                  | Model identifier for OpenAI-compat API |
| `API_BASE_URL`| `https://api.openai.com/v1`    | API base URL (supports any OpenAI-compat endpoint) |

### Submission Validator

Run the local validator script against the public Space URL:

```bash
./scripts/validate-submission.sh https://your-space.hf.space .
```

---

## HTTP API

The deployed Space exposes both the Gradio UI and the validator endpoints:

- `POST /reset` with optional JSON body: `{"task":"easy|medium|hard","seed":42,"max_steps":12}`
- `POST /step` with action payload: `{"action":"flag","confidence":0.78}`
- `GET /state` to read the current observation without advancing
- `GET /healthz` for a basic service check
- `GET /health` for the OpenEnv runtime validator
- `GET /metadata` for environment metadata
- `GET /schema` for action, observation, and state schemas
- `POST /mcp` for the JSON-RPC compatibility check used by the OpenEnv runtime validator

These endpoints return standard OpenEnv-style JSON responses and allow the official submission validator to ping the Space directly.

---

## 🏆 What Makes This Submission Stand Out

| Feature | Detail |
|---------|--------|
| **Confidence-gated escalation** | Models human review queue; novel in OpenEnv |
| **Cross-modal interaction features** | 10 explicit conflict terms in the 64-dim feature vector |
| **Severity-weighted rewards** | Nudity/violence false-negatives penalised 1.5× harder |
| **Multi-metric grading** | ECE, FNR, fairness gap, per-class F1 alongside accuracy |
| **PPO-Clip from scratch** | Full backprop, GAE, KL stop — no external RL library |
| **Vectorised environment** | Batch rollout collection for scalable RL training |
| **Curriculum learning** | easy → medium → hard difficulty progression |
| **Adversarial hard tier** | 14 edge cases designed to break keyword matching |
| **Multilingual fairness** | Arabic-language post in dataset (post #41) |
| **Chain-of-thought LLM** | 3-shot CoT with 4-signal reasoning protocol |
| **6-tab Gradio demo** | Play / Auto-Pilot / Training / Dataset / Leaderboard / About |
| **Multi-stage Dockerfile** | Lean production image, health check, pre-trained checkpoint |

---

## 📄 License

MIT — see `LICENSE`.

---

*Built for the Meta × Hugging Face × PyTorch OpenEnv Hackathon.*


