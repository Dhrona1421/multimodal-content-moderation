# ─────────────────────────────────────────────────────────────────────────────
# Multimodal Content Moderation Environment — v2
# Production Dockerfile for Hugging Face Spaces / Docker deployment
#
# Multi-stage build:
#   Stage 1 (builder) — install deps into a venv
#   Stage 2 (runtime) — slim final image, no build tools
#
# Ports:  7860 (Gradio)
# CMD:    python app.py     → interactive demo (default)
#         python inference.py --rule-based --verbose
#         python train.py --updates 200
#         python train.py --eval-only --checkpoint ppo_checkpoint_best
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Multimodal Content Moderation Env"
LABEL org.opencontainers.image.version="3.0.0"
LABEL org.opencontainers.image.description="OpenEnv RL environment — Meta × HF × PyTorch Hackathon"
LABEL org.opencontainers.image.licenses="MIT"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Copy source files ─────────────────────────────────────────────────────────
COPY moderation_dataset.json .
COPY features.py             .
COPY network.py              .
COPY env.py                  .
COPY openenv_env.py          .
COPY schemas.py              .
COPY tasks.py                .
COPY grader.py               .
COPY inference.py            .
COPY api_inference.py        .
COPY train.py                .
COPY app.py                  .
COPY api.py                  .
COPY __init__.py             .
COPY pyproject.toml          .
COPY uv.lock                 .
COPY openenv.yaml            .
COPY requirements.txt        .
COPY Dockerfile              .
COPY validate_submission.py  .
COPY README.md               .
COPY LICENSE                 .
COPY server                  ./server
COPY scripts                 ./scripts

# ── Copy pre-trained checkpoint (if present) ──────────────────────────────────
# This lets judges evaluate immediately without re-training.
# Falls back gracefully if file is absent.
COPY ppo_final.npz           ./ppo_final.npz
COPY ppo_checkpoint_best.npz ./ppo_checkpoint_best.npz
COPY ppo_checkpoint_final.npz ./ppo_checkpoint_final.npz

# ── Runtime secrets (override at launch; no defaults baked in) ───────────────
# Gradio demo auto-detects LLM availability from HF_TOKEN.
# If HF_TOKEN is unset, the rule-based agent runs automatically.
ENV MODEL_NAME=""
ENV API_BASE_URL="https://api.openai.com/v1"

# ── Gradio port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c " \
import json, numpy as np; \
from env import ContentModerationEnv; \
from features import extract_features, FEATURE_DIM; \
e = ContentModerationEnv(task='easy', seed=0); \
obs = e.reset(); \
assert len(obs['features']) == FEATURE_DIM, 'bad feature shape'; \
_, r, done, info = e.step({'action': 'flag', 'confidence': 0.8}); \
assert -1.5 <= r <= 1.1, f'reward out of range: {r}'; \
print('HEALTH OK') \
"

# ── Default: launch Gradio demo ───────────────────────────────────────────────
CMD ["python", "api.py"]

# ── Override examples ─────────────────────────────────────────────────────────
# docker run -e HF_TOKEN=sk-... -p 7860:7860 <image>           # LLM agent
# docker run <image> python inference.py --rule-based --verbose # CLI eval
# docker run <image> python train.py --updates 200              # train PPO
# docker run <image> python train.py --eval-only                # eval ckpt
