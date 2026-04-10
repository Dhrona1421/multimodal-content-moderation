"""
features.py — Advanced multimodal feature extractor for content moderation.

Produces a 64-dimensional observation vector from a raw post observation dict.

Feature groups:
  [0:5]   image_tag one-hot          (5)
  [5:8]   user_type one-hot          (3)
  [8:11]  difficulty one-hot         (3)
  [11:20] text keyword group scores  (9) — semantic buckets
  [20:25] bigram / phrase signals    (5)
  [25:30] surface text statistics    (6)  ← index corrected below
  [30:40] cross-modal interaction    (10)
  [40:45] session / history signals  (5)
  [45:64] reserved / zero-padded     (19)

Total: 64
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_TAGS   = ["safe", "nudity", "violence", "drugs", "misleading"]
USER_TYPES   = ["new", "trusted", "suspicious"]
DIFFICULTIES = ["easy", "medium", "hard"]

ACTIONS      = ["allow", "flag", "remove"]
ACTION_IDX   = {a: i for i, a in enumerate(["allow", "flag", "remove"])}
FEATURE_DIM  = 64

# ── Semantic keyword groups ───────────────────────────────────────────────────

KEYWORD_GROUPS: Dict[str, List[str]] = {
    "spam_scam": [
        "click here", "dm me", "limited time", "gift card", "act now",
        "verify your", "claim your", "win a", "free offer", "selected to",
        "send 0.", "double back", "worldwide shipping", "no questions",
        "fast discreet", "dm for", "link in bio", "bit.ly", "urgent",
    ],
    "hate_speech": [
        "don't belong", "go back to", "your kind", "those people",
        "crime rates", "they all", "get out", "not welcome", "filth",
        "subhuman", "vermin", "slur", "deport", "invasion",
    ],
    "violence": [
        "kill", "hurt", "attack", "destroy", "blood", "weapon",
        "shoot", "bomb", "threat", "watch out", "warning", "you will",
        "punish", "brutal", "fight", "beat", "eliminated",
    ],
    "drugs": [
        "oxy", "xanax", "adderall", "valium", "tramadol", "oxycontin",
        "no script", "pharmaceutical", "meds available", "pill",
        "batch", "discreet pack", "420", "rolled", "weed", "cannabis",
    ],
    "misleading": [
        "doctors are hiding", "big pharma", "wake up", "sheeple",
        "they don't want", "secret cure", "natural remedy", "leaked",
        "mainstream media", "fake news", "cover up", "conspiracy",
        "do your own research", "share before", "gets deleted",
    ],
    "self_harm": [
        "what's the point", "burden to everyone", "nobody cares",
        "want to disappear", "end it", "can't go on", "hopeless",
        "not here anymore", "just ignore", "sorry for being",
    ],
    "financial_fraud": [
        "crypto", "bitcoin", "invest", "returns", "profit", "wallet",
        "trading method", "passive income", "dm profit", "turned into",
        "elon", "giving away", "giveaway", "double your",
    ],
    "safe_positive": [
        "love", "happy", "beautiful", "grateful", "blessed", "family",
        "friend", "birthday", "travel", "recipe", "adopted", "rescued",
        "nature", "sunrise", "sunset", "goal", "journey", "fitness",
        "proud", "amazing", "congratulations",
    ],
    "professional_educational": [
        "research", "study", "medical", "nurse", "doctor", "according to",
        "evidence", "published", "awareness", "mental health", "support",
        "resources", "thread", "breakdown", "primary sources", "false",
    ],
}

# ── Phrase-level signals ──────────────────────────────────────────────────────

BIGRAM_SIGNALS: Dict[str, List[str]] = {
    "credible_threat":    ["watch out tomorrow", "this is your only warning",
                           "i have something planned", "none of you will forget"],
    "coded_sales":        ["you know what", "regulars get", "first pick",
                           "dm if you know", "fresh batch", "premium quality"],
    "satire_marker":      ["lmao", "literally", "just saying", "imagine",
                           "the audacity", "no cap", "fr fr", "lowkey"],
    "authoritative_deny": ["this is false", "debunked", "primary sources",
                           "confirmed disinfo", "fact check", "misinformation"],
    "community_positive": ["check on your", "reach out", "you matter",
                           "not alone", "here for you", "save a life"],
}

# ── Surface statistics ────────────────────────────────────────────────────────

def _surface_stats(text: str) -> np.ndarray:
    """6 normalised surface-level text signals."""
    words    = text.split()
    n_words  = len(words)
    n_caps   = sum(1 for w in words if w.isupper() and len(w) > 1)
    n_emoji  = len(re.findall(r"[^\w\s,.]", text))
    n_excl   = text.count("!")
    n_url    = len(re.findall(r"http|bit\.ly|\.com|\.net|link in", text.lower()))
    n_at     = text.count("@") + text.count("#")

    return np.array([
        min(n_words  / 60.0,  1.0),   # normalised length
        min(n_caps   / 5.0,   1.0),   # ALL-CAPS words
        min(n_emoji  / 8.0,   1.0),   # special chars / emoji
        min(n_excl   / 4.0,   1.0),   # exclamation marks
        min(n_url    / 2.0,   1.0),   # URL / link signals
        min(n_at     / 3.0,   1.0),   # mentions / hashtags
    ], dtype=np.float32)


# ── Cross-modal interaction features ─────────────────────────────────────────

def _cross_modal(
    image_tag: str,
    user_type: str,
    keyword_scores: np.ndarray,
) -> np.ndarray:
    """
    10 hand-crafted interaction terms that capture conflicting signals.
    These are the key features for hard-tier edge cases.
    """
    safe_img       = float(image_tag == "safe")
    high_risk_img  = float(image_tag in {"nudity", "drugs"})
    mislead_img    = float(image_tag == "misleading")
    violence_img   = float(image_tag == "violence")
    trusted_user   = float(user_type == "trusted")
    suspicious_usr = float(user_type == "suspicious")
    new_user       = float(user_type == "new")

    spam_score     = keyword_scores[0]
    hate_score     = keyword_scores[1]
    violence_score = keyword_scores[2]
    drug_score     = keyword_scores[3]
    misinfo_score  = keyword_scores[4]
    safe_score     = keyword_scores[7]

    return np.array([
        high_risk_img  * safe_score,       # safe text + harmful image (hard case)
        trusted_user   * misinfo_score,    # trusted user spreading misinfo
        suspicious_usr * safe_score,       # suspicious user + safe content
        safe_img       * hate_score,       # coded hate (no image signal)
        violence_img   * (1 - violence_score),  # violent image + safe text
        suspicious_usr * spam_score,       # suspicious + spam
        trusted_user   * drug_score,       # trusted + drug content
        new_user       * (spam_score + drug_score),  # new user risk amplifier
        mislead_img    * trusted_user,     # trusted spreading misleading
        high_risk_img  * suspicious_usr,   # highest-risk combination
    ], dtype=np.float32)


# ── Session / history signals ─────────────────────────────────────────────────

def _session_signals(obs: Dict) -> np.ndarray:
    """5 session-level signals."""
    history   = obs.get("user_history", 0)
    step      = obs.get("step",         1)
    max_steps = obs.get("max_steps",   12)

    return np.array([
        min(history  / 5.0,  1.0),         # normalised session flag count
        step         / max_steps,           # episode progress
        float(history >= 2),               # repeat offender binary
        float(history >= 3),               # high-repeat offender
        min(history * 0.2, 1.0),           # continuous risk signal
    ], dtype=np.float32)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_features(obs: Dict) -> np.ndarray:
    """
    Convert a raw observation dict into a 64-dim feature vector.

    All values are in [0, 1].  Zero-padding fills dims 45–63.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    text       = obs.get("text",       "").lower()
    image_tag  = obs.get("image_tag",  "safe")
    user_type  = obs.get("user_type",  "new")
    difficulty = obs.get("difficulty", "medium")

    # [0:5]  image one-hot
    if image_tag in IMAGE_TAGS:
        feat[IMAGE_TAGS.index(image_tag)] = 1.0

    # [5:8]  user one-hot
    if user_type in USER_TYPES:
        feat[5 + USER_TYPES.index(user_type)] = 1.0

    # [8:11] difficulty one-hot
    if difficulty in DIFFICULTIES:
        feat[8 + DIFFICULTIES.index(difficulty)] = 1.0

    # [11:20] keyword group scores
    group_scores = np.zeros(len(KEYWORD_GROUPS), dtype=np.float32)
    for i, (_, keywords) in enumerate(KEYWORD_GROUPS.items()):
        hits = sum(1 for kw in keywords if kw in text)
        group_scores[i] = min(hits / max(len(keywords) * 0.3, 1.0), 1.0)
    feat[11:20] = group_scores

    # [20:25] bigram / phrase signals
    for i, (_, phrases) in enumerate(BIGRAM_SIGNALS.items()):
        feat[20 + i] = float(any(p in text for p in phrases))

    # [25:31] surface stats (6 dims)
    feat[25:31] = _surface_stats(obs.get("text", ""))

    # [31:41] cross-modal interactions (10 dims)
    feat[31:41] = _cross_modal(image_tag, user_type, group_scores)

    # [41:46] session signals (5 dims)
    feat[41:46] = _session_signals(obs)

    # [46:64] zero-padded reserved

    return feat


# ── Batch version ─────────────────────────────────────────────────────────────

def extract_features_batch(observations: List[Dict]) -> np.ndarray:
    """Extract features for a list of observations → (N, FEATURE_DIM)."""
    return np.stack([extract_features(obs) for obs in observations])


if __name__ == "__main__":
    import json
    data   = json.load(open("moderation_dataset.json"))
    sample = {
        "text":         data[0]["text"],
        "image_tag":    data[0]["image_tag"],
        "user_type":    data[0]["user_type"],
        "difficulty":   data[0]["difficulty"],
        "step":         1,
        "max_steps":    12,
        "user_history": 0,
    }
    f = extract_features(sample)
    print(f"Feature vector shape : {f.shape}")
    print(f"Value range          : [{f.min():.3f}, {f.max():.3f}]")
    print(f"Non-zero dims        : {(f != 0).sum()}")
    print(f"\nFirst 20 dims: {f[:20].round(3)}")
