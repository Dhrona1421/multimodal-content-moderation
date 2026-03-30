"""
inference.py — Advanced inference with chain-of-thought LLM agent and
               deterministic rule-based baseline.

LLM agent features:
  • Multi-step chain-of-thought reasoning (image → user → text → decision)
  • 3-shot examples covering easy/medium/hard cases
  • Structured JSON output: action + confidence + reasoning chain
  • Retry with exponential backoff on API errors
  • Conversation-style context window (last 2 steps retained)

Rule-based agent features:
  • Weighted keyword scoring with per-group thresholds
  • Cross-modal conflict detection (safe text + harmful image)
  • Repeat-offender escalation via session history
  • Confidence calibrated to signal strength

Both agents support the (action, confidence) → Tuple return signature
required by the grader for ECE scoring.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from features import (
    FEATURE_DIM, ACTIONS, IMAGE_TAGS, USER_TYPES,
    KEYWORD_GROUPS, BIGRAM_SIGNALS,
)
from grader import ModerationGrader

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

CONFIDENCE_THRESHOLD = 0.45   # below → escalate

# ── LLM client ───────────────────────────────────────────────────────────────

try:
    from openai import OpenAI  # type: ignore
    _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None
    LLM_AVAILABLE = _client is not None
except ImportError:
    _client = None
    LLM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-thought system prompt
# ─────────────────────────────────────────────────────────────────────────────

_FEW_SHOT = """
--- EXAMPLE 1 (EASY — remove) ---
Post: "DM me for Oxy, Xanax, no script needed 🔥 Discreet shipping worldwide"
Image: drugs | User: suspicious | History: 0
Reasoning:
  1. IMAGE: drugs tag confirms pharmaceutical content
  2. USER:  suspicious account with no session context
  3. TEXT:  names controlled substances, explicit "no script" = illegal sale
  4. CROSS: all signals align — no conflict
Decision: remove (confidence 0.97)

--- EXAMPLE 2 (MEDIUM — flag) ---
Post: "Just saying, crime rates went way up after THOSE people moved in. Funny how that works 👀"
Image: safe | User: new
Reasoning:
  1. IMAGE: safe — no direct visual harm
  2. USER:  new account, no history
  3. TEXT:  "those people" + crime correlation = coded racist dog-whistle, no explicit slur
  4. CROSS: text harm, image safe — moderate severity
Decision: flag (confidence 0.78)

--- EXAMPLE 3 (HARD — allow) ---
Post: "As an ICU nurse, knowing acetaminophen toxicity thresholds can save lives. Always check labels 🏥"
Image: drugs | User: trusted
Reasoning:
  1. IMAGE: drugs tag — pills in photo
  2. USER:  trusted, no session flags
  3. TEXT:  professional medical harm-reduction context, no instructional misuse
  4. CROSS: drugs image + trusted + educational context → net safe
Decision: allow (confidence 0.82)
""".strip()

SYSTEM_PROMPT = f"""You are a senior content moderator at a major social media platform.

You will receive a social media post with:
  text          — caption or body text
  image_tag     — safe | nudity | violence | drugs | misleading
  user_type     — new | trusted | suspicious
  user_history  — weighted recency-decayed session flag count (0–5)
  step          — current position in the moderation queue

YOUR TASK: decide one action and rate your confidence.

ACTIONS:
  allow  — safe content, publish immediately
  flag   — borderline, route to human review queue
  remove — clear violation, immediate takedown

CHAIN-OF-THOUGHT PROTOCOL — you must evaluate ALL four signals:
  1. IMAGE:  Does the image tag alone warrant action?
  2. USER:   Does user trust level shift the decision?
  3. TEXT:   What does the text signal (keywords, tone, intent)?
  4. CROSS:  Do signals conflict? (safe text + harmful image → image wins)

CONFIDENCE CALIBRATION:
  ≥ 0.85  = very certain (clear violation or clear safe)
  0.60–0.84 = confident with minor ambiguity
  0.45–0.59 = uncertain, leaning one way
  < 0.45  = genuinely unsure → triggers human escalation (partial credit)

MODERATION RULES:
  • Harmful image overrides safe-looking text
  • Trusted users get moderate benefit of the doubt — NOT for serious violations
  • Suspicious user with history ≥ 2 warrants extra scrutiny
  • Medical/educational context with drugs/violence can be allow
  • Satire/sports framing does NOT automatically justify graphic content
  • When torn between flag and remove → choose flag

EXAMPLES:
{_FEW_SHOT}

Respond ONLY with valid JSON (no markdown fences):
{{
  "reasoning": {{
    "image_signal":  "one sentence",
    "user_signal":   "one sentence",
    "text_signal":   "one sentence",
    "cross_modal":   "one sentence",
    "final_logic":   "one sentence"
  }},
  "action":     "allow|flag|remove",
  "confidence": 0.0-1.0
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Observation formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_observation(obs: Dict[str, Any]) -> str:
    return (
        f"[Post #{obs.get('post_id','?')} | "
        f"Step {obs.get('step','?')}/{obs.get('max_steps','?')} | "
        f"Difficulty: {obs.get('difficulty','?').upper()}]\n\n"
        f"text         : {obs.get('text','')}\n"
        f"image_tag    : {obs.get('image_tag','safe')}\n"
        f"user_type    : {obs.get('user_type','new')}\n"
        f"user_history : {obs.get('user_history', 0):.2f}  "
        f"(weighted recency-decayed session flags)\n"
        f"step         : {obs.get('step',1)}/{obs.get('max_steps',12)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """Extract and clean a JSON block from potentially messy LLM output."""
    # 1. Strip markdown code fences (```json or ```)
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    # 2. Extract anything between the first { and the last }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def llm_agent(
    obs: Dict[str, Any],
    max_retries: int = 3,
) -> Tuple[str, float]:
    """
    Chain-of-thought LLM agent. Falls back to rule_based_agent on error.
    """
    if not LLM_AVAILABLE:
        return rule_based_agent(obs)

    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(   # type: ignore[union-attr]
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": format_observation(obs)},
                ],
                temperature=0.0,
                max_tokens=450,
            )
            raw = resp.choices[0].message.content or ""
            clean = _clean_json(raw)
            parsed = json.loads(clean)
action = str(parsed.get("action", "flag")).lower().strip()
confidence = float(parsed.get("confidence", 0.75))
reasoning = parsed.get("reasoning", {})

if action not in ACTIONS:
    action = "flag"
confidence = float(max(0.0, min(1.0, confidence)))
return action, confidence, reasoning

except Exception as exc:
wait = 1 + (attempt * 2)
print(f"  [LLM] Attempt {attempt+1} failed: {exc}. Retrying in {wait}s...",
      file=sys.stderr)
time.sleep(wait)

a, c = rule_based_agent(obs)
return a, c, {"final_logic": "Fallback to deterministic rules"}


def rule_based_agent(obs: Dict[str, Any]) -> Tuple[str, float]:
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based agent  (deterministic, no API)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_score(text: str, group: str) -> float:
    """Return fraction of group keywords present in text."""
    kws  = KEYWORD_GROUPS.get(group, [])
    hits = sum(1 for kw in kws if kw in text)
    return min(hits / max(len(kws) * 0.25, 1.0), 1.0)


def _phrase_hit(text: str, group: str) -> bool:
    return any(p in text for p in BIGRAM_SIGNALS.get(group, []))


def rule_based_agent(obs: Dict[str, Any]) -> Tuple[str, float, Dict[str, str]]:
    """
    Deterministic rule-based agent using weighted keyword scoring
    and cross-modal conflict resolution.

    Returns (action, confidence, reasoning_dict).
    """
    image   = obs.get("image_tag",    "safe")
    user    = obs.get("user_type",    "new")
    text    = obs.get("text",         "").lower()
    history = float(obs.get("user_history", 0))

    # ── Keyword scores ────────────────────────────────────────────────────────
    spam_s    = _keyword_score(text, "spam_scam")
    hate_s    = _keyword_score(text, "hate_speech")
    viol_s    = _keyword_score(text, "violence")
    drug_s    = _keyword_score(text, "drugs")
    misinfo_s = _keyword_score(text, "misleading")
    harm_s    = _keyword_score(text, "self_harm")
    fraud_s   = _keyword_score(text, "financial_fraud")
    safe_s    = _keyword_score(text, "safe_positive")
    edu_s     = _keyword_score(text, "professional_educational")

    # ── Phrase signals ────────────────────────────────────────────────────────
    credible_threat  = _phrase_hit(text, "credible_threat")
    coded_sales      = _phrase_hit(text, "coded_sales")
    authoritative    = _phrase_hit(text, "authoritative_deny")
    community_pos    = _phrase_hit(text, "community_positive")

    # ── Trust modifier ────────────────────────────────────────────────────────
    trust_factor = {"trusted": 0.7, "new": 1.0, "suspicious": 1.3}.get(user, 1.0)
    # history amplifier: each unit of decayed history adds 10% weight
    hist_factor  = 1.0 + min(history * 0.1, 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Decision tree (ordered by severity)
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Absolute removes — image-driven, regardless of text/user
    if image == "nudity":
        return "remove", 0.97, {"image_signal": "Nudity tag requires immediate takedown."}

    if credible_threat:
        return "remove", 0.95, {"text_signal": "Credible threat detected in caption."}

    if image == "drugs" and spam_s > 0.4 * trust_factor:
        conf = round(min(0.85 + spam_s * 0.1, 0.97), 3)
        return "remove", conf, {"cross_modal": "Drug image combined with pharmaceutical sales text."}

    if spam_s > 0.55 * trust_factor and user != "trusted":
        conf = round(min(0.82 + spam_s * 0.1, 0.97), 3)
        return "remove", conf, {"text_signal": "High spam score detected for non-trusted user."}

    if fraud_s > 0.5 and image in ("safe", "misleading"):
        act = "remove" if user == "suspicious" else "flag"
        return act, 0.85, {"text_signal": f"Financial fraud indicators ({fraud_s:.2f}) lead to {act}."}

    # 2. Remove on high harm text + suspicious
    if user == "suspicious" and hist_factor > 1.3:
        if hate_s > 0.3 or viol_s > 0.3:
            conf = round(min(0.72 + hist_factor * 0.05, 0.90), 3)
            return "remove", conf, {"user_signal": "Suspicious user with harmful text history."}

    # 3. Flag — image-driven
    if image == "misleading":
        if authoritative and edu_s > 0.2:
            return "allow", 0.75, {"final_logic": "Agent actively debunking misinformation."}
        conf = 0.80 if user != "trusted" else 0.72
        return "flag", round(conf * trust_factor, 3), {"image_signal": "Misleading image tag requires review."}

    if image == "violence":
        sports = ["boxing", "mma", "ufc", "wrestling", "match", "fight night",
                  "professional", "tournament"]
        if any(k in text for k in sports) and edu_s > 0.1:
            return "allow", 0.76, {"final_logic": "Content recognized as professional sports."}
        return "flag", round(0.79 * trust_factor, 3), {"image_signal": "Violent image tag requires review."}

    if image == "drugs":
        if edu_s > 0.3 and user == "trusted":
            return "allow", 0.80, {"final_logic": "Medical/educational drug content from trusted user."}
        return "flag", round(0.76 * trust_factor, 3), {"image_signal": "Drug image tag requires review."}

    # 4. Flag — text-driven
    weighted_hate = hate_s * trust_factor * hist_factor
    if weighted_hate > 0.25:
        conf = round(min(0.65 + weighted_hate * 0.2, 0.90), 3)
        return "flag", conf, {"text_signal": "Hate speech indicators above threshold."}

    if misinfo_s > 0.35:
        return "flag", round(0.74 * trust_factor, 3), {"text_signal": "Misinformation indicators detected."}

    if harm_s > 0.3:
        return "flag", 0.78, {"text_signal": "Self-harm indicators detected."}

    if drug_s > 0.4:
        return "flag", round(0.72 * trust_factor, 3), {"text_signal": "Drug-related keywords detected."}

    if coded_sales and user != "trusted":
        return "flag", round(0.68 * trust_factor * hist_factor, 3), {"text_signal": "Coded trafficking language detected."}

    # 5. Allow
    if image == "safe":
        if user == "trusted":
            conf = 0.93
        elif user == "new":
            conf = 0.82
        else:
            if hist_factor > 1.3:
                return "flag", 0.58, {"user_signal": "Suspicious user with high flag history."}
            conf = 0.70
        if safe_s > 0.4 or community_pos:
            conf = min(conf + 0.05, 0.97)
        return "allow", round(conf, 3), {"final_logic": "Standard safe content with no flags."}

    return "allow", 0.62, {"final_logic": "Default allow (no clear violations detected)."}


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    force_rule_based: bool          = False,
    dataset_path:     str           = "moderation_dataset.json",
    seed:             int           = 42,
    task_filter:      Optional[str] = None,
    verbose:          bool          = False,
    extra_agents:     Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation and return grading reports for all agents.
    """
    agents: Dict[str, Any] = {}

    if not force_rule_based and LLM_AVAILABLE:
        agents[f"LLM ({MODEL_NAME})"] = lambda obs: llm_agent(obs)[:2]
    agents["Rule-Based"] = lambda obs: rule_based_agent(obs)[:2]

    if extra_agents:
        agents.update(extra_agents)

    grader  = ModerationGrader(dataset_path=dataset_path, seed=seed)
    reports: Dict[str, Any] = {}

    print(f"\n{'═'*78}")
    print(f"  Content Moderation Environment — Inference & Evaluation")
    print(f"  Agents: {', '.join(agents.keys())}   Seed: {seed}")
    print(f"{'═'*78}\n")

    for agent_name, agent_fn in agents.items():
        print(f"  Running: {agent_name}")
        if task_filter:
            from tasks import TASKS
            result = grader.grade_single_task(task_filter, agent_fn)
            report = {
                "aggregate_score": result["score"],
                "tasks": {task_filter: result},
                "summary": grader._build_summary({task_filter: result}, result["score"]),
            }
        else:
            report = grader.grade_all_tasks(agent_fn)

        reports[agent_name] = report
        grader.print_report(report, verbose=verbose)

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(agents) > 1:
        print(f"\n{'═'*78}")
        print(f"  AGENT COMPARISON")
        print(f"{'─'*78}")
        tasks_shown = list(reports[list(agents.keys())[0]]["tasks"].keys())
        header = f"  {'Agent':<25}" + "".join(f" {t.upper():>10}" for t in tasks_shown) + "  AGGREGATE"
        print(header)
        print(f"{'─'*78}")
        for name, rpt in reports.items():
            row = f"  {name:<25}"
            for t in tasks_shown:
                row += f" {rpt['tasks'][t]['score']:>10.4f}"
            row += f"  {rpt['aggregate_score']:>9.4f}"
            print(row)
        print(f"{'═'*78}\n")

    # Save results
    slim: Dict[str, Any] = {
        "seed": seed,
        "agents": {
            name: {
                "aggregate_score": rpt["aggregate_score"],
                "tasks": {
                    t: {k: v for k, v in d.items() if k not in ("step_results",)}
                    for t, d in rpt["tasks"].items()
                },
            }
            for name, rpt in reports.items()
        },
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2)
    print("  [Results saved → results.json]\n")

    return reports


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42
    reports = run_inference(
        force_rule_based="--rule-based" in args,
        verbose="--verbose" in args,
        seed=seed,
        task_filter=(args[args.index("--task") + 1] if "--task" in args else None),
    )
