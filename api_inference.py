"""
HTTP-based inference loop for the OpenEnv moderation API.

This script evaluates agents by calling:
  - POST /reset
  - POST /step
  - GET /state

It is useful for validating deployed Hugging Face Space behavior instead of
running the local Python environment directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from inference import llm_agent, rule_based_agent


def _request_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            body = json.loads(raw) if raw else {}
            resp_headers = {k: v for k, v in resp.headers.items()}
            return body, resp_headers
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def run_api_inference(
    base_url: str,
    task: str,
    seed: int,
    max_steps: int,
    agent_name: str,
    env_id: Optional[str],
    timeout: int,
) -> Dict[str, Any]:
    base_url = base_url.rstrip("/")
    agent = rule_based_agent if agent_name == "rule-based" else llm_agent

    reset_payload: Dict[str, Any] = {
        "task": task,
        "seed": seed,
        "max_steps": max_steps,
    }
    if env_id:
        reset_payload["env_id"] = env_id

    obs, reset_headers = _request_json(
        f"{base_url}/reset",
        method="POST",
        payload=reset_payload,
        timeout=timeout,
    )
    if not obs:
        raise RuntimeError("Reset returned an empty observation.")

    session_id = reset_headers.get("X-Env-Id", env_id or "default")
    session_headers = {"X-Env-Id": session_id}

    print(f"Session: {session_id}")
    print(f"Task: {task} | Agent: {agent_name} | Max steps: {max_steps}")

    total_reward = 0.0
    history = []
    info: Dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        action, confidence, reasoning = agent(obs)
        payload = {
            "action": action,
            "confidence": float(confidence),
            "agent_reasoning": reasoning,
        }
        result, _ = _request_json(
            f"{base_url}/step",
            method="POST",
            payload=payload,
            headers=session_headers,
            timeout=timeout,
        )

        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        info = dict(result.get("info", {}))
        obs = dict(result.get("observation", {}))

        total_reward += reward
        history.append(info)

        print(
            f"step={step:02d} action={action:<6} conf={confidence:.2f} "
            f"reward={reward:+.3f} correct={info.get('correct_action', '?')} done={done}"
        )
        if done:
            break

    state, _ = _request_json(
        f"{base_url}/state",
        method="GET",
        headers=session_headers,
        timeout=timeout,
    )

    summary = {
        "session_id": session_id,
        "steps": len(history),
        "total_reward": round(total_reward, 4),
        "episode_score": info.get("episode_score"),
        "state_after_run_is_empty": (state == {}),
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference through OpenEnv HTTP endpoints.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:7860")
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--agent", type=str, default="rule-based", choices=["rule-based", "llm"])
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_api_inference(
            base_url=args.base_url,
            task=args.task,
            seed=args.seed,
            max_steps=args.max_steps,
            agent_name=args.agent,
            env_id=args.env_id,
            timeout=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"api inference failed: {exc}", file=sys.stderr)
        raise
