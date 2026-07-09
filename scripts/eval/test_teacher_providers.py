#!/usr/bin/env python3
"""Small, explicit provider smoke test for rollout judging and generation speed.

This script intentionally does not persist API keys. Configure providers with:

  OPENROUTER_API_KEY=... python scripts/eval/test_teacher_providers.py hy3
  OPENCODE_API_KEY=... python scripts/eval/test_teacher_providers.py opencode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_ROLLOUTS = "/Users/bilawalriaz/rollouts/rollouts.jsonl"
DEFAULT_OUTDIR = "results/provider_tests"

HY3_MODEL = "tencent/hy3:free"
HY3_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENCODE_MODEL = "mimo-v2.5"
OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"


SCORING_PROMPT = """You are judging small-model rollout generations.

Rank every generation by answer quality, format adherence, and clarity.
Return compact JSON only:
{{"ranking":[best_to_worst_indices],"winner_index":0,"quality_score":1-10,"format_valid":true,"winner_reason":"short reason"}}

Original prompt:
{prompt}

Generations:
{generations}
"""

SCORING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        "winner_index": {"type": "integer"},
        "quality_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "format_valid": {"type": "boolean"},
        "winner_reason": {"type": "string"},
    },
    "required": ["ranking", "winner_index", "quality_score", "format_valid", "winner_reason"],
}


OPENCODE_PROMPTS = [
    {
        "id": "json_extract",
        "prompt": (
            "Extract JSON only, no markdown. Text: Alice Chen joined Orbital Labs in 2024 "
            "as VP of robotics. Return schema {\"name\": string, \"company\": string, "
            "\"year\": number, \"role\": string}."
        ),
        "quality_checks": ["valid_json", "contains_alice", "contains_orbital", "contains_2024"],
    },
    {
        "id": "python_bugfix",
        "prompt": (
            "Fix this Python function and briefly explain the bug:\n"
            "def average(xs):\n"
            "    total = 0\n"
            "    for i in range(len(xs) - 1):\n"
            "        total += xs[i]\n"
            "    return total / len(xs)\n"
            "Return corrected code plus one sentence."
        ),
        "quality_checks": ["mentions_all_items", "has_code", "mentions_empty_or_zero"],
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_rollout_groups(path: Path, limit: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ph = rec.get("prompt_hash") or prompt_hash(rec.get("prompt", ""))
            group = groups.setdefault(
                ph,
                {
                    "prompt_hash": ph,
                    "prompt": rec.get("prompt", ""),
                    "prompt_meta": rec.get("prompt_meta", {}),
                    "generations": [],
                },
            )
            group["generations"].append(rec)

    selected: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group["generations"]) >= 2:
            selected.append(group)
        if len(selected) >= limit:
            break
    return selected


def format_generations(gens: list[dict[str, Any]]) -> str:
    parts = []
    for i, gen in enumerate(gens):
        reasoning = gen.get("reasoning", "")
        response = gen.get("response", "")
        parts.append(
            f"--- Generation {i} ---\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Response:\n{response}\n"
        )
    return "\n\n".join(parts)


def call_chat(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in url:
        headers["HTTP-Referer"] = "https://github.com/bilawalriaz/autonomous-small-model-exploration"
        headers["X-Title"] = "autonomous-small-model-exploration"
    started = time.monotonic()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.monotonic() - started
    try:
        data = response.json()
    except ValueError:
        data = {"text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {json.dumps(data)[:1000]}")
    return data, elapsed


def message_text(data: dict[str, Any]) -> str:
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text") or x.get("content") or x) for x in content)
    return ""


def parse_jsonish(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def response_metrics(data: dict[str, Any], elapsed: float, text: str) -> dict[str, Any]:
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    return {
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") or usage.get("total") or 0,
        "completion_tokens_per_second": round(completion_tokens / elapsed, 3) if completion_tokens else None,
        "response_chars": len(text),
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
    }


def run_hy3(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for hy3")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = load_rollout_groups(Path(args.rollouts), args.n)
    results = []
    for group in groups:
        gens = group["generations"]
        prompt = SCORING_PROMPT.format(prompt=group["prompt"], generations=format_generations(gens))
        payload = {
            "model": HY3_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": args.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "provider_smoke_judge",
                    "strict": True,
                    "schema": SCORING_SCHEMA,
                },
            },
        }
        data, elapsed = call_chat(HY3_URL, api_key, payload, args.timeout)
        text = message_text(data)
        parsed = parse_jsonish(text)
        results.append(
            {
                "provider": "openrouter",
                "model": HY3_MODEL,
                "endpoint": HY3_URL,
                "prompt_hash": group["prompt_hash"],
                "prompt_preview": group["prompt"][:240],
                "n_generations": len(gens),
                "n_generations_sent": len(gens),
                "all_generations_included": len(gens) == len(gens),
                "metrics": response_metrics(data, elapsed, text),
                "parseable_json": parsed is not None,
                "parsed": parsed,
                "raw_text": text,
                "timestamp": now_iso(),
            }
        )

    path = outdir / f"hy3_rollout_judge_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


def local_quality_checks(prompt_id: str, text: str) -> dict[str, Any]:
    lowered = text.lower()
    checks: dict[str, Any] = {}
    if prompt_id == "json_extract":
        parsed = parse_jsonish(text)
        checks["valid_json"] = isinstance(parsed, dict)
        checks["contains_alice"] = "alice" in lowered
        checks["contains_orbital"] = "orbital" in lowered
        checks["contains_2024"] = "2024" in text
    elif prompt_id == "python_bugfix":
        checks["has_code"] = "def average" in text and "for" in lowered
        checks["mentions_all_items"] = ("len(xs)" in text or "all" in lowered) and "range(len(xs) - 1)" not in text
        checks["mentions_empty_or_zero"] = "empty" in lowered or "zero" in lowered or "division" in lowered
    checks["passed"] = sum(bool(v) for v in checks.values())
    checks["total"] = len([k for k in checks if k not in ("passed", "total")])
    return checks


def run_opencode(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("OPENCODE_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENCODE_API_KEY is required for opencode")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for rec in OPENCODE_PROMPTS[: args.n]:
        payload = {
            "model": OPENCODE_MODEL,
            "messages": [{"role": "user", "content": rec["prompt"]}],
            "temperature": 0.2,
            "max_tokens": args.max_tokens,
        }
        data, elapsed = call_chat(OPENCODE_URL, api_key, payload, args.timeout)
        text = message_text(data)
        results.append(
            {
                "provider": "opencode-go",
                "model": OPENCODE_MODEL,
                "endpoint": OPENCODE_URL,
                "prompt_id": rec["id"],
                "prompt": rec["prompt"],
                "metrics": response_metrics(data, elapsed, text),
                "quality_checks": local_quality_checks(rec["id"], text),
                "raw_text": text,
                "timestamp": now_iso(),
            }
        )

    path = outdir / f"opencode_go_mimo_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["hy3", "opencode"])
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.provider == "hy3":
        path = run_hy3(args)
    else:
        path = run_opencode(args)
    print(path)


if __name__ == "__main__":
    main()
