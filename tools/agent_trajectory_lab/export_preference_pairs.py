#!/usr/bin/env python3
"""Export same-task preference pairs from Hermes trajectory runs.

The preference heuristic is intentionally simple and auditable:
- prefer verifier-passed runs
- prefer prompt-matched trace exports
- prefer lower elapsed time, fewer tool calls, and shorter traces
- reject failed, missing-trace, overlong, or less efficient same-task rollouts

This creates DPO-ready JSONL with `prompt`, `chosen`, `rejected`, and metadata.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


TRANSIENT_PATTERNS = (
    "HTTP 429",
    "rate limit",
    "exceeded the rate limit",
    "API call failed after",
    "temporarily unavailable",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def iter_result_paths(roots: list[Path]):
    seen = set()
    for root in roots:
        for path in sorted(root.glob("*/result.json")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield path


def matched_trace_path(result: dict[str, Any]) -> str:
    for export in result.get("session_id_exports", []):
        if export.get("matches_prompt") and export.get("trace_exists"):
            return export.get("trace_path", "")
    return ""


def trace_stats(trace_path: str) -> dict[str, int]:
    if not trace_path:
        return {"trace_lines": 0, "tool_uses": 0, "tool_results": 0}
    path = Path(trace_path)
    if not path.exists():
        return {"trace_lines": 0, "tool_uses": 0, "tool_results": 0}

    trace_lines = 0
    tool_uses = 0
    tool_results = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        trace_lines += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = item.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if msg.get("role") == "assistant":
            tool_uses += sum(1 for part in content if part.get("type") == "tool_use")
        if msg.get("role") == "user":
            tool_results += sum(1 for part in content if part.get("type") == "tool_result")
    return {"trace_lines": trace_lines, "tool_uses": tool_uses, "tool_results": tool_results}


def trace_transcript(trace_path: str, limit: int) -> str:
    """Render a compact transcript preserving assistant tool calls and observations."""
    if not trace_path or not Path(trace_path).exists():
        return ""
    chunks: list[str] = []
    for line in Path(trace_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = item.get("message", {})
        role = msg.get("role", "unknown")
        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for part in content:
            kind = part.get("type")
            if kind == "text" and part.get("text", "").strip():
                chunks.append(f"{role}: {part['text'].strip()}")
            elif kind == "tool_use":
                chunks.append(f"assistant tool_call {part.get('name')}: {json.dumps(part.get('input', {}), sort_keys=True)}")
            elif kind == "tool_result":
                content_text = str(part.get("content", ""))
                chunks.append(f"tool_result: {content_text[:1200]}")
        rendered = "\n".join(chunks)
        if len(rendered) >= limit:
            return rendered[:limit]
    return "\n".join(chunks)[:limit]


def score_record(record: dict[str, Any]) -> tuple[float, list[str]]:
    result = record["result"]
    stats = record["stats"]
    reasons = []
    score = 0.0
    if result.get("passed"):
        score += 100.0
        reasons.append("verifier_passed")
    else:
        score -= 100.0
        reasons.append("verifier_failed")
    if record["matched_trace_path"]:
        score += 30.0
        reasons.append("matched_trace")
    else:
        score -= 30.0
        reasons.append("missing_matched_trace")
    if result.get("hermes_returncode") == 0:
        score += 10.0
    else:
        score -= 20.0
        reasons.append("nonzero_hermes_returncode")

    elapsed = float(result.get("elapsed_seconds") or 9999)
    tool_uses = stats["tool_uses"]
    trace_lines = stats["trace_lines"]
    score -= min(elapsed / 20.0, 40.0)
    score -= min(tool_uses * 0.5, 25.0)
    score -= min(trace_lines * 0.1, 20.0)
    if elapsed > 300:
        reasons.append("overlong_elapsed")
    if tool_uses > 40:
        reasons.append("overlong_tool_loop")
    return score, reasons


def is_transient_result(result: dict[str, Any]) -> bool:
    if result.get("transient_api_failure"):
        return True
    text = f"{result.get('hermes_stdout_tail', '')}\n{result.get('hermes_stderr_tail', '')}".lower()
    return any(pattern.lower() in text for pattern in TRANSIENT_PATTERNS)


def build_records(roots: list[Path], transcript_limit: int, include_transient: bool) -> tuple[list[dict[str, Any]], int]:
    records = []
    skipped_transient = 0
    for result_path in iter_result_paths(roots):
        result = load_json(result_path)
        if not result:
            continue
        if is_transient_result(result) and not include_transient:
            skipped_transient += 1
            continue
        run_dir = result_path.parent
        trace_path = matched_trace_path(result)
        stats = trace_stats(trace_path)
        record = {
            "task_id": result.get("task_id"),
            "run_id": result.get("run_id") or result.get("task_id"),
            "family": result.get("family"),
            "template_id": result.get("template_id"),
            "variant_id": result.get("variant_id"),
            "rollout_index": result.get("rollout_index"),
            "prompt": read_text(run_dir / "prompt.txt"),
            "assistant": trace_transcript(trace_path, transcript_limit) or read_text(run_dir / "stdout.txt", transcript_limit),
            "source_result": str(result_path),
            "matched_trace_path": trace_path,
            "result": result,
            "stats": stats,
        }
        score, reasons = score_record(record)
        record["preference_score"] = round(score, 3)
        record["preference_reasons"] = reasons
        records.append(record)
    return records, skipped_transient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transcript-limit", type=int, default=24000)
    parser.add_argument("--include-all-passed-contrast", action="store_true")
    parser.add_argument("--include-transient", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    records, skipped_transient = build_records(roots, args.transcript_limit, args.include_transient)
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record["task_id"])].append(record)

    pairs = []
    skipped = {}
    for task_id, group in sorted(groups.items()):
        if len(group) < 2:
            skipped[task_id] = "fewer_than_two_rollouts"
            continue
        ranked = sorted(group, key=lambda item: item["preference_score"], reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        if chosen["source_result"] == rejected["source_result"]:
            skipped[task_id] = "same_chosen_rejected"
            continue
        if chosen["preference_score"] == rejected["preference_score"] and not args.include_all_passed_contrast:
            skipped[task_id] = "equal_scores"
            continue
        pairs.append(
            {
                "task_id": task_id,
                "family": chosen["family"],
                "template_id": chosen["template_id"],
                "variant_id": chosen["variant_id"],
                "prompt": chosen["prompt"],
                "chosen": chosen["assistant"],
                "rejected": rejected["assistant"],
                "chosen_score": chosen["preference_score"],
                "rejected_score": rejected["preference_score"],
                "chosen_reasons": chosen["preference_reasons"],
                "rejected_reasons": rejected["preference_reasons"],
                "chosen_stats": chosen["stats"],
                "rejected_stats": rejected["stats"],
                "chosen_source": chosen["source_result"],
                "rejected_source": rejected["source_result"],
                "preference_type": "same_task_rollout",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "pairs": len(pairs),
        "records_seen": len(records),
        "task_groups": len(groups),
        "skipped_groups": len(skipped),
        "skipped_transient_results": skipped_transient,
        "include_transient": args.include_transient,
        "families": {},
        "output": str(out),
    }
    for pair in pairs:
        summary["families"].setdefault(pair["family"], 0)
        summary["families"][pair["family"]] += 1
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
