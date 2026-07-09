#!/usr/bin/env python3
"""Export same-task preference pairs from Hermes trajectory runs.

The preference heuristic is intentionally simple and auditable:
- prefer verifier-passed runs
- prefer prompt-matched trace exports
- prefer lower elapsed time, fewer tool calls, and shorter traces
- reject failed, missing-trace, invalid-tool, ignored-observation, wrong-tool,
  premature-final, hallucinated-file, unsafe-command, overlong, or less
  efficient same-task rollouts

This creates DPO-ready JSONL with `prompt`, `chosen`, `rejected`, and metadata.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


TRANSIENT_PATTERNS = (
    "HTTP 429",
    "rate limit",
    "exceeded the rate limit",
    "API call failed after",
    "temporarily unavailable",
)
UNSAFE_COMMAND_PATTERNS = (
    "rm -rf /",
    "mkfs.",
    "dd if=",
    ":(){",
    "chmod -r 777 /",
    "chmod 777 /",
    "chown -r ",
    "sudo ",
)
FILE_ERROR_PATTERNS = (
    "no such file or directory",
    "cannot stat",
    "can't open file",
    "not found",
    "does not exist",
)
TOOL_CALL_TEXT_PATTERNS = (
    '"tool"',
    '"tool_call"',
    '"tool_calls"',
    '"function"',
    "<tool_call>",
    "```json",
)
NOISY_PATH_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}
NOISY_PATH_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".o",
    ".a",
    ".class",
    ".jar",
    ".zip",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".sqlite",
    ".db",
}
QUOTED_NOISY_PATH_RE = re.compile(
    r"""(["'])(?:\./)?(?:\.git|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules)[^"']*\1"""
)
NOISY_PATH_FRAGMENT_RE = re.compile(
    r"""(?:\./)?(?:\.git|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules)/[^\s"'`,;)]*"""
)
STANDALONE_NOISY_NAME_RE = re.compile(r"(?<!\w)(?:\.git|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|node_modules)(?!\w)")


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


def is_noisy_path(value: str) -> bool:
    normalized = value[2:] if value.startswith("./") else value
    path = Path(normalized)
    if any(part in NOISY_PATH_PARTS for part in path.parts):
        return True
    if path.name.startswith(".coverage"):
        return True
    return path.suffix.lower() in NOISY_PATH_SUFFIXES


def sanitize_tool_obj(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"path", "cwd", "file"} and isinstance(item, str) and is_noisy_path(item):
                cleaned[key] = "[omitted noisy path]"
                continue
            if key == "files" and isinstance(item, list):
                kept = [x for x in item if not (isinstance(x, str) and is_noisy_path(x))]
                cleaned[key] = kept
                omitted = len(item) - len(kept)
                if omitted:
                    cleaned["omitted_noisy_files"] = omitted
                continue
            cleaned[key] = sanitize_tool_obj(item)
        return cleaned
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [item for item in value if not is_noisy_path(item)]
        return [sanitize_tool_obj(item) for item in value]
    if isinstance(value, str):
        return STANDALONE_NOISY_NAME_RE.sub(
            "[omitted noisy path]",
            NOISY_PATH_FRAGMENT_RE.sub("[omitted noisy path]", QUOTED_NOISY_PATH_RE.sub('"[omitted noisy path]"', value)),
        )
    return value


def sanitize_tool_result_content(content: Any, limit: int = 1200) -> str:
    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            text = NOISY_PATH_FRAGMENT_RE.sub(
                "[omitted noisy path]",
                QUOTED_NOISY_PATH_RE.sub('"[omitted noisy path]"', content),
            )
            text = STANDALONE_NOISY_NAME_RE.sub("[omitted noisy path]", text)
            lines = [line for line in text.splitlines() if not is_noisy_path(line.strip().strip('"'))]
            return "\n".join(lines)[:limit]
    if isinstance(parsed, (dict, list)):
        return json.dumps(sanitize_tool_obj(parsed), sort_keys=True, ensure_ascii=False)[:limit]
    return str(parsed)[:limit]


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


def trace_text(trace_path: str, limit: int = 120000) -> str:
    if not trace_path or not Path(trace_path).exists():
        return ""
    return Path(trace_path).read_text(encoding="utf-8", errors="replace")[:limit]


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
                chunks.append(f"{role}: {sanitize_tool_obj(part['text'].strip())}")
            elif kind == "tool_use":
                tool_input = sanitize_tool_obj(part.get("input", {}))
                chunks.append(f"assistant tool_call {part.get('name')}: {json.dumps(tool_input, sort_keys=True)}")
            elif kind == "tool_result":
                chunks.append(f"tool_result: {sanitize_tool_result_content(part.get('content', ''))}")
        rendered = "\n".join(chunks)
        if len(rendered) >= limit:
            return rendered[:limit]
    return "\n".join(chunks)[:limit]


def score_record(record: dict[str, Any]) -> tuple[float, list[str]]:
    result = record["result"]
    stats = record["stats"]
    reasons = []
    behavior_reasons = analyze_behavior_failures(record)
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
        reasons.append("hermes_returncode_zero")
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
    if result.get("passed") and record["matched_trace_path"] and tool_uses > 0:
        reasons.append("successful_valid_tool_trace")
    if result.get("passed") and elapsed <= 180 and tool_uses <= 20:
        reasons.append("short_successful_rollout")
    for reason in behavior_reasons:
        if reason not in reasons:
            reasons.append(reason)
    score += behavior_penalty(behavior_reasons)
    return score, reasons


def behavior_penalty(reasons: list[str]) -> float:
    weights = {
        "invalid_tool_call_text": -45.0,
        "wrong_tool_or_no_tool_use": -70.0,
        "premature_final_answer": -60.0,
        "ignored_observation_or_failed_recovery": -35.0,
        "hallucinated_file_or_path": -35.0,
        "unsafe_command": -80.0,
        "overlong_tool_loop": -25.0,
        "overlong_elapsed": -15.0,
    }
    return sum(weights.get(reason, 0.0) for reason in reasons)


def analyze_behavior_failures(record: dict[str, Any]) -> list[str]:
    """Tag concrete agent-failure modes useful for DPO rejected examples."""
    result = record["result"]
    stats = record["stats"]
    assistant = record.get("assistant", "")
    stdout = read_text(Path(record["source_result"]).parent / "stdout.txt", limit=40000)
    stderr = read_text(Path(record["source_result"]).parent / "stderr.txt", limit=12000)
    trace_raw = record.get("trace_raw", "")
    combined = "\n".join([assistant, stdout, stderr, trace_raw]).lower()
    command_text = "\n".join(
        line for line in assistant.lower().splitlines() if "tool_call terminal" in line or '"command"' in line
    )
    reasons: list[str] = []

    tool_uses = stats["tool_uses"]
    tool_results = stats["tool_results"]
    passed = bool(result.get("passed"))
    hermes_ok = result.get("hermes_returncode") == 0

    if not passed and tool_uses == 0:
        reasons.append("wrong_tool_or_no_tool_use")
        if hermes_ok:
            reasons.append("premature_final_answer")

    if tool_uses == 0 and any(pattern in combined for pattern in TOOL_CALL_TEXT_PATTERNS):
        reasons.append("invalid_tool_call_text")

    if not passed and tool_results > 0:
        verifier_text = "\n".join(
            f"{v.get('stdout', '')}\n{v.get('stderr', '')}" for v in result.get("verifiers", [])
        ).lower()
        if verifier_text or any(word in combined for word in ("failed", "traceback", "assertionerror", "error")):
            reasons.append("ignored_observation_or_failed_recovery")

    if not passed and any(pattern in combined for pattern in FILE_ERROR_PATTERNS):
        reasons.append("hallucinated_file_or_path")

    if any(pattern in command_text for pattern in UNSAFE_COMMAND_PATTERNS):
        reasons.append("unsafe_command")

    if stats["tool_uses"] > 40:
        reasons.append("overlong_tool_loop")

    if float(result.get("elapsed_seconds") or 0) > 300:
        reasons.append("overlong_elapsed")

    # Preserve order while deduplicating.
    return list(dict.fromkeys(reasons))


def failure_reasons(reasons: list[str]) -> list[str]:
    failure_tags = {
        "invalid_tool_call_text",
        "wrong_tool_or_no_tool_use",
        "premature_final_answer",
        "ignored_observation_or_failed_recovery",
        "hallucinated_file_or_path",
        "unsafe_command",
        "overlong_tool_loop",
        "overlong_elapsed",
        "missing_matched_trace",
        "nonzero_hermes_returncode",
        "verifier_failed",
    }
    return [reason for reason in reasons if reason in failure_tags]


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
        raw_trace = trace_text(trace_path)
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
            "trace_raw": raw_trace,
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
                "chosen_failure_reasons": failure_reasons(chosen["preference_reasons"]),
                "rejected_failure_reasons": failure_reasons(rejected["preference_reasons"]),
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
        "rejected_failure_reasons": {},
        "output": str(out),
    }
    for pair in pairs:
        summary["families"].setdefault(pair["family"], 0)
        summary["families"][pair["family"]] += 1
        for reason in pair["rejected_failure_reasons"]:
            summary["rejected_failure_reasons"].setdefault(reason, 0)
            summary["rejected_failure_reasons"][reason] += 1
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
