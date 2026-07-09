#!/usr/bin/env python3
"""Create deterministic training-ready SFT and DPO splits from trajectory exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stable_bucket(key: str, modulo: int = 1000) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


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


def sanitize_transcript_text(text: str) -> str:
    """Remove high-volume workspace noise from already-rendered transcripts."""
    cleaned_lines = []
    for line in text.splitlines():
        if line.startswith("tool_result: "):
            payload = line.removeprefix("tool_result: ")
            cleaned_lines.append(f"tool_result: {sanitize_tool_result_content(payload)}")
            continue
        cleaned_lines.append(
            STANDALONE_NOISY_NAME_RE.sub(
                "[omitted noisy path]",
                NOISY_PATH_FRAGMENT_RE.sub(
                    "[omitted noisy path]",
                    QUOTED_NOISY_PATH_RE.sub('"[omitted noisy path]"', line),
                ),
            )
        )
    return "\n".join(cleaned_lines)


def trace_transcript(trace_paths: list[str], limit: int, include_user_text: bool = False) -> str:
    """Render the matched Hermes trace as a compact tool-use transcript."""
    if not trace_paths:
        return ""
    path = Path(trace_paths[0])
    if not path.exists():
        return ""
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            if role == "user" and kind == "text" and not include_user_text:
                continue
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


def strip_leading_user_turn(transcript: str) -> str:
    """Remove a leading rendered user prompt from compact transcript text."""
    if not transcript.startswith("user: "):
        return transcript
    marker = "\nassistant"
    idx = transcript.find(marker)
    if idx == -1:
        return transcript
    return transcript[idx + 1 :]


def split_rows(rows: list[dict[str, Any]], val_pct: int) -> dict[str, list[dict[str, Any]]]:
    """Stratify by family with stable row ordering and at least one val row per family."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family", "unknown"))].append(row)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for family, family_rows in sorted(by_family.items()):
        ranked = sorted(
            family_rows,
            key=lambda row: stable_bucket(
                f"{family}:{row.get('task_id') or row.get('source_result') or row.get('prompt', '')}",
                modulo=1_000_000,
            ),
        )
        n_val = round(len(ranked) * (val_pct / 100.0))
        if len(ranked) >= 10:
            n_val = max(1, n_val)
        n_val = min(n_val, max(len(ranked) - 1, 0))
        val_ids = {id(row) for row in ranked[:n_val]}
        for row in family_rows:
            splits["val" if id(row) in val_ids else "train"].append(row)
    return splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_failure_reason(row: dict[str, Any]) -> bool:
    return bool(row.get("rejected_failure_reasons"))


def family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("family", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def sft_row(row: dict[str, Any], trajectory_limit: int) -> dict[str, Any]:
    trace_paths = row.get("matched_session_trace_files", [])
    trajectory = trace_transcript(trace_paths, trajectory_limit, include_user_text=False)
    return {
        "task_id": row.get("task_id"),
        "family": row.get("family"),
        "prompt": row.get("prompt"),
        "messages": row.get("messages", []),
        "trajectory": trajectory,
        "trajectory_messages": [
            {
                "role": "system",
                "content": "You are a pragmatic code/sysadmin agent. Use tools to inspect, modify, and verify before finalizing.",
            },
            {"role": "user", "content": row.get("prompt")},
            {"role": "assistant", "content": trajectory or row.get("assistant_final")},
        ],
        "assistant_final": row.get("assistant_final"),
        "diff": row.get("diff"),
        "diff_stat": row.get("diff_stat"),
        "verifiers": row.get("verifiers", []),
        "usage": row.get("usage", {}),
        "matched_session_trace_files": trace_paths,
        "source_result": row.get("source_result"),
    }


def dpo_row(row: dict[str, Any]) -> dict[str, Any]:
    chosen = sanitize_transcript_text(strip_leading_user_turn(row.get("chosen") or ""))
    rejected = sanitize_transcript_text(strip_leading_user_turn(row.get("rejected") or ""))
    prompt = row.get("prompt")
    return {
        "task_id": row.get("task_id"),
        "family": row.get("family"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected_messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        "chosen_score": row.get("chosen_score"),
        "rejected_score": row.get("rejected_score"),
        "chosen_reasons": row.get("chosen_reasons", []),
        "rejected_reasons": row.get("rejected_reasons", []),
        "chosen_failure_reasons": row.get("chosen_failure_reasons", []),
        "rejected_failure_reasons": row.get("rejected_failure_reasons", []),
        "chosen_stats": row.get("chosen_stats", {}),
        "rejected_stats": row.get("rejected_stats", {}),
        "chosen_source": row.get("chosen_source"),
        "rejected_source": row.get("rejected_source"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-pct", type=int, default=5)
    parser.add_argument("--trajectory-limit", type=int, default=32000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sft_rows = [sft_row(row, args.trajectory_limit) for row in load_jsonl(Path(args.sft))]
    dpo_rows = [dpo_row(row) for row in load_jsonl(Path(args.dpo))]

    sft_splits = split_rows(sft_rows, args.val_pct)
    dpo_splits = split_rows(dpo_rows, args.val_pct)
    dpo_hard_rows = [row for row in dpo_rows if has_failure_reason(row)]
    dpo_efficiency_rows = [row for row in dpo_rows if not has_failure_reason(row)]
    dpo_hard_splits = split_rows(dpo_hard_rows, args.val_pct)
    dpo_efficiency_splits = split_rows(dpo_efficiency_rows, args.val_pct)

    outputs = {
        "sft_train": out_dir / "sft_train.jsonl",
        "sft_val": out_dir / "sft_val.jsonl",
        "dpo_train": out_dir / "dpo_train.jsonl",
        "dpo_val": out_dir / "dpo_val.jsonl",
        "dpo_hard_train": out_dir / "dpo_hard_train.jsonl",
        "dpo_hard_val": out_dir / "dpo_hard_val.jsonl",
        "dpo_efficiency_train": out_dir / "dpo_efficiency_train.jsonl",
        "dpo_efficiency_val": out_dir / "dpo_efficiency_val.jsonl",
    }
    write_jsonl(outputs["sft_train"], sft_splits["train"])
    write_jsonl(outputs["sft_val"], sft_splits["val"])
    write_jsonl(outputs["dpo_train"], dpo_splits["train"])
    write_jsonl(outputs["dpo_val"], dpo_splits["val"])
    write_jsonl(outputs["dpo_hard_train"], dpo_hard_splits["train"])
    write_jsonl(outputs["dpo_hard_val"], dpo_hard_splits["val"])
    write_jsonl(outputs["dpo_efficiency_train"], dpo_efficiency_splits["train"])
    write_jsonl(outputs["dpo_efficiency_val"], dpo_efficiency_splits["val"])

    manifest = {
        "source_sft": args.sft,
        "source_dpo": args.dpo,
        "val_pct": args.val_pct,
        "trajectory_limit": args.trajectory_limit,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "sft": {
            "total": len(sft_rows),
            "train": len(sft_splits["train"]),
            "val": len(sft_splits["val"]),
            "with_trajectory": sum(1 for row in sft_rows if row.get("trajectory")),
            "families": family_counts(sft_rows),
            "train_families": family_counts(sft_splits["train"]),
            "val_families": family_counts(sft_splits["val"]),
        },
        "dpo": {
            "total": len(dpo_rows),
            "train": len(dpo_splits["train"]),
            "val": len(dpo_splits["val"]),
            "hard_total": len(dpo_hard_rows),
            "hard_train": len(dpo_hard_splits["train"]),
            "hard_val": len(dpo_hard_splits["val"]),
            "efficiency_total": len(dpo_efficiency_rows),
            "efficiency_train": len(dpo_efficiency_splits["train"]),
            "efficiency_val": len(dpo_efficiency_splits["val"]),
            "families": family_counts(dpo_rows),
            "train_families": family_counts(dpo_splits["train"]),
            "val_families": family_counts(dpo_splits["val"]),
            "hard_families": family_counts(dpo_hard_rows),
            "efficiency_families": family_counts(dpo_efficiency_rows),
            "rejected_failure_reasons": dict(
                sorted(
                    Counter(
                        reason
                        for row in dpo_rows
                        for reason in row.get("rejected_failure_reasons", [])
                    ).items()
                )
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
