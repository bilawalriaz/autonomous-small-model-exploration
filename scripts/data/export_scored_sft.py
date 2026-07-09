#!/usr/bin/env python3
"""Export teacher-scored rollout labels into clean SFT candidate files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assistant_content(row: dict, include_reasoning: bool) -> str:
    response = (row.get("winner_response") or "").strip()
    reasoning = (row.get("condensed_reasoning") or "").strip()
    if include_reasoning and reasoning:
        return f"Reasoning:\n{reasoning}\n\nAnswer:\n{response}".strip()
    return response


def convert_row(row: dict, include_reasoning: bool) -> dict:
    prompt = row.get("prompt") or ""
    content = assistant_content(row, include_reasoning=include_reasoning)
    return {
        "prompt_hash": row.get("prompt_hash"),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ],
        "prompt": prompt,
        "assistant_response": content,
        "winner_response": row.get("winner_response", ""),
        "condensed_reasoning": row.get("condensed_reasoning", ""),
        "quality_score": row.get("quality_score"),
        "correctness": row.get("correctness"),
        "format_valid": row.get("format_valid"),
        "winner_index": row.get("winner_index"),
        "ranking": row.get("ranking", []),
        "teacher_provider": row.get("teacher_provider"),
        "teacher_model": row.get("teacher_model"),
        "teacher_name": row.get("teacher_name"),
        "source_timestamp": row.get("timestamp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="/Users/bilawalriaz/scored/scored.jsonl")
    parser.add_argument("--errors", default="/Users/bilawalriaz/scored/errors.jsonl")
    parser.add_argument("--outdir", default="/Users/bilawalriaz/scored/exports")
    parser.add_argument("--min-quality", type=int, default=8)
    parser.add_argument("--include-reasoning", action="store_true")
    args = parser.parse_args()

    scored_path = Path(args.scored)
    errors_path = Path(args.errors)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(scored_path)
    errors = load_jsonl(errors_path)

    clean = [
        row for row in rows
        if row.get("format_valid") is True
        and row.get("quality_score", 0) >= args.min_quality
        and row.get("correctness") in ("correct", "na")
    ]
    quarantine = [row for row in rows if row not in clean]

    suffix = f"q{args.min_quality}_{'reasoning' if args.include_reasoning else 'response'}"
    clean_path = outdir / f"sft_clean_{suffix}.jsonl"
    quarantine_path = outdir / f"sft_quarantine_{suffix}.jsonl"
    manifest_path = outdir / f"sft_manifest_{suffix}.json"

    with clean_path.open("w") as f:
        for row in clean:
            f.write(json.dumps(convert_row(row, include_reasoning=args.include_reasoning), ensure_ascii=False) + "\n")

    with quarantine_path.open("w") as f:
        for row in quarantine:
            f.write(json.dumps(convert_row(row, include_reasoning=args.include_reasoning), ensure_ascii=False) + "\n")

    manifest = {
        "source_scored": str(scored_path),
        "source_errors": str(errors_path),
        "clean_output": str(clean_path),
        "quarantine_output": str(quarantine_path),
        "total_scored_rows": len(rows),
        "clean_rows": len(clean),
        "quarantine_rows": len(quarantine),
        "error_rows": len(errors),
        "filters": {
            "format_valid": True,
            "min_quality": args.min_quality,
            "correctness": ["correct", "na"],
            "include_reasoning": args.include_reasoning,
        },
        "quality_histogram": dict(sorted(Counter(row.get("quality_score") for row in rows).items())),
        "clean_quality_histogram": dict(sorted(Counter(row.get("quality_score") for row in clean).items())),
        "correctness_histogram": dict(Counter(row.get("correctness") for row in rows)),
        "format_valid_rows": sum(1 for row in rows if row.get("format_valid") is True),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
