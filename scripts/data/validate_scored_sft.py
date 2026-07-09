#!/usr/bin/env python3
"""Deterministically validate teacher-scored rollout labels before SFT export."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jsonschema
import yaml


FENCE_RE = re.compile(r"^\s*```(?:json|yaml|yml)?\s*\n(?P<body>.*?)(?:\n```\s*)$", re.DOTALL | re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def strip_outer_fence(text: str) -> tuple[str, bool]:
    match = FENCE_RE.match(text.strip())
    if match:
        return match.group("body").strip(), True
    return text.strip(), False


def parse_response(text: str, output_format: str | None) -> tuple[object, bool]:
    body, had_fence = strip_outer_fence(text)
    fmt = (output_format or "").lower()
    if fmt == "json":
        return json.loads(body), had_fence
    if fmt == "yaml":
        return yaml.safe_load(body), had_fence

    # Unknown format: accept either JSON or YAML if it parses.
    try:
        return json.loads(body), had_fence
    except Exception:
        return yaml.safe_load(body), had_fence


def validate_against_schema(parsed: object, schema_text: str | None) -> tuple[bool, str]:
    if not schema_text:
        return True, ""
    schema = json.loads(schema_text)
    try:
        jsonschema.validate(parsed, schema)
        return True, ""
    except jsonschema.ValidationError as first_error:
        # Some prompts ask for a wrapper object around an array while prompt_meta
        # stores only the array item schema. If the response is a single-key object
        # containing a list, validate that list as a conservative compatibility path.
        if schema.get("type") == "array" and isinstance(parsed, dict) and len(parsed) == 1:
            value = next(iter(parsed.values()))
            try:
                jsonschema.validate(value, schema)
                return True, "validated_single_key_wrapper_value"
            except jsonschema.ValidationError:
                pass
        return False, first_error.message


def assistant_content(row: dict, include_reasoning: bool) -> str:
    response = (row.get("winner_response") or "").strip()
    reasoning = (row.get("condensed_reasoning") or "").strip()
    if include_reasoning and reasoning:
        return f"Reasoning:\n{reasoning}\n\nAnswer:\n{response}".strip()
    return response


def convert_row(row: dict, include_reasoning: bool, validation: dict) -> dict:
    prompt = row.get("prompt") or ""
    return {
        "prompt_hash": row.get("prompt_hash"),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content(row, include_reasoning)},
        ],
        "prompt": prompt,
        "assistant_response": assistant_content(row, include_reasoning),
        "winner_response": row.get("winner_response", ""),
        "condensed_reasoning": row.get("condensed_reasoning", ""),
        "quality_score": row.get("quality_score"),
        "correctness": row.get("correctness"),
        "format_valid": row.get("format_valid"),
        "teacher_model": row.get("teacher_model"),
        "teacher_name": row.get("teacher_name"),
        "source_timestamp": row.get("timestamp"),
        "deterministic_validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="/Users/bilawalriaz/scored/scored.jsonl")
    parser.add_argument("--outdir", default="/Users/bilawalriaz/scored/exports")
    parser.add_argument("--min-quality", type=int, default=8)
    parser.add_argument("--include-reasoning", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.scored))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    teacher_clean = [
        row for row in rows
        if row.get("format_valid") is True
        and row.get("quality_score", 0) >= args.min_quality
        and row.get("correctness") in ("correct", "na")
    ]

    validated = []
    rejected = []
    reasons = Counter()
    for row in teacher_clean:
        meta = row.get("prompt_meta") or {}
        output_format = meta.get("output_format")
        schema_text = meta.get("json_schema")
        try:
            parsed, had_fence = parse_response(row.get("winner_response") or "", output_format)
        except Exception as exc:
            reason = f"parse_failed:{type(exc).__name__}"
            reasons[reason] += 1
            rejected.append({"row": row, "validation": {"valid": False, "reason": reason}})
            continue

        schema_ok, schema_note = validate_against_schema(parsed, schema_text)
        if not schema_ok:
            reason = f"schema_failed:{schema_note}"
            reasons[reason] += 1
            rejected.append({"row": row, "validation": {"valid": False, "reason": reason, "had_fence": had_fence}})
            continue

        validation = {
            "valid": True,
            "output_format": output_format,
            "had_fence": had_fence,
            "schema_note": schema_note,
        }
        validated.append((row, validation))

    suffix = f"q{args.min_quality}_{'reasoning' if args.include_reasoning else 'response'}"
    valid_path = outdir / f"sft_strict_{suffix}.jsonl"
    reject_path = outdir / f"sft_strict_rejected_{suffix}.jsonl"
    manifest_path = outdir / f"sft_strict_manifest_{suffix}.json"

    with valid_path.open("w") as f:
        for row, validation in validated:
            f.write(json.dumps(convert_row(row, args.include_reasoning, validation), ensure_ascii=False) + "\n")

    with reject_path.open("w") as f:
        for item in rejected:
            f.write(json.dumps({
                "prompt_hash": item["row"].get("prompt_hash"),
                "quality_score": item["row"].get("quality_score"),
                "correctness": item["row"].get("correctness"),
                "format_valid": item["row"].get("format_valid"),
                "validation": item["validation"],
            }, ensure_ascii=False) + "\n")

    manifest = {
        "source_scored": str(Path(args.scored)),
        "valid_output": str(valid_path),
        "rejected_output": str(reject_path),
        "total_scored_rows": len(rows),
        "teacher_clean_rows": len(teacher_clean),
        "strict_valid_rows": len(validated),
        "strict_rejected_rows": len(rejected),
        "filters": {
            "format_valid": True,
            "min_quality": args.min_quality,
            "correctness": ["correct", "na"],
            "include_reasoning": args.include_reasoning,
            "parse_output": True,
            "validate_prompt_meta_json_schema": True,
        },
        "rejection_reasons": dict(reasons.most_common(30)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
