#!/usr/bin/env python3
"""Validate all format variants in the format ablation dataset directory.

Checks:
- Same canonical IDs across all variants
- No missing outputs
- Valid JSONL (each line parses)
- Reasonable token lengths (< 2048 tokens estimated)
- Role alternation for chat data (user/assistant/user/assistant)
- No assistant-empty rows
- No duplicate IDs
- No eval contamination (compare IDs against eval dataset)
- Each format has correct schema

Usage:
    python scripts/data/validate_dataset_formats.py \
        --dataset-dir data/sft/format_ablation/ \
        --canonical data/canonical/phase9_pilot_300.jsonl

Exits 0 on success, 1 on failure.
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_TOKEN_ESTIMATE = 2048

# Expected schemas per format
SCHEMA_CHECKS = {
    "alpaca_flat": lambda r: all(k in r for k in ("instruction", "input", "output")),
    "single_turn_chat": lambda r: "messages" in r and isinstance(r["messages"], list),
    "multi_turn_concise": lambda r: "messages" in r and isinstance(r["messages"], list),
    "multi_turn_verbose": lambda r: "messages" in r and isinstance(r["messages"], list),
    "structured_terse": lambda r: "messages" in r and isinstance(r["messages"], list),
    "bad_format_control": lambda r: "messages" in r and isinstance(r["messages"], list),
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def load_jsonl_ids(path: str) -> set[str]:
    """Load all IDs from a JSONL file (uses _canonical_id or id field)."""
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("_canonical_id") or obj.get("id", "")
            if cid:
                ids.add(cid)
    return ids


def validate_format_file(path: str, fmt: str) -> list[str]:
    """Validate a single format file. Returns list of error messages."""
    errors = []
    basename = os.path.basename(path)

    if not os.path.exists(path):
        errors.append(f"[{basename}] File does not exist: {path}")
        return errors

    schema_check = SCHEMA_CHECKS.get(fmt)
    seen_ids = set()
    line_num = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line_num += 1
            line = raw_line.strip()
            if not line:
                continue

            # Valid JSONL
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"[{basename}] Line {line_num}: invalid JSON: {e}")
                continue

            # Schema check
            if schema_check and not schema_check(record):
                errors.append(
                    f"[{basename}] Line {line_num}: schema mismatch for format '{fmt}'"
                )

            # Canonical ID present
            cid = record.get("_canonical_id") or record.get("id", "")
            if not cid:
                errors.append(f"[{basename}] Line {line_num}: missing _canonical_id")

            # Duplicate ID check
            if cid in seen_ids:
                errors.append(f"[{basename}] Line {line_num}: duplicate ID '{cid}'")
            seen_ids.add(cid)

            # Token length check
            total_text = json.dumps(record, ensure_ascii=False)
            tok_est = estimate_tokens(total_text)
            if tok_est > MAX_TOKEN_ESTIMATE:
                errors.append(
                    f"[{basename}] Line {line_num} ({cid}): estimated {tok_est} tokens > {MAX_TOKEN_ESTIMATE}"
                )

            # Chat-specific checks
            if "messages" in record:
                messages = record["messages"]
                if not isinstance(messages, list):
                    errors.append(f"[{basename}] Line {line_num}: 'messages' is not a list")
                    continue

                if len(messages) == 0:
                    errors.append(f"[{basename}] Line {line_num} ({cid}): empty messages array")
                    continue

                # Role alternation and no empty assistant
                for i, msg in enumerate(messages):
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role not in ("user", "assistant", "system"):
                        errors.append(
                            f"[{basename}] Line {line_num} ({cid}): message {i} has invalid role '{role}'"
                        )

                    if role == "assistant" and not content.strip():
                        errors.append(
                            f"[{basename}] Line {line_num} ({cid}): message {i} has empty assistant content"
                        )

                # Check role alternation (user/assistant should alternate)
                non_system = [m for m in messages if m.get("role") != "system"]
                for i in range(len(non_system) - 1):
                    if non_system[i].get("role") == non_system[i + 1].get("role"):
                        errors.append(
                            f"[{basename}] Line {line_num} ({cid}): consecutive '{non_system[i]['role']}' roles at messages {i},{i+1}"
                        )

            # Alpaca-specific: check non-empty output
            if fmt == "alpaca_flat":
                output = record.get("output", "")
                if not output.strip():
                    errors.append(
                        f"[{basename}] Line {line_num} ({cid}): empty 'output' field"
                    )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate all format variants in the ablation dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing format variant JSONL files.",
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to canonical JSONL file (for ID cross-check).",
    )
    parser.add_argument(
        "--eval-dataset",
        default="data/eval/small_model_eval_v1.jsonl",
        help="Path to eval dataset for contamination check.",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    format_files: dict[str, str] = {}

    # Discover format files
    expected_formats = [
        "alpaca_flat",
        "single_turn_chat",
        "multi_turn_concise",
        "multi_turn_verbose",
        "structured_terse",
        "bad_format_control",
    ]

    for fmt in expected_formats:
        path = os.path.join(args.dataset_dir, f"{fmt}.jsonl")
        if os.path.exists(path):
            format_files[fmt] = path
        else:
            all_errors.append(f"Missing format file: {path}")

    if not format_files:
        logger.error("No format files found in %s", args.dataset_dir)
        print("\n=== VALIDATION FAILED ===")
        print(f"Errors: {len(all_errors)}")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    logger.info("Found %d format files: %s", len(format_files), list(format_files.keys()))

    # Load canonical IDs
    if os.path.exists(args.canonical):
        canonical_ids = load_jsonl_ids(args.canonical)
        logger.info("Canonical dataset has %d unique IDs", len(canonical_ids))
    else:
        canonical_ids = set()
        logger.warning("Canonical file not found: %s (skipping ID cross-check)", args.canonical)

    # Load eval IDs for contamination check
    eval_ids = set()
    if os.path.exists(args.eval_dataset):
        eval_ids = load_jsonl_ids(args.eval_dataset)
        logger.info("Eval dataset has %d IDs for contamination check", len(eval_ids))
    else:
        logger.info("Eval dataset not found: %s (skipping contamination check)", args.eval_dataset)

    # Validate each format file
    format_ids: dict[str, set[str]] = {}
    for fmt, path in format_files.items():
        logger.info("Validating %s ...", path)
        errs = validate_format_file(path, fmt)
        all_errors.extend(errs)
        format_ids[fmt] = load_jsonl_ids(path)

    # Cross-format ID consistency
    if format_ids:
        reference_fmt = list(format_ids.keys())[0]
        reference_ids = format_ids[reference_fmt]
        for fmt, ids in format_ids.items():
            if fmt == reference_fmt:
                continue
            missing = reference_ids - ids
            extra = ids - reference_ids
            if missing:
                all_errors.append(
                    f"[{fmt}] Missing {len(missing)} IDs present in {reference_fmt}: "
                    f"{list(missing)[:5]}..."
                )
            if extra:
                all_errors.append(
                    f"[{fmt}] Has {len(extra)} extra IDs not in {reference_fmt}: "
                    f"{list(extra)[:5]}..."
                )

    # Cross-check against canonical
    if canonical_ids and format_ids:
        first_fmt = list(format_ids.keys())[0]
        format_id_set = format_ids[first_fmt]
        missing_from_format = canonical_ids - format_id_set
        extra_in_format = format_id_set - canonical_ids
        if missing_from_format:
            all_errors.append(
                f"Format files missing {len(missing_from_format)} canonical IDs: "
                f"{list(missing_from_format)[:5]}..."
            )
        if extra_in_format:
            all_errors.append(
                f"Format files have {len(extra_in_format)} IDs not in canonical: "
                f"{list(extra_in_format)[:5]}..."
            )

    # Eval contamination check
    if eval_ids and format_ids:
        for fmt, ids in format_ids.items():
            overlap = ids & eval_ids
            if overlap:
                all_errors.append(
                    f"[{fmt}] EVAL CONTAMINATION: {len(overlap)} IDs overlap with eval set: "
                    f"{list(overlap)[:5]}..."
                )

    # Print report
    print("\n" + "=" * 60)
    print("FORMAT ABLATION VALIDATION REPORT")
    print("=" * 60)
    print(f"\nDataset directory: {args.dataset_dir}")
    print(f"Canonical source:  {args.canonical}")
    print(f"Formats validated: {len(format_files)}")
    for fmt in expected_formats:
        if fmt in format_files:
            count = len(format_ids.get(fmt, set()))
            print(f"  ✓ {fmt}: {count} examples")
        else:
            print(f"  ✗ {fmt}: MISSING")

    if canonical_ids:
        print(f"\nCanonical IDs: {len(canonical_ids)}")
    if eval_ids:
        print(f"Eval IDs (contamination check): {len(eval_ids)}")

    print()
    if all_errors:
        print(f"ERRORS FOUND: {len(all_errors)}")
        for i, err in enumerate(all_errors, 1):
            print(f"  {i}. {err}")
        print("\n=== VALIDATION FAILED ===")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print("\n=== VALIDATION SUCCEEDED ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
