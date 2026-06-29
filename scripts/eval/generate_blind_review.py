#!/usr/bin/env python3
"""Generate stratified blind review samples for human assessment.

Produces a markdown file with side-by-side model outputs, anonymized as
Model X / Model Y (randomized assignment per example). At least 60 examples
stratified across all 9 eval categories. The reviewer does not know which
model is which until the review is unblinded.

CLI:
    # Single model vs base (most common)
    python scripts/eval/generate_blind_review.py \
        --run-ids lfm2_230m_quality_multi_turn_verbose_20260629 \
                  lfm2_230m_base_20260629 \
        --min-per-category 6 \
        --seed 42 \
        --output results/evals/blind_review_multi_turn_verbose_vs_base.md

    # Multi-model comparison (9-way)
    python scripts/eval/generate_blind_review.py \
        --run-ids lfm2_230m_base_20260629 \
                  lfm2_230m_quality_alpaca_flat_20260629 \
                  lfm2_230m_quality_single_turn_chat_20260629 \
                  lfm2_230m_quality_multi_turn_concise_20260629 \
                  lfm2_230m_quality_multi_turn_verbose_20260629 \
                  lfm2_230m_quality_structured_terse_20260629 \
                  lfm2_230m_quality_bad_format_control_20260629 \
                  lfm2_230m_quality_bsmagpie_surgical_20260629 \
                  lfm2_230m_surgical_bsmagpie_surgical_20260629 \
        --pairwise \
        --min-per-category 7 \
        --output results/evals/blind_review_all_formats.md
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EVAL_CATEGORIES = [
    "instruction_following",
    "json_structured",
    "gamefaq_extraction",
    "coding",
    "deobfuscation",
    "reasoning",
    "concision_antislip",
    "factual_qa",
    "multi_turn",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stable_shuffle(items: list, seed: int) -> list:
    """Deterministic shuffle using hashlib, not Python's hash()."""
    indexed = list(enumerate(items))
    def sort_key(pair):
        idx, _ = pair
        h = hashlib.sha256(f"shuffle:{seed}:{idx}".encode()).hexdigest()
        return h
    indexed.sort(key=sort_key)
    return [item for _, item in indexed]


def stable_choice(options: list, seed_str: str) -> int:
    """Deterministic choice index using hashlib."""
    h = hashlib.sha256(f"choice:{seed_str}".encode()).hexdigest()
    return int(h[:8], 16) % len(options)


def truncate(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def build_pairwise_blind_review(
    run_ids: list[str],
    run_outputs: dict[str, dict[str, dict]],
    eval_data: dict[str, dict],
    min_per_category: int,
    seed: int,
    max_total: int = 80,
) -> list[dict]:
    """Build stratified blind review examples for pairwise comparison.

    Each example has: prompt, response_X, response_Y, category, eval_id.
    X/Y assignment is randomized per example.
    """
    # For pairwise: compare first run_id against all others
    primary = run_ids[0]
    comparators = run_ids[1:]

    # Collect eval_ids present across all runs
    all_eval_ids = set(run_outputs[primary].keys())
    for comp in comparators:
        all_eval_ids &= set(run_outputs[comp].keys())

    # Group by category
    by_category = defaultdict(list)
    for eid in all_eval_ids:
        cat = eval_data.get(eid, {}).get("category", "unknown")
        by_category[cat].append(eid)

    # Stratified sampling
    selected = []
    used_ids = set()

    for cat in EVAL_CATEGORIES:
        available = [eid for eid in by_category.get(cat, []) if eid not in used_ids]
        if not available:
            available = [eid for eid in by_category.get(cat, [])]

        if len(available) <= min_per_category:
            chosen = available
        else:
            # Stable selection
            shuffled = stable_shuffle(available, seed + hash(cat) % 10000)
            chosen = shuffled[:min_per_category]

        for eid in chosen:
            used_ids.add(eid)
            # Pick a random comparator for this example
            if len(comparators) == 0:
                comp = primary  # self-comparison fallback
            elif len(comparators) == 1:
                comp = comparators[0]
            else:
                ci = stable_choice(comparators, f"{eid}:{seed}")
                comp = comparators[ci]

            # Randomize X/Y assignment
            flip = stable_choice(["A", "B"], f"flip:{eid}:{seed}") == 1
            if flip:
                label_x, label_y = comp, primary
            else:
                label_x, label_y = primary, comp

            selected.append({
                "eval_id": eid,
                "category": cat,
                "prompt": eval_data.get(eid, {}).get("prompt", "N/A"),
                "response_x": run_outputs.get(label_x, {}).get(eid, {}).get("generated_response", "N/A"),
                "response_y": run_outputs.get(label_y, {}).get(eid, {}).get("generated_response", "N/A"),
                "model_x_real": label_x,
                "model_y_real": label_y,
                "flipped": flip,
            })

    # Fill to max_total with random from unused
    remaining = [eid for eid in all_eval_ids if eid not in used_ids]
    remaining = stable_shuffle(remaining, seed + 9999)
    for eid in remaining:
        if len(selected) >= max_total:
            break
        cat = eval_data.get(eid, {}).get("category", "unknown")
        ci = stable_choice(comparators, f"{eid}:{seed}:fill")
        comp = comparators[ci] if comparators else primary
        flip = stable_choice(["A", "B"], f"flip:{eid}:{seed}:fill") == 1
        if flip:
            label_x, label_y = comp, primary
        else:
            label_x, label_y = primary, comp

        selected.append({
            "eval_id": eid,
            "category": cat,
            "prompt": eval_data.get(eid, {}).get("prompt", "N/A"),
            "response_x": run_outputs.get(label_x, {}).get(eid, {}).get("generated_response", "N/A"),
            "response_y": run_outputs.get(label_y, {}).get(eid, {}).get("generated_response", "N/A"),
            "model_x_real": label_x,
            "model_y_real": label_y,
            "flipped": flip,
        })

    return selected


def render_markdown(
    selected: list[dict],
    run_ids: list[str],
    seed: int,
) -> str:
    """Render blind review as markdown with anonymized model labels."""
    lines = [
        "# Blind Review: Phase 9 Data Format Ablation",
        "",
        f"**Models under review:** {len(run_ids)}",
        f"**Total examples:** {len(selected)}",
        f"**Seed:** {seed}",
        "",
        "**IMPORTANT:** This review is blinded. Model X and Model Y assignments",
        "are randomized per example. Do NOT try to deduce which model is which",
        "from ordering alone.",
        "",
        "The unblinding key is stored in the JSON companion file.",
        "",
        "---",
        "",
    ]

    # Category distribution
    cat_counts = defaultdict(int)
    for s in selected:
        cat_counts[s["category"]] += 1
    lines.append("## Category Distribution")
    lines.append("")
    for cat in EVAL_CATEGORIES:
        count = cat_counts.get(cat, 0)
        lines.append(f"- **{cat}:** {count} examples")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, item in enumerate(selected, 1):
        cat = item["category"]
        eid = item["eval_id"]
        prompt = item["prompt"]
        resp_x = item["response_x"]
        resp_y = item["response_y"]

        lines.append(f"## Example {i}: `{eid}` [{cat}]")
        lines.append("")
        lines.append("### Prompt")
        lines.append("```")
        lines.append(truncate(prompt, 1500))
        lines.append("```")
        lines.append("")
        lines.append("### Model X")
        lines.append("```")
        lines.append(truncate(resp_x, 1500))
        lines.append("```")
        lines.append("")
        lines.append("### Model Y")
        lines.append("```")
        lines.append(truncate(resp_y, 1500))
        lines.append("```")
        lines.append("")
        lines.append("### Assessment")
        lines.append("- [ ] Model X is clearly better")
        lines.append("- [ ] Model X is slightly better")
        lines.append("- [ ] About the same")
        lines.append("- [ ] Model Y is slightly better")
        lines.append("- [ ] Model Y is clearly better")
        lines.append("- [ ] Both are poor / unhelpful")
        lines.append("")
        lines.append("**What makes the better one better?**")
        lines.append("_[free text]_")
        lines.append("")
        lines.append("**What specific errors or weaknesses did you notice?**")
        lines.append("_[free text]_")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Total examples reviewed: {len(selected)}")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat in EVAL_CATEGORIES:
        lines.append(f"| {cat} | {cat_counts.get(cat, 0)} |")
    lines.append("")

    return "\n".join(lines)


def render_unblinding_json(selected: list[dict], run_ids: list[str], seed: int) -> dict:
    """Generate the unblinding key for later analysis."""
    key = {}
    for i, item in enumerate(selected):
        key[f"example_{i+1}"] = {
            "eval_id": item["eval_id"],
            "category": item["category"],
            "model_x_is": item["model_x_real"],
            "model_y_is": item["model_y_real"],
            "flipped_from_primary": item["flipped"],
        }
    return {
        "blind_review_seed": seed,
        "run_ids": run_ids,
        "primary_model": run_ids[0],
        "example_count": len(selected),
        "unblinding_key": key,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate stratified blind review samples")
    parser.add_argument("--run-ids", nargs="+", required=True,
                        help="Run IDs to compare. First is primary. Provide at least 2.")
    parser.add_argument("--pairwise", action="store_true",
                        help="Compare primary against each comparator pairwise")
    parser.add_argument("--min-per-category", type=int, default=7,
                        help="Minimum examples per eval category (default: 7)")
    parser.add_argument("--max-total", type=int, default=80,
                        help="Maximum total examples (default: 80)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling/blinding")
    parser.add_argument("--output", default=None,
                        help="Output path. Default: results/evals/blind_review_<timestamp>.md")
    args = parser.parse_args()

    if len(args.run_ids) < 2:
        log.error("Need at least 2 run IDs to compare")
        sys.exit(1)

    # Load outputs for all run_ids
    run_outputs = {}
    eval_data = {}
    for rid in args.run_ids:
        run_dir = PROJECT_ROOT / "results" / "evals" / rid
        outputs_path = run_dir / "outputs.jsonl"
        if not outputs_path.exists():
            log.warning(f"Outputs not found for {rid}: {outputs_path}")
            continue
        outputs = load_jsonl(outputs_path)
        run_outputs[rid] = {o["eval_id"]: o for o in outputs}
        # Also build eval_data lookup
        for o in outputs:
            eid = o["eval_id"]
            if eid not in eval_data:
                eval_data[eid] = {
                    "eval_id": eid,
                    "category": o.get("category", "unknown"),
                    "prompt": o.get("prompt", ""),
                }
        log.info(f"Loaded {len(outputs)} outputs for {rid}")

    if len(run_outputs) < 2:
        log.error(f"Need at least 2 run IDs with outputs. Found: {list(run_outputs.keys())}")
        sys.exit(1)

    available_ids = list(run_outputs.keys())
    log.info(f"Available run IDs with outputs: {available_ids}")

    # Build review
    selected = build_pairwise_blind_review(
        available_ids, run_outputs, eval_data,
        args.min_per_category, args.seed, args.max_total,
    )

    if not selected:
        log.error("No examples selected — check that outputs exist")
        sys.exit(1)

    # Render
    md_content = render_markdown(selected, available_ids, args.seed)
    unblinding = render_unblinding_json(selected, available_ids, args.seed)

    # Write output
    if args.output:
        md_path = Path(args.output)
    else:
        md_path = PROJECT_ROOT / "results" / "evals" / "blind_review.md"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(md_content)
    log.info(f"Wrote blind review to {md_path}")

    # Write unblinding key
    json_path = md_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(unblinding, f, indent=2)
    log.info(f"Wrote unblinding key to {json_path}")

    # Summary
    cat_counts = defaultdict(int)
    for s in selected:
        cat_counts[s["category"]] += 1
    print(f"\n{'='*60}")
    print(f"  BLIND REVIEW GENERATED")
    print(f"{'='*60}")
    print(f"  Total examples: {len(selected)}")
    print(f"  Run IDs compared: {len(available_ids)}")
    print(f"  Min per category target: {args.min_per_category}")
    print(f"  Categories covered: {len(cat_counts)}/{len(EVAL_CATEGORIES)}")
    for cat in EVAL_CATEGORIES:
        count = cat_counts.get(cat, 0)
        status = "OK" if count >= args.min_per_category else "SHORT"
        print(f"    {cat}: {count} [{status}]")
    print(f"\n  Review file: {md_path}")
    print(f"  Unblinding key: {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
