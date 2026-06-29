#!/usr/bin/env python3
"""Aggregate judge scores into summary metrics and comparison tables.

CLI:
    python scripts/eval/aggregate_eval_results.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --compare-with lfm2_230m_base_20260629 lfm2_230m_format_ablation_alpaca_flat_20260629
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SLOP_PHRASES = [
    "as an ai", "i apologize", "i'm sorry, but", "as a language model",
    "i don't have personal", "it's important to note that",
    "please note that", "i hope this helps",
]

DIMENSIONS = [
    "correctness", "instruction_following", "output_format",
    "concision", "usefulness", "hallucination_risk", "overall",
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


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_json_validity(text: str) -> bool:
    """Check if response contains valid JSON."""
    # Look for JSON-like content
    json_pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = json_pattern.findall(text)
    for m in matches:
        try:
            json.loads(m)
            return True
        except (json.JSONDecodeError, ValueError):
            continue
    # Try the whole text
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_slop(text: str) -> int:
    """Count slop phrases in text."""
    text_lower = text.lower()
    return sum(1 for p in SLOP_PHRASES if p in text_lower)


def aggregate_pointwise(scores: list[dict]) -> dict:
    """Aggregate pointwise scores into summary metrics."""
    if not scores:
        return {}

    by_category = defaultdict(list)
    for s in scores:
        cat = s.get("category", "unknown")
        by_category[cat].append(s)

    # Overall averages
    avg_scores = {}
    for dim in DIMENSIONS:
        vals = [s["scores"].get(dim, 0) for s in scores if s.get("scores")]
        avg_scores[dim] = round(sum(vals) / len(vals), 3) if vals else 0

    # Per-category averages
    category_scores = {}
    for cat, cat_scores in by_category.items():
        cat_avg = {}
        for dim in DIMENSIONS:
            vals = [s["scores"].get(dim, 0) for s in cat_scores if s.get("scores")]
            cat_avg[dim] = round(sum(vals) / len(vals), 3) if vals else 0
        category_scores[cat] = {
            "avg_scores": cat_avg,
            "count": len(cat_scores),
        }

    return {
        "count": len(scores),
        "avg_scores": avg_scores,
        "category_scores": category_scores,
    }


def compute_format_metrics(outputs: list[dict]) -> dict:
    """Compute format-specific metrics from outputs."""
    json_tasks = [o for o in outputs if o.get("category") in ("json_structured", "gamefaq_extraction", "structured_terse")]
    json_valid = sum(1 for o in json_tasks if check_json_validity(o.get("generated_response", "")))
    json_rate = round(json_valid / len(json_tasks), 3) if json_tasks else None

    lengths = [o.get("tokens_generated", 0) for o in outputs]
    avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0

    slop_counts = [check_slop(o.get("generated_response", "")) for o in outputs]
    total_slop = sum(slop_counts)
    slop_rate = round(sum(1 for c in slop_counts if c > 0) / len(outputs), 3) if outputs else 0

    return {
        "json_format_validity_rate": json_rate,
        "json_task_count": len(json_tasks),
        "avg_output_length": avg_length,
        "slop_phrase_rate": slop_rate,
        "slop_phrase_total": total_slop,
        "total_outputs": len(outputs),
    }


def compute_regression_summary(current: dict, baseline: dict) -> dict:
    """Compare current vs baseline, count regressions."""
    if not baseline:
        return {"regressions": 0, "improvements": 0, "details": []}

    current_cats = current.get("category_scores", {})
    baseline_cats = baseline.get("category_scores", {})

    regressions = []
    improvements = []

    for cat in set(list(current_cats.keys()) + list(baseline_cats.keys())):
        cur = current_cats.get(cat, {}).get("avg_scores", {})
        base = baseline_cats.get(cat, {}).get("avg_scores", {})
        cur_overall = cur.get("overall", 0)
        base_overall = base.get("overall", 0)

        if cur_overall < base_overall - 0.3:
            regressions.append({
                "category": cat,
                "current_overall": cur_overall,
                "baseline_overall": base_overall,
                "delta": round(cur_overall - base_overall, 3),
            })
        elif cur_overall > base_overall + 0.3:
            improvements.append({
                "category": cat,
                "current_overall": cur_overall,
                "baseline_overall": base_overall,
                "delta": round(cur_overall - base_overall, 3),
            })

    return {
        "regressions": len(regressions),
        "improvements": len(improvements),
        "regression_details": regressions,
        "improvement_details": improvements,
    }


def compute_winrate(current: dict, baseline: dict) -> float | None:
    """Compute overall win rate from pairwise scores or pointwise comparison."""
    if not baseline:
        return None
    cur_overall = current.get("avg_scores", {}).get("overall", 0)
    base_overall = baseline.get("avg_scores", {}).get("overall", 0)
    if cur_overall + base_overall == 0:
        return 0.5
    return round(cur_overall / (cur_overall + base_overall), 3)


def print_summary_table(run_id: str, metrics: dict, comparisons: dict):
    """Print a readable summary table."""
    print(f"\n{'='*80}")
    print(f"  EVAL SUMMARY: {run_id}")
    print(f"{'='*80}")

    # Scores
    avg = metrics.get("pointwise", {}).get("avg_scores", {})
    if avg:
        print(f"\n  Overall Scores (1-5):")
        print(f"  {'Dimension':<25} {'Score':>6}")
        print(f"  {'-'*31}")
        for dim in DIMENSIONS:
            val = avg.get(dim, "N/A")
            bar = "█" * int(float(val) * 4) if isinstance(val, (int, float)) else ""
            print(f"  {dim:<25} {val:>6}  {bar}")

    # Format metrics
    fmt = metrics.get("format", {})
    if fmt:
        print(f"\n  Format Metrics:")
        print(f"  {'Metric':<35} {'Value':>8}")
        print(f"  {'-'*43}")
        for k, v in fmt.items():
            print(f"  {k:<35} {v:>8}")

    # Comparisons
    for comp_name, comp_data in comparisons.items():
        wr = comp_data.get("win_rate")
        reg = comp_data.get("regression", {})
        if wr is not None:
            print(f"\n  vs {comp_name}: win_rate={wr}")
        if reg.get("regressions", 0) > 0:
            print(f"  ⚠ Regressions: {reg['regressions']}")
            for r in reg.get("regression_details", []):
                print(f"    - {r['category']}: {r['baseline_overall']} → {r['current_overall']} ({r['delta']:+.2f})")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval results into summary metrics")
    parser.add_argument("--run-id", required=True, help="Run ID to aggregate")
    parser.add_argument("--compare-with", nargs="*", default=[], help="Baseline run IDs to compare against")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / "evals" / args.run_id

    # Load data
    scores_path = run_dir / "judge_scores.jsonl"
    outputs_path = run_dir / "outputs.jsonl"

    scores = load_jsonl(scores_path)
    outputs = load_jsonl(outputs_path)

    if not scores:
        log.error(f"No judge scores found at {scores_path}")
        sys.exit(1)
    if not outputs:
        log.error(f"No outputs found at {outputs_path}")
        sys.exit(1)

    log.info(f"Loaded {len(scores)} scores, {len(outputs)} outputs")

    # Aggregate pointwise scores
    pointwise_scores = [s for s in scores if s.get("mode") == "pointwise"]
    pairwise_scores = [s for s in scores if s.get("mode") == "pairwise"]

    pointwise_agg = aggregate_pointwise(pointwise_scores)
    format_metrics = compute_format_metrics(outputs)

    # Pairwise win rates
    pairwise_agg = {}
    if pairwise_scores:
        wins = sum(1 for s in pairwise_scores if s.get("winner") == "model_a")
        losses = sum(1 for s in pairwise_scores if s.get("winner") == "model_b")
        ties = sum(1 for s in pairwise_scores if s.get("winner") == "tie")
        total = len(pairwise_scores)
        pairwise_agg = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(wins / total, 3) if total else 0,
            "total": total,
        }

    # Comparisons with baselines
    comparisons = {}
    for baseline_id in args.compare_with:
        baseline_dir = PROJECT_ROOT / "results" / "evals" / baseline_id
        baseline_agg_path = baseline_dir / "aggregate.json"
        baseline_agg = load_json(baseline_agg_path)
        if not baseline_agg:
            log.warning(f"No aggregate found for baseline {baseline_id}, skipping")
            continue

        baseline_pointwise = baseline_agg.get("pointwise", {})
        win_rate = compute_winrate(pointwise_agg, baseline_pointwise)
        regression = compute_regression_summary(pointwise_agg, baseline_pointwise)
        comparisons[baseline_id] = {
            "win_rate": win_rate,
            "regression": regression,
            "baseline_avg_scores": baseline_pointwise.get("avg_scores", {}),
        }

    # Build aggregate
    aggregate = {
        "run_id": args.run_id,
        "pointwise": pointwise_agg,
        "pairwise": pairwise_agg,
        "format": format_metrics,
        "comparisons": comparisons,
    }

    # Save
    agg_path = run_dir / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    log.info(f"Saved aggregate to {agg_path}")

    # Print summary
    print_summary_table(args.run_id, aggregate, comparisons)


if __name__ == "__main__":
    main()
