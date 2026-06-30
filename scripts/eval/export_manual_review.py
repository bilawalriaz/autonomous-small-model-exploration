#!/usr/bin/env python3
"""Export selected examples for human manual review.

CLI:
    python scripts/eval/export_manual_review.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --baseline-run-id lfm2_230m_base_20260629
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SLOP_PHRASES = [
    "as an ai", "i apologize", "i'm sorry, but", "as a language model",
    "i don't have personal", "it's important to note that",
    "please note that", "i hope this helps",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def main():
    parser = argparse.ArgumentParser(description="Export examples for manual review")
    parser.add_argument("--run-id", required=True, help="Run ID (model A)")
    parser.add_argument("--baseline-run-id", required=True, help="Baseline run ID (model B)")
    parser.add_argument("--max-examples", type=int, default=30, help="Max examples to export")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    run_dir = PROJECT_ROOT / "results" / "evals" / args.run_id
    baseline_dir = PROJECT_ROOT / "results" / "evals" / args.baseline_run_id

    # Load pairwise scores
    scores = load_jsonl(run_dir / "judge_scores.jsonl")
    pairwise = [s for s in scores if s.get("mode") == "pairwise"]

    # Load outputs
    outputs_a = {o["eval_id"]: o for o in load_jsonl(run_dir / "outputs.jsonl")}
    outputs_b = {o["eval_id"]: o for o in load_jsonl(baseline_dir / "outputs.jsonl")}

    if not pairwise:
        log.warning("No pairwise scores found; falling back to pointwise comparison")
        pointwise = [s for s in scores if s.get("mode") == "pointwise"]
        baseline_scores_raw = load_jsonl(baseline_dir / "judge_scores.jsonl")
        baseline_pointwise = {s["eval_id"]: s for s in baseline_scores_raw if s.get("mode") == "pointwise"}

        pairwise = []
        for s in pointwise:
            bs = baseline_pointwise.get(s["eval_id"])
            if bs:
                cur_overall = s.get("scores", {}).get("overall", 0)
                base_overall = bs.get("scores", {}).get("overall", 0)
                if cur_overall > base_overall:
                    winner = "model_a"
                elif cur_overall < base_overall:
                    winner = "model_b"
                else:
                    winner = "tie"
                pairwise.append({
                    "eval_id": s["eval_id"],
                    "category": s.get("category", "unknown"),
                    "winner": winner,
                    "delta": cur_overall - base_overall,
                    "scores": {"model_a": s.get("scores", {}), "model_b": bs.get("scores", {})},
                    "reason": f"Pointwise overall: {cur_overall} vs {base_overall}",
                })

    if not pairwise:
        log.error("No comparison data available")
        sys.exit(1)

    # Categorize examples
    wins = [p for p in pairwise if p.get("winner") == "model_a"]
    losses = [p for p in pairwise if p.get("winner") == "model_b"]
    ties = [p for p in pairwise if p.get("winner") == "tie"]

    # Sort by delta magnitude where available
    wins.sort(key=lambda x: abs(x.get("delta", 0)), reverse=True)
    losses.sort(key=lambda x: abs(x.get("delta", 0)), reverse=True)

    selected = []

    # Top wins (up to 8)
    for item in wins[:8]:
        selected.append(("strong_win", item))

    # Top losses (up to 8)
    for item in losses[:8]:
        selected.append(("strong_loss", item))

    # Slop/concision failures (up to 5)
    for item in pairwise:
        if len(selected) >= args.max_examples:
            break
        eid = item["eval_id"]
        out_a = outputs_a.get(eid, {})
        resp = out_a.get("generated_response", "")
        if any(p in resp.lower() for p in SLOP_PHRASES):
            if ("slop", item) not in [(s[0], s[1]) for s in selected]:
                selected.append(("slop_failure", item))

    # JSON category failures (up to 5)
    for item in pairwise:
        if len(selected) >= args.max_examples:
            break
        if item.get("category") in ("json_structured", "gamefaq_extraction"):
            eid = item["eval_id"]
            out_a = outputs_a.get(eid, {})
            scores_a = item.get("scores", {}).get("model_a", {})
            if scores_a.get("output_format", 5) <= 2:
                selected.append(("json_failure", item))

    # Reasoning failures (up to 5)
    for item in pairwise:
        if len(selected) >= args.max_examples:
            break
        if item.get("category") == "reasoning":
            scores_a = item.get("scores", {}).get("model_a", {})
            if scores_a.get("correctness", 5) <= 2:
                selected.append(("reasoning_failure", item))

    # Fill remaining with random ties or weird decisions
    remaining = [p for p in pairwise if p not in [s[1] for s in selected]]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= args.max_examples:
            break
        selected.append(("random_sample", item))

    # Deduplicate by eval_id
    seen = set()
    deduped = []
    for tag, item in selected:
        eid = item["eval_id"]
        if eid not in seen:
            seen.add(eid)
            deduped.append((tag, item))
    selected = deduped[:args.max_examples]

    # Build markdown
    lines = [
        f"# Manual Review: {args.run_id}",
        f"",
        f"**Baseline:** {args.baseline_run_id}",
        f"**Total pairwise comparisons:** {len(pairwise)}",
        f"**Wins:** {len(wins)} | **Losses:** {len(losses)} | **Ties:** {len(ties)}",
        f"**Exported for review:** {len(selected)}",
        f"",
        f"---",
        f"",
    ]

    for i, (tag, item) in enumerate(selected, 1):
        eid = item["eval_id"]
        out_a = outputs_a.get(eid, {})
        out_b = outputs_b.get(eid, {})
        prompt = out_a.get("prompt", out_b.get("prompt", "N/A"))
        resp_a = out_a.get("generated_response", "N/A")
        resp_b = out_b.get("generated_response", "N/A")
        winner = item.get("winner", "unknown")
        reason = item.get("reason", "N/A")

        lines.append(f"## Example {i}: `{eid}` [{tag}]")
        lines.append(f"")
        lines.append(f"**Category:** {item.get('category', 'N/A')}  ")
        lines.append(f"**Judge Winner:** `{winner}`  ")
        lines.append(f"**Judge Reason:** {truncate(reason, 500)}")
        lines.append(f"")
        lines.append(f"### Prompt")
        lines.append(f"```")
        lines.append(truncate(prompt, 1500))
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"### Model A ({args.run_id})")
        lines.append(f"```")
        lines.append(truncate(resp_a, 1500))
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"### Model B ({args.baseline_run_id})")
        lines.append(f"```")
        lines.append(truncate(resp_b, 1500))
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"### Manual Assessment")
        lines.append(f"- [ ] I agree with the judge's decision")
        lines.append(f"- [ ] I disagree — I think Model A is better")
        lines.append(f"- [ ] I disagree — I think Model B is better")
        lines.append(f"- [ ] Both are about the same")
        lines.append(f"- [ ] Both are bad")
        lines.append(f"")
        lines.append(f"**Notes:**")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    # Write
    output_path = run_dir / "manual_review_sample.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"Wrote {len(selected)} examples to {output_path}")


if __name__ == "__main__":
    main()
