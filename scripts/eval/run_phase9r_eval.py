#!/usr/bin/env python3
"""Run the complete Phase 9R evaluation pipeline on aero.

This script runs eval harness + judge + aggregate + blind review for all
Phase 9 adapters. Designed to run on aero (RTX 2070 Super 8GB).

Requirements:
  - Model weights downloaded (LiquidAI/LFM2.5-230M)
  - Adapters trained (all 8 in adapters/ directory)
  - JUDGE_API_URL and JUDGE_API_KEY env vars set for real judge

Usage:
  # Full run with real judge
  python scripts/eval/run_phase9r_eval.py --judge-api-url http://localhost:8080 --judge-api-key sk-xxx

  # Full run with mock judge (for pipeline validation only)
  python scripts/eval/run_phase9r_eval.py --mock-judge

  # Dry run (show what would be executed)
  python scripts/eval/run_phase9r_eval.py --dry-run --mock-judge
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Adapter definitions: (run_id, adapter_rel_path, adapter_type)
# adapter_rel_path is relative to PROJECT_ROOT/adapters/
ADAPTERS = [
    ("lfm2_230m_quality_alpaca_flat_20260629", "lfm2_230m_quality_format_ablation_alpaca_flat_20260629", "quality"),
    ("lfm2_230m_quality_single_turn_chat_20260629", "lfm2_230m_quality_format_ablation_single_turn_chat_20260629", "quality"),
    ("lfm2_230m_quality_multi_turn_concise_20260629", "lfm2_230m_quality_format_ablation_multi_turn_concise_20260629", "quality"),
    ("lfm2_230m_quality_multi_turn_verbose_20260629", "lfm2_230m_quality_format_ablation_multi_turn_verbose_20260629", "quality"),
    ("lfm2_230m_quality_structured_terse_20260629", "lfm2_230m_quality_format_ablation_structured_terse_20260629", "quality"),
    ("lfm2_230m_quality_bad_format_control_20260629", "lfm2_230m_quality_format_ablation_bad_format_control_20260629", "quality"),
    ("lfm2_230m_quality_bsmagpie_surgical_20260629", "lfm2_230m_quality_bilawal_smol_magpie_v1_20260629", "quality"),
    ("lfm2_230m_surgical_bsmagpie_surgical_20260629", "lfm2_230m_surgical_bilawal_smol_magpie_v1_20260629", "surgical"),
]

BASE_RUN_ID = "lfm2_230m_base_20260629"
EVAL_CONFIG = "configs/eval/lfm2_small_model_eval.yaml"


def run_cmd(cmd: list[str], dry_run: bool = False, check: bool = True) -> int:
    """Run a command, log it, return exit code."""
    cmd_str = " ".join(cmd)
    log.info(f"  $ {cmd_str}")
    if dry_run:
        log.info("  [DRY RUN — skipping]")
        return 0
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if check and result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}: {cmd_str}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Phase 9R evaluation pipeline")
    parser.add_argument("--judge-api-url", default=os.environ.get("JUDGE_API_URL", "http://localhost:8080"))
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--mock-judge", action="store_true",
                        help="Use mock judge (pipeline validation only, not behavioral evidence)")
    parser.add_argument("--strict-report-mode", action="store_true",
                        help="Fail if judge API is unavailable and --mock-judge is not set")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--skip-harness", action="store_true", help="Skip eval harness (reuse existing outputs)")
    parser.add_argument("--skip-blind-review", action="store_true", help="Skip blind review generation")
    parser.add_argument("--only-base", action="store_true", help="Only run base model eval")
    parser.add_argument("--only", default=None, help="Only run eval for run_id containing this substring")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_config_path = str(PROJECT_ROOT / EVAL_CONFIG)

    # Build judge args
    judge_args = []
    if args.mock_judge:
        judge_args.append("--mock")
        log.warning("Using MOCK judge — results are pipeline validation only, NOT behavioral evidence")
    else:
        if args.strict_report_mode:
            judge_args.append("--strict-report-mode")
        if args.judge_api_key:
            judge_args.extend(["--api-key", args.judge_api_key])
        if args.judge_api_url:
            judge_args.extend(["--api-url", args.judge_api_url])

    # ---- STEP 1: Base model eval ----
    if not args.only:
        log.info("=" * 60)
        log.info("STEP 1: Evaluate base model")
        log.info("=" * 60)

        if not args.skip_harness:
            base_outputs = PROJECT_ROOT / "results" / "evals" / BASE_RUN_ID / "outputs.jsonl"
            if base_outputs.exists() and not args.skip_harness:
                log.info(f"  Base outputs exist at {base_outputs}, skipping harness")
            else:
                rc = run_cmd([
                    sys.executable, "scripts/eval/run_eval_harness.py",
                    "--config", eval_config_path,
                    "--base-only",
                    "--run-id", BASE_RUN_ID,
                ], dry_run=args.dry_run)
                if rc != 0:
                    log.error("Base model eval failed — aborting")
                    sys.exit(1)

        # Judge base
        base_scores = PROJECT_ROOT / "results" / "evals" / BASE_RUN_ID / "judge_scores.jsonl"
        if not base_scores.exists() or not args.skip_harness:
            rc = run_cmd([
                sys.executable, "scripts/eval/judge_outputs.py",
                "--run-id", BASE_RUN_ID,
                "--mode", "pointwise",
            ] + judge_args, dry_run=args.dry_run, check=False)

    # ---- STEP 2: Evaluate all adapters ----
    log.info("=" * 60)
    log.info("STEP 2: Evaluate adapters")
    log.info("=" * 60)

    adapter_run_ids = [a[0] for a in ADAPTERS]

    for run_id, adapter_path, adapter_type in ADAPTERS:
        if args.only and args.only not in run_id:
            continue

        full_adapter_path = str(PROJECT_ROOT / "adapters" / adapter_path)
        log.info(f"\n--- {run_id} ({adapter_type}) ---")

        # Check adapter exists
        if not args.dry_run and not Path(full_adapter_path).exists():
            log.warning(f"  Adapter not found: {full_adapter_path} — skipping")
            continue

        # Eval harness
        if not args.skip_harness:
            outputs_path = PROJECT_ROOT / "results" / "evals" / run_id / "outputs.jsonl"
            if outputs_path.exists():
                log.info(f"  Outputs exist, skipping harness")
            else:
                rc = run_cmd([
                    sys.executable, "scripts/eval/run_eval_harness.py",
                    "--config", eval_config_path,
                    "--adapter", full_adapter_path,
                    "--run-id", run_id,
                ], dry_run=args.dry_run)
                if rc != 0:
                    log.warning(f"  Harness failed for {run_id}, continuing")
                    continue

        # Judge
        scores_path = PROJECT_ROOT / "results" / "evals" / run_id / "judge_scores.jsonl"
        if not scores_path.exists() or not args.skip_harness:
            rc = run_cmd([
                sys.executable, "scripts/eval/judge_outputs.py",
                "--run-id", run_id,
                "--mode", "pointwise",
            ] + judge_args, dry_run=args.dry_run, check=False)

            # Also run pairwise vs base
            rc = run_cmd([
                sys.executable, "scripts/eval/judge_outputs.py",
                "--run-id", run_id,
                "--mode", "pairwise",
                "--baseline-run-id", BASE_RUN_ID,
            ] + judge_args, dry_run=args.dry_run, check=False)

        # Aggregate
        rc = run_cmd([
            sys.executable, "scripts/eval/aggregate_eval_results.py",
            "--run-id", run_id,
            "--compare-with", BASE_RUN_ID,
        ], dry_run=args.dry_run, check=False)

    # ---- STEP 3: Blind review ----
    if not args.skip_blind_review:
        log.info("=" * 60)
        log.info("STEP 3: Generate blind review samples")
        log.info("=" * 60)

        all_run_ids = [BASE_RUN_ID] + adapter_run_ids
        if args.only:
            all_run_ids = [BASE_RUN_ID] + [a[0] for a in ADAPTERS if args.only in a[0]]

        review_path = f"results/evals/blind_review_phase9r_{ts}.md"
        rc = run_cmd([
            sys.executable, "scripts/eval/generate_blind_review.py",
            "--run-ids", *all_run_ids,
            "--min-per-category", "7",
            "--output", review_path,
        ], dry_run=args.dry_run, check=False)

    # ---- STEP 4: Summary ----
    log.info("=" * 60)
    log.info("STEP 4: Pipeline complete")
    log.info("=" * 60)

    judge_source = "mock" if args.mock_judge else "api"
    log.info(f"  Judge source: {judge_source}")
    if args.mock_judge:
        log.warning("  RESULTS ARE PIPELINE VALIDATION ONLY — NOT BEHAVIORAL EVIDENCE")
        log.warning("  Rerun without --mock-judge for publishable results")

    log.info(f"  Eval config: {EVAL_CONFIG}")
    log.info(f"  Run IDs evaluated: {len(adapter_run_ids)}")
    log.info(f"  Blind review: {'generated' if not args.skip_blind_review else 'skipped'}")

    if not args.dry_run:
        # Write run log
        log_path = PROJECT_ROOT / "results" / "evals" / f"phase9r_run_log_{ts}.json"
        run_log = {
            "phase": "9R",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "judge_source": judge_source,
            "eval_config": EVAL_CONFIG,
            "run_ids": [BASE_RUN_ID] + adapter_run_ids,
            "mock_judge": args.mock_judge,
            "strict_report_mode": args.strict_report_mode,
        }
        with open(log_path, "w") as f:
            json.dump(run_log, f, indent=2)
        log.info(f"  Run log: {log_path}")


if __name__ == "__main__":
    main()
