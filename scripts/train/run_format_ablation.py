#!/usr/bin/env python3
"""Orchestrate the full format ablation experiment.

CLI:
    # Full run (all formats)
    python scripts/train/run_format_ablation.py \
        --config configs/experiments/format_ablation_quality.yaml

    # Dry run (show plan)
    python scripts/train/run_format_ablation.py \
        --config configs/experiments/format_ablation_quality.yaml --dry-run

    # Skip training (eval only)
    python scripts/train/run_format_ablation.py \
        --config configs/experiments/format_ablation_quality.yaml --skip-training

    # Single format
    python scripts/train/run_format_ablation.py \
        --config configs/experiments/format_ablation_quality.yaml --format multi_turn_concise
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], desc: str) -> int:
    """Run a subprocess command, log output, return exit code."""
    log.info(f"[{desc}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if result.returncode != 0:
        log.error(f"[{desc}] Failed with exit code {result.returncode}")
    else:
        log.info(f"[{desc}] Completed successfully")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Orchestrate format ablation experiment")
    parser.add_argument("--config", required=True, help="Experiment config YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only eval existing adapters")
    parser.add_argument("--format", default=None, help="Run only one specific format")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval/judge steps")
    parser.add_argument("--base-run-id", default=None, help="Base model run ID for comparison")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs exist")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    model_name = config["model"]["name"]
    training_cfg = config.get("training", {})
    eval_cfg_path = config.get("eval", {}).get("config_path", "configs/eval/lfm2_small_model_eval.yaml")
    formats = config.get("formats", [])

    if args.format:
        formats = [f for f in formats if f["name"] == args.format]
        if not formats:
            log.error(f"Format '{args.format}' not found in config")
            sys.exit(1)

    # Date stamp for run IDs
    date_stamp = datetime.now().strftime("%Y%m%d")

    # Base run ID
    base_run_id = args.base_run_id or f"lfm2_230m_base_{date_stamp}"

    # Build plan
    plan = []
    for fmt in formats:
        run_id = f"lfm2_230m_format_ablation_{fmt['name']}_{date_stamp}"
        adapter_dir = PROJECT_ROOT / "adapters" / run_id
        plan.append({
            "format": fmt["name"],
            "description": fmt.get("description", ""),
            "dataset": fmt["output_path"],
            "run_id": run_id,
            "adapter_dir": str(adapter_dir),
        })

    log.info(f"Experiment: {experiment_name}")
    log.info(f"Model: {model_name}")
    log.info(f"Formats to process: {len(plan)}")
    log.info(f"Base run ID: {base_run_id}")

    if args.dry_run:
        print("\n=== DRY RUN PLAN ===\n")
        print(f"1. Ensure base model eval exists: {base_run_id}")
        for i, step in enumerate(plan, 2):
            print(f"{i}. [{step['format']}]")
            if not args.skip_training:
                print(f"   Train: python scripts/train/train_lfm2_sft.py \\")
                print(f"     --config configs/sft/baseline_lfm2_230m_quality.yaml \\")
                print(f"     --dataset {step['dataset']} --run-id {step['run_id']}")
            print(f"   Eval:  python scripts/eval/run_eval_harness.py \\")
            print(f"     --config {eval_cfg_path} \\")
            print(f"     --adapter adapters/{step['run_id']} --run-id {step['run_id']}")
            print(f"   Judge: python scripts/eval/judge_outputs.py \\")
            print(f"     --run-id {step['run_id']} --mode pairwise --baseline-run-id {base_run_id}")
            print(f"   Agg:   python scripts/eval/aggregate_eval_results.py \\")
            print(f"     --run-id {step['run_id']} --compare-with {base_run_id}")
            print()
        print(f"{len(plan)+2}. Build comparison table and manifest")
        print(f"\nTotal steps: {len(plan) * 4 + 2}")
        return

    # Ensure base model eval exists
    base_eval_dir = PROJECT_ROOT / "results" / "evals" / base_run_id
    if not base_eval_dir.exists():
        log.info(f"Base eval not found. Running base model eval first...")
        rc = run_cmd([
            sys.executable, "scripts/eval/run_eval_harness.py",
            "--config", eval_cfg_path,
            "--base-only",
            "--run-id", base_run_id,
        ], "base-eval")
        if rc != 0:
            log.error("Base model eval failed")
            sys.exit(1)

    # Process each format
    manifest = {
        "experiment": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_run_id": base_run_id,
        "formats": [],
    }

    for step in plan:
        fmt_name = step["format"]
        run_id = step["run_id"]
        log.info(f"\n{'='*60}")
        log.info(f"Processing format: {fmt_name} (run_id: {run_id})")
        log.info(f"{'='*60}")

        step_result = {
            "format": fmt_name,
            "run_id": run_id,
            "status": "pending",
        }

        # Step 1: Train
        if not args.skip_training:
            rc = run_cmd([
                sys.executable, "scripts/train/train_lfm2_sft.py",
                "--config", "configs/sft/baseline_lfm2_230m_quality.yaml",
                "--dataset", step["dataset"],
                "--run-id", run_id,
            ] + (["--force"] if args.force else []), f"train-{fmt_name}")
            if rc != 0:
                step_result["status"] = "training_failed"
                manifest["formats"].append(step_result)
                continue

        # Step 2: Eval
        if not args.skip_eval:
            rc = run_cmd([
                sys.executable, "scripts/eval/run_eval_harness.py",
                "--config", eval_cfg_path,
                "--adapter", f"adapters/{run_id}/adapter",
                "--run-id", run_id,
            ] + (["--force"] if args.force else []), f"eval-{fmt_name}")
            if rc != 0:
                step_result["status"] = "eval_failed"
                manifest["formats"].append(step_result)
                continue

            # Step 3: Judge
            rc = run_cmd([
                sys.executable, "scripts/eval/judge_outputs.py",
                "--run-id", run_id,
                "--mode", "pairwise",
                "--baseline-run-id", base_run_id,
            ] + (["--force"] if args.force else []), f"judge-{fmt_name}")
            if rc != 0:
                step_result["status"] = "judge_failed"
                manifest["formats"].append(step_result)
                continue

            # Step 4: Aggregate
            rc = run_cmd([
                sys.executable, "scripts/eval/aggregate_eval_results.py",
                "--run-id", run_id,
                "--compare-with", base_run_id,
            ], f"aggregate-{fmt_name}")
            if rc != 0:
                step_result["status"] = "aggregate_failed"
                manifest["formats"].append(step_result)
                continue

        # Load aggregate if exists
        agg_path = PROJECT_ROOT / "results" / "evals" / run_id / "aggregate.json"
        if agg_path.exists():
            with open(agg_path) as f:
                step_result["aggregate"] = json.load(f)

        step_result["status"] = "complete"
        manifest["formats"].append(step_result)

    # Save manifest
    output_dir = PROJECT_ROOT / "results" / "evals" / f"ablation_{experiment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"\nSaved manifest to {manifest_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  FORMAT ABLATION RESULTS: {experiment_name}")
    print(f"{'='*80}")
    print(f"\n  {'Format':<25} {'Status':<15} {'Overall':>8} {'Win Rate':>10} {'Regress':>8}")
    print(f"  {'-'*66}")

    for entry in manifest["formats"]:
        agg = entry.get("aggregate", {})
        pw = agg.get("pointwise", {}).get("avg_scores", {})
        overall = pw.get("overall", "N/A")
        comps = agg.get("comparisons", {})
        win_rate = "N/A"
        regressions = "N/A"
        if base_run_id in comps:
            win_rate = comps[base_run_id].get("win_rate", "N/A")
            reg = comps[base_run_id].get("regression", {})
            regressions = reg.get("regressions", "N/A")
        print(f"  {entry['format']:<25} {entry['status']:<15} {overall:>8} {win_rate:>10} {regressions:>8}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
