#!/usr/bin/env python3
"""Phase 3: Run all remaining experiments in priority order.

Runs L1-L5, C1-C4, P1-P3, Q1-Q3, G1-G4 sequentially.
Syncs results to git after each block.
Skips blocks whose output files already exist (unless --force).

Usage:
    python3 scripts/run_phase3_remaining.py [--force] [--start-block L1]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Block definitions: (block_id, script, args, est_minutes, expected_output)
BLOCKS = [
    # Priority 2: Atlas-guided LoRA
    ("L1", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "json_schema"], 30, "phase3_atlas_lora_*_json_schema.json"),
    ("L2", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "factual_recall"], 30, "phase3_atlas_lora_*_factual_recall.json"),
    ("L3", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "code_semantics"], 30, "phase3_atlas_lora_*_code_semantics.json"),
    ("L4", "scripts/run_phase3_rank_sweep_with_accuracy.py", [], 45, "phase3_rank_sweep_*.json"),
    ("L5", "scripts/run_phase3_module_sweep_with_accuracy.py", [], 45, "phase3_module_sweep_*.json"),

    # Priority 3: Better causal tests
    ("C1", "scripts/run_phase3_ablation_method_comparison.py", [], 30, "phase3_ablation_methods_*.json"),
    ("C2", "scripts/run_phase3_position_ablation_all_layers.py", [], 45, "phase3_position_ablation_*.json"),
    ("C3", "scripts/run_phase3_module_ablation.py", [], 30, "phase3_module_ablation_*.json"),
    ("C4", "scripts/run_phase3_steering_controls.py", [], 20, "phase3_steering_controls_*.json"),

    # Priority 4: Prompt robustness
    ("P1", "scripts/run_phase3_natural_language_hubs.py", [], 30, "phase3_natural_language_hubs_*.json"),
    ("P2", "scripts/run_phase3_steering_by_length.py", [], 20, "phase3_steering_by_length_*.json"),
    ("P3", "scripts/run_phase3_coder_atlas.py", [], 15, "phase3_coder_atlas*.json"),

    # Priority 5: Quantization
    ("Q1", "scripts/run_phase3_quantization_atlas.py", ["--quant", "4bit", "--experiment", "ablation"], 20, "phase3_quant_atlas_*_4bit.json"),
    ("Q2", "scripts/run_phase3_quantization_atlas.py", ["--quant", "4bit", "--experiment", "steering"], 20, "phase3_quant_atlas_*_4bit.json"),

    # Priority 6: Gem hunting
    ("G1", "scripts/run_phase3_steering_direction_transfer.py", [], 30, "phase3_steering_transfer*.json"),
    ("G2", "scripts/run_phase3_knockout_controls.py", [], 20, "phase3_knockout_controls_*.json"),
    ("G3", "scripts/run_phase3_checkpoint_lockin.py", [], 60, "phase3_checkpoint_lockin_*.json"),
    ("G4", "scripts/run_phase3_atlas_guided_skip.py", [], 45, "phase3_atlas_guided_skip_*.json"),
]

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
# Some blocks use different models
MODEL_OVERRIDES = {
    "P3": "Qwen/Qwen2.5-Coder-0.5B",
}


def run_block(block_id, script, extra_args, model, force=False, dry_run=False):
    """Run a single experiment block."""
    cmd = [sys.executable, script, "--model", model] + extra_args
    if force:
        cmd.append("--force")

    print(f"\n{'='*60}")
    print(f"Block {block_id}: {script}")
    print(f"Model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    if dry_run:
        print("  [DRY RUN]")
        return "dry_run"

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=7200)
        elapsed = time.time() - start
        status = "success" if result.returncode == 0 else "failed"
        print(f"\n  Block {block_id}: {status} ({elapsed:.0f}s)")
        return status
    except subprocess.TimeoutExpired:
        print(f"\n  Block {block_id}: TIMEOUT")
        return "timeout"
    except Exception as e:
        print(f"\n  Block {block_id}: ERROR: {e}")
        return "error"


def sync_results(block_id):
    """Commit and push results after a block."""
    results_dir = PROJECT_ROOT / "experiments" / "results"
    new_files = list(results_dir.glob("phase3_*.json"))
    if not new_files:
        return

    try:
        subprocess.run(["git", "add", "experiments/results/", "experiments/registry.jsonl"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Phase 3 {block_id}: results synced"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        subprocess.run(["git", "push", "origin", "master"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        print(f"  Results synced to git after {block_id}")
    except Exception as e:
        print(f"  Warning: git sync failed after {block_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 remaining experiments")
    parser.add_argument("--force", action="store_true", help="Re-run completed blocks")
    parser.add_argument("--start-block", type=str, default=None, help="Start from this block")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Phase 3: Remaining experiments runner")
    print(f"Default model: {args.model}")
    print(f"Blocks: {len(BLOCKS)}")
    print(f"Force: {args.force}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}", flush=True)

    results_log = []
    started = args.start_block is None

    for block_id, script, extra_args, est_min, expected_output in BLOCKS:
        if not started:
            if block_id == args.start_block:
                started = True
            else:
                continue

        model = MODEL_OVERRIDES.get(block_id, args.model)

        status = run_block(block_id, script, extra_args, model, args.force, args.dry_run)

        results_log.append({
            "block": block_id,
            "status": status,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Sync results after each block
        if status == "success" and not args.dry_run:
            sync_results(block_id)

    # Final summary
    print(f"\n{'='*60}")
    print("PHASE 3 RUN SUMMARY")
    print(f"{'='*60}")
    succeeded = sum(1 for r in results_log if r["status"] == "success")
    failed = sum(1 for r in results_log if r["status"] == "failed")
    print(f"Completed: {succeeded}/{len(results_log)}")
    if failed:
        print(f"Failed: {failed}")
        for r in results_log:
            if r["status"] != "success":
                print(f"  {r['block']}: {r['status']}")

    # Save log
    log_path = PROJECT_ROOT / "experiments" / "runs" / f"phase3_full_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"results": results_log, "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"\nLog: {log_path}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
