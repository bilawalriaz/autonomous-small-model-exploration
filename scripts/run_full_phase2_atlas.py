#!/usr/bin/env python3
"""Phase 2 orchestrator: run all Phase 2 experiment blocks in order.

Usage:
    python scripts/run_full_phase2_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all
    python scripts/run_full_phase2_atlas.py --model Qwen/Qwen2.5-1.5B --blocks A,B,C
    python scripts/run_full_phase2_atlas.py --blocks D --model Qwen/Qwen2.5-3B
    python scripts/run_full_phase2_atlas.py --list-blocks

Blocks:
    A - Parity verification (fill missing 1.5B experiments)
    B - Steering migration test
    C - Better ablation controls
    D - Third scale point (3B)
    E - Cross-family replication
    F - Adapter surgery
    G - Skill separability benchmark
    H - Deobfuscation surgery
    I - Long-prompt robustness
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Block -> script mapping
BLOCK_SCRIPTS = {
    "A": {
        "script": "scripts/run_phase2_parity.py",
        "description": "Parity verification - fill missing experiments for 1.5B",
        "models": ["Qwen/Qwen2.5-1.5B"],
        "priority": 1,
    },
    "B": {
        "script": "scripts/run_phase2_steering_migration.py",
        "description": "Steering migration - test steering at hub layers",
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 2,
    },
    "C": {
        "script": "scripts/run_phase2_ablation_controls.py",
        "description": "Ablation controls - zero/mean/resample/patch comparison",
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 3,
    },
    "F": {
        "script": "scripts/run_phase2_adapter_surgery.py",
        "description": "Adapter surgery + compatibility matrix",
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 4,
    },
    "H": {
        "script": "scripts/run_phase2_deobfuscation.py",
        "description": "Deobfuscation subskill surgery",
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 5,
    },
    "G": {
        "script": "scripts/run_phase2_skill_separability.py",
        "description": "Skill separability benchmark",
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 6,
    },
    "D": {
        "script": "scripts/run_phase2_third_scale.py",
        "description": "Third scale point (3B reduced atlas)",
        "models": ["Qwen/Qwen2.5-3B"],
        "priority": 7,
    },
    "E": {
        "script": "scripts/run_phase2_cross_family.py",
        "description": "Cross-family replication (Gemma/SmolLM)",
        "models": ["google/gemma-2-2b", "HuggingFaceTB/SmolLM2-1.7B"],
        "priority": 8,
    },
    "I": {
        "script": "scripts/run_phase2_long_task_robustness.py",
        "description": "Long-prompt and real-task robustness",
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 9,
    },
}


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_block(block_id, model, extra_args=None, dry_run=False):
    """Run a single experiment block."""
    if block_id not in BLOCK_SCRIPTS:
        print(f"ERROR: Unknown block '{block_id}'")
        return False

    block = BLOCK_SCRIPTS[block_id]
    script = block["script"]
    script_path = Path(__file__).parent.parent / script

    if not script_path.exists():
        print(f"WARNING: Script not found: {script_path}")
        print(f"  Block {block_id} ({block['description']}) - SKIPPED")
        return None

    if dry_run:
        print(f"  [DRY RUN] Would run: python {script} --model {model}")
        return True

    cmd = [sys.executable, str(script_path), "--model", model]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Block {block_id}: {block['description']}")
    print(f"Model: {model}")
    print(f"Script: {script}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(script_path.parent.parent))
        elapsed = time.time() - start_time
        status = "success" if result.returncode == 0 else "failed"
        print(f"\nBlock {block_id} {status} in {elapsed:.0f}s")
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\nBlock {block_id} interrupted by user")
        return False
    except Exception as e:
        print(f"\nBlock {block_id} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 2 orchestrator")
    parser.add_argument("--model", type=str, default=None,
                       help="Override model (default: use block's default)")
    parser.add_argument("--blocks", type=str, default="all",
                       help="Comma-separated block IDs (A-I) or 'all'")
    parser.add_argument("--list-blocks", action="store_true",
                       help="List available blocks and exit")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would run without executing")
    parser.add_argument("--force", action="store_true",
                       help="Re-run completed experiments")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override seed")
    args = parser.parse_args()

    if args.list_blocks:
        print("Phase 2 Experiment Blocks (in priority order):")
        print("=" * 60)
        for bid in sorted(BLOCK_SCRIPTS, key=lambda x: BLOCK_SCRIPTS[x]["priority"]):
            b = BLOCK_SCRIPTS[bid]
            script_path = Path(__file__).parent.parent / b["script"]
            exists = "READY" if script_path.exists() else "MISSING"
            print(f"  {bid} [{exists}] (priority {b['priority']}): {b['description']}")
            print(f"     Models: {', '.join(b['models'])}")
            print(f"     Script: {b['script']}")
        return

    # Parse blocks
    if args.blocks == "all":
        blocks = sorted(BLOCK_SCRIPTS.keys(), key=lambda x: BLOCK_SCRIPTS[x]["priority"])
    else:
        blocks = [b.strip().upper() for b in args.blocks.split(",")]
        for b in blocks:
            if b not in BLOCK_SCRIPTS:
                print(f"ERROR: Unknown block '{b}'. Use --list-blocks to see options.")
                sys.exit(1)

    print(f"Phase 2 Orchestrator")
    print(f"Git commit: {get_git_hash()}")
    print(f"Blocks to run: {', '.join(blocks)}")
    print(f"Force: {args.force}")
    print(f"Dry run: {args.dry_run}")
    print()

    extra_args = []
    if args.force:
        extra_args.append("--force")
    if args.seed is not None:
        extra_args.extend(["--seed", str(args.seed)])

    results = {}
    for block_id in blocks:
        block = BLOCK_SCRIPTS[block_id]
        model = args.model or block["models"][0]

        # For blocks with multiple models, run each
        models_to_run = [model] if args.model else block["models"]

        for m in models_to_run:
            success = run_block(block_id, m, extra_args, args.dry_run)
            results[f"{block_id}:{m}"] = success

            if success is False and not args.dry_run:
                print(f"\nWARNING: Block {block_id} failed for {m}")
                print("Continuing with next block...")

    # Summary
    print(f"\n{'='*60}")
    print("Phase 2 Summary")
    print(f"{'='*60}")
    for key, status in results.items():
        block_id, model = key.split(":", 1)
        icon = "PASS" if status else ("SKIP" if status is None else "FAIL")
        print(f"  [{icon}] Block {block_id} ({model})")

    # Save run log
    run_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_hash(),
        "blocks": {k: v for k, v in results.items()},
        "force": args.force,
        "seed": args.seed,
    }
    log_path = Path(__file__).parent.parent / "experiments" / "runs" / f"phase2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nRun log saved: {log_path}")


if __name__ == "__main__":
    main()
