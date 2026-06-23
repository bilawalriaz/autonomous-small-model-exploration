#!/usr/bin/env python3
"""Phase 3 orchestrator: gap closure and gem discovery.

Runs experiments in priority order to close methodological gaps and hunt for
surprising exceptions. Designed for a single RTX 2070 Super (8GB VRAM).

Usage:
    python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all
    python scripts/run_full_phase3_atlas.py --model Qwen/Qwen2.5-0.5B --blocks R1,R2,R3
    python scripts/run_full_phase3_atlas.py --list-blocks
    python scripts/run_full_phase3_atlas.py --blocks all --dry-run

Blocks:
    R1-R5  - Multi-seed replication (Priority 1)
    L1-L5  - Atlas-guided LoRA (Priority 2)
    C1-C4  - Better causal tests (Priority 3)
    P1-P3  - Prompt robustness (Priority 4)
    Q1-Q3  - Quantization causal surface (Priority 5)
    G1-G4  - Gem hunting (Priority 6)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

BLOCK_SCRIPTS = {
    # Priority 1: Multi-seed replication
    "R1": {
        "script": "scripts/run_phase3_multiseed_replication.py",
        "description": "0.5B layer ablation x3 seeds (42, 137, 256)",
        "args": ["--experiment", "layer_ablation"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 1,
        "est_minutes": 15,
    },
    "R2": {
        "script": "scripts/run_phase3_multiseed_replication.py",
        "description": "1.5B layer ablation x3 seeds",
        "args": ["--experiment", "layer_ablation"],
        "models": ["Qwen/Qwen2.5-1.5B"],
        "priority": 1,
        "est_minutes": 30,
    },
    "R3": {
        "script": "scripts/run_phase3_multiseed_replication.py",
        "description": "3B layer ablation x3 seeds",
        "args": ["--experiment", "layer_ablation"],
        "models": ["Qwen/Qwen2.5-3B"],
        "priority": 1,
        "est_minutes": 60,
    },
    "R4": {
        "script": "scripts/run_phase3_multiseed_replication.py",
        "description": "0.5B steering x3 seeds (at hub layers)",
        "args": ["--experiment", "steering"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 1,
        "est_minutes": 20,
    },
    "R5": {
        "script": "scripts/run_phase3_multiseed_replication.py",
        "description": "1.5B steering x3 seeds (at hub layers)",
        "args": ["--experiment", "steering"],
        "models": ["Qwen/Qwen2.5-1.5B"],
        "priority": 1,
        "est_minutes": 40,
    },

    # Priority 2: Atlas-guided LoRA
    "L1": {
        "script": "scripts/run_phase3_atlas_guided_lora.py",
        "description": "Atlas-guided vs random vs all-linear LoRA on JSON (0.5B)",
        "args": ["--family", "json_schema"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 2,
        "est_minutes": 30,
    },
    "L2": {
        "script": "scripts/run_phase3_atlas_guided_lora.py",
        "description": "Atlas-guided vs random vs all-linear LoRA on factual (0.5B)",
        "args": ["--family", "factual_recall"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 2,
        "est_minutes": 30,
    },
    "L3": {
        "script": "scripts/run_phase3_atlas_guided_lora.py",
        "description": "Atlas-guided vs random vs all-linear LoRA on code (0.5B)",
        "args": ["--family", "code_semantics"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 2,
        "est_minutes": 30,
    },
    "L4": {
        "script": "scripts/run_phase3_rank_sweep_with_accuracy.py",
        "description": "Rank sweep (r=2,4,8,16) with task accuracy on 0.5B",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 2,
        "est_minutes": 45,
    },
    "L5": {
        "script": "scripts/run_phase3_module_sweep_with_accuracy.py",
        "description": "Module sweep with task accuracy on 0.5B",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 2,
        "est_minutes": 45,
    },

    # Priority 3: Better causal tests
    "C1": {
        "script": "scripts/run_phase3_ablation_method_comparison.py",
        "description": "Full ablation method comparison at ALL layers (0.5B)",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 3,
        "est_minutes": 30,
    },
    "C2": {
        "script": "scripts/run_phase3_position_ablation_all_layers.py",
        "description": "Token-position ablation at ALL layers (0.5B)",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 3,
        "est_minutes": 45,
    },
    "C3": {
        "script": "scripts/run_phase3_module_ablation.py",
        "description": "Module-specific ablation (q/k/v/o/up/down/gate) at hub layers",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 3,
        "est_minutes": 30,
    },
    "C4": {
        "script": "scripts/run_phase3_steering_controls.py",
        "description": "Random-vector and shuffled-label controls for steering",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 3,
        "est_minutes": 20,
    },

    # Priority 4: Prompt robustness
    "P1": {
        "script": "scripts/run_phase3_natural_language_hubs.py",
        "description": "Hub ID with 50+ natural language prompts (0.5B)",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 4,
        "est_minutes": 30,
    },
    "P2": {
        "script": "scripts/run_phase3_steering_by_length.py",
        "description": "Steering effectiveness vs prompt length",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 4,
        "est_minutes": 20,
    },
    "P3": {
        "script": "scripts/run_phase3_coder_atlas.py",
        "description": "Hub ID on Qwen2.5-Coder-0.5B",
        "args": [],
        "models": ["Qwen/Qwen2.5-Coder-0.5B"],
        "priority": 4,
        "est_minutes": 15,
    },

    # Priority 5: Quantization causal surface
    "Q1": {
        "script": "scripts/run_phase3_quantization_atlas.py",
        "description": "Layer ablation on 4-bit NF4 0.5B",
        "args": ["--quant", "4bit"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 5,
        "est_minutes": 20,
    },
    "Q2": {
        "script": "scripts/run_phase3_quantization_atlas.py",
        "description": "Steering on 4-bit NF4 0.5B",
        "args": ["--quant", "4bit", "--experiment", "steering"],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 5,
        "est_minutes": 20,
    },
    "Q3": {
        "script": "scripts/run_phase3_quantization_atlas.py",
        "description": "Layer ablation on 4-bit NF4 1.5B",
        "args": ["--quant", "4bit"],
        "models": ["Qwen/Qwen2.5-1.5B"],
        "priority": 5,
        "est_minutes": 40,
    },

    # Priority 6: Gem hunting
    "G1": {
        "script": "scripts/run_phase3_steering_direction_transfer.py",
        "description": "Steering direction transfer across scales",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"],
        "priority": 6,
        "est_minutes": 30,
    },
    "G2": {
        "script": "scripts/run_phase3_knockout_controls.py",
        "description": "Random-vector and shuffled-label knockout controls",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 6,
        "est_minutes": 20,
    },
    "G3": {
        "script": "scripts/run_phase3_checkpoint_lockin.py",
        "description": "Checkpoint lock-in at 1.5B (step 10/25/50/100)",
        "args": [],
        "models": ["Qwen/Qwen2.5-1.5B"],
        "priority": 6,
        "est_minutes": 60,
    },
    "G4": {
        "script": "scripts/run_phase3_atlas_guided_skip.py",
        "description": "Atlas-guided layer skip + recovery finetune",
        "args": [],
        "models": ["Qwen/Qwen2.5-0.5B"],
        "priority": 6,
        "est_minutes": 45,
    },
}

PRIORITY_ORDER = ["R1", "R2", "R3", "R4", "R5",
                   "L1", "L2", "L3", "L4", "L5",
                   "C1", "C2", "C3", "C4",
                   "P1", "P2", "P3",
                   "Q1", "Q2", "Q3",
                   "G1", "G2", "G3", "G4"]


def list_blocks():
    print("Phase 3 blocks (in priority order):\n")
    current_priority = 0
    for block_id in PRIORITY_ORDER:
        block = BLOCK_SCRIPTS[block_id]
        if block["priority"] != current_priority:
            current_priority = block["priority"]
            priority_names = {
                1: "MULTI-SEED REPLICATION",
                2: "ATLAS-GUIDED LoRA",
                3: "BETTER CAUSAL TESTS",
                4: "PROMPT ROBUSTNESS",
                5: "QUANTIZATION CAUSAL SURFACE",
                6: "GEM HUNTING",
            }
            print(f"\n  Priority {current_priority}: {priority_names.get(current_priority, '')}")
        models = ", ".join(m.split("/")[-1] for m in block["models"])
        print(f"    {block_id:3s}  {block['description'][:55]:55s}  [{models}] ~{block['est_minutes']}min")


def run_block(block_id, model, dry_run=False, force=False):
    block = BLOCK_SCRIPTS[block_id]
    script = PROJECT_ROOT / block["script"]

    if not script.exists():
        print(f"  WARNING: Script not found: {script}")
        print(f"  Skipping block {block_id}")
        return {"block": block_id, "status": "skipped", "reason": "script_not_found"}

    cmd = [sys.executable, str(script), "--model", model] + block.get("args", [])
    if force:
        cmd.append("--force")

    print(f"\n{'='*60}")
    print(f"Block {block_id}: {block['description']}")
    print(f"Model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return {"block": block_id, "status": "dry_run"}

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=block["est_minutes"] * 120)
        elapsed = time.time() - start
        status = "success" if result.returncode == 0 else "failed"
        return {
            "block": block_id,
            "status": status,
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired:
        return {"block": block_id, "status": "timeout"}
    except Exception as e:
        return {"block": block_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 orchestrator")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--blocks", type=str, default="all",
                       help="Comma-separated block IDs or 'all'")
    parser.add_argument("--list-blocks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--priority", type=int, default=None,
                       help="Only run blocks at this priority level")
    args = parser.parse_args()

    if args.list_blocks:
        list_blocks()
        return

    if args.blocks == "all":
        blocks = PRIORITY_ORDER
    else:
        blocks = [b.strip().upper() for b in args.blocks.split(",")]

    # Filter by priority if specified
    if args.priority is not None:
        blocks = [b for b in blocks if BLOCK_SCRIPTS[b]["priority"] == args.priority]

    # Filter by model compatibility
    blocks = [b for b in blocks if b in BLOCK_SCRIPTS]

    print(f"Phase 3 orchestrator")
    print(f"Model: {args.model}")
    print(f"Blocks: {blocks}")
    print(f"Force: {args.force}")
    print(f"Dry run: {args.dry_run}")

    results = []
    for block_id in blocks:
        if block_id not in BLOCK_SCRIPTS:
            print(f"Unknown block: {block_id}")
            continue
        result = run_block(block_id, args.model, dry_run=args.dry_run, force=args.force)
        results.append(result)

    # Save run log
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "blocks": [r["block"] for r in results],
        "results": results,
        "force": args.force,
        "dry_run": args.dry_run,
    }
    log_path = PROJECT_ROOT / "experiments" / "runs" / f"phase3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print("Phase 3 run complete.")
    print(f"Results: {len([r for r in results if r['status'] == 'success'])}/{len(results)} succeeded")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
