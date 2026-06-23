#!/usr/bin/env python3
"""Re-run failed Phase 3 blocks after fixes."""
import subprocess, sys, time
from pathlib import Path

PROJECT_ROOT = Path("/home/billz/work/autonomous-small-model-exploration")

blocks = [
    ("L1", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "json_schema"]),
    ("L2", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "factual_recall"]),
    ("L3", "scripts/run_phase3_atlas_guided_lora.py", ["--family", "code_semantics"]),
    ("G3", "scripts/run_phase3_checkpoint_lockin.py", []),
    ("G4", "scripts/run_phase3_atlas_guided_skip.py", []),
]

for bid, script, args in blocks:
    cmd = [sys.executable, script, "--model", "Qwen/Qwen2.5-0.5B"] + args
    print(f"\n{'='*60}", flush=True)
    print(f"Block {bid}: {script}", flush=True)
    print(f"{'='*60}", flush=True)
    start = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=7200)
        status = "success" if r.returncode == 0 else "failed"
    except Exception as e:
        status = "error: " + str(e)
    elapsed = time.time() - start
    print(f"  Block {bid}: {status} ({elapsed:.0f}s)", flush=True)

    # Sync results
    if status == "success":
        subprocess.run(["git", "add", "experiments/results/", "experiments/registry.jsonl"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Phase 3 {bid}: rerun results"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        subprocess.run(["git", "push", "origin", "master"],
                      cwd=str(PROJECT_ROOT), capture_output=True)
        print(f"  Results synced after {bid}", flush=True)

print("\nAll reruns done.", flush=True)
