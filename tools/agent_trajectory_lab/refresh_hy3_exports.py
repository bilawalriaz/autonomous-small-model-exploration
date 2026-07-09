#!/usr/bin/env python3
"""Periodically refresh HY3 clean, DPO, and training-ready trajectory exports."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOTS = [
    "runs/hy3_agentic_scale_pilot",
    "runs/hy3_agentic_scale_shard_00_r4",
    "runs/hy3_agentic_scale_shard_01_r4",
    "runs/hy3_agentic_scale_shard_02_r4",
    "runs/hy3_agentic_scale_shard_03_r4",
    "runs/hy3_agentic_balanced_shard_00_r4",
    "runs/hy3_agentic_balanced_shard_01_r4",
    "runs/hy3_agentic_balanced_shard_02_r4",
    "runs/hy3_agentic_balanced_shard_03_r4",
    "runs/hy3_agentic_balanced_shard_04_r4",
    "runs/hy3_agentic_balanced_shard_05_r4",
    "runs/hy3_agentic_balanced_shard_06_r4",
    "runs/hy3_agentic_balanced_shard_07_r4",
    "runs/hy3_agentic_balanced_shard_08_r4",
    "runs/hy3_agentic_balanced_shard_09_r4",
    "runs/hy3_agentic_balanced_shard_10_r4",
    "runs/hy3_agentic_balanced_shard_11_r4",
    "runs/hy3_agentic_balanced_shard_12_r4",
    "runs/hy3_agentic_balanced_shard_13_r4",
    "runs/hy3_agentic_balanced_shard_14_r4",
    "runs/hy3_agentic_balanced_shard_15_r4",
    "runs/hy3_state_compaction_recovery_0_r4",
    "runs/hy3_state_compaction_recovery_1_r4",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def refresh_once(args: argparse.Namespace) -> dict:
    roots = [root for root in ROOTS if Path(root).exists()]
    clean_out = Path(args.clean_out)
    dpo_out = Path(args.dpo_out)
    split_dir = Path(args.split_dir)

    commands = [
        [
            "python3",
            "export_trajectory_dataset.py",
            "--roots",
            *roots,
            "--out",
            str(clean_out),
            "--require-matched-trace",
        ],
        [
            "python3",
            "export_preference_pairs.py",
            "--roots",
            *roots,
            "--out",
            str(dpo_out),
            "--include-all-passed-contrast",
        ],
        [
            "python3",
            "export_training_ready_splits.py",
            "--sft",
            str(clean_out),
            "--dpo",
            str(dpo_out),
            "--out-dir",
            str(split_dir),
            "--val-pct",
            str(args.val_pct),
        ],
    ]
    command_results = []
    for cmd in commands:
        proc = run(cmd)
        command_results.append(
            {
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
        if proc.returncode != 0:
            break

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roots": roots,
        "commands": command_results,
        "clean_summary": load_json(clean_out.with_suffix(".summary.json")),
        "dpo_summary": load_json(dpo_out.with_suffix(".summary.json")),
        "split_manifest": load_json(split_dir / "manifest.json"),
    }
    Path(args.state).write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-out", default="datasets/hy3_agentic_scale_partial_clean.jsonl")
    parser.add_argument("--dpo-out", default="datasets/hy3_agentic_scale_partial_dpo.jsonl")
    parser.add_argument("--split-dir", default="datasets/hy3_agentic_scale_training_ready")
    parser.add_argument("--state", default="datasets/hy3_agentic_scale_refresh_state.json")
    parser.add_argument("--val-pct", type=int, default=5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep", type=int, default=600)
    args = parser.parse_args()

    while True:
        refresh_once(args)
        if not args.loop:
            return 0
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
