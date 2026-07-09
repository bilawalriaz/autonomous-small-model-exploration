#!/usr/bin/env python3
"""Keep HY3 balanced shard collection moving without exceeding a worker cap."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


MODEL_ARGS = [
    "--hermes-arg=--provider",
    "--hermes-arg=nous",
    "--hermes-arg=--model",
    "--hermes-arg=tencent/hy3:free",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def pid_is_alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def active_shards() -> dict[str, str]:
    proc = run(["pgrep", "-af", "run_hermes_tasks.py --tasks agentic_scale_tasks_balanced_shard_"])
    active: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "run_hermes_tasks.py" not in line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, cmdline = parts
        for token in shlex.split(cmdline):
            if token.startswith("agentic_scale_tasks_balanced_shard_") and token.endswith(".json"):
                shard = token.removeprefix("agentic_scale_tasks_balanced_shard_").removesuffix(".json")
                active[shard] = pid
    return active


def shard_status(shard: str, rollouts: int) -> dict[str, int | bool]:
    task_path = Path(f"agentic_scale_tasks_balanced_shard_{shard}.json")
    out_dir = Path(f"runs/hy3_agentic_balanced_shard_{shard}_r{rollouts}")
    tasks = json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else []
    expected = len(tasks) * rollouts
    results = list(out_dir.glob("*/result.json"))
    passed = 0
    failed = 0
    for path in results:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failed += 1
            continue
        if result.get("passed"):
            passed += 1
        else:
            failed += 1
    return {
        "expected": expected,
        "results": len(results),
        "passed": passed,
        "failed": failed,
        "complete": expected > 0 and len(results) >= expected and failed == 0,
    }


def launch_shard(shard: str, args: argparse.Namespace) -> int:
    out_dir = Path(f"runs/hy3_agentic_balanced_shard_{shard}_r{args.rollouts}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"runs/hy3_agentic_balanced_shard_{shard}_r{args.rollouts}.log")
    pid_path = Path(f"runs/hy3_agentic_balanced_shard_{shard}_r{args.rollouts}.pid")
    if pid_path.exists() and pid_is_alive(pid_path):
        return int(pid_path.read_text(encoding="utf-8").strip())

    cmd = [
        "python3",
        "run_hermes_tasks.py",
        "--tasks",
        f"agentic_scale_tasks_balanced_shard_{shard}.json",
        "--out",
        str(out_dir),
        "--skip-existing",
        "--rerun-failed",
        "--rollouts",
        str(args.rollouts),
        "--timeout",
        str(args.timeout),
        "--transient-retries",
        str(args.transient_retries),
        "--transient-sleep",
        str(args.transient_sleep),
        "--clean-incomplete",
        *MODEL_ARGS,
    ]
    if args.dry_run:
        print("DRY-RUN launch", shard, shlex.join(cmd), flush=True)
        return 0
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"launched shard {shard} pid={proc.pid}", flush=True)
    return proc.pid


def supervise_once(args: argparse.Namespace) -> dict:
    active = active_shards()
    launched = []
    statuses = {}
    for index in range(args.start_shard, args.end_shard + 1):
        shard = f"{index:02d}"
        statuses[shard] = shard_status(shard, args.rollouts)

    slots = max(args.max_workers - len(active), 0)
    for index in range(args.start_shard, args.end_shard + 1):
        if slots <= 0:
            break
        shard = f"{index:02d}"
        if shard in active:
            continue
        if statuses[shard]["complete"]:
            continue
        if not Path(f"agentic_scale_tasks_balanced_shard_{shard}.json").exists():
            continue
        pid = launch_shard(shard, args)
        launched.append({"shard": shard, "pid": pid})
        slots -= 1

    state = {"active": active, "launched": launched, "statuses": statuses}
    print(json.dumps(state, indent=2), flush=True)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=15)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--transient-retries", type=int, default=4)
    parser.add_argument("--transient-sleep", type=int, default=90)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    while True:
        supervise_once(args)
        if not args.loop:
            return 0
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
