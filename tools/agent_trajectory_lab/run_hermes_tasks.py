#!/usr/bin/env python3
"""Run Hermes Agent tasks in isolated workspaces and capture trace artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERMES_BIN = Path(os.environ.get("HERMES_BIN", "~/.hermes/hermes-agent/venv/bin/hermes")).expanduser()
SESSION_ID_RE = re.compile(r"\b\d{8}_\d{6}_[0-9a-f]{6}\b")
TRANSIENT_PATTERNS = (
    "HTTP 429",
    "rate limit",
    "exceeded the rate limit",
    "API call failed after",
    "temporarily unavailable",
)


def run(cmd: list[str] | str, cwd: Path, timeout: int = 300, env: dict[str, str] | None = None):
    shell = isinstance(cmd, str)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def is_transient_api_failure(proc: subprocess.CompletedProcess[str]) -> bool:
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    return any(pattern.lower() in text for pattern in TRANSIENT_PATTERNS)


def write_seed_files(workspace: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def write_pytest_shim(workspace: Path) -> None:
    shim = workspace / "pytest.py"
    if shim.exists():
        return
    shim.write_text(
        """#!/usr/bin/env python3
import importlib.util
import pathlib
import sys
import traceback

failures = 0
for path in sorted(pathlib.Path('.').glob('test_*.py')):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
        for name, obj in sorted(vars(mod).items()):
            if name.startswith('test_') and callable(obj):
                try:
                    obj()
                    print(f'{path.name}::{name} PASSED')
                except Exception:
                    failures += 1
                    print(f'{path.name}::{name} FAILED')
                    traceback.print_exc()
    except Exception:
        failures += 1
        print(f'{path.name} import FAILED')
        traceback.print_exc()
sys.exit(1 if failures else 0)
""",
        encoding="utf-8",
    )


def tree_snapshot(workspace: Path) -> list[str]:
    paths = []
    for path in sorted(workspace.rglob("*")):
        if ".git" in path.parts:
            continue
        if path.is_file():
            paths.append(str(path.relative_to(workspace)))
    return paths


def export_sessions(workspace: Path, out_path: Path) -> dict:
    cmd = [
        str(HERMES_BIN),
        "sessions",
        "export",
        "--format",
        "jsonl",
        "--cwd",
        str(workspace),
        "--newer-than",
        "2h",
        "--yes",
        str(out_path),
    ]
    proc = run(cmd, cwd=workspace, timeout=120)
    return {
        "cmd": shlex.join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "path": str(out_path),
        "exists": out_path.exists(),
    }


def list_recent_session_ids(cwd: Path, limit: int = 40) -> list[str]:
    cmd = [str(HERMES_BIN), "sessions", "list", "--source", "cli", "--limit", str(limit)]
    proc = run(cmd, cwd=cwd, timeout=60)
    seen = []
    for match in SESSION_ID_RE.findall(proc.stdout):
        if match not in seen:
            seen.append(match)
    return seen


def export_session_ids(cwd: Path, session_ids: list[str], out_dir: Path, prompt_marker: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for session_id in session_ids:
        jsonl_path = out_dir / f"{session_id}.jsonl"
        trace_path = out_dir / f"{session_id}.trace.jsonl"
        json_cmd = [
            str(HERMES_BIN),
            "sessions",
            "export",
            "--format",
            "jsonl",
            "--session-id",
            session_id,
            "--yes",
            str(jsonl_path),
        ]
        trace_cmd = [
            str(HERMES_BIN),
            "sessions",
            "export",
            "--format",
            "trace",
            "--session-id",
            session_id,
            "--yes",
            str(trace_path),
        ]
        json_proc = run(json_cmd, cwd=cwd, timeout=120)
        trace_proc = run(trace_cmd, cwd=cwd, timeout=120)
        matches_prompt = False
        if jsonl_path.exists():
            try:
                matches_prompt = prompt_marker in jsonl_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                matches_prompt = False
        if not matches_prompt:
            jsonl_path.unlink(missing_ok=True)
            trace_path.unlink(missing_ok=True)
        exports.append(
            {
                "session_id": session_id,
                "matches_prompt": matches_prompt,
                "jsonl_path": str(jsonl_path),
                "trace_path": str(trace_path),
                "jsonl_exists": jsonl_path.exists(),
                "trace_exists": trace_path.exists(),
                "jsonl_returncode": json_proc.returncode,
                "trace_returncode": trace_proc.returncode,
                "jsonl_stdout": json_proc.stdout[-1000:],
                "jsonl_stderr": json_proc.stderr[-1000:],
                "trace_stdout": trace_proc.stdout[-1000:],
                "trace_stderr": trace_proc.stderr[-1000:],
            }
        )
    return exports


def run_task(
    task: dict,
    out_root: Path,
    hermes_args: list[str],
    timeout: int,
    rollout_index: int | None = None,
    transient_retries: int = 0,
    transient_sleep: int = 60,
) -> dict:
    task_id = task["id"]
    run_id = task_id if rollout_index is None else f"{task_id}__r{rollout_index:02d}"
    started = datetime.now(timezone.utc).isoformat()
    out_root = out_root.resolve()
    run_dir = out_root / run_id
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    write_seed_files(workspace, task.get("files", {}))
    write_pytest_shim(workspace)
    run(["git", "init", "-q"], cwd=workspace, timeout=30)
    run(["git", "add", "."], cwd=workspace, timeout=30)
    run(["git", "commit", "-qm", "seed"], cwd=workspace, timeout=30)
    before_tree = tree_snapshot(workspace)

    prompt_marker = f"Task ID: {task_id}"
    if rollout_index is not None:
        prompt_marker += f" | Rollout: {rollout_index:02d}"
    prompt = prompt_marker + "\n\n" + task["prompt"].strip() + (
        "\n\nOperate only inside the current working directory. "
        "Use tools as needed. When finished, briefly summarize what changed and what verification you ran."
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    usage_file = run_dir / "usage.json"

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    env.setdefault("HERMES_ACCEPT_HOOKS", "1")
    before_sessions = set(list_recent_session_ids(workspace))

    cmd = [
        str(HERMES_BIN),
        "--ignore-rules",
        "--yolo",
        "--usage-file",
        str(usage_file),
        "-z",
        prompt,
        *hermes_args,
    ]
    t0 = time.monotonic()
    hermes_attempts = []
    proc = None
    for attempt in range(transient_retries + 1):
        proc = run(cmd, cwd=workspace, timeout=timeout, env=env)
        transient = is_transient_api_failure(proc)
        hermes_attempts.append(
            {
                "attempt": attempt,
                "returncode": proc.returncode,
                "transient_api_failure": transient,
                "stdout_tail": proc.stdout[-1000:],
                "stderr_tail": proc.stderr[-1000:],
            }
        )
        if not transient or attempt >= transient_retries:
            break
        time.sleep(transient_sleep * (attempt + 1))
    assert proc is not None
    elapsed = round(time.monotonic() - t0, 3)
    after_sessions = list_recent_session_ids(workspace)
    new_sessions = [sid for sid in after_sessions if sid not in before_sessions]

    (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    verifier_results = []
    for spec in task.get("verifiers", []):
        v_cmd = spec["cmd"]
        v_proc = run(v_cmd, cwd=workspace, timeout=spec.get("timeout", 120))
        verifier_results.append(
            {
                "name": spec.get("name", v_cmd),
                "cmd": v_cmd,
                "returncode": v_proc.returncode,
                "stdout": v_proc.stdout[-8000:],
                "stderr": v_proc.stderr[-8000:],
                "passed": v_proc.returncode == 0,
            }
        )

    diff_proc = run(["git", "diff", "--", "."], cwd=workspace, timeout=60)
    stat_proc = run(["git", "diff", "--stat", "--", "."], cwd=workspace, timeout=60)
    (run_dir / "diff.patch").write_text(diff_proc.stdout, encoding="utf-8")
    (run_dir / "diff_stat.txt").write_text(stat_proc.stdout, encoding="utf-8")

    session_export = export_sessions(workspace, run_dir / "hermes_sessions.jsonl")
    session_id_exports = export_session_ids(workspace, new_sessions, run_dir / "session_exports", prompt_marker)
    usage = {}
    if usage_file.exists():
        try:
            usage = json.loads(usage_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            usage = {"parse_error": True, "raw": usage_file.read_text(encoding="utf-8")[:2000]}

    result = {
        "task_id": task_id,
        "run_id": run_id,
        "rollout_index": rollout_index,
        "family": task.get("family"),
        "template_id": task.get("template_id"),
        "variant_id": task.get("variant_id"),
        "started": started,
        "elapsed_seconds": elapsed,
        "workspace": str(workspace.resolve()),
        "hermes_cmd": shlex.join(cmd),
        "hermes_returncode": proc.returncode,
        "hermes_stdout_tail": proc.stdout[-4000:],
        "hermes_stderr_tail": proc.stderr[-4000:],
        "hermes_attempts": hermes_attempts,
        "transient_api_failure": is_transient_api_failure(proc),
        "usage": usage,
        "before_tree": before_tree,
        "after_tree": tree_snapshot(workspace),
        "diff_stat": stat_proc.stdout,
        "verifiers": verifier_results,
        "passed": proc.returncode == 0 and all(v["passed"] for v in verifier_results),
        "session_export": session_export,
        "session_ids": new_sessions,
        "prompt_marker": prompt_marker,
        "session_id_exports": session_id_exports,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="agent_tasks.json")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--rollout-offset", type=int, default=0)
    parser.add_argument("--transient-retries", type=int, default=0)
    parser.add_argument("--transient-sleep", type=int, default=60)
    parser.add_argument("--hermes-arg", action="append", default=[])
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit:
        tasks = tasks[: args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"

    for task in tasks:
        for rollout in range(args.rollout_offset, args.rollout_offset + args.rollouts):
            rollout_index = None if args.rollouts == 1 and args.rollout_offset == 0 else rollout
            run_id = task["id"] if rollout_index is None else f"{task['id']}__r{rollout_index:02d}"
            existing = out_root / run_id / "result.json"
            if args.skip_existing and existing.exists():
                try:
                    prior = json.loads(existing.read_text(encoding="utf-8"))
                    if prior.get("passed") or not args.rerun_failed:
                        print(f"==> {run_id}: skip existing passed={prior.get('passed')}", flush=True)
                        continue
                    print(f"==> {run_id}: rerun failed existing", flush=True)
                    shutil.rmtree(existing.parent)
                except json.JSONDecodeError:
                    pass
            print(f"==> {run_id}: {task.get('family', '')}", flush=True)
            result = run_task(
                task,
                out_root,
                args.hermes_arg,
                args.timeout,
                rollout_index,
                args.transient_retries,
                args.transient_sleep,
            )
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print(f"    passed={result['passed']} elapsed={result['elapsed_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
