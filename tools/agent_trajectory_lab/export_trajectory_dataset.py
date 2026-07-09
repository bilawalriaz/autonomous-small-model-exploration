#!/usr/bin/env python3
"""Export passed Hermes task runs into a durable trajectory dataset JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_results(roots: list[Path]):
    seen = set()
    for root in roots:
        for result_path in sorted(root.glob("*/result.json")):
            key = str(result_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield result_path


def matched_trace_files(result: dict) -> list[str]:
    matched = []
    for export in result.get("session_id_exports", []):
        if export.get("matches_prompt") and export.get("trace_exists"):
            matched.append(export.get("trace_path", ""))
    return [path for path in matched if path]


def build_record(result_path: Path, include_failed: bool, require_matched_trace: bool) -> dict | None:
    result = load_json(result_path)
    if not result:
        return None
    if not include_failed and not result.get("passed"):
        return None
    matched_traces = matched_trace_files(result)
    if require_matched_trace and not matched_traces:
        return None

    run_dir = result_path.parent
    prompt = read_text(run_dir / "prompt.txt")
    stdout = read_text(run_dir / "stdout.txt")
    stderr = read_text(run_dir / "stderr.txt")
    diff = read_text(run_dir / "diff.patch")
    diff_stat = read_text(run_dir / "diff_stat.txt")

    trace_files = sorted((run_dir / "session_exports").glob("*.trace.jsonl"))
    jsonl_files = sorted((run_dir / "session_exports").glob("*.jsonl"))
    trace_preview = []
    for path in trace_files[:3]:
        trace_preview.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "preview": read_text(path, limit=4000),
            }
        )

    changed_files = []
    workspace = Path(result.get("workspace", ""))
    before = set(result.get("before_tree", []))
    after = set(result.get("after_tree", []))
    for rel in sorted(after - before):
        path = workspace / rel
        changed_files.append({"path": rel, "content": read_text(path, limit=12000)})
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            rel = line.removeprefix("+++ b/")
            if rel not in {x["path"] for x in changed_files}:
                path = workspace / rel
                changed_files.append({"path": rel, "content": read_text(path, limit=12000)})

    assistant_content = stdout.strip()
    if diff.strip():
        assistant_content += "\n\nPatch produced:\n```diff\n" + diff[:16000] + "\n```"

    return {
        "task_id": result.get("task_id"),
        "family": result.get("family"),
        "passed": result.get("passed"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "prompt": prompt,
        "assistant_final": stdout,
        "stderr": stderr,
        "diff_stat": diff_stat,
        "diff": diff,
        "changed_files": changed_files,
        "verifiers": result.get("verifiers", []),
        "usage": result.get("usage", {}),
        "workspace": result.get("workspace"),
        "session_ids": result.get("session_ids", []),
        "matched_session_trace_files": matched_traces,
        "session_trace_files": [str(p) for p in trace_files],
        "session_jsonl_files": [str(p) for p in jsonl_files if not str(p).endswith(".trace.jsonl")],
        "trace_preview": trace_preview,
        "messages": [
            {
                "role": "system",
                "content": "You are a pragmatic sysadmin/code agent. Use tools to inspect, modify, and verify. Prefer safe, reproducible operations.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content.strip()},
        ],
        "source_result": str(result_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--require-matched-trace", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for result_path in iter_results(roots):
        record = build_record(result_path, args.include_failed, args.require_matched_trace)
        if record:
            records.append(record)

    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "records": len(records),
        "passed": sum(1 for r in records if r["passed"]),
        "families": {},
        "with_trace_files": sum(1 for r in records if r["session_trace_files"]),
        "with_matched_trace_files": sum(1 for r in records if r["matched_session_trace_files"]),
        "require_matched_trace": args.require_matched_trace,
        "output": str(out),
    }
    for record in records:
        summary["families"].setdefault(record["family"], 0)
        summary["families"][record["family"]] += 1
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
