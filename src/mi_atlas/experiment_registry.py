"""Experiment registry: track all runs in experiments/registry.jsonl."""

from typing import Any

from .utils import append_jsonl, load_jsonl, now_iso, git_commit_hash, PROJECT_ROOT


def next_experiment_id() -> str:
    """Generate the next experiment ID."""
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    existing = load_jsonl(registry_path)
    max_num = 0
    for record in existing:
        try:
            num = int(record.get("id", "exp_000000").split("_")[1])
            max_num = max(max_num, num)
        except (ValueError, IndexError):
            pass
    return f"exp_{max_num + 1:06d}"


def register_experiment(
    type: str,
    model: str,
    backend: str,
    config: str,
    inputs: list[str],
    outputs: list[str],
    status: str,
    summary: str,
    key_metrics: dict[str, Any] | None = None,
    failure: str | None = None,
    next: str | None = None,
) -> dict:
    """Register an experiment in the registry and return the record."""
    record = {
        "id": next_experiment_id(),
        "timestamp": now_iso(),
        "type": type,
        "model": model,
        "backend": backend,
        "git_commit": git_commit_hash(),
        "config": config,
        "inputs": inputs,
        "outputs": outputs,
        "status": status,
        "summary": summary,
        "key_metrics": key_metrics or {},
        "failure": failure,
        "next": next,
    }
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    append_jsonl(record, registry_path)
    return record


def load_registry() -> list[dict]:
    """Load the full experiment registry."""
    return load_jsonl(PROJECT_ROOT / "experiments" / "registry.jsonl")


def get_experiment(exp_id: str) -> dict | None:
    """Get a specific experiment by ID."""
    for record in load_registry():
        if record.get("id") == exp_id:
            return record
    return None


def count_by_status() -> dict[str, int]:
    """Count experiments by status."""
    counts: dict[str, int] = {}
    for record in load_registry():
        s = record.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    return counts


def count_by_type() -> dict[str, int]:
    """Count experiments by type."""
    counts: dict[str, int] = {}
    for record in load_registry():
        t = record.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts
