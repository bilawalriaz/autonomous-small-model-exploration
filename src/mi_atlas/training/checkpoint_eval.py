"""Checkpoint evaluation during training."""

from pathlib import Path

from ..eval_runner import evaluate_suite
from ..backend import BackendBase, create_backend
from ..model_loader import load_model_hf
from ..task_suite import TaskSuite


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    suite: TaskSuite,
    split: str | None = "test",
) -> dict:
    """Evaluate a training checkpoint.

    Args:
        checkpoint_path: Path to HF checkpoint directory
        suite: Task suite to evaluate on
        split: Which split to evaluate

    Returns:
        dict with evaluation results
    """
    bundle = load_model_hf(str(checkpoint_path))
    backend = create_backend(bundle)
    results = evaluate_suite(backend, suite, split=split)
    return results


def evaluate_checkpoints(
    checkpoint_paths: list[str | Path],
    suite: TaskSuite,
    split: str | None = "test",
) -> list[dict]:
    """Evaluate multiple checkpoints."""
    results = []
    for path in checkpoint_paths:
        try:
            result = evaluate_checkpoint(path, suite, split)
            result["checkpoint_path"] = str(path)
            results.append(result)
        except Exception as e:
            results.append({
                "checkpoint_path": str(path),
                "error": str(e),
            })
    return results
