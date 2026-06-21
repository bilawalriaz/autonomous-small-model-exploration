"""Checkpoint comparison: eval metrics across training checkpoints."""

import torch
import numpy as np


def compare_checkpoint_metrics(
    results: list[dict],
) -> dict:
    """Compare evaluation metrics across checkpoints.

    Args:
        results: list of eval result dicts from checkpoint_eval

    Returns:
        dict with comparison tables
    """
    # Extract primary metric per family per checkpoint
    comparison = {}
    for result in results:
        path = result.get("checkpoint_path", "unknown")
        summary = result.get("summary", {})
        family_means = summary.get("primary_metric_by_family", {})
        comparison[path] = family_means

    return {
        "checkpoints": list(comparison.keys()),
        "family_scores": comparison,
    }
