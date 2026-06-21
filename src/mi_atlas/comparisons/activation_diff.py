"""Activation distribution comparison across models/checkpoints."""

import torch
import numpy as np


def compare_activation_means(
    cache_a: dict[str, torch.Tensor],
    cache_b: dict[str, torch.Tensor],
    layer_names: list[str],
) -> dict:
    """Compare mean activations between two models/checkpoints.

    Args:
        cache_a: Activation cache from model A
        cache_b: Activation cache from model B
        layer_names: Layers to compare

    Returns:
        dict with cosine similarities and L2 distances per layer
    """
    results = {}
    for name in layer_names:
        if name in cache_a and name in cache_b:
            a = cache_a[name].float()
            b = cache_b[name].float()

            if a.dim() > 2:
                a = a.mean(dim=tuple(range(a.dim() - 1)))
            if b.dim() > 2:
                b = b.mean(dim=tuple(range(b.dim() - 1)))

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
            ).item()

            # L2 distance
            l2_dist = (a.flatten() - b.flatten()).norm().item()

            results[name] = {
                "cosine_similarity": cos_sim,
                "l2_distance": l2_dist,
            }

    return results
