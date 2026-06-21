"""Activation caching utilities."""

from pathlib import Path

import torch
import numpy as np

from .backend import BackendBase
from .utils import save_json, load_json, PROJECT_ROOT, ensure_dir


def cache_activations(
    backend: BackendBase,
    prompts: list[str],
    save_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Run model on prompts and cache all activations.

    Returns dict mapping layer_name -> stacked activations tensor.
    """
    all_activations: dict[str, list[torch.Tensor]] = {}

    for prompt in prompts:
        inputs = backend.tokenize(prompt)
        input_ids = inputs["input_ids"].to(backend.device)
        logits, cache = backend.run_with_cache(input_ids)

        for key, tensor in cache.items():
            if key not in all_activations:
                all_activations[key] = []
            all_activations[key].append(tensor.detach().cpu())

    # Stack along batch dimension
    stacked = {}
    for key, tensors in all_activations.items():
        try:
            stacked[key] = torch.cat(tensors, dim=0)
        except RuntimeError:
            # Different shapes — store as list
            stacked[key] = tensors

    if save_path:
        ensure_dir(Path(save_path).parent)
        torch.save(stacked, save_path)

    return stacked


def load_cached_activations(path: str | Path) -> dict[str, torch.Tensor]:
    """Load cached activations from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def get_layer_activations(
    cache: dict[str, torch.Tensor], layer_idx: int
) -> torch.Tensor | None:
    """Get activations for a specific layer from cache."""
    keys_to_try = [
        f"layer_{layer_idx:02d}",
        f"blocks.{layer_idx}.hook_resid_post",
        f"blocks.{layer_idx}.hook_resid_pre",
    ]
    for key in keys_to_try:
        if key in cache:
            return cache[key]
    return None
