"""Adapter analysis: norm, effective rank, specificity."""

import torch
import numpy as np
from pathlib import Path
from peft import PeftModel


def load_adapter_weights(adapter_path: str | Path) -> dict[str, torch.Tensor]:
    """Load adapter weights from a PEFT adapter directory."""
    from safetensors.torch import load_file
    adapter_path = Path(adapter_path)
    weight_file = adapter_path / "adapter_model.safetensors"
    if weight_file.exists():
        return load_file(str(weight_file))
    # Fallback: try .bin
    bin_file = adapter_path / "adapter_model.bin"
    if bin_file.exists():
        return torch.load(str(bin_file), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def adapter_norms_by_layer(weights: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute L2 norm of adapter weights per layer."""
    layer_norms: dict[str, float] = {}
    for key, tensor in weights.items():
        # Extract layer index from key like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is not None:
            norm = tensor.float().norm().item()
            layer_key = f"layer_{layer_idx:02d}"
            layer_norms[layer_key] = layer_norms.get(layer_key, 0.0) + norm
    return layer_norms


def effective_rank(weights: dict[str, torch.Tensor]) -> dict[str, float]:
    """Approximate effective rank of adapter weight matrices."""
    results = {}
    for key, tensor in weights.items():
        if "lora_A" in key:
            t = tensor.float()
            if t.dim() >= 2:
                try:
                    _, S, _ = torch.linalg.svd(t, full_matrices=False)
                    # Effective rank = exp(entropy of singular value distribution)
                    S_norm = S / S.sum()
                    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
                    results[key] = entropy.exp().item()
                except Exception:
                    results[key] = 0.0
    return results
