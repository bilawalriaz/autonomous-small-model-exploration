"""Weight delta analysis for fine-tuned models."""

import torch


def compute_weight_deltas(
    base_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
) -> dict[str, dict]:
    """Compute per-layer weight deltas between base and trained models.

    Args:
        base_state: Base model state dict
        trained_state: Trained model state dict

    Returns:
        dict mapping layer name to delta stats
    """
    results = {}
    for key in base_state:
        if key in trained_state:
            base_w = base_state[key].float()
            trained_w = trained_state[key].float()

            if base_w.shape != trained_w.shape:
                continue

            delta = trained_w - base_w
            results[key] = {
                "l2_norm": delta.norm().item(),
                "frobenius_norm": delta.norm("fro").item() if delta.dim() >= 2 else delta.norm().item(),
                "mean_abs_delta": delta.abs().mean().item(),
                "max_abs_delta": delta.abs().max().item(),
                "relative_norm": (delta.norm() / (base_w.norm() + 1e-10)).item(),
                "shape": list(base_w.shape),
            }

    return results


def summarize_deltas_by_layer(deltas: dict[str, dict]) -> dict[str, float]:
    """Aggregate weight deltas by layer."""
    layer_sums: dict[str, float] = {}
    for key, stats in deltas.items():
        # Extract layer index
        parts = key.split(".")
        layer_key = key
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_key = f"layer_{int(parts[i+1]):02d}"
                break
        layer_sums[layer_key] = layer_sums.get(layer_key, 0.0) + stats["l2_norm"]
    return layer_sums
