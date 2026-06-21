"""Ablation methods for component localisation."""

import torch
import numpy as np

from .backend import BackendBase, TransformerLensBackend, HFBackend
from .task_suite import TaskSuite, TaskExample
from .metrics import exact_match_score
from .utils import save_json, now_iso, PROJECT_ROOT


def mean_ablation_hook(activation: torch.Tensor, mean_values: torch.Tensor) -> torch.Tensor:
    """Replace activation with mean values (broadcastable)."""
    return mean_values.expand_as(activation)


def zero_ablation_hook(activation: torch.Tensor) -> torch.Tensor:
    """Zero out activation."""
    return torch.zeros_like(activation)


def resample_ablation_hook(
    activation: torch.Tensor,
    donor_activation: torch.Tensor,
) -> torch.Tensor:
    """Replace activation with values from a different input."""
    return donor_activation[:activation.shape[0]]


def compute_mean_activations(
    cache: dict[str, torch.Tensor],
    layer_names: list[str],
) -> dict[str, torch.Tensor]:
    """Compute mean activation across batch for each layer."""
    means = {}
    for name in layer_names:
        if name in cache:
            tensor = cache[name]
            if isinstance(tensor, list):
                tensor = torch.cat(tensor, dim=0)
            means[name] = tensor.mean(dim=0, keepdim=True)
    return means


def ablate_layer(
    backend: BackendBase,
    prompt: str,
    layer_name: str,
    ablation_type: str = "mean",
    mean_values: torch.Tensor | None = None,
) -> dict:
    """Ablate a single layer and measure effect.

    Args:
        backend: Model backend
        prompt: Input prompt
        layer_name: Name of layer to ablate
        ablation_type: "mean", "zero", or "resample"
        mean_values: Pre-computed mean activations for mean ablation

    Returns:
        dict with original and ablated logits
    """
    inputs = backend.tokenize(prompt)
    input_ids = inputs["input_ids"].to(backend.device)

    # Get original logits
    with torch.no_grad():
        original_logits = backend.model(input_ids)
        if hasattr(original_logits, "logits"):
            original_logits = original_logits.logits

    # Apply ablation — approach depends on backend
    if isinstance(backend, TransformerLensBackend):
        ablated_logits = _ablate_tl(backend, input_ids, layer_name, ablation_type, mean_values)
    else:
        ablated_logits = _ablate_hf(backend, input_ids, layer_name, ablation_type, mean_values)

    return {
        "original_logits": original_logits.detach().cpu(),
        "ablated_logits": ablated_logits.detach().cpu(),
        "layer_name": layer_name,
        "ablation_type": ablation_type,
    }


def _ablate_tl(
    backend: TransformerLensBackend,
    input_ids: torch.Tensor,
    layer_name: str,
    ablation_type: str,
    mean_values: torch.Tensor | None,
) -> torch.Tensor:
    """Ablate using TransformerLens hooks."""
    def hook_fn(activation, hook):
        if ablation_type == "zero":
            return torch.zeros_like(activation)
        elif ablation_type == "mean" and mean_values is not None:
            return mean_values.to(activation.device).expand_as(activation)
        return activation

    hooks = [(layer_name, hook_fn)]
    return backend.run_with_hooks(input_ids, fwd_hooks=hooks)


def _ablate_hf(
    backend: HFBackend,
    input_ids: torch.Tensor,
    layer_name: str,
    ablation_type: str,
    mean_values: torch.Tensor | None,
) -> torch.Tensor:
    """Ablate using HF forward hooks."""
    # Parse layer index from name like "layer_05"
    try:
        layer_idx = int(layer_name.split("_")[-1])
    except (ValueError, IndexError):
        layer_idx = 0

    # Register forward hook
    hooks = []
    handle = None

    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if ablation_type == "zero":
            modified = torch.zeros_like(hidden)
        elif ablation_type == "mean" and mean_values is not None:
            modified = mean_values.to(hidden.device).expand_as(hidden)
        else:
            modified = hidden

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    # Try to find the layer module
    model = backend.model
    layer_module = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_module = model.transformer.h[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_module = model.model.layers[layer_idx]
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layer_module = model.gpt_neox.layers[layer_idx]

    if layer_module is not None:
        handle = layer_module.register_forward_hook(ablation_hook)

    with torch.no_grad():
        outputs = backend.model(input_ids)
        result = outputs.logits

    if handle is not None:
        handle.remove()

    return result


def run_layer_ablation_suite(
    backend: BackendBase,
    suite: TaskSuite,
    metric_fn=None,
    ablation_type: str = "mean",
    split: str | None = None,
) -> dict:
    """Run layer ablation across all layers and task families.

    Returns dict with ablation effect matrix.
    """
    if split:
        suite = suite.filter_by_split(split)

    n_layers = backend.n_layers
    families = suite.families
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]

    # Default metric: exact match
    if metric_fn is None:
        metric_fn = exact_match_score

    # Results matrix: layers x families
    effect_matrix = np.zeros((n_layers, len(families)))

    for fam_idx, family in enumerate(families):
        family_suite = suite.filter_by_family(family)
        family_examples = list(family_suite)[:5]  # Limit for speed

        for layer_idx in range(n_layers):
            layer_name = layer_names[layer_idx]
            effects = []

            for example in family_examples:
                result = ablate_layer(
                    backend, example.clean_prompt, layer_name, ablation_type
                )
                # Compare original vs ablated next-token distribution
                orig_logits = result["original_logits"][0, -1, :]
                abl_logits = result["ablated_logits"][0, -1, :]
                orig_probs = torch.softmax(orig_logits, dim=-1)
                abl_probs = torch.softmax(abl_logits, dim=-1)

                # KL divergence as effect measure
                kl = torch.nn.functional.kl_div(
                    abl_probs.log(), orig_probs, reduction="sum"
                ).item()
                effects.append(kl)

            effect_matrix[layer_idx, fam_idx] = np.mean(effects) if effects else 0.0

    return {
        "effect_matrix": effect_matrix.tolist(),
        "layer_names": layer_names,
        "families": families,
        "ablation_type": ablation_type,
        "n_layers": n_layers,
    }
