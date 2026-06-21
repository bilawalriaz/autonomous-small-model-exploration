"""Steering vector methods."""

import torch
import numpy as np

from .backend import BackendBase


def compute_steering_vector(
    backend: BackendBase,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_name: str,
) -> torch.Tensor:
    """Compute steering vector as mean(positive) - mean(negative) activations.

    Args:
        backend: Model backend
        positive_prompts: Prompts exhibiting target behaviour
        negative_prompts: Prompts not exhibiting target behaviour
        layer_name: Layer to extract activations from

    Returns:
        Steering vector tensor
    """
    pos_acts = []
    neg_acts = []

    for prompt in positive_prompts:
        inputs = backend.tokenize(prompt)
        input_ids = inputs["input_ids"].to(backend.device)
        _, cache = backend.run_with_cache(input_ids)
        if layer_name in cache:
            # Take last token activation
            pos_acts.append(cache[layer_name][0, -1, :].detach().cpu())

    for prompt in negative_prompts:
        inputs = backend.tokenize(prompt)
        input_ids = inputs["input_ids"].to(backend.device)
        _, cache = backend.run_with_cache(input_ids)
        if layer_name in cache:
            neg_acts.append(cache[layer_name][0, -1, :].detach().cpu())

    if not pos_acts or not neg_acts:
        return torch.zeros(backend.d_model)

    mean_pos = torch.stack(pos_acts).mean(dim=0)
    mean_neg = torch.stack(neg_acts).mean(dim=0)

    return mean_pos - mean_neg


def inject_steering_vector(
    backend: BackendBase,
    prompt: str,
    layer_name: str,
    steering_vector: torch.Tensor,
    strength: float = 1.0,
    position: str = "last",
) -> dict:
    """Inject steering vector at a layer and measure effect.

    Args:
        backend: Model backend
        prompt: Input prompt
        layer_name: Layer to inject at
        steering_vector: The steering direction
        strength: Multiplier for the steering vector
        position: "last" for last token, "all" for all positions

    Returns:
        dict with original and steered results
    """
    inputs = backend.tokenize(prompt)
    input_ids = inputs["input_ids"].to(backend.device)

    # Original
    original_logits, _ = backend.run_with_cache(input_ids)

    # Steered (requires TransformerLens)
    if not hasattr(backend.model, 'run_with_hooks'):
        return {
            "status": "failed",
            "error": "Steering requires TransformerLens backend",
        }

    sv = steering_vector.to(backend.device) * strength

    def steer_hook(activation, hook):
        if position == "last":
            activation[0, -1, :] += sv
        else:
            activation += sv.unsqueeze(0).unsqueeze(0)
        return activation

    steered_logits = backend.run_with_hooks(
        input_ids, fwd_hooks=[(layer_name, steer_hook)]
    )

    return {
        "status": "success",
        "original_logits": original_logits.detach().cpu(),
        "steered_logits": steered_logits.detach().cpu(),
        "strength": strength,
        "layer": layer_name,
    }


def steering_strength_sweep(
    backend: BackendBase,
    prompt: str,
    layer_name: str,
    steering_vector: torch.Tensor,
    strengths: list[float] | None = None,
) -> list[dict]:
    """Sweep steering vector strengths."""
    if strengths is None:
        strengths = [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8]

    results = []
    for s in strengths:
        result = inject_steering_vector(backend, prompt, layer_name, steering_vector, s)
        result["strength"] = s
        results.append(result)

    return results
