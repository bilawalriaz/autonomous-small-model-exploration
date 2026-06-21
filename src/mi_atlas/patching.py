"""Activation patching for causal evidence."""

import torch
import numpy as np

from .backend import BackendBase, TransformerLensBackend
from .metrics import normalized_recovery, patch_score
from .utils import save_json, now_iso, PROJECT_ROOT


def activation_patching(
    backend: BackendBase,
    clean_prompt: str,
    corrupt_prompt: str,
    component_name: str,
    metric_fn=None,
) -> dict:
    """Run activation patching for a single component.

    Patches clean activations into corrupt run at the specified component.

    Args:
        backend: Model backend
        clean_prompt: Prompt that produces correct behaviour
        corrupt_prompt: Prompt that produces incorrect behaviour
        component_name: Name of component to patch
        metric_fn: Function(logits, target) -> float metric

    Returns:
        dict with patching results
    """
    # Run clean
    clean_inputs = backend.tokenize(clean_prompt)
    clean_ids = clean_inputs["input_ids"].to(backend.device)
    clean_logits, clean_cache = backend.run_with_cache(clean_ids)

    # Run corrupt
    corrupt_inputs = backend.tokenize(corrupt_prompt)
    corrupt_ids = corrupt_inputs["input_ids"].to(backend.device)
    corrupt_logits, corrupt_cache = backend.run_with_cache(corrupt_ids)

    # Get clean activation for the component
    clean_activation = None
    for key in [component_name, f"blocks.{component_name}"]:
        if key in clean_cache:
            clean_activation = clean_cache[key]
            break

    if clean_activation is None:
        return {
            "component": component_name,
            "status": "failed",
            "error": f"Component {component_name} not found in cache",
        }

    # Patch: replace corrupt activation with clean activation
    if isinstance(backend, TransformerLensBackend):
        def patch_hook(corrupt_act, hook):
            return clean_activation.to(corrupt_act.device)

        patched_logits = backend.run_with_hooks(
            corrupt_ids, fwd_hooks=[(component_name, patch_hook)]
        )
    else:
        # Generic approach: not supported for HF backend directly
        return {
            "component": component_name,
            "status": "failed",
            "error": "Activation patching requires TransformerLens or NNsight backend",
        }

    result = {
        "component": component_name,
        "status": "success",
        "clean_logits": clean_logits.detach().cpu(),
        "corrupt_logits": corrupt_logits.detach().cpu(),
        "patched_logits": patched_logits.detach().cpu(),
    }

    # Compute metrics if function provided
    if metric_fn:
        clean_metric = metric_fn(clean_logits)
        corrupt_metric = metric_fn(corrupt_logits)
        patched_metric = metric_fn(patched_logits)

        result["clean_metric"] = clean_metric
        result["corrupt_metric"] = corrupt_metric
        result["patched_metric"] = patched_metric
        result["patch_score"] = patch_score(patched_metric, corrupt_metric)
        result["normalized_recovery"] = normalized_recovery(
            patched_metric, corrupt_metric, clean_metric
        )

    return result


def run_patching_suite(
    backend: BackendBase,
    clean_corrupt_pairs: list[dict],
    components: list[str] | None = None,
    metric_fn=None,
) -> dict:
    """Run patching across components and pairs.

    Args:
        backend: Model backend
        clean_corrupt_pairs: list of {"clean": str, "corrupt": str, "target": str, "family": str}
        components: list of component names to patch. If None, patch all layers.
        metric_fn: Metric function

    Returns:
        dict with results matrix
    """
    if components is None:
        n_layers = backend.n_layers
        components = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    results = []
    for pair in clean_corrupt_pairs:
        for comp in components:
            result = activation_patching(
                backend, pair["clean"], pair["corrupt"], comp, metric_fn
            )
            result["family"] = pair.get("family", "unknown")
            result["pair_id"] = pair.get("id", "unknown")
            results.append(result)

    return {
        "timestamp": now_iso(),
        "n_pairs": len(clean_corrupt_pairs),
        "n_components": len(components),
        "results": results,
    }
