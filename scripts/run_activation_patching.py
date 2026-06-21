"""Activation patching using PyTorch hooks.

For each clean/corrupt pair, patch activations from the clean run into
the corrupt run at specific layers/components and measure recovery.

This provides STRONG causal evidence (not just ablation).
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, load_json


def get_residual_hook(replacement_tensor, position=-1):
    """Hook that replaces residual stream at a specific position."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Replace at the target position
        hidden[:, position, :] = replacement_tensor[:, position, :].to(hidden.device)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook_fn


def get_mlp_hook(replacement_tensor, position=-1):
    """Hook that replaces MLP output at a specific position."""
    def hook_fn(module, input, output):
        output[:, position, :] = replacement_tensor[:, position, :].to(output.device)
        return output
    return hook_fn


def get_attention_hook(replacement_tensor, position=-1):
    """Hook that replaces attention output at a specific position."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        attn_output[:, position, :] = replacement_tensor[:, position, :].to(attn_output.device)

        if isinstance(output, tuple):
            return (attn_output,) + output[1:]
        return attn_output
    return hook_fn


def run_patching_experiment(
    model, tokenizer, clean_prompt, corrupt_prompt, target,
    component_type="residual", layers_to_patch=None, position=-1,
):
    """Run activation patching for a single pair.

    Patches clean activations into corrupt run at specified layers.

    Returns dict with patching results per layer.
    """
    if layers_to_patch is None:
        layers_to_patch = list(range(model.config.num_hidden_layers))

    # Tokenize
    clean_ids = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    # Run clean and corrupt
    with torch.no_grad():
        clean_logits = model(clean_ids).logits
        corrupt_logits = model(corrupt_ids).logits

    # Get clean target logprob (from clean run)
    clean_logprob = torch.log_softmax(clean_logits[0, -1], dim=-1)
    clean_target_lp = sum(clean_logprob[tid].item() for tid in target_ids) if len(target_ids) > 0 else 0

    corrupt_logprob = torch.log_softmax(corrupt_logits[0, -1], dim=-1)
    corrupt_target_lp = sum(corrupt_logprob[tid].item() for tid in target_ids) if len(target_ids) > 0 else 0

    results = {
        "clean_target_logprob": clean_target_lp,
        "corrupt_target_logprob": corrupt_target_lp,
        "patching_results": {},
    }

    # Run clean to get intermediate activations
    clean_activations = {}
    hooks = []

    for layer_idx in layers_to_patch:
        layer = model.model.layers[layer_idx]
        if component_type == "residual":
            def capture_hook(module, input, output, li=layer_idx):
                if isinstance(output, tuple):
                    clean_activations[li] = output[0].detach().clone()
                else:
                    clean_activations[li] = output.detach().clone()
            hooks.append(layer.register_forward_hook(capture_hook))
        elif component_type == "mlp":
            def capture_hook(module, input, output, li=layer_idx):
                clean_activations[li] = output.detach().clone()
            hooks.append(layer.mlp.register_forward_hook(capture_hook))
        elif component_type == "attn":
            def capture_hook(module, input, output, li=layer_idx):
                if isinstance(output, tuple):
                    clean_activations[li] = output[0].detach().clone()
                else:
                    clean_activations[li] = output.detach().clone()
            hooks.append(layer.self_attn.register_forward_hook(capture_hook))

    with torch.no_grad():
        _ = model(clean_ids)

    for h in hooks:
        h.remove()

    # Now patch each layer
    for layer_idx in layers_to_patch:
        if layer_idx not in clean_activations:
            continue

        clean_act = clean_activations[layer_idx]
        layer = model.model.layers[layer_idx]

        if component_type == "residual":
            hook = layer.register_forward_hook(get_residual_hook(clean_act, position))
        elif component_type == "mlp":
            hook = layer.mlp.register_forward_hook(get_mlp_hook(clean_act, position))
        elif component_type == "attn":
            hook = layer.self_attn.register_forward_hook(get_attention_hook(clean_act, position))

        with torch.no_grad():
            patched_logits = model(corrupt_ids).logits

        hook.remove()

        patched_logprob = torch.log_softmax(patched_logits[0, -1], dim=-1)
        patched_target_lp = sum(patched_logprob[tid].item() for tid in target_ids) if len(target_ids) > 0 else 0

        # Normalized recovery
        denom = max(1e-8, clean_target_lp - corrupt_target_lp)
        recovery = (patched_target_lp - corrupt_target_lp) / denom

        results["patching_results"][f"layer_{layer_idx:02d}"] = {
            "patched_target_logprob": patched_target_lp,
            "raw_recovery": patched_target_lp - corrupt_target_lp,
            "normalized_recovery": recovery,
        }

    return results


def main():
    set_seed(42)

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    n_layers = bundle.architecture["n_layers"]

    # Load clean/corrupt pairs
    pairs_path = str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v0.json")
    pairs = load_json(pairs_path)
    print(f"  Loaded {len(pairs)} clean/corrupt pairs")

    # Focus on layers identified as important by ablation
    # Layer 0, 2, 3, 5, 7, 8, 13, 21, 22, 23
    layers_to_patch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 21, 22, 23]

    component_types = ["residual", "mlp", "attn"]
    all_results = {}

    for comp_type in component_types:
        print(f"\n  === {comp_type.upper()} PATCHING ===")
        comp_results = []

        for pair in pairs[:10]:  # Limit to 10 pairs for speed
            pair_id = pair.get("id", "unknown")
            family = pair.get("family", "unknown")
            clean = pair.get("clean", "")
            corrupt = pair.get("corrupt", "")
            target = pair.get("target", "")

            if not clean or not corrupt:
                continue

            print(f"    {pair_id} ({family}): ", end="", flush=True)
            try:
                result = run_patching_experiment(
                    model, tokenizer, clean, corrupt, target,
                    component_type=comp_type,
                    layers_to_patch=layers_to_patch,
                )
                result["pair_id"] = pair_id
                result["family"] = family
                comp_results.append(result)

                # Print top recovery layers
                best_layer = max(
                    result["patching_results"].items(),
                    key=lambda x: x[1]["normalized_recovery"],
                    default=("none", {"normalized_recovery": 0})
                )
                print(f"best={best_layer[0]}(recovery={best_layer[1]['normalized_recovery']:.3f})")
            except Exception as e:
                print(f"ERROR: {e}")

        all_results[comp_type] = comp_results

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / "activation_patching.json"
    # Convert to serializable format
    serializable = {}
    for comp_type, results_list in all_results.items():
        serializable[comp_type] = results_list
    save_json(serializable, output_path)
    print(f"\n  Results saved to {output_path}")

    # Print summary
    print(f"\n  PATCHING SUMMARY:")
    for comp_type, results_list in all_results.items():
        if not results_list:
            continue
        print(f"\n  {comp_type}:")
        # Average recovery per layer across all pairs
        layer_recoveries = {}
        for result in results_list:
            for layer_key, layer_data in result.get("patching_results", {}).items():
                if layer_key not in layer_recoveries:
                    layer_recoveries[layer_key] = []
                layer_recoveries[layer_key].append(layer_data["normalized_recovery"])

        for layer_key in sorted(layer_recoveries.keys()):
            recs = layer_recoveries[layer_key]
            mean_rec = np.mean(recs)
            print(f"    {layer_key}: mean_recovery={mean_rec:.3f} (n={len(recs)})")

    register_experiment(
        type="patching",
        model=bundle.model_name,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[pairs_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"Activation patching: {len(pairs[:10])} pairs x {len(layers_to_patch)} layers x 3 component types",
        next="Steering vectors on top layers",
    )
    print("\n  Activation patching complete!")


if __name__ == "__main__":
    main()
