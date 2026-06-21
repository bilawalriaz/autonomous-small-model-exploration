"""Activation patching with aligned clean/corrupt pairs.

This is the core causal experiment — patch clean activations into
corrupt runs and measure recovery. STRONG evidence level.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, load_json


def main():
    set_seed(42)

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()
    n_layers = bundle.architecture["n_layers"]

    # Load aligned pairs
    pairs = load_json(str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"))
    print(f"  Loaded {len(pairs)} aligned pairs")

    families = sorted(set(p["family"] for p in pairs))

    # Patch at each layer (residual stream)
    layers_to_patch = list(range(n_layers))
    effect_matrix = np.zeros((n_layers, len(families)))
    counts = np.zeros((n_layers, len(families)))

    for pair in pairs:
        fam = pair["family"]
        fam_idx = families.index(fam)
        prefix_len = pair["prefix_len"]
        total_len = pair["total_len"]
        target = pair["target"]
        target_ids = tokenizer.encode(target, add_special_tokens=False)

        clean_ids = tokenizer(pair["clean"], return_tensors="pt")["input_ids"].to(model.device)
        corrupt_ids = tokenizer(pair["corrupt"], return_tensors="pt")["input_ids"].to(model.device)

        # Run clean and corrupt
        with torch.no_grad():
            clean_logits = model(clean_ids).logits
            corrupt_logits = model(corrupt_ids).logits

        # Target logprob at the divergence point (last prefix position)
        div_pos = prefix_len - 1  # position where we predict the next token

        clean_lp = torch.log_softmax(clean_logits[0, div_pos], dim=-1)
        corrupt_lp = torch.log_softmax(corrupt_logits[0, div_pos], dim=-1)

        clean_target_lp = sum(clean_lp[tid].item() for tid in target_ids)
        corrupt_target_lp = sum(corrupt_lp[tid].item() for tid in target_ids)

        # Skip if no meaningful difference
        if abs(clean_target_lp - corrupt_target_lp) < 0.01:
            continue

        # Patch each layer
        for layer_idx in layers_to_patch:
            # Get clean activation at this layer
            clean_act = {}
            def capture_hook(module, input, output, li=layer_idx):
                if isinstance(output, tuple):
                    clean_act[li] = output[0].detach().clone()
                else:
                    clean_act[li] = output.detach().clone()

            layer = model.model.layers[layer_idx]
            handle = layer.register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(clean_ids)
            handle.remove()

            if layer_idx not in clean_act:
                continue

            # Patch into corrupt run
            replacement = clean_act[layer_idx]
            def patch_hook(module, input, output, rep=replacement):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden[:, :rep.shape[1], :] = rep[:, :hidden.shape[1], :].to(hidden.device)
                    return (hidden,) + output[1:]
                else:
                    output[:, :rep.shape[1], :] = rep[:, :output.shape[1], :].to(output.device)
                    return output

            handle = layer.register_forward_hook(patch_hook)
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            handle.remove()

            patched_lp = torch.log_softmax(patched_logits[0, div_pos], dim=-1)
            patched_target_lp = sum(patched_lp[tid].item() for tid in target_ids)

            # Normalized recovery
            denom = max(1e-8, clean_target_lp - corrupt_target_lp)
            recovery = (patched_target_lp - corrupt_target_lp) / denom

            effect_matrix[layer_idx, fam_idx] += recovery
            counts[layer_idx, fam_idx] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        effect_matrix = np.where(counts > 0, effect_matrix / counts, 0.0)

    # Save
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
    results = {
        "n_layers": n_layers,
        "families": families,
        "layer_names": layer_names,
        "effect_matrix": effect_matrix.tolist(),
        "counts": counts.tolist(),
        "n_pairs": len(pairs),
    }
    output_path = PROJECT_ROOT / "experiments" / "results" / "activation_patching_v1.json"
    save_json(results, output_path)
    print(f"\nResults saved to {output_path}")

    # Print results
    print(f"\nPATCHING RESULTS (normalized recovery, {len(pairs)} pairs)")
    print("=" * 60)
    for fam_idx, fam in enumerate(families):
        layer_effects = effect_matrix[:, fam_idx]
        top5 = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(f"L{i}({v:.3f})" for i, v in top5)
        print(f"  {fam}: {top5_str}")

    # Overall most important layers
    print(f"\nOVERALL TOP LAYERS (mean recovery across families):")
    mean_recovery = effect_matrix.mean(axis=1)
    top_overall = sorted(enumerate(mean_recovery), key=lambda x: x[1], reverse=True)[:5]
    for idx, val in top_overall:
        print(f"  L{idx:02d}: mean_recovery={val:.3f}")

    # Plot
    plot_ablation_heatmap(
        effect_matrix, layer_names, families,
        title="Activation Patching: Normalized Recovery",
        name="activation_patching_heatmap_v1",
    )
    print("\nPlot saved.")

    register_experiment(
        type="patching", model=bundle.model_name, backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json")],
        outputs=[str(output_path)], status="success",
        summary=f"Activation patching: {len(pairs)} aligned pairs, max recovery={effect_matrix.max():.3f}",
        next="Path patching on top components",
    )
    print("Done!")


if __name__ == "__main__":
    main()
