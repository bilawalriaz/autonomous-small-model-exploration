"""Activation patching v2: KL divergence at last position (more robust)."""
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

    pairs = load_json(str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"))
    print(f"  {len(pairs)} aligned pairs")
    families = sorted(set(p["family"] for p in pairs))

    effect_matrix = np.zeros((n_layers, len(families)))
    counts = np.zeros((n_layers, len(families)))

    for pair_idx, pair in enumerate(pairs):
        fam = pair["family"]
        fam_idx = families.index(fam)
        clean_ids = tokenizer(pair["clean"], return_tensors="pt")["input_ids"].to(model.device)
        corrupt_ids = tokenizer(pair["corrupt"], return_tensors="pt")["input_ids"].to(model.device)

        # Clean distribution at last position
        with torch.no_grad():
            clean_logits = model(clean_ids).logits
        clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)

        # Get clean activations for each layer
        clean_acts = {}
        def capture(layer_idx):
            def hook(module, input, output):
                clean_acts[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach().clone()
            return hook

        handles = []
        for li in range(n_layers):
            handles.append(model.model.layers[li].register_forward_hook(capture(li)))
        with torch.no_grad():
            _ = model(clean_ids)
        for h in handles:
            h.remove()

        # Patch each layer
        for layer_idx in range(n_layers):
            if layer_idx not in clean_acts:
                continue
            rep = clean_acts[layer_idx]

            def patch_hook(module, input, output, r=rep):
                h = output[0] if isinstance(output, tuple) else output
                seq_len = min(r.shape[1], h.shape[1])
                h[:, :seq_len, :] = r[:, :seq_len, :].to(h.device)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            handle = model.model.layers[layer_idx].register_forward_hook(patch_hook)
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            handle.remove()

            patched_probs = torch.softmax(patched_logits[0, -1], dim=-1)
            kl = torch.nn.functional.kl_div(
                patched_probs.log(), clean_probs, reduction="sum"
            ).item()
            effect_matrix[layer_idx, fam_idx] += kl
            counts[layer_idx, fam_idx] += 1

        if (pair_idx + 1) % 5 == 0:
            print(f"  {pair_idx + 1}/{len(pairs)} pairs done")

    # Average
    with np.errstate(divide="ignore", invalid="ignore"):
        effect_matrix = np.where(counts > 0, effect_matrix / counts, 0.0)

    # Print
    print(f"\nPATCHING RESULTS (KL divergence at last pos, lower=more recovery)")
    print("=" * 60)
    for fam_idx, fam in enumerate(families):
        le = effect_matrix[:, fam_idx]
        best = sorted(enumerate(le), key=lambda x: x[1])[:3]
        best_str = ", ".join(f"L{i}(KL={v:.3f})" for i, v in best)
        print(f"  {fam}: best layers: {best_str}")

    print(f"\nOVERALL BEST LAYERS (mean KL across families, lower=better):")
    mean_kl = effect_matrix.mean(axis=1)
    best_overall = sorted(enumerate(mean_kl), key=lambda x: x[1])[:5]
    for idx, val in best_overall:
        print(f"  L{idx:02d}: mean_KL={val:.3f}")

    # Save
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
    save_json({
        "n_layers": n_layers, "families": families,
        "layer_names": layer_names,
        "effect_matrix": effect_matrix.tolist(),
        "counts": counts.tolist(),
        "n_pairs": len(pairs),
    }, str(PROJECT_ROOT / "experiments" / "results" / "patching_kl_v1.json"))

    plot_ablation_heatmap(
        effect_matrix, layer_names, families,
        title="Activation Patching: KL Divergence (lower=recovery)",
        name="patching_kl_heatmap_v1",
    )

    register_experiment(
        type="patching", model=bundle.model_name, backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json")],
        outputs=[str(PROJECT_ROOT / "experiments" / "results" / "patching_kl_v1.json")],
        status="success",
        summary=f"Patching KL: {len(pairs)} pairs, min KL={effect_matrix.min():.3f}",
        next="Path patching on top components",
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
