"""MLP output ablation on critical layers using PyTorch hooks.

Tests how much each layer's MLP contributes to task performance.
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
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def run_with_mlp_ablation(model, input_ids, layer_idx):
    """Run model with MLP output zeroed at a specific layer."""
    mlp = model.model.layers[layer_idx].mlp

    def zero_mlp_hook(module, input, output):
        return torch.zeros_like(output)

    hook = mlp.register_forward_hook(zero_mlp_hook)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    hook.remove()
    return logits


def compute_kl(original_logits, ablated_logits, position=-1):
    orig_probs = torch.softmax(original_logits[0, position], dim=-1)
    abl_log_probs = torch.log_softmax(ablated_logits[0, position], dim=-1)
    return torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()


def main():
    set_seed(42)

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    n_layers = bundle.architecture["n_layers"]

    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    families = suite.families

    # Test all layers
    layers_to_test = list(range(n_layers))

    print(f"  Testing MLP ablation for {n_layers} layers x {len(families)} families")

    effect_matrix = np.zeros((n_layers, len(families)))

    for fam_idx, family in enumerate(families):
        examples = list(suite.filter_by_family(family))[:5]

        for layer_idx in layers_to_test:
            kl_effects = []
            for example in examples:
                inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(model.device)

                with torch.no_grad():
                    orig_logits = model(input_ids).logits
                abl_logits = run_with_mlp_ablation(model, input_ids, layer_idx)

                kl = compute_kl(orig_logits, abl_logits)
                kl_effects.append(kl)

            effect_matrix[layer_idx, fam_idx] = np.mean(kl_effects) if kl_effects else 0.0

        print(f"  '{family}' done.")

    # Save
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
    results = {
        "n_layers": n_layers,
        "families": families,
        "layer_names": layer_names,
        "effect_matrix": effect_matrix.tolist(),
    }
    output_path = PROJECT_ROOT / "experiments" / "results" / "mlp_ablation.json"
    save_json(results, output_path)

    # Print top layers per family
    print(f"\n  TOP 3 MLP LAYERS PER FAMILY:")
    for fam_idx, fam in enumerate(families):
        layer_effects = effect_matrix[:, fam_idx]
        top3 = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"L{i}({v:.3f})" for i, v in top3)
        print(f"    {fam}: {top3_str}")

    # Plot
    plot_path = plot_ablation_heatmap(
        effect_matrix, layer_names, families,
        title="MLP Zero-Ablation Heatmap (KL Divergence)",
        name="mlp_ablation_heatmap",
    )
    print(f"  Heatmap: {plot_path}")

    register_experiment(
        type="ablation", model=bundle.model_name, backend="hf_hooks",
        config="config/experiment_plan.yaml", inputs=[suite_path],
        outputs=[str(output_path), str(plot_path)], status="success",
        summary=f"MLP ablation: {n_layers} layers, max KL={effect_matrix.max():.3f}",
        next="Activation patching",
    )
    print("  MLP ablation complete!")


if __name__ == "__main__":
    main()
