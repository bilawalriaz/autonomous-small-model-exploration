"""Compare layer ablation before/after LoRA training.

This is the core training perturbation experiment:
does fine-tuning shift which components are important?
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


def run_layer_ablation(model, tokenizer, suite, families, n_layers, ablation_type="zero"):
    """Run layer ablation and return effect matrix."""
    effect_matrix = np.zeros((n_layers, len(families)))

    for fam_idx, family in enumerate(families):
        examples = list(suite.filter_by_family(family))[:5]

        for layer_idx in range(n_layers):
            kl_effects = []
            for example in examples:
                inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(model.device)

                # Original
                with torch.no_grad():
                    orig_logits = model(input_ids).logits

                # Ablated
                # Get MLP module — path differs for PeftModel vs raw model
                from peft import PeftModel
                if isinstance(model, PeftModel):
                    mlp = model.base_model.model.model.layers[layer_idx].mlp
                else:
                    mlp = model.model.layers[layer_idx].mlp

                def zero_hook(module, input, output):
                    return torch.zeros_like(output)

                handle = mlp.register_forward_hook(zero_hook)
                with torch.no_grad():
                    abl_logits = model(input_ids).logits
                handle.remove()

                # KL divergence
                orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
                abl_log_probs = torch.log_softmax(abl_logits[0, -1], dim=-1)
                kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
                kl_effects.append(kl)

            effect_matrix[layer_idx, fam_idx] = np.mean(kl_effects) if kl_effects else 0.0

        print(f"    {family} done")

    return effect_matrix


def main():
    set_seed(42)

    print("=" * 60)
    print("LoRA BEFORE/AFTER COMPARISON: MLP ABLATION")
    print("=" * 60)

    # Load base model
    print("\n1. Loading BASE model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    base_model = bundle.model
    tokenizer = bundle.tokenizer
    base_model.eval()
    n_layers = bundle.architecture["n_layers"]

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    families = suite.families

    # Run base ablation
    print("\n2. Running BASE model MLP ablation...")
    base_matrix = run_layer_ablation(base_model, tokenizer, suite, families, n_layers)

    # Load LoRA adapter
    print("\n3. Loading LoRA adapter...")
    from peft import PeftModel
    adapter_path = str(PROJECT_ROOT / "experiments" / "adapters" / "lora_json_r8" / "adapter")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.eval()
    print(f"  Adapter loaded from {adapter_path}")

    # Run LoRA ablation
    print("\n4. Running LoRA-adapted model MLP ablation...")
    lora_matrix = run_layer_ablation(lora_model, tokenizer, suite, families, n_layers)

    # Compute diff
    diff_matrix = lora_matrix - base_matrix

    # Save results
    results = {
        "n_layers": n_layers,
        "families": families,
        "base_effect_matrix": base_matrix.tolist(),
        "lora_effect_matrix": lora_matrix.tolist(),
        "diff_matrix": diff_matrix.tolist(),
    }
    output_path = PROJECT_ROOT / "experiments" / "results" / "lora_ablation_comparison.json"
    save_json(results, output_path)
    print(f"\n5. Results saved to {output_path}")

    # Print comparison
    print("\n6. COMPARISON: Top changed layers per family")
    layer_names = [f"L{i:02d}" for i in range(n_layers)]
    for fam_idx, fam in enumerate(families):
        base_top3 = sorted(enumerate(base_matrix[:, fam_idx]), key=lambda x: x[1], reverse=True)[:3]
        lora_top3 = sorted(enumerate(lora_matrix[:, fam_idx]), key=lambda x: x[1], reverse=True)[:3]
        diff_sorted = sorted(enumerate(diff_matrix[:, fam_idx]), key=lambda x: abs(x[1]), reverse=True)[:3]

        base_str = ", ".join(f"L{i}({v:.2f})" for i, v in base_top3)
        lora_str = ", ".join(f"L{i}({v:.2f})" for i, v in lora_top3)
        diff_str = ", ".join(f"L{i}({v:+.2f})" for i, v in diff_sorted)

        print(f"\n  {fam}:")
        print(f"    Base top3: {base_str}")
        print(f"    LoRA top3: {lora_str}")
        print(f"    Biggest changes: {diff_str}")

    # Overall summary
    print("\n7. OVERALL SHIFTS")
    # Average absolute change per layer across families
    avg_abs_change = np.abs(diff_matrix).mean(axis=1)
    top_shifted = sorted(enumerate(avg_abs_change), key=lambda x: x[1], reverse=True)[:5]
    print("  Layers with largest average absolute change:")
    for idx, change in top_shifted:
        base_avg = base_matrix[idx].mean()
        lora_avg = lora_matrix[idx].mean()
        print(f"    L{idx:02d}: base_avg={base_avg:.2f}, lora_avg={lora_avg:.2f}, change={lora_avg-base_avg:+.2f}")

    # Check if JSON family specifically changed more
    json_idx = families.index("json_schema") if "json_schema" in families else None
    if json_idx is not None:
        json_diff = diff_matrix[:, json_idx]
        top_json_shift = sorted(enumerate(json_diff), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"\n  JSON schema specific shifts:")
        for idx, change in top_json_shift:
            print(f"    L{idx:02d}: {base_matrix[idx, json_idx]:.2f} → {lora_matrix[idx, json_idx]:.2f} ({change:+.2f})")

    # Generate plots
    print("\n8. Generating plots...")

    # Base heatmap
    plot_ablation_heatmap(
        base_matrix, layer_names, families,
        title="BASE Model MLP Ablation", name="lora_comparison_base",
    )

    # LoRA heatmap
    plot_ablation_heatmap(
        lora_matrix, layer_names, families,
        title="LoRA-adapted MLP Ablation", name="lora_comparison_lora",
    )

    # Diff heatmap
    plot_ablation_heatmap(
        diff_matrix, layer_names, families,
        title="LoRA vs BASE: MLP Ablation Difference", name="lora_comparison_diff",
    )

    print("  Plots saved.")

    register_experiment(
        type="comparison",
        model=bundle.model_name,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[adapter_path, suite_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"LoRA vs BASE comparison: max diff={np.abs(diff_matrix).max():.2f}",
        next="LoRA rank sweep, better activation patching pairs",
    )

    print("\n9. DONE")


if __name__ == "__main__":
    main()
