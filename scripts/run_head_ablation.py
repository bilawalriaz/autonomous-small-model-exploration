"""Head-level ablation using PyTorch hooks on Qwen2.5-0.5B.

Qwen2.5-0.5B architecture:
- 24 layers
- 14 query heads, 2 KV heads (GQA), head_dim=64
- Each attention block has q_proj, k_proj, v_proj, o_proj

Strategy: hook the attention output (o_proj input) and zero individual heads.
"""
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def get_attention_module(model, layer_idx):
    """Get the attention module for a given layer."""
    return model.model.layers[layer_idx].self_attn


def ablate_head_forward_hook(head_idx, n_heads=14, head_dim=64):
    """Create a forward hook that zeros out a specific attention head's output."""
    def hook_fn(module, input, output):
        # output is (attn_output, attn_weights, past_key_value)
        # attn_output shape: (batch, seq_len, hidden_size)
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        # Reshape to (batch, seq_len, n_heads, head_dim)
        batch, seq_len, hidden = attn_output.shape
        attn_output = attn_output.view(batch, seq_len, n_heads, head_dim)

        # Zero out the target head
        attn_output[:, :, head_idx, :] = 0.0

        # Reshape back
        attn_output = attn_output.view(batch, seq_len, hidden)

        if isinstance(output, tuple):
            return (attn_output,) + output[1:]
        return attn_output
    return hook_fn


def run_with_head_ablation(model, input_ids, layer_idx, head_idx, n_heads=14, head_dim=64):
    """Run model with one head zeroed out."""
    attn_module = get_attention_module(model, layer_idx)
    hook = attn_module.register_forward_hook(
        ablate_head_forward_hook(head_idx, n_heads, head_dim)
    )
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    hook.remove()
    return logits


def compute_kl_divergence(original_logits, ablated_logits, position=-1):
    """KL divergence between original and ablated next-token distributions."""
    orig_probs = torch.softmax(original_logits[0, position], dim=-1)
    abl_log_probs = torch.log_softmax(ablated_logits[0, position], dim=-1)
    return torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()


def main():
    set_seed(42)

    print("Loading model on aero GPU...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    n_layers = bundle.architecture["n_layers"]  # 24
    n_heads = bundle.architecture["n_heads"]    # 14
    n_kv_heads = bundle.architecture["n_kv_heads"]  # 2
    head_dim = bundle.architecture["d_head"]    # 64

    print(f"  Layers: {n_layers}, Q Heads: {n_heads}, KV Heads: {n_kv_heads}, Head dim: {head_dim}")

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    families = suite.families

    # Focus on the critical layers identified from layer ablation
    # Layer 2 (dominant), 7, 8, 21, 22 (secondary)
    critical_layers = [2, 7, 8, 21, 22]
    # Also test a low-importance layer as control
    control_layers = [13]

    layers_to_test = sorted(set(critical_layers + control_layers))

    print(f"  Testing layers: {layers_to_test}")
    print(f"  Families: {families}")

    # Results matrix: (layers * heads) x families
    all_row_labels = []
    for layer_idx in layers_to_test:
        for head_idx in range(n_heads):
            all_row_labels.append(f"L{layer_idx}H{head_idx}")

    effect_matrix = np.zeros((len(layers_to_test) * n_heads, len(families)))

    for fam_idx, family in enumerate(families):
        family_examples = list(suite.filter_by_family(family))
        # Use up to 5 examples per family for speed
        examples = family_examples[:5]

        for layer_pos, layer_idx in enumerate(layers_to_test):
            for head_idx in range(n_heads):
                kl_effects = []

                for example in examples:
                    inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=512)
                    input_ids = inputs["input_ids"].to(model.device)

                    # Original
                    with torch.no_grad():
                        orig_logits = model(input_ids).logits

                    # Ablated
                    abl_logits = run_with_head_ablation(
                        model, input_ids, layer_idx, head_idx, n_heads, head_dim
                    )

                    kl = compute_kl_divergence(orig_logits, abl_logits)
                    kl_effects.append(kl)

                row_idx = layer_pos * n_heads + head_idx
                effect_matrix[row_idx, fam_idx] = np.mean(kl_effects) if kl_effects else 0.0

        print(f"  Family '{family}' done.")

    # Save results
    results = {
        "layers_tested": layers_to_test,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "families": families,
        "row_labels": all_row_labels,
        "effect_matrix": effect_matrix.tolist(),
    }
    output_path = PROJECT_ROOT / "experiments" / "results" / "head_ablation.json"
    save_json(results, output_path)
    print(f"  Results saved to {output_path}")

    # Print top heads per family
    print(f"\n  TOP 5 HEADS PER FAMILY:")
    for fam_idx, fam in enumerate(families):
        head_effects = effect_matrix[:, fam_idx]
        top5 = sorted(enumerate(head_effects), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(f"{all_row_labels[i]}({v:.3f})" for i, v in top5)
        print(f"    {fam}: {top5_str}")

    # Generate heatmap
    plot_path = plot_ablation_heatmap(
        effect_matrix,
        row_labels=all_row_labels,
        col_labels=families,
        title="Head Ablation Heatmap (Critical Layers)",
        name="head_ablation_heatmap",
    )
    print(f"  Heatmap saved to {plot_path}")

    # Register experiment
    register_experiment(
        type="ablation",
        model=bundle.model_name,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(output_path), str(plot_path)],
        status="success",
        summary=f"Head ablation: {len(layers_to_test)} layers x {n_heads} heads, max KL={effect_matrix.max():.3f}",
        next="Activation patching on top heads, MLP ablation",
    )

    print("\n  Head ablation complete!")


if __name__ == "__main__":
    main()
