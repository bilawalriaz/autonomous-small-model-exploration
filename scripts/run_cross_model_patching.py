"""exp_000019: Cross-model activation patching (trained -> base).

Patch trained model (base+LoRA) activations into the base model at each layer.
Measures whether trained activations can transfer learned behaviour into the base model.

High recovery (low KL to trained) at layer X = trained activations at X carry the learned behavior.
Low recovery at layer X = that layer alone is insufficient to transfer the behavior.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT

from peft import PeftModel


def get_layers(model):
    """Get the transformer layers list, handling PeftModel wrapping."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        # PeftModel: model.model.model.layers
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Direct Qwen2ForCausalLM: model.model.layers
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_layer_activations(model, input_ids, n_layers):
    """Cache residual stream output at each layer using forward hooks."""
    activations = {}
    handles = []
    layers = get_layers(model)

    for i in range(n_layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[idx] = output[0].detach().clone()
                else:
                    activations[idx] = output.detach().clone()
            return hook_fn
        h = layers[i].register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        _ = model(input_ids)

    for h in handles:
        h.remove()

    return activations


def patch_layer_and_run(model, input_ids, layer_idx, donor_activation):
    """Run model with a single layer's output replaced by donor_activation."""
    layers = get_layers(model)

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            return (donor_activation,) + output[1:]
        return donor_activation

    handle = layers[layer_idx].register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids).logits
    handle.remove()
    return logits


def compute_kl(logits_a, logits_b):
    """KL(P_a || P_b) at last token position."""
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    kl = torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()
    return kl


def compute_target_logprob(logits, target_id, position=-1):
    """Log probability of a specific target token."""
    log_probs = torch.log_softmax(logits[0, position, :], dim=-1)
    return log_probs[target_id].item()


def main():
    set_seed(42)

    # Load pairs
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    adapter_path = str(PROJECT_ROOT / "experiments" / "adapters" / "lora_json_r8" / "adapter")

    print("Loading base model...")
    base_bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    tokenizer = base_bundle.tokenizer
    n_layers = base_bundle.architecture["n_layers"]
    device = base_bundle.device

    print("Loading trained model (base + LoRA JSON)...")
    trained_model = PeftModel.from_pretrained(base_bundle.model, adapter_path)
    trained_model.eval()

    results = []

    for pair in pairs:
        pair_id = pair["id"]
        family = pair["family"]
        prompt = pair["prefix"]
        target = pair["target"]

        # Tokenize
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if len(target_ids) == 0:
            continue
        target_id = target_ids[0]

        print(f"\n  [{pair_id}] ({family}) target={target}")

        # Get base logits (adapter disabled)
        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits

        # Get trained logits (adapter enabled)
        trained_logits = trained_model(ids).logits

        base_kl = compute_kl(trained_logits, base_logits)
        base_target_lp = compute_target_logprob(base_logits, target_id)
        trained_target_lp = compute_target_logprob(trained_logits, target_id)

        print(f"    Base KL to trained: {base_kl:.4f}")
        print(f"    Base target logprob: {base_target_lp:.4f}, Trained: {trained_target_lp:.4f}")

        # Get trained model's layer activations
        trained_acts = get_layer_activations(trained_model, ids, n_layers)

        # Patch each layer's trained activation into base model
        layer_results = []
        for layer_idx in range(n_layers):
            if layer_idx not in trained_acts:
                continue

            donor = trained_acts[layer_idx].to(device)
            with trained_model.disable_adapter():
                patched_logits = patch_layer_and_run(trained_model, ids, layer_idx, donor)

            patched_kl = compute_kl(trained_logits, patched_logits)
            patched_target_lp = compute_target_logprob(patched_logits, target_id)

            # Recovery: how much of the base->trained gap is closed?
            if base_kl > 1e-8:
                recovery = 1.0 - (patched_kl / base_kl)
            else:
                recovery = 0.0

            # Target logprob change
            lp_delta = patched_target_lp - base_target_lp
            trained_lp_delta = trained_target_lp - base_target_lp
            if abs(trained_lp_delta) > 1e-8:
                lp_recovery = lp_delta / trained_lp_delta
            else:
                lp_recovery = 0.0

            layer_results.append({
                "layer": layer_idx,
                "patched_kl_to_trained": round(patched_kl, 6),
                "kl_recovery": round(recovery, 6),
                "patched_target_logprob": round(patched_target_lp, 6),
                "lp_delta": round(lp_delta, 6),
                "lp_recovery": round(lp_recovery, 6),
            })

            if recovery > 0.5:
                print(f"    L{layer_idx:02d}: KL={patched_kl:.4f} recovery={recovery:.3f} lp_delta={lp_delta:.4f}")

        results.append({
            "pair_id": pair_id,
            "family": family,
            "target": target,
            "base_kl_to_trained": round(base_kl, 6),
            "base_target_logprob": round(base_target_lp, 6),
            "trained_target_logprob": round(trained_target_lp, 6),
            "layer_results": layer_results,
        })

    # Compute summary: mean recovery per layer across all pairs
    n_layers_actual = n_layers
    layer_recoveries = {i: [] for i in range(n_layers_actual)}
    layer_lp_recoveries = {i: [] for i in range(n_layers_actual)}

    for r in results:
        for lr in r["layer_results"]:
            layer_recoveries[lr["layer"]].append(lr["kl_recovery"])
            layer_lp_recoveries[lr["layer"]].append(lr["lp_recovery"])

    summary = []
    for i in range(n_layers_actual):
        recs = layer_recoveries.get(i, [])
        lp_recs = layer_lp_recoveries.get(i, [])
        summary.append({
            "layer": i,
            "mean_kl_recovery": round(np.mean(recs), 6) if recs else 0.0,
            "std_kl_recovery": round(np.std(recs), 6) if recs else 0.0,
            "mean_lp_recovery": round(np.mean(lp_recs), 6) if lp_recs else 0.0,
            "n_pairs": len(recs),
        })

    # Find best transfer layers
    best_layers = sorted(summary, key=lambda x: x["mean_kl_recovery"], reverse=True)[:5]
    print(f"\n  Top 5 transfer layers (mean KL recovery):")
    for bl in best_layers:
        print(f"    L{bl['layer']:02d}: recovery={bl['mean_kl_recovery']:.4f} +/- {bl['std_kl_recovery']:.4f}")

    output = {
        "experiment": "cross_model_patching_trained_to_base",
        "n_pairs": len(results),
        "n_layers": n_layers,
        "adapter": "lora_json_r8",
        "results": results,
        "summary": summary,
        "best_transfer_layers": best_layers,
    }

    output_path = PROJECT_ROOT / "experiments" / "results" / "cross_model_patching.json"
    save_json(output, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="patching",
        model="Qwen/Qwen2.5-0.5B",
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[str(pairs_path), adapter_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"Cross-model patching: {len(results)} pairs, {n_layers} layers, trained->base",
        key_metrics={
            "best_transfer_layer": best_layers[0]["layer"] if best_layers else -1,
            "best_recovery": best_layers[0]["mean_kl_recovery"] if best_layers else 0.0,
        },
        next="Skill knockout (negative steering on trained model)",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
