"""exp_000021: Adapter-only ablation — selectively remove LoRA adapter contribution at each layer.

Tests H6: "Adapter weights write to late layers but effects propagate upstream."

Method:
1. Load base model and trained model (base + LoRA JSON adapter)
2. For each prompt, cache base model layer outputs AND trained model layer outputs
3. Run trained model, but at layer X, replace the output with the BASE model's output
   (effectively removing the adapter's contribution at layer X only)
4. Measure KL divergence between ablated-trained and full-trained

If adapter ablation at L0-L2 has large effect despite small adapter norms there:
  -> effects propagate upstream (H6 supported)
If adapter ablation at L20-L23 has large effect matching norm distribution:
  -> norms are where the action is (H6 rejected)
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
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_layer_activations(model, input_ids, n_layers):
    """Cache residual stream output at each layer."""
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
    """Run model with a single layer's output replaced."""
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
    """KL(P_a || P_b) at last token."""
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()


def compute_adapter_norms(peft_model, n_layers):
    """Compute LoRA adapter weight norms per layer."""
    norms = {}
    state_dict = peft_model.state_dict()

    for i in range(n_layers):
        layer_norms = []
        for key in state_dict:
            if f"layers.{i}." in key and ("lora_A" in key or "lora_B" in key):
                w = state_dict[key]
                if w.dim() > 0:
                    layer_norms.append(w.norm().item())

        if layer_norms:
            # Combined norm: sqrt(sum of squared norms)
            combined = np.sqrt(sum(n**2 for n in layer_norms))
            norms[i] = {
                "combined_norm": combined,
                "individual_norms": layer_norms,
                "n_modules": len(layer_norms),
            }
        else:
            norms[i] = {"combined_norm": 0.0, "individual_norms": [], "n_modules": 0}

    return norms


def main():
    set_seed(42)

    # Load task suite for broader testing
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    # Also use task suite prompts for broader coverage
    suite_path = PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"
    with open(suite_path) as f:
        suite = json.load(f)

    # Select representative prompts from key families
    test_prompts = []
    families_seen = set()
    for ex in suite:
        fam = ex.get("family", "")
        if fam not in families_seen and fam in ["json_schema", "factual_recall", "copying", "delimiter_tracking", "code_syntax", "arithmetic"]:
            test_prompts.append({
                "prompt": ex["clean_prompt"],
                "family": fam,
            })
            families_seen.add(fam)
            if len(test_prompts) >= 8:
                break

    # Add clean/corrupt pair prompts too
    for pair in pairs[:6]:
        test_prompts.append({
            "prompt": pair["prefix"],
            "family": pair["family"],
        })

    # Deduplicate
    seen = set()
    unique_prompts = []
    for tp in test_prompts:
        if tp["prompt"] not in seen:
            seen.add(tp["prompt"])
            unique_prompts.append(tp)
    test_prompts = unique_prompts[:12]

    adapter_path = str(PROJECT_ROOT / "experiments" / "adapters" / "lora_json_r8" / "adapter")

    print("Loading base model...")
    base_bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    tokenizer = base_bundle.tokenizer
    n_layers = base_bundle.architecture["n_layers"]
    device = base_bundle.device

    print("Loading trained model (base + LoRA JSON)...")
    trained_model = PeftModel.from_pretrained(base_bundle.model, adapter_path)
    trained_model.eval()

    # Compute adapter weight norms per layer
    print("\nComputing adapter weight norms...")
    adapter_norms = compute_adapter_norms(trained_model, n_layers)
    for i in range(n_layers):
        norm = adapter_norms[i]["combined_norm"]
        n_mods = adapter_norms[i]["n_modules"]
        if norm > 0:
            print(f"  L{i:02d}: norm={norm:.4f} ({n_mods} modules)")

    results = []

    for tp in test_prompts:
        prompt = tp["prompt"]
        family = tp["family"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        print(f"\n  [{family}] {prompt[:50]}...")

        # Get base logits (adapter disabled)
        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits

        # Get trained logits (adapter enabled)
        trained_logits = trained_model(ids).logits

        # Baseline: KL between base and trained (total adapter effect)
        total_adapter_kl = compute_kl(trained_logits, base_logits)
        print(f"    Total adapter KL (base vs trained): {total_adapter_kl:.4f}")

        # Get base model layer activations (adapter disabled)
        with trained_model.disable_adapter():
            base_acts = get_layer_activations(trained_model, ids, n_layers)

        # For each layer, patch base activation into trained model (remove adapter at that layer)
        layer_results = []
        for layer_idx in range(n_layers):
            if layer_idx not in base_acts:
                continue

            donor = base_acts[layer_idx].to(device)
            ablated_logits = patch_layer_and_run(trained_model, ids, layer_idx, donor)

            # KL between ablated-trained and full-trained
            ablation_kl = compute_kl(trained_logits, ablated_logits)

            # How much of the total adapter effect is lost?
            if total_adapter_kl > 1e-8:
                effect_fraction = ablation_kl / total_adapter_kl
            else:
                effect_fraction = 0.0

            layer_results.append({
                "layer": layer_idx,
                "ablation_kl": round(ablation_kl, 6),
                "effect_fraction": round(effect_fraction, 6),
                "adapter_norm": round(adapter_norms[layer_idx]["combined_norm"], 6),
            })

            if ablation_kl > 0.5:
                print(f"    L{layer_idx:02d}: KL={ablation_kl:.4f} fraction={effect_fraction:.3f} norm={adapter_norms[layer_idx]['combined_norm']:.4f}")

        results.append({
            "prompt": prompt,
            "family": family,
            "total_adapter_kl": round(total_adapter_kl, 6),
            "layer_results": layer_results,
        })

    # Compute summary: mean ablation effect per layer
    layer_ablation_kls = {i: [] for i in range(n_layers)}
    layer_effect_fracs = {i: [] for i in range(n_layers)}

    for r in results:
        for lr in r["layer_results"]:
            layer_ablation_kls[lr["layer"]].append(lr["ablation_kl"])
            layer_effect_fracs[lr["layer"]].append(lr["effect_fraction"])

    summary = []
    for i in range(n_layers):
        kls = layer_ablation_kls.get(i, [])
        fracs = layer_effect_fracs.get(i, [])
        summary.append({
            "layer": i,
            "mean_ablation_kl": round(np.mean(kls), 6) if kls else 0.0,
            "std_ablation_kl": round(np.std(kls), 6) if kls else 0.0,
            "mean_effect_fraction": round(np.mean(fracs), 6) if fracs else 0.0,
            "adapter_norm": round(adapter_norms[i]["combined_norm"], 6),
            "n_prompts": len(kls),
        })

    # Correlation between adapter norm and ablation effect
    norms = [s["adapter_norm"] for s in summary]
    effects = [s["mean_ablation_kl"] for s in summary]
    if np.std(norms) > 0 and np.std(effects) > 0:
        correlation = np.corrcoef(norms, effects)[0, 1]
    else:
        correlation = 0.0

    print(f"\n  Correlation (adapter norm vs ablation effect): {correlation:.4f}")

    # Find layers where norm is low but effect is high (upstream propagation)
    norm_effect_mismatch = []
    for s in summary:
        if s["adapter_norm"] < np.median(norms) and s["mean_ablation_kl"] > np.median(effects):
            norm_effect_mismatch.append(s)
    print(f"  Norm-effect mismatch layers (low norm, high effect): {[s['layer'] for s in norm_effect_mismatch]}")

    # Top effect layers
    top_effect = sorted(summary, key=lambda x: x["mean_ablation_kl"], reverse=True)[:5]
    print(f"\n  Top 5 adapter ablation effect layers:")
    for te in top_effect:
        print(f"    L{te['layer']:02d}: KL={te['mean_ablation_kl']:.4f} norm={te['adapter_norm']:.4f}")

    output = {
        "experiment": "adapter_only_ablation",
        "n_prompts": len(results),
        "n_layers": n_layers,
        "adapter": "lora_json_r8",
        "adapter_norms": adapter_norms,
        "results": results,
        "summary": summary,
        "norm_effect_correlation": round(correlation, 6),
        "norm_effect_mismatch_layers": [s["layer"] for s in norm_effect_mismatch],
        "top_effect_layers": top_effect,
    }

    output_path = PROJECT_ROOT / "experiments" / "results" / "adapter_ablation.json"
    save_json(output, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="ablation",
        model="Qwen/Qwen2.5-0.5B",
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[adapter_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"Adapter-only ablation: {len(results)} prompts, {n_layers} layers, norm-effect analysis",
        key_metrics={
            "norm_effect_correlation": round(correlation, 4),
            "best_effect_layer": top_effect[0]["layer"] if top_effect else -1,
        },
        next="Generate comprehensive plots and publication report",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
