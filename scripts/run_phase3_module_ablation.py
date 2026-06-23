#!/usr/bin/env python3
"""Phase 3: Module-specific ablation at hub layers.

For each hub layer (from MODEL_HUB_LAYERS dict), ablate individually:
  q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
outputs. Measure KL per family. This reveals which MODULE within a layer
matters most. Compare to full-layer ablation.

For GQA models (e.g. Qwen2.5-0.5B: 14 Q heads, 2 KV heads), k_proj and
v_proj have fewer heads than q_proj/o_proj, so their ablation effect is
expected to be smaller per-head.

Usage:
    python scripts/run_phase3_module_ablation.py --model Qwen/Qwen2.5-0.5B
    python scripts/run_phase3_module_ablation.py --model Qwen/Qwen2.5-1.5B --force
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, now_iso, git_commit_hash, PROJECT_ROOT

# Hub layers per model (from Phase 1-2 atlas)
MODEL_HUB_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [2, 8, 12, 19, 21, 22, 23],
    "Qwen/Qwen2.5-1.5B": [2, 6, 14, 21, 25, 26, 27],
    "Qwen/Qwen2.5-3B": [2, 13, 18, 26, 33, 34, 35],
    "HuggingFaceTB/SmolLM2-1.7B": [0, 6, 12, 18, 23],
}

# Linear modules inside each transformer layer
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["up_proj", "down_proj", "gate_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES


def get_layer_module(model, layer_idx, module_name):
    """Get a specific sub-module within a transformer layer."""
    layer = model.model.layers[layer_idx]
    if module_name in ATTN_MODULES:
        return getattr(layer.self_attn, module_name)
    elif module_name in MLP_MODULES:
        return getattr(layer.mlp, module_name)
    raise ValueError(f"Unknown module: {module_name}")


def ablate_module_output(model, input_ids, layer_idx, module_name, ablation_type="zero"):
    """Run model with a single module's output zeroed/mean-ablated.

    Hooks the output of the specified linear projection within the layer.
    """
    module = get_layer_module(model, layer_idx, module_name)

    def zero_hook(mod, inp, out):
        return torch.zeros_like(out)

    def mean_hook(mod, inp, out):
        # Replace with mean across sequence dimension
        return out.mean(dim=1, keepdim=True).expand_as(out)

    hook_fn = zero_hook if ablation_type == "zero" else mean_hook
    handle = module.register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(input_ids)
        ablated_logits = outputs.logits

    handle.remove()
    return ablated_logits


def compute_kl(original_logits, ablated_logits, position=-1):
    """KL divergence between original and ablated next-token distributions."""
    orig_probs = torch.softmax(original_logits[0, position].float(), dim=-1)
    abl_log_probs = torch.log_softmax(ablated_logits[0, position].float(), dim=-1)
    return torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()


def run_full_layer_ablation(model, input_ids, layer_idx):
    """Ablate entire layer output (for comparison)."""
    layer = model.model.layers[layer_idx]

    def zero_hook(mod, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)

    handle = layer.register_forward_hook(zero_hook)
    with torch.no_grad():
        ablated_logits = model(input_ids).logits
    handle.remove()
    return ablated_logits


def main():
    parser = argparse.ArgumentParser(description="Phase 3 module-specific ablation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    hub_layers = MODEL_HUB_LAYERS.get(args.model, [2, 12, 22])

    print(f"Phase 3: Module-specific ablation")
    print(f"Model: {args.model}")
    print(f"Hub layers: {hub_layers}")
    print(f"Modules: {ALL_MODULES}")
    print(f"Seed: {args.seed}")

    set_seed(args.seed)

    # Load model
    print("\nLoading model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    n_kv_heads = bundle.architecture.get("n_kv_heads", bundle.architecture.get("n_heads", 14))
    n_heads = bundle.architecture.get("n_heads", 14)
    print(f"  Q heads: {n_heads}, KV heads: {n_kv_heads} (GQA ratio: {n_heads}/{n_kv_heads})")

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    families = suite.families

    # Results structure
    all_results = {}

    for layer_idx in hub_layers:
        print(f"\n  === Layer {layer_idx} ===")
        layer_results = {"layer": layer_idx, "modules": {}, "full_layer": {}}

        for fam_idx, family in enumerate(families):
            family_examples = list(suite.filter_by_family(family))[:5]
            if not family_examples:
                continue

            print(f"    Family: {family} ({len(family_examples)} examples)")

            # Ablate each module
            for module_name in ALL_MODULES:
                kl_effects = []
                for example in family_examples:
                    ids = tokenizer(example.clean_prompt, return_tensors="pt",
                                    truncation=True, max_length=512)["input_ids"].to(model.device)

                    with torch.no_grad():
                        orig_logits = model(ids).logits
                    abl_logits = ablate_module_output(model, ids, layer_idx, module_name, "zero")
                    kl = compute_kl(orig_logits, abl_logits)
                    kl_effects.append(kl)

                mean_kl = float(np.mean(kl_effects)) if kl_effects else 0.0

                if module_name not in layer_results["modules"]:
                    layer_results["modules"][module_name] = {}
                layer_results["modules"][module_name][family] = {
                    "mean_kl": round(mean_kl, 6),
                    "std_kl": round(float(np.std(kl_effects)), 6) if kl_effects else 0.0,
                    "n": len(kl_effects),
                }

            # Full-layer ablation for comparison
            kl_effects_full = []
            for example in family_examples:
                ids = tokenizer(example.clean_prompt, return_tensors="pt",
                                truncation=True, max_length=512)["input_ids"].to(model.device)
                with torch.no_grad():
                    orig_logits = model(ids).logits
                abl_logits = run_full_layer_ablation(model, ids, layer_idx)
                kl = compute_kl(orig_logits, abl_logits)
                kl_effects_full.append(kl)

            mean_kl_full = float(np.mean(kl_effects_full)) if kl_effects_full else 0.0
            layer_results["full_layer"][family] = {
                "mean_kl": round(mean_kl_full, 6),
                "std_kl": round(float(np.std(kl_effects_full)), 6) if kl_effects_full else 0.0,
                "n": len(kl_effects_full),
            }

        all_results[str(layer_idx)] = layer_results

        # Print summary for this layer
        print(f"\n    Module importance at L{layer_idx} (mean KL across families):")
        module_importance = {}
        for mod in ALL_MODULES:
            fam_kls = [v["mean_kl"] for v in layer_results["modules"].get(mod, {}).values()]
            mean_importance = float(np.mean(fam_kls)) if fam_kls else 0.0
            module_importance[mod] = mean_importance

        full_kls = list(layer_results["full_layer"].values())
        full_mean = float(np.mean([v["mean_kl"] for v in full_kls])) if full_kls else 1e-10

        sorted_mods = sorted(module_importance.items(), key=lambda x: x[1], reverse=True)
        for mod, imp in sorted_mods:
            fraction = imp / max(full_mean, 1e-10)
            print(f"      {mod:10s}: KL={imp:.4f} ({fraction:.1%} of full layer)")
        print(f"      {'FULL LAYER':10s}: KL={full_mean:.4f}")

    # Aggregate: which module matters most overall?
    print(f"\n{'='*60}")
    print("  AGGREGATE MODULE IMPORTANCE")
    print(f"{'='*60}")

    aggregate = {}
    for mod in ALL_MODULES:
        mod_kls = []
        for layer_key, layer_data in all_results.items():
            for family, fam_data in layer_data["modules"].get(mod, {}).items():
                mod_kls.append(fam_data["mean_kl"])
        aggregate[mod] = {
            "mean_kl": float(np.mean(mod_kls)) if mod_kls else 0.0,
            "std_kl": float(np.std(mod_kls)) if mod_kls else 0.0,
        }

    # Normalize by full-layer sum
    full_layer_kls = []
    for layer_key, layer_data in all_results.items():
        for family, fam_data in layer_data["full_layer"].items():
            full_layer_kls.append(fam_data["mean_kl"])
    total_full = float(np.sum(full_layer_kls)) if full_layer_kls else 1e-10

    print(f"  Module               Mean KL    % of Full Layer")
    print(f"  {'─'*50}")
    for mod, stats in sorted(aggregate.items(), key=lambda x: x[1]["mean_kl"], reverse=True):
        pct = stats["mean_kl"] / max(float(np.mean(full_layer_kls)), 1e-10) * 100
        print(f"  {mod:20s} {stats['mean_kl']:.4f}    {pct:.1f}%")

    # Save
    summary = {
        "experiment": "phase3_module_ablation",
        "model": args.model,
        "seed": args.seed,
        "hub_layers": hub_layers,
        "modules_tested": ALL_MODULES,
        "n_kv_heads": n_kv_heads,
        "n_q_heads": n_heads,
        "gqa_ratio": f"{n_heads}/{n_kv_heads}",
        "layers": all_results,
        "aggregate": aggregate,
        "timestamp": now_iso(),
        "git_commit": git_commit_hash(),
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_module_ablation_{model_slug}.json"
    save_json(summary, out_path)
    print(f"\n  Results: {out_path}")

    # Register
    best_module = max(aggregate.items(), key=lambda x: x[1]["mean_kl"])
    register_experiment(
        type="ablation",
        model=args.model,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(out_path)],
        status="success",
        summary=f"Module ablation at {len(hub_layers)} hub layers: "
                f"most important module={best_module[0]} (KL={best_module[1]['mean_kl']:.4f})",
        next="Compare module importance to LoRA target-module findings",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
