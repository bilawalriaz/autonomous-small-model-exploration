#!/usr/bin/env python3
"""Phase 3: Steering replication using HF native hooks (no TransformerLens).

Computes steering vectors from clean/corrupt pairs, injects them at hub layers,
and measures KL divergence at multiple strengths across multiple seeds.

Usage:
    python scripts/run_phase3_steering_hf.py --model Qwen/Qwen2.5-0.5B --seeds 42,137,256
    python scripts/run_phase3_steering_hf.py --model Qwen/Qwen2.5-1.5B --seeds 42,137,256
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

from mi_atlas.utils import save_json, PROJECT_ROOT, set_seed, now_iso, git_commit_hash

MODEL_HUB_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [2, 8, 12, 19, 21, 22, 23],
    "Qwen/Qwen2.5-1.5B": [6, 14, 21, 25, 26, 27],
    "Qwen/Qwen2.5-3B": [13, 18, 26, 33, 34, 35],
}


def load_model_hf(model_name):
    """Load model with HF Transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use 4-bit for 3B+ models
    n_params = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).parameters()) if "3B" in model_name or "3b" in model_name else 0

    if "3B" in model_name or "3b" in model_name:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float32
        )

    return model, tokenizer


def get_activations_at_layer(model, tokenizer, prompt, layer_idx):
    """Get activations at a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    activations = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    return activations.get("hidden", None)


def compute_steering_vector_hf(model, tokenizer, positive_prompts, negative_prompts, layer_idx):
    """Compute steering vector: mean(positive activations) - mean(negative activations)."""
    pos_acts = []
    neg_acts = []

    for prompt in positive_prompts:
        act = get_activations_at_layer(model, tokenizer, prompt, layer_idx)
        if act is not None:
            pos_acts.append(act.mean(dim=1).squeeze())  # mean over tokens

    for prompt in negative_prompts:
        act = get_activations_at_layer(model, tokenizer, prompt, layer_idx)
        if act is not None:
            neg_acts.append(act.mean(dim=1).squeeze())

    if pos_acts and neg_acts:
        sv = torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)
        return sv
    return None


def inject_and_measure(model, tokenizer, prompt, sv, layer_idx, strength):
    """Inject steering vector and measure KL divergence from baseline."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Baseline
    with torch.no_grad():
        baseline_logits = model(**inputs).logits[:, -1, :]

    # Steered
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            modified = output[0].clone()
            modified[:, -1, :] += sv.to(modified.device) * strength
            return (modified,) + output[1:]
        modified = output.clone()
        modified[:, -1, :] += sv.to(modified.device) * strength
        return modified

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        steered_logits = model(**inputs).logits[:, -1, :]
    hook.remove()

    # Compute KL
    baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
    steered_probs = torch.softmax(steered_logits.float(), dim=-1)
    kl = torch.nn.functional.kl_div(
        steered_probs.log(), baseline_probs, reduction="sum"
    ).item()

    # Target logit delta
    baseline_top = baseline_logits.argmax(dim=-1)
    target_delta = (steered_logits[0, baseline_top] - baseline_logits[0, baseline_top]).item()

    return abs(kl), target_delta


def main():
    parser = argparse.ArgumentParser(description="Phase 3 steering replication (HF)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seeds", type=str, default="42,137,256")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=5)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    model_slug = args.model.split("/")[-1]
    hub_layers = MODEL_HUB_LAYERS.get(args.model, [2, 12, 22])
    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    print(f"Phase 3: Steering replication (HF native)")
    print(f"Model: {args.model}")
    print(f"Hub layers: {hub_layers}")
    print(f"Seeds: {seeds}")

    # Load clean/corrupt pairs
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    if pairs_path.exists():
        with open(pairs_path) as f:
            pairs = json.load(f)
    else:
        # Generate simple factual pairs
        pairs = [
            {"clean": "The capital of France is", "corrupt": "The capital of Australia is", "family": "factual_recall"},
            {"clean": "The capital of Italy is", "corrupt": "The capital of Brazil is", "family": "factual_recall"},
            {"clean": "The largest planet is", "corrupt": "The smallest planet is", "family": "factual_recall"},
        ]

    eval_prompts = [
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Germany is",
        "2 + 2 =",
        "The sun rises in the",
    ]

    all_results = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*50}")
        print(f"Seed: {seed}")
        print(f"{'='*50}")

        model, tokenizer = load_model_hf(args.model)
        seed_results = {"seed": seed, "layers": {}}

        for layer_idx in hub_layers:
            layer_results = {"layer": layer_idx, "strengths": {}}

            # Compute steering vector from pairs
            positive_prompts = [p["clean"] for p in pairs[:args.n_pairs]]
            negative_prompts = [p["corrupt"] for p in pairs[:args.n_pairs]]

            sv = compute_steering_vector_hf(model, tokenizer, positive_prompts, negative_prompts, layer_idx)
            if sv is None:
                print(f"  L{layer_idx}: steering vector computation failed")
                continue

            sv_norm = float(torch.norm(sv).item())
            print(f"  L{layer_idx}: sv_norm={sv_norm:.3f}")

            for strength in strengths:
                kl_values = []
                delta_values = []
                for prompt in eval_prompts:
                    try:
                        kl, delta = inject_and_measure(model, tokenizer, prompt, sv, layer_idx, strength)
                        kl_values.append(kl)
                        delta_values.append(delta)
                    except Exception as e:
                        pass

                if kl_values:
                    layer_results["strengths"][str(strength)] = {
                        "mean_kl": float(np.mean(kl_values)),
                        "std_kl": float(np.std(kl_values)),
                        "mean_delta": float(np.mean(delta_values)),
                        "n": len(kl_values),
                    }

            seed_results["layers"][str(layer_idx)] = layer_results

        # Save per-seed
        out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_hf_{model_slug}_seed{seed}.json"
        save_json(seed_results, out_path)
        print(f"  Saved: {out_path}")

        all_results[f"seed_{seed}"] = seed_results

        del model, tokenizer
        torch.cuda.empty_cache()

    # Cross-seed summary
    summary = {
        "experiment": "steering_replication_hf",
        "model": args.model,
        "seeds": seeds,
        "hub_layers": hub_layers,
        "per_layer": {},
    }

    for layer_idx in hub_layers:
        layer_key = str(layer_idx)
        layer_stats = {"layer": layer_idx, "strengths": {}}

        for strength in strengths:
            s_key = str(strength)
            kl_values = []
            for seed_key, seed_data in all_results.items():
                if layer_key in seed_data["layers"]:
                    s_data = seed_data["layers"][layer_key]["strengths"].get(s_key)
                    if s_data:
                        kl_values.append(s_data["mean_kl"])

            if kl_values:
                layer_stats["strengths"][s_key] = {
                    "mean_across_seeds": float(np.mean(kl_values)),
                    "std_across_seeds": float(np.std(kl_values)),
                    "cv": float(np.std(kl_values) / np.mean(kl_values)) if np.mean(kl_values) > 0 else 0,
                }

        summary["per_layer"][layer_key] = layer_stats

    # Find best steering layer
    best_layer = None
    best_kl = 0
    for layer_key, layer_data in summary["per_layer"].items():
        for s_key, s_data in layer_data["strengths"].items():
            if s_data["mean_across_seeds"] > best_kl:
                best_kl = s_data["mean_across_seeds"]
                best_layer = int(layer_key)
                best_strength = float(s_key)

    summary["best_layer"] = best_layer
    summary["best_strength"] = best_strength
    summary["best_kl"] = best_kl
    summary["timestamp"] = now_iso()
    summary["git_commit"] = git_commit_hash()

    summary_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_hf_replication_{model_slug}.json"
    save_json(summary, summary_path)

    print(f"\n  Best steering: L{best_layer} s={best_strength} KL={best_kl:.3f}")
    print(f"  Summary: {summary_path}")

    # Register
    from mi_atlas.experiment_registry import register_experiment
    register_experiment(
        type="steering",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(pairs_path)],
        outputs=[str(summary_path)],
        status="success",
        summary=f"Steering replication: best L{best_layer} s={best_strength} KL={best_kl:.3f}",
        next="Test steering controls (random vectors)" if best_kl > 0.5 else "Investigate steering failure",
    )


if __name__ == "__main__":
    main()
