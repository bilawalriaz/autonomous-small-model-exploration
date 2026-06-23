#!/usr/bin/env python3
"""Phase 3: Steering controls — random-vector and shuffled-label baselines.

Tests whether steering effects are task-specific or just activation perturbation.
Compares:
1. Target steering vector (from clean/corrupt pairs)
2. Random same-norm vector
3. Shuffled-label vector (same pairs, wrong labels)
4. Unrelated-task vector (from a different task family)

If random vectors give similar effects, the steering finding is not task-specific.

Usage:
    python scripts/run_phase3_steering_controls.py --model Qwen/Qwen2.5-0.5B
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

from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.steering import compute_steering_vector, inject_steering_vector
from mi_atlas.metrics import kl_divergence_from_logits
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def compute_random_vector(hidden_dim, norm, seed=42):
    """Generate a random vector with specified norm."""
    rng = np.random.RandomState(seed)
    v = rng.randn(hidden_dim).astype(np.float32)
    v = v / np.linalg.norm(v) * norm
    return torch.from_numpy(v)


def compute_shuffled_label_vector(backend, pairs, layer_idx, seed=42):
    """Compute steering vector with shuffled clean/corrupt labels."""
    rng = np.random.RandomState(seed)
    # Shuffle the pairing: pair clean[i] with corrupt[j]
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    activations_pos = []
    activations_neg = []

    for i, pair in enumerate(pairs):
        j = indices[i]
        # Use clean from pair i, corrupt from pair j (shuffled)
        pos_out = backend.get_activations(pair["clean"], layers=[layer_idx])
        neg_out = backend.get_activations(pairs[j]["corrupt"], layers=[layer_idx])
        activations_pos.append(pos_out[layer_idx].mean(dim=1))  # mean over tokens
        activations_neg.append(neg_out[layer_idx].mean(dim=1))

    if activations_pos:
        mean_pos = torch.stack(activations_pos).mean(dim=0)
        mean_neg = torch.stack(activations_neg).mean(dim=0)
        sv = mean_pos - mean_neg
        return sv
    return None


def run_steering_with_vector(backend, prompts, sv, layer_idx, strengths):
    """Run steering with a given vector and return KL at each strength."""
    results = []
    for strength in strengths:
        kl_values = []
        for prompt_info in prompts:
            prompt = prompt_info["prompt"] if isinstance(prompt_info, dict) else prompt_info
            try:
                output = inject_steering_vector(backend, prompt, sv, layer_idx, strength)
                kl = kl_divergence_from_logits(output["baseline_logits"], output["steered_logits"])
                kl_values.append(kl)
            except Exception as e:
                pass
        if kl_values:
            results.append({
                "strength": strength,
                "mean_kl": float(np.mean(kl_values)),
                "std_kl": float(np.std(kl_values)),
                "n": len(kl_values),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 steering controls")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    hub_layers = {
        "Qwen/Qwen2.5-0.5B": [2, 8, 12, 19],
        "Qwen/Qwen2.5-1.5B": [6, 21, 26],
    }.get(args.model, [2, 12, 22])

    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    print(f"Phase 3: Steering controls")
    print(f"Model: {args.model}")
    print(f"Hub layers: {hub_layers}")
    print(f"Controls: target, random_same_norm, shuffled_label, unrelated_task")

    # Load model and pairs
    bundle = load_model(args.model)
    backend = create_backend(bundle)

    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    # Separate pairs by family
    factual_pairs = [p for p in pairs if p.get("family") == "factual_recall"]
    json_pairs = [p for p in pairs if p.get("family") == "json_schema"]
    other_pairs = [p for p in pairs if p.get("family") not in ("factual_recall", "json_schema")]

    eval_prompts = [p["clean"] for p in factual_pairs[:5]]  # Evaluate on factual prompts

    hidden_dim = bundle.model.config.hidden_size

    all_results = {}

    for layer_idx in hub_layers:
        print(f"\n  Layer {layer_idx}...")

        # 1. Target steering vector (from factual pairs)
        sv_target = compute_steering_vector(backend, factual_pairs[0] if factual_pairs else pairs[0], layer_idx)
        target_norm = float(torch.norm(sv_target).item())

        target_results = run_steering_with_vector(backend, eval_prompts, sv_target, layer_idx, strengths)

        # 2. Random same-norm vector
        sv_random = compute_random_vector(hidden_dim, target_norm, seed=args.seed)
        random_results = run_steering_with_vector(backend, eval_prompts, sv_random, layer_idx, strengths)

        # 3. Shuffled-label vector
        sv_shuffled = compute_shuffled_label_vector(backend, factual_pairs or pairs, layer_idx, seed=args.seed)
        shuffled_results = []
        if sv_shuffled is not None:
            shuffled_results = run_steering_with_vector(backend, eval_prompts, sv_shuffled, layer_idx, strengths)

        # 4. Unrelated-task vector (from JSON pairs, applied to factual prompts)
        unrelated_results = []
        if json_pairs:
            sv_unrelated = compute_steering_vector(backend, json_pairs[0], layer_idx)
            unrelated_results = run_steering_with_vector(backend, eval_prompts, sv_unrelated, layer_idx, strengths)

        layer_result = {
            "layer": layer_idx,
            "target_norm": target_norm,
            "target": target_results,
            "random_same_norm": random_results,
            "shuffled_label": shuffled_results,
            "unrelated_task": unrelated_results,
        }
        all_results[str(layer_idx)] = layer_result

        # Print summary
        for control_name, control_results in [
            ("target", target_results), ("random", random_results),
            ("shuffled", shuffled_results), ("unrelated", unrelated_results)
        ]:
            if control_results:
                best = max(control_results, key=lambda x: x["mean_kl"])
                print(f"    {control_name:10s}: best_kl={best['mean_kl']:.3f} at s={best['strength']}")

    # Compute selectivity ratios
    print(f"\n  SELECTIVITY ANALYSIS")
    selectivity = {}
    for layer_key, layer_data in all_results.items():
        target_best = max(layer_data["target"], key=lambda x: x["mean_kl"])["mean_kl"] if layer_data["target"] else 0
        random_best = max(layer_data["random_same_norm"], key=lambda x: x["mean_kl"])["mean_kl"] if layer_data["random_same_norm"] else 0

        selectivity_ratio = target_best / max(random_best, 1e-10)
        selectivity[layer_key] = {
            "target_best_kl": target_best,
            "random_best_kl": random_best,
            "selectivity_ratio": selectivity_ratio,
            "task_specific": selectivity_ratio > 2.0,
        }
        print(f"    L{layer_key}: target={target_best:.3f}, random={random_best:.3f}, "
              f"ratio={selectivity_ratio:.1f}x, task_specific={'YES' if selectivity_ratio > 2.0 else 'NO'}")

    # Save
    summary = {
        "experiment": "steering_controls",
        "model": args.model,
        "seed": args.seed,
        "hub_layers": hub_layers,
        "layers": all_results,
        "selectivity": selectivity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_controls_{model_slug}.json"
    save_json(summary, out_path)
    print(f"\n  Results: {out_path}")

    register_experiment(
        type="control",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(pairs_path)],
        outputs=[str(out_path)],
        status="success",
        summary=f"Steering controls: {len([v for v in selectivity.values() if v['task_specific']])} of {len(selectivity)} layers are task-specific",
        next="If selectivity > 2x, steering is task-specific. If < 2x, it's just perturbation.",
    )


if __name__ == "__main__":
    main()
