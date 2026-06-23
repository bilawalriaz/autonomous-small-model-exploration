#!/usr/bin/env python3
"""Phase 3: Multi-seed replication experiments.

Runs layer ablation or steering with multiple seeds to establish variance.
This is the single highest-priority Phase 3 experiment because it affects
confidence in ALL other claims.

Usage:
    python scripts/run_phase3_multiseed_replication.py --model Qwen/Qwen2.5-0.5B --experiment layer_ablation
    python scripts/run_phase3_multiseed_replication.py --model Qwen/Qwen2.5-1.5B --experiment steering
    python scripts/run_phase3_multiseed_replication.py --model Qwen/Qwen2.5-0.5B --experiment layer_ablation --seeds 42,137,256
"""

import argparse
import json
import os
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
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.steering import compute_steering_vector, inject_steering_vector
from mi_atlas.metrics import kl_divergence_from_logits
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT

DEFAULT_SEEDS = [42, 137, 256]

# Hub layers per model (from Phase 1-2 atlas)
MODEL_HUB_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [2, 8, 12, 19, 21, 22, 23],
    "Qwen/Qwen2.5-1.5B": [2, 6, 14, 21, 25, 26, 27],
    "Qwen/Qwen2.5-3B": [2, 13, 18, 26, 33, 34, 35],
    "HuggingFaceTB/SmolLM2-1.7B": [0, 6, 12, 18, 23],
}


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_layer_ablation_replication(model_name, seeds, suite_path=None, force=False):
    """Run layer ablation with multiple seeds and compute variance."""
    results = []

    for seed in seeds:
        set_seed(seed)
        run_id = f"P3_REPL_ablation_{model_name.split('/')[-1]}_seed{seed}"

        print(f"\n{'='*50}")
        print(f"Layer ablation: seed={seed}, model={model_name}")
        print(f"{'='*50}")

        bundle = load_model(model_name)
        backend = create_backend(bundle)

        suite = TaskSuite.load(suite_path or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"))

        start = time.time()
        ablation_result = run_layer_ablation_suite(backend, suite, ablation_type="zero", split="test")
        elapsed = time.time() - start

        seed_result = {
            "run_id": run_id,
            "seed": seed,
            "model": model_name,
            "n_layers": ablation_result["n_layers"],
            "families": ablation_result["families"],
            "effect_matrix": ablation_result["effect_matrix"],
            "mean_per_layer": ablation_result.get("mean_per_layer", []),
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(seed_result)

        # Save individual seed result
        out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_ablation_{model_name.split('/')[-1]}_seed{seed}.json"
        save_json(seed_result, out_path)
        print(f"  Saved: {out_path}")

        # Free VRAM
        del backend, bundle
        torch.cuda.empty_cache()

    # Compute cross-seed statistics
    if len(results) >= 2:
        effect_matrices = [np.array(r["effect_matrix"]) for r in results]
        stacked = np.stack(effect_matrices)  # (n_seeds, n_layers, n_families)

        mean_across_seeds = stacked.mean(axis=0).tolist()
        std_across_seeds = stacked.std(axis=0).tolist()
        max_cv_per_layer = []  # coefficient of variation per layer

        for layer_idx in range(stacked.shape[1]):
            layer_effects = stacked[:, layer_idx, :]  # (n_seeds, n_families)
            layer_mean = layer_effects.mean()
            layer_std = layer_effects.std()
            cv = (layer_std / layer_mean) if layer_mean > 0 else 0
            max_cv_per_layer.append(float(cv))

        # Find hub per seed
        hubs_per_seed = []
        for r in results:
            mean_per = r.get("mean_per_layer", [])
            if mean_per:
                hubs_per_seed.append(mean_per.index(max(mean_per)))
            else:
                # Compute from effect_matrix
                em = np.array(r["effect_matrix"])
                hubs_per_seed.append(int(em.mean(axis=1).argmax()))

        hub_std = np.std(hubs_per_seed) if len(hubs_per_seed) > 1 else 0

        summary = {
            "experiment": "multi_seed_layer_ablation",
            "model": model_name,
            "seeds": seeds,
            "n_seeds": len(results),
            "hub_per_seed": hubs_per_seed,
            "hub_mean": float(np.mean(hubs_per_seed)),
            "hub_std": float(hub_std),
            "max_cv_per_layer": max_cv_per_layer,
            "mean_cv": float(np.mean(max_cv_per_layer)),
            "mean_across_seeds": mean_across_seeds,
            "std_across_seeds": std_across_seeds,
            "verdict": "robust" if hub_std <= 1 else "fragile",
        }

        # Save summary
        summary_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_ablation_replication_{model_name.split('/')[-1]}.json"
        save_json(summary, summary_path)
        print(f"\n  Summary saved: {summary_path}")
        print(f"  Hub per seed: {hubs_per_seed} (std={hub_std:.1f})")
        print(f"  Mean CV: {summary['mean_cv']:.3f}")
        print(f"  Verdict: {summary['verdict']}")

        # Register
        register_experiment(
            type="replication",
            model=model_name,
            backend="hf",
            config="config/experiment_plan.yaml",
            inputs=[suite_path or "data/eval_sets/task_suite_v0.json"],
            outputs=[str(summary_path)],
            status="success",
            summary=f"Multi-seed ablation: hub_std={hub_std:.1f}, mean_cv={summary['mean_cv']:.3f}, verdict={summary['verdict']}",
            next="Run steering replication" if summary["verdict"] == "robust" else "Investigate hub variance",
        )

        return summary

    return {"experiment": "multi_seed_layer_ablation", "seeds": seeds, "n_seeds": len(results)}


def run_steering_replication(model_name, seeds, suite_path=None, force=False):
    """Run steering at hub layers with multiple seeds."""
    hub_layers = MODEL_HUB_LAYERS.get(model_name, [2, 12, 22])
    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
    results = []

    for seed in seeds:
        set_seed(seed)
        run_id = f"P3_REPL_steering_{model_name.split('/')[-1]}_seed{seed}"

        print(f"\n{'='*50}")
        print(f"Steering replication: seed={seed}, model={model_name}")
        print(f"Hub layers: {hub_layers}")
        print(f"{'='*50}")

        bundle = load_model(model_name)
        backend = create_backend(bundle)

        suite = TaskSuite.load(suite_path or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"))

        # Compute steering vectors from clean/corrupt pairs
        pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
        if pairs_path.exists():
            with open(pairs_path) as f:
                pairs = json.load(f)
        else:
            pairs = []

        start = time.time()
        seed_results = {"seed": seed, "layers": {}}

        for layer_idx in hub_layers:
            layer_results = {"layer": layer_idx, "strengths": {}}

            for strength in strengths:
                # Run steering on factual recall prompts
                kl_values = []
                for pair in pairs[:5]:  # Limit to 5 pairs for speed
                    try:
                        sv = compute_steering_vector(backend, pair, layer_idx)
                        output = inject_steering_vector(backend, pair["clean"], sv, layer_idx, strength)
                        kl = kl_divergence_from_logits(output["baseline_logits"], output["steered_logits"])
                        kl_values.append(kl)
                    except Exception as e:
                        print(f"  Warning: steering failed at L{layer_idx} s={strength}: {e}")

                if kl_values:
                    layer_results["strengths"][str(strength)] = {
                        "mean_kl": float(np.mean(kl_values)),
                        "std_kl": float(np.std(kl_values)),
                        "n": len(kl_values),
                    }

            seed_results["layers"][str(layer_idx)] = layer_results

        elapsed = time.time() - start
        seed_results["elapsed_seconds"] = round(elapsed, 1)
        results.append(seed_results)

        # Save
        out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_{model_name.split('/')[-1]}_seed{seed}.json"
        save_json(seed_results, out_path)
        print(f"  Saved: {out_path}")

        del backend, bundle
        torch.cuda.empty_cache()

    # Compute cross-seed steering variance
    if len(results) >= 2:
        # For each layer and strength, compute mean and std across seeds
        summary = {"experiment": "multi_seed_steering", "model": model_name, "seeds": seeds, "layers": {}}

        for layer_idx in hub_layers:
            layer_key = str(layer_idx)
            layer_stats = {"layer": layer_idx, "strengths": {}}

            for strength in strengths:
                s_key = str(strength)
                kl_values = []
                for r in results:
                    if layer_key in r["layers"] and s_key in r["layers"][layer_key]["strengths"]:
                        kl_values.append(r["layers"][layer_key]["strengths"][s_key]["mean_kl"])

                if kl_values:
                    layer_stats["strengths"][s_key] = {
                        "mean_across_seeds": float(np.mean(kl_values)),
                        "std_across_seeds": float(np.std(kl_values)),
                        "cv": float(np.std(kl_values) / np.mean(kl_values)) if np.mean(kl_values) > 0 else 0,
                    }

            summary["layers"][layer_key] = layer_stats

        summary_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_replication_{model_name.split('/')[-1]}.json"
        save_json(summary, summary_path)
        print(f"\n  Summary: {summary_path}")

        return summary

    return {"experiment": "multi_seed_steering", "seeds": seeds}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 multi-seed replication")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--experiment", type=str, choices=["layer_ablation", "steering"], required=True)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else DEFAULT_SEEDS

    print(f"Phase 3 multi-seed replication")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.experiment}")
    print(f"Seeds: {seeds}")

    if args.experiment == "layer_ablation":
        result = run_layer_ablation_replication(args.model, seeds, args.suite, args.force)
    elif args.experiment == "steering":
        result = run_steering_replication(args.model, seeds, args.suite, args.force)

    print(f"\nDone. Result: {json.dumps(result, indent=2)[:500]}")


if __name__ == "__main__":
    main()
