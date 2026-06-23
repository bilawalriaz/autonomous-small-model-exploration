#!/usr/bin/env python3
"""Phase 3: Full ablation method comparison at ALL layers.

For each layer 0..n_layers-1: runs zero ablation, mean ablation, gaussian
resample ablation. Computes KL divergence per family.
Outputs: full rank-order correlation matrix (Spearman) between methods.
Tests whether zero=mean holds at all layers (not just hub layers).

Usage:
    python -u scripts/run_phase3_ablation_method_comparison.py --model Qwen/Qwen2.5-0.5B
    python -u scripts/run_phase3_ablation_method_comparison.py --model Qwen/Qwen2.5-0.5B --force --seed 137
"""

import argparse
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from scipy import stats

from mi_atlas.model_loader import load_model_hf
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.ablations import (
    run_layer_ablation_suite, zero_ablation_hook,
    mean_ablation_hook, resample_ablation_hook,
)
from mi_atlas.experiment_registry import register_experiment, load_registry
from mi_atlas.metrics import kl_divergence
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, now_iso


def log(msg):
    """Print with timestamp and flush."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_already_run(model_slug, force):
    """Check if this experiment already completed."""
    if force:
        return False
    registry = load_registry()
    for rec in registry:
        if (rec.get("type") == "phase3_ablation_methods"
                and model_slug in rec.get("model", "")
                and rec.get("status") == "success"):
            return True
    return False


def compute_mean_activations_for_suite(backend, suite, layer_idx, n_samples=10):
    """Compute mean activation at a layer across multiple examples."""
    layer_name = f"layer_{layer_idx:02d}"
    activations = []

    examples = list(suite)[:n_samples]
    for example in examples:
        try:
            inputs = backend.tokenize(example.clean_prompt)
            input_ids = inputs["input_ids"].to(backend.device)
            _, cache = backend.run_with_cache(input_ids)
            if layer_name in cache:
                # Take mean over sequence positions
                act = cache[layer_name][0]  # (seq_len, d_model)
                activations.append(act.mean(dim=0).detach().cpu())
        except Exception:
            continue

    if not activations:
        return None
    return torch.stack(activations).mean(dim=0)


def run_single_layer_ablation(backend, model, tokenizer, suite, layer_idx,
                               ablation_type, mean_act=None, n_examples=5):
    """Run ablation of a specific layer across families, return KL per family."""
    families = suite.families
    family_kl = {}

    for family in families:
        family_suite = suite.filter_by_family(family)
        examples = list(family_suite)[:n_examples]
        kl_effects = []

        for example in examples:
            try:
                inputs = backend.tokenize(example.clean_prompt)
                input_ids = inputs["input_ids"].to(backend.device)

                # Get original logits
                with torch.no_grad():
                    orig_out = model(input_ids)
                    orig_logits = orig_out.logits[0, -1, :]

                # Set up ablation hook
                layer_module = model.model.layers[layer_idx]

                def ablation_hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    if ablation_type == "zero":
                        modified = torch.zeros_like(hidden)
                    elif ablation_type == "mean" and mean_act is not None:
                        modified = mean_act.to(hidden.device).expand_as(hidden)
                    elif ablation_type == "resample":
                        # Gaussian noise with same statistics
                        modified = torch.randn_like(hidden) * hidden.std() + hidden.mean()
                    else:
                        modified = hidden

                    if isinstance(output, tuple):
                        return (modified,) + output[1:]
                    return modified

                handle = layer_module.register_forward_hook(ablation_hook_fn)
                with torch.no_grad():
                    abl_out = model(input_ids)
                    abl_logits = abl_out.logits[0, -1, :]
                handle.remove()

                # Compute KL divergence
                orig_probs = torch.softmax(orig_logits.float(), dim=-1)
                abl_log_probs = torch.log_softmax(abl_logits.float(), dim=-1)
                kl = torch.nn.functional.kl_div(
                    abl_log_probs, orig_probs, reduction="sum"
                ).item()
                kl_effects.append(kl)
            except Exception as e:
                log(f"    Error at L{layer_idx} {family}: {e}")
                continue

        family_kl[family] = float(np.mean(kl_effects)) if kl_effects else 0.0

    return family_kl


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Ablation method comparison at all layers"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true",
                       help="Re-run even if already completed")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    ablation_types = ["zero", "mean", "resample"]

    log("=" * 60)
    log(f"Phase 3: Ablation Method Comparison at ALL Layers")
    log(f"Model: {args.model}")
    log(f"Ablation types: {ablation_types}")
    log(f"Seed: {args.seed}")
    log("=" * 60)

    if check_already_run(model_slug, args.force):
        log("Already completed. Use --force to re-run.")
        return

    set_seed(args.seed)
    start_time = time.time()

    # Load model
    log("Loading model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    backend = create_backend(bundle)
    n_layers = bundle.architecture["n_layers"]
    families_list = []
    log(f"  Loaded: {n_layers} layers, device={bundle.device}")

    # Build suite
    suite = build_default_suite(seed=args.seed)
    families_list = suite.families
    log(f"  Families: {families_list}")

    # Pre-compute mean activations for mean ablation
    log("\nComputing mean activations across dataset...")
    mean_activations = {}
    for layer_idx in range(n_layers):
        mean_act = compute_mean_activations_for_suite(backend, suite, layer_idx)
        if mean_act is not None:
            mean_activations[layer_idx] = mean_act
        if (layer_idx + 1) % 5 == 0:
            log(f"  Computed mean activations for {layer_idx+1}/{n_layers} layers")

    # Run ablation for each method at each layer
    all_method_results = {}

    for abl_type in ablation_types:
        log(f"\n{'='*50}")
        log(f"Running {abl_type} ablation at all {n_layers} layers...")
        log(f"{'='*50}")

        # effect_matrix: (n_layers, n_families)
        effect_matrix = {}
        method_start = time.time()

        for layer_idx in range(n_layers):
            try:
                mean_act = mean_activations.get(layer_idx) if abl_type == "mean" else None
                family_kl = run_single_layer_ablation(
                    backend, model, tokenizer, suite, layer_idx,
                    abl_type, mean_act=mean_act, n_examples=5
                )
                effect_matrix[layer_idx] = family_kl

                # Progress
                max_kl = max(family_kl.values()) if family_kl else 0.0
                top_family = max(family_kl, key=family_kl.get) if family_kl else "none"
                if (layer_idx + 1) % 3 == 0 or layer_idx == 0 or layer_idx == n_layers - 1:
                    log(f"  L{layer_idx:02d}: max_KL={max_kl:.4f} ({top_family})")
            except Exception as e:
                log(f"  L{layer_idx:02d}: FAILED - {e}")
                effect_matrix[layer_idx] = {fam: 0.0 for fam in families_list}

        method_elapsed = time.time() - method_start

        # Convert to matrix form
        matrix = np.zeros((n_layers, len(families_list)))
        for layer_idx in range(n_layers):
            for fam_idx, fam in enumerate(families_list):
                matrix[layer_idx, fam_idx] = effect_matrix.get(layer_idx, {}).get(fam, 0.0)

        all_method_results[abl_type] = {
            "effect_matrix": matrix.tolist(),
            "per_layer": {str(k): v for k, v in effect_matrix.items()},
            "elapsed_seconds": round(method_elapsed, 1),
        }

        log(f"  {abl_type} completed in {method_elapsed:.0f}s")
        log(f"  Max effect: {matrix.max():.4f}, Mean effect: {matrix.mean():.4f}")

    # Compute Spearman rank-order correlations between methods
    log(f"\n{'='*60}")
    log("SPEARMAN RANK-ORDER CORRELATION BETWEEN METHODS")
    log(f"{'='*60}")

    correlations = {}
    method_names = list(all_method_results.keys())

    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if j <= i:
                continue
            mat1 = np.array(all_method_results[m1]["effect_matrix"])
            mat2 = np.array(all_method_results[m2]["effect_matrix"])

            # Flatten for overall correlation
            flat1 = mat1.flatten()
            flat2 = mat2.flatten()
            rho, p_val = stats.spearmanr(flat1, flat2)

            # Per-family correlations
            per_family_corr = {}
            for fam_idx, fam in enumerate(families_list):
                v1 = mat1[:, fam_idx]
                v2 = mat2[:, fam_idx]
                if np.std(v1) > 0 and np.std(v2) > 0:
                    r, p = stats.spearmanr(v1, v2)
                    per_family_corr[fam] = {"rho": float(r), "p_value": float(p)}
                else:
                    per_family_corr[fam] = {"rho": 0.0, "p_value": 1.0}

            # Per-layer correlations
            per_layer_corr = {}
            for layer_idx in range(n_layers):
                v1 = mat1[layer_idx, :]
                v2 = mat2[layer_idx, :]
                if np.std(v1) > 0 and np.std(v2) > 0:
                    r, p = stats.spearmanr(v1, v2)
                    per_layer_corr[f"L{layer_idx:02d}"] = {
                        "rho": float(r), "p_value": float(p)
                    }
                else:
                    per_layer_corr[f"L{layer_idx:02d}"] = {
                        "rho": 0.0, "p_value": 1.0
                    }

            pair_key = f"{m1}_vs_{m2}"
            correlations[pair_key] = {
                "overall_rho": float(rho),
                "overall_p_value": float(p_val),
                "per_family": per_family_corr,
                "per_layer": per_layer_corr,
            }

            log(f"\n  {m1} vs {m2}:")
            log(f"    Overall Spearman rho = {rho:.4f} (p = {p_val:.2e})")
            fam_strs = [f"{f}={c['rho']:.3f}" for f, c in per_family_corr.items()]
            log(f"    Per-family: {', '.join(fam_strs)}")

    # Test zero=mean hypothesis at all layers
    log(f"\n{'='*60}")
    log("ZERO=MEAN HYPOTHESIS TEST (ALL LAYERS)")
    log(f"{'='*60}")

    zero_mean_test = {}
    if "zero" in all_method_results and "mean" in all_method_results:
        zero_mat = np.array(all_method_results["zero"]["effect_matrix"])
        mean_mat = np.array(all_method_results["mean"]["effect_matrix"])

        for layer_idx in range(n_layers):
            zero_vals = zero_mat[layer_idx, :]
            mean_vals = mean_mat[layer_idx, :]
            diff = np.abs(zero_vals - mean_vals)
            max_diff = diff.max()
            mean_diff = diff.mean()

            # Paired t-test across families
            if len(zero_vals) > 2:
                t_stat, p_val = stats.ttest_rel(zero_vals, mean_vals)
            else:
                t_stat, p_val = 0.0, 1.0

            zero_mean_test[f"L{layer_idx:02d}"] = {
                "max_abs_diff": float(max_diff),
                "mean_abs_diff": float(mean_diff),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "zero_equals_mean": bool(p_val > 0.05),
                "zero_effect": zero_vals.tolist(),
                "mean_effect": mean_vals.tolist(),
            }

            significance = "YES" if p_val > 0.05 else "NO"
            if (layer_idx + 1) % 3 == 0 or layer_idx == 0 or layer_idx == n_layers - 1:
                log(f"  L{layer_idx:02d}: max_diff={max_diff:.4f}, "
                    f"p={p_val:.4f}, zero=mean? {significance}")

        # Overall summary
        n_equal = sum(1 for v in zero_mean_test.values() if v["zero_equals_mean"])
        log(f"\n  Layers where zero=mean holds (p>0.05): {n_equal}/{n_layers}")
        log(f"  Layers where zero≠mean: {n_layers - n_equal}/{n_layers}")

    # Assemble final results
    all_results = {
        "experiment": "phase3_ablation_method_comparison",
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "n_layers": n_layers,
        "families": families_list,
        "ablation_types": ablation_types,
        "timestamp": now_iso(),
        "method_results": all_method_results,
        "correlations": correlations,
        "zero_mean_test": zero_mean_test,
        "summary": {
            "n_layers": n_layers,
            "n_families": len(families_list),
            "correlations": {k: v["overall_rho"] for k, v in correlations.items()},
            "zero_mean_holds_at_layers": sum(
                1 for v in zero_mean_test.values() if v["zero_equals_mean"]
            ),
            "elapsed_seconds": round(time.time() - start_time, 1),
        },
    }

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_ablation_methods_{model_slug}.json"
    save_json(all_results, output_path)
    log(f"\nResults saved to {output_path}")

    # Register experiment
    register_experiment(
        type="phase3_ablation_methods",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(output_path)],
        status="success",
        summary=f"Phase 3 ablation method comparison: {n_layers} layers x "
                f"{len(ablation_types)} methods x {len(families_list)} families. "
                f"Correlations: {', '.join(k + '=' + f'{v:.3f}' for k, v in all_results['summary']['correlations'].items())}. "
                f"Zero=mean holds at {all_results['summary']['zero_mean_holds_at_layers']}/{n_layers} layers.",
        key_metrics={
            "overall_correlations": all_results["summary"]["correlations"],
            "zero_mean_layers": all_results["summary"]["zero_mean_holds_at_layers"],
        },
        next="Investigate layers where zero≠mean, compare with position ablation",
    )

    elapsed = time.time() - start_time
    log(f"\nAblation method comparison complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
