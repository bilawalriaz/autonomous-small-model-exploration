#!/usr/bin/env python3
"""Phase 3: Steering direction transfer across scales.

Tests whether steering DIRECTION transfers across model scales. This is a
gem-hunting experiment that probes whether steering migration is about
direction, layer position, or both.

Protocol:
  1. Compute steering vector at 0.5B hub layer (L2)
  2. Apply that same vector at 1.5B L2
  3. Also compute a fresh vector at 1.5B L2
  4. Compare: KL divergence, target logit delta
  5. Test: does the 0.5B L2 direction have any effect at 1.5B L26 (the 1.5B hub)?

This disentangles whether cross-scale transfer is about:
  - Direction (same vector, different layer)
  - Layer position (same layer, different vector)
  - Both

Usage:
    python scripts/run_phase3_steering_direction_transfer.py
    python scripts/run_phase3_steering_direction_transfer.py --small-model Qwen/Qwen2.5-0.5B --large-model Qwen/Qwen2.5-1.5B
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
from mi_atlas.steering import compute_steering_vector, inject_steering_vector
from mi_atlas.metrics import kl_divergence
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, PROJECT_ROOT, set_seed, now_iso


# ── Hub layers by model ────────────────────────────────────────────────
HUB_LAYERS = {
    "Qwen/Qwen2.5-0.5B": {"primary": 2, "secondary": [21, 22, 23]},
    "Qwen/Qwen2.5-1.5B": {"primary": 26, "secondary": [2, 21, 25, 27]},
}

# Eval prompts (factual recall)
EVAL_PROMPTS = [
    "The capital of France is ",
    "The capital of Japan is ",
    "The capital of Germany is ",
    "The capital of Italy is ",
    "The capital of Spain is ",
    "The largest planet in our solar system is ",
    "Water boils at 100 degrees ",
    "The chemical symbol for gold is ",
]

EVAL_PROMPTS_JSON = [
    "Return valid JSON: name=Alice, age=30\n",
    "Return valid JSON: color=red, count=5\n",
    "JSON with keys city and pop. city=London, pop=9\n",
    "JSON: fruit=apple, qty=10\n",
]


def get_steering_pairs():
    """Load or construct clean/corrupt pairs for steering vector computation."""
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    if pairs_path.exists():
        pairs = load_json(pairs_path)
        factual = [p for p in pairs if p.get("family") == "factual_recall"]
        if factual:
            return factual

    # Fallback: construct pairs manually
    return [
        {"clean": "The capital of France is ", "corrupt": "The capital of Germany is ", "family": "factual_recall"},
        {"clean": "The capital of Japan is ", "corrupt": "The capital of Italy is ", "family": "factual_recall"},
        {"clean": "The capital of Germany is ", "corrupt": "The capital of Spain is ", "family": "factual_recall"},
        {"clean": "The capital of Italy is ", "corrupt": "The capital of France is ", "family": "factual_recall"},
    ]


def compute_sv_from_pairs(backend, pairs, layer_name):
    """Compute steering vector from clean/corrupt pairs."""
    if not pairs:
        return None

    # Use first pair for consistency with other experiments
    pair = pairs[0]
    clean = pair["clean"] if isinstance(pair["clean"], str) else pair["clean"][0]
    corrupt = pair["corrupt"] if isinstance(pair["corrupt"], str) else pair["corrupt"][0]

    return compute_steering_vector(backend, [clean], [corrupt], layer_name)


def measure_steering_effect(backend, prompts, sv, layer_name, strengths):
    """Measure steering effect across strengths for a set of prompts.

    Returns per-strength aggregated metrics.
    """
    results_by_strength = {s: {"kl": [], "target_delta": []} for s in strengths}

    for prompt in prompts:
        for strength in strengths:
            try:
                output = inject_steering_vector(
                    backend, prompt, layer_name, sv, strength
                )
                if output.get("status") != "success":
                    continue

                baseline_logits = output["original_logits"][0, -1, :]
                steered_logits = output["steered_logits"][0, -1, :]

                kl = kl_divergence(
                    baseline_logits.unsqueeze(0), steered_logits.unsqueeze(0)
                )

                # Target logit delta: how much the argmax changes
                baseline_top5 = set(torch.topk(baseline_logits, 5).indices.tolist())
                steered_top5 = set(torch.topk(steered_logits, 5).indices.tolist())
                top5_overlap = len(baseline_top5 & steered_top5) / 5.0

                results_by_strength[strength]["kl"].append(float(kl))
                results_by_strength[strength]["target_delta"].append(float(1.0 - top5_overlap))

            except Exception:
                pass

    # Aggregate
    aggregated = []
    for s in strengths:
        kls = results_by_strength[s]["kl"]
        deltas = results_by_strength[s]["target_delta"]
        if kls:
            aggregated.append({
                "strength": s,
                "mean_kl": float(np.mean(kls)),
                "std_kl": float(np.std(kls)),
                "mean_top5_change": float(np.mean(deltas)),
                "n": len(kls),
            })

    return aggregated


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(description="Phase 3 steering direction transfer")
    parser.add_argument("--model", type=str, default=None,
                       help="Primary model (used as small model if --small-model not set)")
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--large-model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # If --model is passed (from orchestrator), use it as --small-model
    if args.model is not None:
        args.small_model = args.model

    set_seed(args.seed)
    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    small_slug = args.small_model.split("/")[-1]
    large_slug = args.large_model.split("/")[-1]

    small_hubs = HUB_LAYERS.get(args.small_model, {"primary": 2, "secondary": []})
    large_hubs = HUB_LAYERS.get(args.large_model, {"primary": 26, "secondary": [2]})

    small_hub = small_hubs["primary"]  # e.g., L2 for 0.5B
    large_hub = large_hubs["primary"]  # e.g., L26 for 1.5B
    large_same_layer = small_hub       # L2 on 1.5B (same absolute layer)

    print(f"Phase 3: Steering direction transfer")
    print(f"Small model: {args.small_model} (hub L{small_hub})")
    print(f"Large model: {args.large_model} (hub L{large_hub})")
    print(f"Strengths: {strengths}")
    print(f"Seed: {args.seed}")

    # Load steering pairs
    pairs = get_steering_pairs()
    print(f"\nSteering pairs: {len(pairs)}")

    # ── Step 1: Load small model and compute steering vector ────────
    print(f"\n{'='*60}")
    print(f"Step 1: Compute steering vector at {small_slug} L{small_hub}")
    start = time.time()
    small_bundle = load_model(args.small_model)
    small_backend = create_backend(small_bundle)
    print(f"  Loaded in {time.time() - start:.1f}s")

    small_layer_name = f"layer_{small_hub:02d}"
    sv_small = compute_sv_from_pairs(small_backend, pairs, small_layer_name)
    if sv_small is None:
        print("  ERROR: Failed to compute steering vector on small model")
        return
    sv_small_norm = float(torch.norm(sv_small).item())
    print(f"  SV norm: {sv_small_norm:.4f}")

    # Measure effect on small model at its hub
    print(f"  Measuring effect at {small_slug} L{small_hub}...")
    small_effects = measure_steering_effect(
        small_backend, EVAL_PROMPTS, sv_small, small_layer_name, strengths
    )
    best_small = max(small_effects, key=lambda x: x["mean_kl"]) if small_effects else None
    if best_small:
        print(f"  Best: KL={best_small['mean_kl']:.3f} at strength={best_small['strength']}")

    # Free small model memory
    del small_backend
    del small_bundle.model
    del small_bundle
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Step 2: Load large model ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2: Load {large_slug}")
    start = time.time()
    large_bundle = load_model(args.large_model)
    large_backend = create_backend(large_bundle)
    print(f"  Loaded in {time.time() - start:.1f}s")

    # ── Step 3: Transfer test ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 3: Transfer tests")

    # Project small SV to large model dimension if needed
    d_small = sv_small.shape[0]
    d_large = large_backend.d_model

    if d_small != d_large:
        print(f"  Dimension mismatch: {d_small} vs {d_large}")
        print(f"  Truncating/padding small SV to match large model")
        sv_small_proj = torch.zeros(d_large)
        copy_len = min(d_small, d_large)
        sv_small_proj[:copy_len] = sv_small[:copy_len]
        # Renormalize to same norm
        sv_small_proj = sv_small_proj / (torch.norm(sv_small_proj) + 1e-10) * sv_small_norm
    else:
        sv_small_proj = sv_small.clone()

    # 3a: Compute fresh steering vector at large model L{small_hub} (same absolute layer)
    large_same_layer_name = f"layer_{large_same_layer:02d}"
    print(f"\n  3a: Fresh SV at {large_slug} L{large_same_layer}...")
    sv_large_same = compute_sv_from_pairs(large_backend, pairs, large_same_layer_name)
    if sv_large_same is None:
        print(f"      ERROR: Failed to compute SV at {large_slug} L{large_same_layer}")
        sv_large_same = torch.zeros(d_large)
    sv_large_same_norm = float(torch.norm(sv_large_same).item())
    print(f"      Fresh SV norm: {sv_large_same_norm:.4f}")

    # Cosine similarity between small and fresh (same layer)
    cos_same_layer = cosine_similarity(sv_small_proj, sv_large_same)
    print(f"      Cosine similarity (small L{small_hub} vs large L{large_same_layer}): {cos_same_layer:.4f}")

    # 3b: Compute fresh steering vector at large model hub (L{large_hub})
    large_hub_layer_name = f"layer_{large_hub:02d}"
    print(f"\n  3b: Fresh SV at {large_slug} L{large_hub} (large hub)...")
    sv_large_hub = compute_sv_from_pairs(large_backend, pairs, large_hub_layer_name)
    if sv_large_hub is None:
        print(f"      ERROR: Failed to compute SV at {large_slug} L{large_hub}")
        sv_large_hub = torch.zeros(d_large)
    sv_large_hub_norm = float(torch.norm(sv_large_hub).item())
    print(f"      Fresh SV norm: {sv_large_hub_norm:.4f}")

    # Cosine similarity between directions
    cos_same_hub = cosine_similarity(sv_small_proj, sv_large_hub)
    cos_fresh = cosine_similarity(sv_large_same, sv_large_hub)
    print(f"      Cosine (small L{small_hub} vs large L{large_hub}): {cos_same_hub:.4f}")
    print(f"      Cosine (large L{large_same_layer} vs large L{large_hub}): {cos_fresh:.4f}")

    # ── Step 4: Effect measurements ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: Effect measurements")

    experiments = {
        "A_fresh_at_large_same_layer": {
            "description": f"Fresh SV computed at {large_slug} L{large_same_layer}, applied at L{large_same_layer}",
            "sv_source": f"{large_slug} L{large_same_layer}",
            "apply_layer": large_same_layer,
            "sv": sv_large_same,
        },
        "B_fresh_at_large_hub": {
            "description": f"Fresh SV computed at {large_slug} L{large_hub}, applied at L{large_hub}",
            "sv_source": f"{large_slug} L{large_hub}",
            "apply_layer": large_hub,
            "sv": sv_large_hub,
        },
        "C_small_at_large_same_layer": {
            "description": f"Small model SV ({small_slug} L{small_hub}), applied at {large_slug} L{large_same_layer}",
            "sv_source": f"{small_slug} L{small_hub}",
            "apply_layer": large_same_layer,
            "sv": sv_small_proj,
        },
        "D_small_at_large_hub": {
            "description": f"Small model SV ({small_slug} L{small_hub}), applied at {large_slug} L{large_hub}",
            "sv_source": f"{small_slug} L{small_hub}",
            "apply_layer": large_hub,
            "sv": sv_small_proj,
        },
    }

    all_experiment_results = {}

    for exp_name, exp_info in experiments.items():
        print(f"\n  {exp_name}: {exp_info['description']}")
        apply_layer_name = f"layer_{exp_info['apply_layer']:02d}"

        effects = measure_steering_effect(
            large_backend, EVAL_PROMPTS, exp_info["sv"], apply_layer_name, strengths
        )

        best = max(effects, key=lambda x: x["mean_kl"]) if effects else None
        if best:
            print(f"    Best: KL={best['mean_kl']:.3f} at s={best['strength']}")

        all_experiment_results[exp_name] = {
            "description": exp_info["description"],
            "sv_source": exp_info["sv_source"],
            "apply_layer": exp_info["apply_layer"],
            "sv_norm": float(torch.norm(exp_info["sv"]).item()),
            "by_strength": effects,
            "best_kl": best["mean_kl"] if best else 0,
            "best_strength": best["strength"] if best else None,
        }

    # Also run JSON prompts for experiment C and D
    print(f"\n  JSON prompt tests...")
    for exp_name in ["C_small_at_large_same_layer", "D_small_at_large_hub"]:
        exp_info = experiments[exp_name]
        apply_layer_name = f"layer_{exp_info['apply_layer']:02d}"
        effects = measure_steering_effect(
            large_backend, EVAL_PROMPTS_JSON, exp_info["sv"], apply_layer_name, strengths
        )
        best = max(effects, key=lambda x: x["mean_kl"]) if effects else None
        if best:
            print(f"    {exp_name} (JSON): best KL={best['mean_kl']:.3f}")
        all_experiment_results[f"{exp_name}_json"] = {
            "description": exp_info["description"] + " (JSON prompts)",
            "by_strength": effects,
            "best_kl": best["mean_kl"] if best else 0,
        }

    # ── Step 5: Transfer analysis ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRANSFER ANALYSIS")

    A_best = all_experiment_results.get("A_fresh_at_large_same_layer", {}).get("best_kl", 0)
    B_best = all_experiment_results.get("B_fresh_at_large_hub", {}).get("best_kl", 0)
    C_best = all_experiment_results.get("C_small_at_large_same_layer", {}).get("best_kl", 0)
    D_best = all_experiment_results.get("D_small_at_large_hub", {}).get("best_kl", 0)

    # Transfer ratios
    transfer_same_layer = C_best / max(A_best, 1e-10)  # How much of fresh effect does small-SV capture at same layer?
    transfer_hub = D_best / max(B_best, 1e-10)          # How much of fresh effect does small-SV capture at hub?
    direction_vs_position = D_best / max(C_best, 1e-10)  # Does same direction work better at hub than same layer?

    analysis = {
        "transfer_same_layer_ratio": round(transfer_same_layer, 3),
        "transfer_hub_ratio": round(transfer_hub, 3),
        "direction_vs_position_ratio": round(direction_vs_position, 3),
        "direction_transfers": transfer_same_layer > 0.3,
        "position_matters": direction_vs_position > 1.5,
        "cosine_same_layer": cos_same_layer,
        "cosine_same_hub": cos_same_hub,
        "cosine_fresh_fresh": cos_fresh,
    }

    print(f"  Transfer at same layer (C/A): {transfer_same_layer:.3f}")
    print(f"  Transfer at hub (D/B):        {transfer_hub:.3f}")
    print(f"  Direction vs position (D/C):  {direction_vs_position:.3f}")
    print(f"  Direction transfers (>0.3):   {'YES' if analysis['direction_transfers'] else 'NO'}")
    print(f"  Position matters (>1.5x):     {'YES' if analysis['position_matters'] else 'NO'}")
    print(f"  Direction similarity:          same_layer={cos_same_layer:.3f}, same_hub={cos_same_hub:.3f}")

    # Interpretation
    if analysis["direction_transfers"] and analysis["position_matters"]:
        interpretation = "Both direction and position contribute. Direction partially transfers, but hub layer matters."
    elif analysis["direction_transfers"] and not analysis["position_matters"]:
        interpretation = "Direction is the key factor. Position matters less."
    elif not analysis["direction_transfers"] and analysis["position_matters"]:
        interpretation = "Position is the key factor. Direction doesn't transfer across scales."
    else:
        interpretation = "Neither direction nor absolute position reliably transfers. Steering is scale-specific."

    print(f"\n  Interpretation: {interpretation}")

    # Save results
    results = {
        "experiment": "steering_direction_transfer",
        "phase": 3,
        "small_model": args.small_model,
        "large_model": args.large_model,
        "small_slug": small_slug,
        "large_slug": large_slug,
        "seed": args.seed,
        "small_hub": small_hub,
        "large_hub": large_hub,
        "strengths": strengths,
        "n_eval_prompts_factual": len(EVAL_PROMPTS),
        "n_eval_prompts_json": len(EVAL_PROMPTS_JSON),
        "timestamp": now_iso(),
        "direction_similarities": {
            "cosine_same_layer": cos_same_layer,
            "cosine_same_hub": cos_same_hub,
            "cosine_fresh_fresh": cos_fresh,
        },
        "experiments": all_experiment_results,
        "analysis": analysis,
        "interpretation": interpretation,
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / "phase3_steering_transfer.json"
    save_json(results, out_path)
    print(f"\nResults saved: {out_path}")

    # Register
    register_experiment(
        type="cross_scale_transfer",
        model=f"{args.small_model} -> {args.large_model}",
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json")],
        outputs=[str(out_path)],
        status="success",
        summary=(
            f"Steering transfer {small_slug}->{large_slug}: "
            f"same_layer_ratio={transfer_same_layer:.2f}, "
            f"hub_ratio={transfer_hub:.2f}, "
            f"direction_cosine={cos_same_layer:.3f}. "
            f"{interpretation}"
        ),
        key_metrics={
            "transfer_same_layer_ratio": round(transfer_same_layer, 3),
            "transfer_hub_ratio": round(transfer_hub, 3),
            "direction_vs_position_ratio": round(direction_vs_position, 3),
            "cosine_same_layer": round(cos_same_layer, 3),
            "direction_transfers": analysis["direction_transfers"],
            "position_matters": analysis["position_matters"],
        },
        next=(
            "If direction transfers, steering vectors can be computed on smaller "
            "models and applied to larger ones. If not, each scale needs its own vectors."
        ),
    )


if __name__ == "__main__":
    main()
