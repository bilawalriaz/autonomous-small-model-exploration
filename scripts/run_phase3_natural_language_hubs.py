#!/usr/bin/env python3
"""Phase 3: Hub identification with 50+ natural language prompts.

Tests T03 (synthetic prompts are toy-like): do hub layers shift when using
natural-language prompts from the Phase 2 canonical task suite instead of
the short synthetic prompts used in Phase 1?

Loads canonical_short task files (JSON per family) and gathers 50+ prompts
per family. Runs layer ablation (zero) across all layers with these expanded
natural-language prompts. Compares hub location to the Phase 1 synthetic-
prompt hub.

Reports:
  - hub_shift: number of layers the hub moved
  - hub_agreement: % of families where hub is the same as Phase 1

Usage:
    python scripts/run_phase3_natural_language_hubs.py --model Qwen/Qwen2.5-0.5B
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
from mi_atlas.ablations import ablate_layer
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, PROJECT_ROOT, set_seed, now_iso


# ── Phase 1 hub reference (from layer_ablation results) ────────────────
PHASE1_HUBS = {
    "Qwen/Qwen2.5-0.5B": {
        # Hub layers identified in Phase 1 with synthetic prompts
        "global_hub": 2,
        "family_hubs": {
            "copying": 2,
            "delimiter_tracking": 2,
            "json_schema": 2,
            "factual_recall": 2,
            "arithmetic": 2,
            "code_syntax": 2,
            "code_semantics": 2,
            "dead_code": 2,
        },
    },
    "Qwen/Qwen2.5-1.5B": {
        "global_hub": 21,
        "family_hubs": {},
    },
}


def load_canonical_short_prompts(min_per_family=50):
    """Load natural-language prompts from canonical_short task files.

    Returns dict mapping family -> list of prompt strings.
    Falls back to generated tasks if canonical files are missing.
    """
    canonical_dir = PROJECT_ROOT / "data" / "tasks" / "canonical_short"
    family_prompts = {}

    # Map canonical file names to our family names
    file_to_family = {
        "factual_recall.json": "factual_recall",
        "json_schema.json": "json_schema",
        "copying.json": "copying",
        "delimiter_tracking.json": "delimiter_tracking",
        "arithmetic.json": "arithmetic",
        "code_syntax.json": "code_syntax",
        "code_semantics.json": "code_semantics",
        "dead_code.json": "dead_code",
        "variable_renaming.json": "variable_renaming",
        "verbosity_control.json": "verbosity_control",
        "uncertainty_expression.json": "uncertainty_signalling",
        "harmless_refusal.json": "refusal_compliance",
        "instruction_following.json": "instruction_following",
        "string_decoding.json": "string_decoding",
        "constant_folding.json": "constant_folding",
        "control_flow_simplification.json": "control_flow_simplification",
    }

    if canonical_dir.exists():
        for fname, family_name in file_to_family.items():
            fpath = canonical_dir / fname
            if not fpath.exists():
                continue
            try:
                data = load_json(fpath)
                examples = data.get("examples", data) if isinstance(data, dict) else data
                prompts = []
                for ex in examples:
                    prompt = ex.get("prompt") or ex.get("clean_prompt") or ""
                    if prompt:
                        prompts.append(prompt)
                if prompts:
                    family_prompts[family_name] = prompts
                    print(f"  {family_name}: {len(prompts)} prompts from {fname}")
            except Exception as e:
                print(f"  Warning: failed to load {fname}: {e}")

    # If any family has fewer than min_per_family, supplement with generated tasks
    from mi_atlas.task_suite import GENERATORS
    for family_name, generator in GENERATORS.items():
        existing = family_prompts.get(family_name, [])
        if len(existing) < min_per_family:
            need = min_per_family - len(existing)
            generated = generator(n=need, seed=42)
            extra_prompts = [ex.clean_prompt for ex in generated]
            family_prompts[family_name] = existing + extra_prompts
            print(f"  {family_name}: supplemented {len(existing)} -> {len(family_prompts[family_name])} prompts")

    return family_prompts


def run_ablation_with_prompts(backend, prompts, ablation_type="zero", n_limit=None):
    """Run layer ablation with a specific set of prompts.

    Returns list of dicts with per-layer KL effects.
    """
    if n_limit:
        prompts = prompts[:n_limit]

    n_layers = backend.n_layers
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]

    per_layer_kl = []

    for layer_idx in range(n_layers):
        layer_name = layer_names[layer_idx]
        kl_values = []

        for prompt in prompts:
            try:
                result = ablate_layer(backend, prompt, layer_name, ablation_type)
                orig_logits = result["original_logits"][0, -1, :]
                abl_logits = result["ablated_logits"][0, -1, :]
                orig_probs = torch.softmax(orig_logits, dim=-1)
                abl_probs = torch.softmax(abl_logits, dim=-1)
                kl = torch.nn.functional.kl_div(
                    abl_probs.log(), orig_probs, reduction="sum"
                ).item()
                kl_values.append(abs(kl))
            except Exception:
                pass

        per_layer_kl.append({
            "layer": layer_idx,
            "mean_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "std_kl": float(np.std(kl_values)) if kl_values else 0.0,
            "n": len(kl_values),
        })

    return per_layer_kl


def find_hub_layer(per_layer_kl):
    """Find the layer with maximum mean KL effect."""
    if not per_layer_kl:
        return 0
    return max(per_layer_kl, key=lambda x: x["mean_kl"])["layer"]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 natural language hubs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-prompts", type=int, default=50,
                        help="Minimum prompts per family")
    args = parser.parse_args()

    set_seed(args.seed)
    model_slug = args.model.split("/")[-1]

    print(f"Phase 3: Natural language hubs")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Min prompts per family: {args.min_prompts}")

    # Load model
    print("\nLoading model...")
    start = time.time()
    bundle = load_model(args.model)
    backend = create_backend(bundle)
    load_time = time.time() - start
    print(f"  Loaded in {load_time:.1f}s, {backend.n_layers} layers")

    # Load canonical short prompts
    print("\nLoading canonical short prompts...")
    family_prompts = load_canonical_short_prompts(min_per_family=args.min_prompts)
    total_prompts = sum(len(v) for v in family_prompts.values())
    print(f"  Total: {total_prompts} prompts across {len(family_prompts)} families")

    # Run ablation per family
    family_results = {}
    for family_name, prompts in family_prompts.items():
        print(f"\n  Running ablation for {family_name} ({len(prompts)} prompts)...")
        t0 = time.time()
        per_layer = run_ablation_with_prompts(backend, prompts, ablation_type="zero")
        elapsed = time.time() - t0
        hub = find_hub_layer(per_layer)
        family_results[family_name] = {
            "per_layer_kl": per_layer,
            "hub_layer": hub,
            "hub_kl": per_layer[hub]["mean_kl"] if hub < len(per_layer) else 0,
            "n_prompts": len(prompts),
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"    Hub: L{hub} (KL={per_layer[hub]['mean_kl']:.3f}) [{elapsed:.1f}s]")

    # Global hub: average KL across all families
    n_layers = backend.n_layers
    global_kl = np.zeros(n_layers)
    global_n = np.zeros(n_layers)
    for fam_data in family_results.values():
        for pl in fam_data["per_layer_kl"]:
            global_kl[pl["layer"]] += pl["mean_kl"]
            global_n[pl["layer"]] += pl["n"]
    global_mean_kl = [float(global_kl[i] / max(global_n[i], 1)) for i in range(n_layers)]
    global_hub = int(np.argmax(global_mean_kl))

    # Compare to Phase 1
    phase1_ref = PHASE1_HUBS.get(args.model, {})
    phase1_global_hub = phase1_ref.get("global_hub", 0)
    phase1_family_hubs = phase1_ref.get("family_hubs", {})

    global_hub_shift = abs(global_hub - phase1_global_hub)

    # Per-family agreement
    agreements = []
    family_comparisons = {}
    for fam_name, fam_data in family_results.items():
        nl_hub = fam_data["hub_layer"]
        p1_hub = phase1_family_hubs.get(fam_name, phase1_global_hub)
        agree = nl_hub == p1_hub
        agreements.append(agree)
        family_comparisons[fam_name] = {
            "nl_hub": nl_hub,
            "phase1_hub": p1_hub,
            "shift": abs(nl_hub - p1_hub),
            "agreement": agree,
        }

    hub_agreement_pct = (sum(agreements) / len(agreements) * 100) if agreements else 0.0

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Global hub (NL): L{global_hub}")
    print(f"  Global hub (Phase 1): L{phase1_global_hub}")
    print(f"  Global hub shift: {global_hub_shift} layers")
    print(f"  Hub agreement: {hub_agreement_pct:.0f}% ({sum(agreements)}/{len(agreements)} families)")
    print(f"\n  Per-family:")
    for fam, comp in family_comparisons.items():
        status = "AGREE" if comp["agreement"] else f"SHIFT={comp['shift']}"
        print(f"    {fam:30s}: NL=L{comp['nl_hub']}, P1=L{comp['phase1_hub']} [{status}]")

    # Save results
    results = {
        "experiment": "natural_language_hubs",
        "phase": 3,
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "n_layers": n_layers,
        "timestamp": now_iso(),
        "min_prompts_per_family": args.min_prompts,
        "total_prompts": total_prompts,
        "global_hub_nl": global_hub,
        "global_hub_phase1": phase1_global_hub,
        "global_hub_shift": global_hub_shift,
        "hub_agreement_pct": round(hub_agreement_pct, 1),
        "family_results": family_results,
        "family_comparisons": family_comparisons,
        "global_mean_kl": global_mean_kl,
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_natural_language_hubs_{model_slug}.json"
    save_json(results, out_path)
    print(f"\nResults saved: {out_path}")

    # Register
    register_experiment(
        type="robustness",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(PROJECT_ROOT / "data" / "tasks" / "canonical_short")],
        outputs=[str(out_path)],
        status="success",
        summary=(
            f"Natural language hubs: global shift={global_hub_shift} layers, "
            f"agreement={hub_agreement_pct:.0f}%, "
            f"{'NL hub aligns with Phase 1' if global_hub_shift == 0 else 'Hub shifted under NL prompts'}"
        ),
        key_metrics={
            "global_hub_shift": global_hub_shift,
            "hub_agreement_pct": round(hub_agreement_pct, 1),
            "n_families": len(family_results),
            "total_prompts": total_prompts,
        },
        next=(
            "If hub_agreement > 80%, synthetic prompts are representative. "
            "If < 60%, Phase 1 hub findings may be toy-specific."
        ),
    )


if __name__ == "__main__":
    main()
