#!/usr/bin/env python3
"""Phase 3: Hub identification on Qwen2.5-Coder-0.5B.

Tests T09 (base vs coder transfer): Does the causal atlas (hub location,
effect distribution, MLP vs attention balance) differ between the base
Qwen2.5-0.5B and the Coder variant?

Runs the same layer ablation suite used for the base model but on the
Coder variant. Compares:
  - Hub location
  - Effect distribution across layers
  - MLP vs attention balance (when available)
  - Per-family hub agreement with the base model

Usage:
    python scripts/run_phase3_coder_atlas.py --model Qwen/Qwen2.5-Coder-0.5B
    python scripts/run_phase3_coder_atlas.py --model Qwen/Qwen2.5-Coder-0.5B --base-model Qwen/Qwen2.5-0.5B
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
from mi_atlas.task_suite import TaskSuite, TaskExample
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, PROJECT_ROOT, set_seed, now_iso


# ── Reference base model results (from Phase 1/2) ─────────────────────
BASE_MODEL_HUBS = {
    "Qwen/Qwen2.5-0.5B": {
        "global_hub": 2,
        "n_layers": 24,
        "family_hubs": {
            "json_schema": 2,
            "factual_recall": 2,
            "code_syntax": 2,
            "copying": 2,
        },
        "effect_distribution": "early-peak",  # Hub in first 15% of layers
    },
}


def load_task_suite_from_canonical():
    """Load a task suite from canonical_short files.

    Returns a TaskSuite with examples from JSON, factual, code, and copying families.
    """
    canonical_dir = PROJECT_ROOT / "data" / "tasks" / "canonical_short"

    # Families to include (matching the base model's suite)
    target_families = {
        "json_schema": "json_schema.json",
        "factual_recall": "factual_recall.json",
        "code_syntax": "code_syntax.json",
        "copying": "copying.json",
    }

    examples = []
    for family_name, fname in target_families.items():
        fpath = canonical_dir / fname
        if not fpath.exists():
            print(f"  Warning: {fname} not found, generating fallback")
            from mi_atlas.task_suite import GENERATORS
            if family_name in GENERATORS:
                generated = GENERATORS[family_name](n=10, seed=42)
                examples.extend(generated)
            continue

        try:
            data = load_json(fpath)
            raw_examples = data.get("examples", data) if isinstance(data, dict) else data

            for i, ex in enumerate(raw_examples[:20]):  # Limit per family for speed
                prompt = ex.get("prompt") or ex.get("clean_prompt") or ""
                target = ex.get("target") or ex.get("target_token") or ""
                corrupt = ex.get("corrupt_prompt") or ex.get("corrupt") or None

                if prompt:
                    examples.append(TaskExample(
                        id=f"{family_name}_{i:04d}",
                        family=family_name,
                        clean_prompt=prompt,
                        corrupt_prompt=corrupt,
                        target=target,
                        metric_type="exact_match",
                        metadata=ex.get("metadata", {}),
                    ))
        except Exception as e:
            print(f"  Warning: failed to load {fname}: {e}")

    return TaskSuite(examples)


def run_layer_ablation_all_families(backend, suite, ablation_type="zero"):
    """Run layer ablation across all layers and families in the suite.

    Returns dict with per-family and global results.
    """
    n_layers = backend.n_layers
    layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
    families = suite.families

    family_results = {}

    for family in families:
        family_suite = suite.filter_by_family(family)
        examples = list(family_suite)[:10]  # Limit for speed

        if not examples:
            continue

        print(f"  {family}: {len(examples)} examples", end="", flush=True)

        per_layer_kl = []
        per_layer_attn_kl = []
        per_layer_mlp_kl = []

        for layer_idx in range(n_layers):
            layer_name = layer_names[layer_idx]
            kl_values = []

            for example in examples:
                try:
                    result = ablate_layer(
                        backend, example.clean_prompt, layer_name, ablation_type
                    )
                    orig_logits = result["original_logits"][0, -1, :]
                    abl_logits = result["ablated_logits"][0, -1, :]
                    orig_probs = torch.softmax(orig_logits.float(), dim=-1)
                    abl_probs = torch.softmax(abl_logits.float(), dim=-1)
                    kl = torch.nn.functional.kl_div(
                        abl_probs.log(), orig_probs, reduction="sum"
                    ).item()
                    kl_values.append(abs(kl))
                except Exception:
                    pass

            mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
            per_layer_kl.append({
                "layer": layer_idx,
                "mean_kl": mean_kl,
                "std_kl": float(np.std(kl_values)) if kl_values else 0.0,
                "n": len(kl_values),
            })

        # Find hub layer
        if per_layer_kl:
            hub_entry = max(per_layer_kl, key=lambda x: x["mean_kl"])
            hub_layer = hub_entry["layer"]
            hub_kl = hub_entry["mean_kl"]
        else:
            hub_layer = 0
            hub_kl = 0.0

        # Compute effect distribution shape
        kls = [p["mean_kl"] for p in per_layer_kl]
        peak_idx = int(np.argmax(kls)) if kls else 0
        relative_position = peak_idx / max(n_layers - 1, 1)
        distribution = (
            "early-peak" if relative_position < 0.33
            else "mid-peak" if relative_position < 0.66
            else "late-peak"
        )

        family_results[family] = {
            "per_layer_kl": per_layer_kl,
            "hub_layer": hub_layer,
            "hub_kl": hub_kl,
            "n_examples": len(examples),
            "effect_distribution": distribution,
            "relative_hub_position": round(relative_position, 3),
            "mean_kl_profile": kls,
        }
        print(f" -> hub L{hub_layer} (KL={hub_kl:.3f}, {distribution})")

    return family_results


def compute_global_hub(family_results, n_layers):
    """Compute global hub by averaging KL across families."""
    global_kl = np.zeros(n_layers)
    count = 0

    for fam_data in family_results.values():
        for pl in fam_data["per_layer_kl"]:
            global_kl[pl["layer"]] += pl["mean_kl"]
        count += 1

    if count > 0:
        global_kl /= count

    global_hub = int(np.argmax(global_kl))
    return global_hub, global_kl.tolist()


def compare_atlases(coder_results, base_ref):
    """Compare coder atlas to base model atlas."""
    comparisons = {}

    # Global hub comparison
    coder_hub = coder_results["global_hub"]
    base_hub = base_ref.get("global_hub", 0)
    hub_shift = abs(coder_hub - base_hub)

    comparisons["global"] = {
        "coder_hub": coder_hub,
        "base_hub": base_hub,
        "hub_shift": hub_shift,
        "same_hub": hub_shift == 0,
    }

    # Per-family comparison
    family_agreements = []
    for family, fam_data in coder_results["family_results"].items():
        coder_fam_hub = fam_data["hub_layer"]
        base_fam_hub = base_ref.get("family_hubs", {}).get(family, base_hub)
        fam_shift = abs(coder_fam_hub - base_fam_hub)
        agree = fam_shift == 0
        family_agreements.append(agree)

        comparisons[family] = {
            "coder_hub": coder_fam_hub,
            "base_hub": base_fam_hub,
            "hub_shift": fam_shift,
            "coder_distribution": fam_data["effect_distribution"],
            "agreement": agree,
        }

    agreement_pct = (
        sum(family_agreements) / len(family_agreements) * 100
        if family_agreements else 0
    )
    comparisons["agreement_pct"] = round(agreement_pct, 1)

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Phase 3 coder atlas")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B",
                        help="Coder model to analyze")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Base model to compare against")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Phase 3: Coder atlas")
    print(f"Coder model: {args.model}")
    print(f"Base model: {args.base_model}")
    print(f"Seed: {args.seed}")

    # Load task suite
    print("\nLoading task suite...")
    suite = load_task_suite_from_canonical()
    print(f"  {len(suite)} examples across families: {suite.families}")

    # Load coder model
    print("\nLoading coder model...")
    start = time.time()
    bundle = load_model(args.model)
    backend = create_backend(bundle)
    load_time = time.time() - start
    print(f"  Loaded in {load_time:.1f}s, {backend.n_layers} layers")
    print(f"  d_model={backend.d_model}")

    # Run ablation suite
    print("\nRunning layer ablation suite on coder model...")
    t0 = time.time()
    family_results = run_layer_ablation_all_families(backend, suite, ablation_type="zero")
    ablation_time = time.time() - t0
    print(f"  Ablation complete in {ablation_time:.1f}s")

    # Compute global hub
    global_hub, global_kl_profile = compute_global_hub(family_results, backend.n_layers)
    print(f"\n  Global hub: L{global_hub}")

    # Compare to base model
    base_ref = BASE_MODEL_HUBS.get(args.base_model, {})
    comparisons = compare_atlases(
        {"global_hub": global_hub, "family_results": family_results},
        base_ref,
    )

    # Print comparison
    print(f"\n{'='*60}")
    print("CODER vs BASE COMPARISON")
    print(f"  Coder global hub: L{global_hub}")
    print(f"  Base global hub:  L{base_ref.get('global_hub', '?')}")
    print(f"  Hub shift: {comparisons.get('global', {}).get('hub_shift', '?')} layers")
    print(f"  Family agreement: {comparisons.get('agreement_pct', 0):.0f}%")
    print()
    for family in family_results:
        comp = comparisons.get(family, {})
        if "coder_hub" in comp:
            print(f"  {family:25s}: Coder=L{comp['coder_hub']}, Base=L{comp['base_hub']}, "
                  f"shift={comp['hub_shift']}, dist={comp['coder_distribution']}, "
                  f"agree={'YES' if comp['agreement'] else 'NO'}")

    # Build summary
    coder_slug = args.model.split("/")[-1]
    results = {
        "experiment": "coder_atlas",
        "phase": 3,
        "coder_model": args.model,
        "coder_slug": coder_slug,
        "base_model": args.base_model,
        "seed": args.seed,
        "n_layers": backend.n_layers,
        "d_model": backend.d_model,
        "timestamp": now_iso(),
        "ablation_time_seconds": round(ablation_time, 1),
        "global_hub": global_hub,
        "global_kl_profile": global_kl_profile,
        "family_results": family_results,
        "comparisons": comparisons,
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / "phase3_coder_atlas.json"
    save_json(results, out_path)
    print(f"\nResults saved: {out_path}")

    # Register
    global_shift = comparisons.get("global", {}).get("hub_shift", -1)
    agreement = comparisons.get("agreement_pct", 0)
    register_experiment(
        type="atlas_comparison",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(PROJECT_ROOT / "data" / "tasks" / "canonical_short")],
        outputs=[str(out_path)],
        status="success",
        summary=(
            f"Coder atlas: hub_shift={global_shift} layers from base, "
            f"family_agreement={agreement:.0f}%, "
            f"{'atlas transfers' if global_shift <= 1 and agreement >= 75 else 'atlas differs'}"
        ),
        key_metrics={
            "global_hub_shift": global_shift,
            "family_agreement_pct": agreement,
            "coder_global_hub": global_hub,
            "base_global_hub": base_ref.get("global_hub", -1),
        },
        next=(
            "If atlas transfers (shift <= 1), coding ability may share the same "
            "circuitry as base capabilities. If atlas differs, coding has its own hub."
        ),
    )


if __name__ == "__main__":
    main()
