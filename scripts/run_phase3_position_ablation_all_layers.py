#!/usr/bin/env python3
"""Phase 3: Token-position-specific ablation at ALL layers.

For each layer 0..n_layers-1 and each position type (BOS, instruction tokens,
content tokens, final answer tokens): ablates that position's contribution
and measures KL divergence.

Uses hooks that selectively zero specific token positions to determine
where in the token sequence each layer's effect is concentrated.

Usage:
    python -u scripts/run_phase3_position_ablation_all_layers.py --model Qwen/Qwen2.5-0.5B
    python -u scripts/run_phase3_position_ablation_all_layers.py --model Qwen/Qwen2.5-0.5B --force --seed 137
"""

import argparse
import re
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from mi_atlas.model_loader import load_model_hf
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite, build_default_suite
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
        if (rec.get("type") == "phase3_position_ablation"
                and model_slug in rec.get("model", "")
                and rec.get("status") == "success"):
            return True
    return False


def classify_token_position(token_str, pos_idx, n_tokens, token_strs):
    """Classify a token into a position type.

    Position types:
    - bos: first token (BOS / start of sequence)
    - instruction: tokens in the instruction/prompt header
    - content: middle content tokens (data, values)
    - answer: final answer tokens (near end of sequence)
    - delimiter: punctuation and structural tokens
    """
    if pos_idx == 0:
        return "bos"

    if pos_idx >= n_tokens - 3:
        return "answer"

    stripped = token_str.strip()

    # Delimiters and structural tokens
    if stripped in (":", ",", ".", "=", "+", "-", "*", "/", "\\", "|",
                    "(", ")", "[", "]", "{", "}", "<", ">",
                    "->", "=>", ":=", "::", "...", "--"):
        return "delimiter"

    if any(c in stripped for c in "([{") or any(c in stripped for c in ")]}"):
        return "delimiter"

    # Instruction tokens (typically in first 20-30% of sequence)
    instruction_boundary = max(3, n_tokens // 4)
    if pos_idx < instruction_boundary:
        return "instruction"

    # Default: content
    return "content"


def run_position_ablation_at_layer(model, tokenizer, examples, layer_idx,
                                    position_types_to_ablate=None):
    """Run position-specific ablation at a given layer.

    For each example, zero out tokens matching the target position types
    at the specified layer and measure KL divergence from baseline.

    Args:
        model: The model
        tokenizer: Tokenizer
        examples: List of TaskExample
        layer_idx: Layer to ablate
        position_types_to_ablate: Set of position types to zero (None = all)

    Returns:
        dict mapping position_type -> mean KL divergence
    """
    if position_types_to_ablate is None:
        position_types_to_ablate = {"bos", "instruction", "content",
                                     "answer", "delimiter"}

    results_by_type = {pt: [] for pt in position_types_to_ablate}

    for example in examples:
        try:
            inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)
            n_tokens = input_ids.shape[1]
            token_strs = [tokenizer.decode([tid]) for tid in input_ids[0]]

            # Classify each position
            positions = []
            for i in range(n_tokens):
                pt = classify_token_position(token_strs[i], i, n_tokens, token_strs)
                positions.append(pt)

            # Get baseline logits
            with torch.no_grad():
                base_out = model(input_ids)
                base_logits = base_out.logits[0, -1, :]

            # For each position type, ablate matching positions
            for pt in position_types_to_ablate:
                positions_to_ablate = [i for i, p in enumerate(positions)
                                       if p == pt]
                if not positions_to_ablate:
                    continue

                # Register hook that zeros specific positions
                def make_hook(ablate_positions):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                        else:
                            hidden = output.clone()

                        for pos in ablate_positions:
                            if pos < hidden.shape[1]:
                                hidden[0, pos, :] = 0.0

                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    return hook_fn

                layer_module = model.model.layers[layer_idx]
                handle = layer_module.register_forward_hook(
                    make_hook(positions_to_ablate)
                )
                with torch.no_grad():
                    abl_out = model(input_ids)
                    abl_logits = abl_out.logits[0, -1, :]
                handle.remove()

                # Compute KL
                kl = kl_divergence(base_logits.unsqueeze(0),
                                  abl_logits.unsqueeze(0))
                results_by_type[pt].append(kl)

        except Exception as e:
            log(f"    Error at L{layer_idx}: {e}")
            continue

    # Average KL per position type
    avg_kl = {}
    for pt, kls in results_by_type.items():
        avg_kl[pt] = float(np.mean(kls)) if kls else 0.0

    return avg_kl


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Position-specific ablation at all layers"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true",
                       help="Re-run even if already completed")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    position_types = ["bos", "instruction", "content", "answer", "delimiter"]

    log("=" * 60)
    log(f"Phase 3: Position-Specific Ablation at ALL Layers")
    log(f"Model: {args.model}")
    log(f"Position types: {position_types}")
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
    n_layers = bundle.architecture["n_layers"]
    log(f"  Loaded: {n_layers} layers, device={bundle.device}")

    # Build suite and collect examples from each family
    suite = build_default_suite(seed=args.seed)
    families = suite.families
    log(f"  Families: {families}")

    # Gather representative examples from each family (test split preferred)
    all_examples = []
    for family in families:
        fam_suite = suite.filter_by_family(family)
        test_suite = fam_suite.filter_by_split("test")
        examples = list(test_suite)[:3] if list(test_suite) else list(fam_suite)[:3]
        all_examples.extend(examples)

    log(f"  Using {len(all_examples)} examples across {len(families)} families")

    # Run ablation at each layer for each position type
    # Result: (n_layers, n_position_types) matrix
    effect_matrix = np.zeros((n_layers, len(position_types)))
    per_layer_per_family = {}

    for layer_idx in range(n_layers):
        try:
            layer_start = time.time()

            # Run on all examples for this layer
            kl_by_type = run_position_ablation_at_layer(
                model, tokenizer, all_examples, layer_idx,
                position_types_to_ablate=set(position_types)
            )

            for pt_idx, pt in enumerate(position_types):
                effect_matrix[layer_idx, pt_idx] = kl_by_type.get(pt, 0.0)

            layer_elapsed = time.time() - layer_start
            max_effect = max(kl_by_type.values()) if kl_by_type else 0.0
            dominant = max(kl_by_type, key=kl_by_type.get) if kl_by_type else "none"

            # Per-family breakdown for this layer
            per_family_layer = {}
            for family in families:
                fam_suite = suite.filter_by_family(family)
                test_suite = fam_suite.filter_by_split("test")
                fam_examples = list(test_suite)[:2] if list(test_suite) else list(fam_suite)[:2]

                fam_kl = run_position_ablation_at_layer(
                    model, tokenizer, fam_examples, layer_idx,
                    position_types_to_ablate=set(position_types)
                )
                per_family_layer[family] = fam_kl

            per_layer_per_family[f"L{layer_idx:02d}"] = per_family_layer

            # Progress logging
            if (layer_idx + 1) % 3 == 0 or layer_idx == 0 or layer_idx == n_layers - 1:
                log(f"  L{layer_idx:02d}: max_KL={max_effect:.4f} (dominant={dominant}) "
                    f"[{layer_elapsed:.1f}s]")
                for pt in position_types:
                    val = kl_by_type.get(pt, 0.0)
                    if val > 0.01:
                        log(f"    {pt:>12s}: KL={val:.4f}")

        except Exception as e:
            log(f"  L{layer_idx:02d}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    # Analysis: which position types dominate at which layers?
    log(f"\n{'='*60}")
    log("POSITION TYPE DOMINANCE BY LAYER")
    log(f"{'='*60}")

    dominance = {}
    for layer_idx in range(n_layers):
        layer_effects = effect_matrix[layer_idx, :]
        if layer_effects.max() > 0:
            dominant_idx = layer_effects.argmax()
            dominant_type = position_types[dominant_idx]
            dominance[f"L{layer_idx:02d}"] = {
                "dominant_type": dominant_type,
                "effects": {pt: float(effect_matrix[layer_idx, i])
                           for i, pt in enumerate(position_types)},
            }
        else:
            dominance[f"L{layer_idx:02d}"] = {
                "dominant_type": "none",
                "effects": {pt: 0.0 for pt in position_types},
            }

    # Print summary table
    log(f"\n{'Layer':>6} {'BOS':>8} {'Instr':>8} {'Content':>8} "
        f"{'Answer':>8} {'Delim':>8} {'Dominant':>10}")
    log("-" * 65)
    for layer_idx in range(n_layers):
        row = [f"L{layer_idx:02d}"]
        for pt_idx, pt in enumerate(position_types):
            val = effect_matrix[layer_idx, pt_idx]
            row.append(f"{val:>8.4f}")
        dom = dominance[f"L{layer_idx:02d}"]["dominant_type"]
        row.append(f"{dom:>10}")
        log(" ".join(row))

    # Aggregate: mean effect of each position type across all layers
    log(f"\n{'='*60}")
    log("AGGREGATE POSITION TYPE EFFECTS")
    log(f"{'='*60}")

    aggregate = {}
    for pt_idx, pt in enumerate(position_types):
        effects = effect_matrix[:, pt_idx]
        aggregate[pt] = {
            "mean_effect": float(effects.mean()),
            "max_effect": float(effects.max()),
            "max_layer": int(effects.argmax()),
            "std": float(effects.std()),
        }
        log(f"  {pt:>12s}: mean={effects.mean():.4f}, max={effects.max():.4f} "
            f"(L{effects.argmax():02d}), std={effects.std():.4f}")

    # Per-family position analysis
    log(f"\n{'='*60}")
    log("PER-FAMILY POSITION ANALYSIS")
    log(f"{'='*60}")

    per_family_summary = {}
    for family in families:
        fam_effects = np.zeros((n_layers, len(position_types)))
        for layer_idx in range(n_layers):
            layer_key = f"L{layer_idx:02d}"
            if layer_key in per_layer_per_family:
                fam_kl = per_layer_per_family[layer_key].get(family, {})
                for pt_idx, pt in enumerate(position_types):
                    fam_effects[layer_idx, pt_idx] = fam_kl.get(pt, 0.0)

        dominant_per_layer = {}
        for layer_idx in range(n_layers):
            row = fam_effects[layer_idx, :]
            if row.max() > 0:
                dominant_per_layer[f"L{layer_idx:02d}"] = position_types[row.argmax()]
            else:
                dominant_per_layer[f"L{layer_idx:02d}"] = "none"

        per_family_summary[family] = {
            "effect_matrix": fam_effects.tolist(),
            "dominant_per_layer": dominant_per_layer,
            "mean_by_type": {pt: float(fam_effects[:, i].mean())
                            for i, pt in enumerate(position_types)},
        }
        max_type = max(per_family_summary[family]["mean_by_type"],
                      key=per_family_summary[family]["mean_by_type"].get)
        log(f"  {family:>25s}: dominant={max_type}")

    # Assemble results
    all_results = {
        "experiment": "phase3_position_ablation_all_layers",
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "n_layers": n_layers,
        "position_types": position_types,
        "families": families,
        "timestamp": now_iso(),
        "effect_matrix": effect_matrix.tolist(),
        "dominance": dominance,
        "aggregate": aggregate,
        "per_layer_per_family": per_layer_per_family,
        "per_family_summary": per_family_summary,
        "summary": {
            "n_layers": n_layers,
            "n_examples": len(all_examples),
            "n_families": len(families),
            "dominant_type_by_layer": {
                k: v["dominant_type"] for k, v in dominance.items()
            },
            "position_type_ranking": sorted(
                aggregate.keys(),
                key=lambda x: aggregate[x]["mean_effect"],
                reverse=True,
            ),
            "elapsed_seconds": round(time.time() - start_time, 1),
        },
    }

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_position_ablation_{model_slug}.json"
    save_json(all_results, output_path)
    log(f"\nResults saved to {output_path}")

    # Register experiment
    ranking = all_results["summary"]["position_type_ranking"]
    register_experiment(
        type="phase3_position_ablation",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(output_path)],
        status="success",
        summary=f"Phase 3 position ablation: {n_layers} layers x "
                f"{len(position_types)} position types x {len(families)} families. "
                f"Position ranking: {' > '.join(ranking)}",
        key_metrics={
            "position_ranking": ranking,
            "aggregate_effects": {
                pt: aggregate[pt]["mean_effect"] for pt in position_types
            },
        },
        next="Compare with full ablation results, investigate position-type circuits",
    )

    elapsed = time.time() - start_time
    log(f"\nPosition ablation complete in {elapsed:.0f}s")
    log(f"Position ranking: {' > '.join(ranking)}")


if __name__ == "__main__":
    main()
