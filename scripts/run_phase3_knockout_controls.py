#!/usr/bin/env python3
"""Phase 3: Knockout controls — random-vector and shuffled-label baselines.

Tests whether the reported ~11654x selectivity ratio in skill knockout
is real or an artifact. At the 0.5B skill knockout layer (L19), tests
negative steering with:
  (a) Target factual vector (from clean/corrupt pairs)
  (b) Random same-norm vector
  (c) Shuffled-label vector (same pairs, wrong labels)
  (d) Unrelated-task (JSON) vector

Measures suppression of factual recall vs collateral damage to
JSON/copying. If random vectors give similar selectivity, the
11654x finding collapses.

Usage:
    python scripts/run_phase3_knockout_controls.py --model Qwen/Qwen2.5-0.5B
    python scripts/run_phase3_knockout_controls.py --model Qwen/Qwen2.5-1.5B --force
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
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, now_iso, git_commit_hash, PROJECT_ROOT

# Knockout layer per model (from Phase 1-2 skill knockout findings)
KNOCKOUT_LAYERS = {
    "Qwen/Qwen2.5-0.5B": 19,
    "Qwen/Qwen2.5-1.5B": 26,
}


def get_layers(model):
    """Get transformer layers list, handling PeftModel wrapping."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Get activation at a specific layer and position."""
    activation = {}
    layers = get_layers(model)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    return activation.get("value")


def compute_steering_vector(model, tokenizer, positive_prompts, negative_prompts, layer_idx, position=-1):
    """Compute steering vector as mean(positive) - mean(negative)."""
    pos_acts = []
    neg_acts = []

    for prompt in positive_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
        act = get_activation_at_layer(model, ids, layer_idx, position)
        if act is not None:
            pos_acts.append(act.cpu())

    for prompt in negative_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
        act = get_activation_at_layer(model, ids, layer_idx, position)
        if act is not None:
            neg_acts.append(act.cpu())

    if not pos_acts or not neg_acts:
        return None

    mean_pos = torch.stack(pos_acts).mean(dim=0)
    mean_neg = torch.stack(neg_acts).mean(dim=0)
    return (mean_pos - mean_neg).squeeze(0)


def compute_random_vector(hidden_dim, norm, seed=42):
    """Generate a random vector with specified norm."""
    rng = np.random.RandomState(seed)
    v = rng.randn(hidden_dim).astype(np.float32)
    v = v / np.linalg.norm(v) * norm
    return torch.from_numpy(v)


def compute_shuffled_label_vector(model, tokenizer, positive_prompts, negative_prompts, layer_idx, seed=42):
    """Compute steering vector with shuffled clean/corrupt labels."""
    rng = np.random.RandomState(seed)
    n = min(len(positive_prompts), len(negative_prompts))
    if n < 2:
        return None

    # Shuffle the pairing: pair positive[i] with negative[shuffled_i]
    indices = list(range(n))
    rng.shuffle(indices)

    pos_acts = []
    neg_acts = []
    for i in range(n):
        j = indices[i]
        ids_pos = tokenizer(positive_prompts[i], return_tensors="pt",
                            truncation=True, max_length=512)["input_ids"].to(model.device)
        ids_neg = tokenizer(negative_prompts[j], return_tensors="pt",
                            truncation=True, max_length=512)["input_ids"].to(model.device)

        act_pos = get_activation_at_layer(model, ids_pos, layer_idx)
        act_neg = get_activation_at_layer(model, ids_neg, layer_idx)
        if act_pos is not None:
            pos_acts.append(act_pos.cpu())
        if act_neg is not None:
            neg_acts.append(act_neg.cpu())

    if not pos_acts or not neg_acts:
        return None

    sv = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
    return sv.squeeze(0)


def inject_negative_steering(model, tokenizer, prompt, layer_idx, steering_vector, strength, position=-1):
    """Apply negative steering and measure effect on next-token distribution."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)

    # Original logits
    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)

    # Steered
    sv = steering_vector.to(model.device) * strength
    layers = get_layers(model)

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hidden[:, position, :] += sv
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = layers[layer_idx].register_forward_hook(steer_hook)
    with torch.no_grad():
        steered_logits = model(ids).logits
    handle.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)

    # KL divergence
    kl = torch.nn.functional.kl_div(
        torch.log(steered_probs + 1e-10), orig_probs, reduction="sum"
    ).item()

    # Top 5 tokens
    orig_top5 = [(tokenizer.decode([tid.item()]), prob.item())
                 for tid, prob in zip(*torch.topk(orig_probs, 5))]
    steered_top5 = [(tokenizer.decode([tid.item()]), prob.item())
                    for tid, prob in zip(*torch.topk(steered_probs, 5))]

    return {
        "kl_divergence": kl,
        "orig_top5": orig_top5,
        "steered_top5": steered_top5,
        "steered_probs": steered_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 knockout controls")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    knockout_layer = KNOCKOUT_LAYERS.get(args.model, 19)

    print(f"Phase 3: Knockout controls")
    print(f"Model: {args.model}")
    print(f"Knockout layer: L{knockout_layer}")
    print(f"Seed: {args.seed}")

    set_seed(args.seed)

    # Load model
    print("\nLoading model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    hidden_dim = bundle.architecture["d_model"]
    model.eval()

    # Define skill-specific prompts for steering vector computation
    # Factual recall: positive = factual statements, negative = generic text
    factual_positive = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Japan is Tokyo.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
    ]
    factual_negative = [
        "France is a beautiful country in Europe.",
        "Germany has many famous cities.",
        "Japan is an island nation in Asia.",
        "Italy is known for its cuisine.",
        "Spain has wonderful beaches.",
    ]

    # JSON: positive = JSON format, negative = plain text
    json_positive = [
        'Return valid JSON: {"name": "Alice", "age": 31}',
        'Return valid JSON: {"x": 1, "y": 2}',
        '{"name": "Bob", "age": 25}',
        '{"city": "London", "country": "UK"}',
        '{"product": "widget", "price": 9.99}',
    ]
    json_negative = [
        "Tell me about Alice who is 31 years old.",
        "What are the values of x and y?",
        "Describe a person named Bob, age 25.",
        "London is the capital of the United Kingdom.",
        "The widget costs nine dollars and ninety-nine cents.",
    ]

    # Test prompts: factual recall targets (to measure suppression)
    factual_test_prompts = [
        {"prompt": "The capital of Italy is ", "target": " Rome", "skill": "factual"},
        {"prompt": "The capital of Spain is ", "target": " Madrid", "skill": "factual"},
        {"prompt": "The capital of France is ", "target": " Paris", "skill": "factual"},
        {"prompt": "The chemical symbol for gold is ", "target": " Au", "skill": "factual"},
    ]

    # Collateral test prompts: JSON and copying (should NOT be suppressed)
    collateral_test_prompts = [
        {"prompt": 'Return valid JSON: {"city": "', "target": '"', "skill": "json"},
        {"prompt": 'Return valid JSON with keys name and age. Eve is 42.\n', "target": "42", "skill": "json"},
        {"prompt": "Complete: A B C A B ", "target": " C", "skill": "copying"},
    ]

    all_test_prompts = factual_test_prompts + collateral_test_prompts

    strengths = [-0.5, -1.0, -2.0, -4.0, -8.0]

    # Control configurations
    controls = {}

    # (a) Target factual vector
    print("\n  Computing target factual steering vector...")
    sv_target = compute_steering_vector(model, tokenizer, factual_positive, factual_negative, knockout_layer)
    if sv_target is not None:
        target_norm = float(torch.norm(sv_target).item())
        controls["target_factual"] = {"vector": sv_target, "norm": target_norm, "description": "factual steering vector"}
        print(f"    Norm: {target_norm:.4f}")
    else:
        print("    FAILED to compute target vector")
        controls["target_factual"] = {"vector": None, "norm": 0}

    # (b) Random same-norm vector
    if controls["target_factual"]["norm"] > 0:
        sv_random = compute_random_vector(hidden_dim, controls["target_factual"]["norm"], seed=args.seed)
        controls["random_same_norm"] = {"vector": sv_random, "norm": float(torch.norm(sv_random).item()),
                                         "description": "random vector with same norm"}
        print(f"  Random vector norm: {controls['random_same_norm']['norm']:.4f}")

    # (c) Shuffled-label vector
    sv_shuffled = compute_shuffled_label_vector(model, tokenizer, factual_positive, factual_negative, knockout_layer, seed=args.seed)
    if sv_shuffled is not None:
        controls["shuffled_label"] = {"vector": sv_shuffled, "norm": float(torch.norm(sv_shuffled).item()),
                                       "description": "shuffled-label steering vector"}
        print(f"  Shuffled-label vector norm: {controls['shuffled_label']['norm']:.4f}")
    else:
        controls["shuffled_label"] = {"vector": None, "norm": 0}

    # (d) Unrelated-task (JSON) vector
    sv_json = compute_steering_vector(model, tokenizer, json_positive, json_negative, knockout_layer)
    if sv_json is not None:
        controls["unrelated_json"] = {"vector": sv_json, "norm": float(torch.norm(sv_json).item()),
                                       "description": "JSON steering vector (unrelated task)"}
        print(f"  Unrelated-task (JSON) vector norm: {controls['unrelated_json']['norm']:.4f}")
    else:
        controls["unrelated_json"] = {"vector": None, "norm": 0}

    # Run each control through negative steering sweep
    all_control_results = {}

    for control_name, control_info in controls.items():
        if control_info["vector"] is None:
            all_control_results[control_name] = {"description": control_info["description"],
                                                  "error": "vector computation failed"}
            continue

        print(f"\n  === Control: {control_name} ===")
        sv = control_info["vector"]

        control_results = {
            "description": control_info["description"],
            "norm": control_info["norm"],
            "strengths": {},
        }

        for strength in strengths:
            strength_results = {"prompts": []}

            for tp in all_test_prompts:
                prompt = tp["prompt"]
                target = tp["target"]
                skill = tp["skill"]

                result = inject_negative_steering(model, tokenizer, prompt, knockout_layer, sv, strength)

                # Get target token prob
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                target_id = target_ids[0] if len(target_ids) > 0 else 0
                target_prob = result["steered_probs"][target_id].item()

                # Baseline prob
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
                with torch.no_grad():
                    baseline_logits = model(ids).logits
                baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
                baseline_target_prob = baseline_probs[target_id].item()

                strength_results["prompts"].append({
                    "prompt": prompt[:80],
                    "skill": skill,
                    "target": target,
                    "baseline_target_prob": round(baseline_target_prob, 6),
                    "steered_target_prob": round(target_prob, 6),
                    "prob_drop": round(baseline_target_prob - target_prob, 6),
                    "kl": round(result["kl_divergence"], 6),
                })

            # Aggregate by skill
            factual_drops = [p["prob_drop"] for p in strength_results["prompts"] if p["skill"] == "factual"]
            json_drops = [p["prob_drop"] for p in strength_results["prompts"] if p["skill"] == "json"]
            copying_drops = [p["prob_drop"] for p in strength_results["prompts"] if p["skill"] == "copying"]

            mean_factual_drop = float(np.mean(factual_drops)) if factual_drops else 0.0
            mean_collateral_drop = float(np.mean(json_drops + copying_drops)) if (json_drops + copying_drops) else 0.0
            selectivity_ratio = mean_factual_drop / max(abs(mean_collateral_drop), 1e-8)

            strength_results["aggregate"] = {
                "mean_factual_drop": round(mean_factual_drop, 6),
                "mean_json_drop": round(float(np.mean(json_drops)), 6) if json_drops else 0.0,
                "mean_copying_drop": round(float(np.mean(copying_drops)), 6) if copying_drops else 0.0,
                "mean_collateral_drop": round(mean_collateral_drop, 6),
                "selectivity_ratio": round(selectivity_ratio, 2),
            }

            control_results["strengths"][str(strength)] = strength_results

            if strength in [-2.0, -4.0]:
                print(f"    s={strength:+.1f}: factual_drop={mean_factual_drop:.4f}, "
                      f"collateral_drop={mean_collateral_drop:.4f}, "
                      f"selectivity={selectivity_ratio:.1f}x")

        all_control_results[control_name] = control_results

    # Final comparison
    print(f"\n{'='*60}")
    print("  KNOCKOUT CONTROL COMPARISON")
    print(f"{'='*60}")

    comparison = {}
    for control_name, control_data in all_control_results.items():
        if "error" in control_data:
            comparison[control_name] = {"error": control_data["error"]}
            continue

        # Find the strength with highest selectivity
        best_selectivity = 0
        best_strength = None
        for s_key, s_data in control_data.get("strengths", {}).items():
            sel = s_data.get("aggregate", {}).get("selectivity_ratio", 0)
            if sel > best_selectivity:
                best_selectivity = sel
                best_strength = s_key

        comparison[control_name] = {
            "description": control_data["description"],
            "best_selectivity_ratio": round(best_selectivity, 2),
            "best_strength": best_strength,
            "norm": control_data.get("norm", 0),
        }
        print(f"  {control_name:20s}: best_selectivity={best_selectivity:.1f}x at s={best_strength}")

    # Is the finding real?
    target_sel = comparison.get("target_factual", {}).get("best_selectivity_ratio", 0)
    random_sel = comparison.get("random_same_norm", {}).get("best_selectivity_ratio", 0)
    shuffled_sel = comparison.get("shuffled_label", {}).get("best_selectivity_ratio", 0)

    if random_sel > 0:
        specificity_over_random = target_sel / max(random_sel, 0.01)
    else:
        specificity_over_random = float('inf')

    finding_verdict = "task_specific" if specificity_over_random > 2.0 else "artifact_or_nonspecific"
    print(f"\n  Target selectivity: {target_sel:.1f}x")
    print(f"  Random selectivity: {random_sel:.1f}x")
    print(f"  Shuffled selectivity: {shuffled_sel:.1f}x")
    print(f"  Specificity over random: {specificity_over_random:.1f}x")
    print(f"  VERDICT: {finding_verdict}")

    # Save
    summary = {
        "experiment": "phase3_knockout_controls",
        "model": args.model,
        "seed": args.seed,
        "knockout_layer": knockout_layer,
        "hidden_dim": hidden_dim,
        "strengths_tested": strengths,
        "controls": all_control_results,
        "comparison": comparison,
        "target_selectivity": target_sel,
        "random_selectivity": random_sel,
        "shuffled_selectivity": shuffled_sel,
        "specificity_over_random": round(specificity_over_random, 2),
        "verdict": finding_verdict,
        "timestamp": now_iso(),
        "git_commit": git_commit_hash(),
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_knockout_controls_{model_slug}.json"
    save_json(summary, out_path)
    print(f"\n  Results: {out_path}")

    # Register
    register_experiment(
        type="control",
        model=args.model,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(out_path)],
        status="success",
        summary=f"Knockout controls at L{knockout_layer}: "
                f"target_sel={target_sel:.1f}x, random_sel={random_sel:.1f}x, "
                f"specificity={specificity_over_random:.1f}x, verdict={finding_verdict}",
        next="If task_specific: result is real. If artifact: re-examine skill knockout methodology.",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
