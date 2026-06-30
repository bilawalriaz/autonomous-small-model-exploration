#!/usr/bin/env python3
"""run_lfm2_layer_atlas.py — LFM2.5-230M Layer Ablation + MLP Ablation + Steering Sweep.

Adapted from the MI-Atlas pipeline for the LFM2.5-230M hybrid architecture
(conv + attention alternating layers).

Runs THREE experiments:
  1. ZERO ablation on ALL 14 layers (residual stream)
  2. MLP ablation on ALL 14 layers (feed_forward output)
  3. Steering sweeps on ALL 14 layers with strengths [-4, -2, -1, -0.5, 0.5, 1, 2, 4]

Architecture:
  L0:  CONV   L1:  CONV   L2:  ATTN   L3:  CONV   L4:  ATTN   L5:  CONV
  L6:  ATTN   L7:  CONV   L8:  ATTN   L9:  CONV   L10: ATTN   L11: CONV
  L12: ATTN   L13: CONV

Usage:
    python -u scripts/run_lfm2_layer_atlas.py
    python -u scripts/run_lfm2_layer_atlas.py --model LiquidAI/LFM2.5-230M --force
    python -u scripts/run_lfm2_layer_atlas.py --seed 137
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent

# LFM2.5-230M architecture constants
NUM_LAYERS = 14
HIDDEN_SIZE = 1024
VOCAB_SIZE = 65536
LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv", "full_attention", "conv",
]
CONV_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "conv"]
ATTN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def git_commit_hash():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  Saved: {path}")


def compute_kl(logits_a, logits_b):
    """KL divergence between last-token distributions."""
    probs_a = torch.softmax(logits_a[0, -1, :].float(), dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :].float(), dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b + 1e-10), probs_a, reduction="sum"
    ).item()


def compute_top1_agreement(logits_a, logits_b):
    """Check if top-1 token agrees between two logit distributions."""
    top_a = logits_a[0, -1, :].argmax().item()
    top_b = logits_b[0, -1, :].argmax().item()
    return top_a == top_b


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Get activation at a specific layer using forward hook."""
    activation = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    return activation.get("value")


# ============ TASK LOADING ============

def load_tasks(max_per_family=5):
    """Load task suite from data/tasks/canonical_short/ or use inline fallback."""
    task_dir = PROJECT_ROOT / "data" / "tasks" / "canonical_short"
    prompts_by_family = {}

    if task_dir.exists():
        for task_file in sorted(task_dir.glob("*.json")):
            if task_file.name in ("tasks.json",):
                continue
            try:
                with open(task_file) as f:
                    data = json.load(f)
                family = data.get("family", task_file.stem)
                examples = data.get("examples", [])
                if not examples:
                    continue
                prompts = []
                for ex in examples[:max_per_family]:
                    p = ex.get("prompt", ex.get("clean_prompt", ""))
                    t = ex.get("target", ex.get("target_token", ""))
                    if p:
                        prompts.append({"prompt": p, "target": t})
                if prompts:
                    prompts_by_family[family] = prompts
            except Exception as e:
                print(f"  Warning: failed to load {task_file}: {e}")

    # Fallback inline tasks
    if not prompts_by_family:
        print("  Using inline fallback tasks")
        prompts_by_family = {
            "factual_recall": [
                {"prompt": "The capital of France is ", "target": "Paris"},
                {"prompt": "The capital of Germany is ", "target": "Berlin"},
                {"prompt": "The largest planet is ", "target": "Jupiter"},
            ],
            "arithmetic": [
                {"prompt": "2 + 3 = ", "target": "5"},
                {"prompt": "10 - 4 = ", "target": "6"},
                {"prompt": "7 * 8 = ", "target": "56"},
            ],
            "copying": [
                {"prompt": "Hello Hello Hello ", "target": "Hello"},
                {"prompt": "abc abc abc ", "target": "abc"},
            ],
            "code_syntax": [
                {"prompt": "def hello():\n    ", "target": "print"},
                {"prompt": "for i in range(10):\n    ", "target": "print"},
            ],
            "json_schema": [
                {"prompt": '{"name": "Alice", "age": ', "target": "30"},
                {"prompt": '{"city": "London", "country": "', "target": "UK"},
            ],
        }

    return prompts_by_family


# ============ EXPERIMENT 1: LAYER ZERO ABLATION ============

def run_layer_zero_ablation(model, tokenizer, prompts_by_family, device):
    """Zero-ablate residual stream output at each of 14 layers."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: LAYER ZERO ABLATION (all 14 layers)")
    print("=" * 60)

    results = {}

    for family, prompts in prompts_by_family.items():
        print(f"\n  Family: {family} ({len(prompts)} prompts)")
        family_kls = []
        family_agreement = []

        for pi, p in enumerate(prompts):
            prompt = p["prompt"]
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            # Baseline
            with torch.no_grad():
                baseline_logits = model(ids).logits

            layer_kls = []
            layer_agree = []

            for layer_idx in range(NUM_LAYERS):
                def ablate_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                handle = model.model.layers[layer_idx].register_forward_hook(ablate_hook)
                with torch.no_grad():
                    abl_logits = model(ids).logits
                handle.remove()

                kl = compute_kl(baseline_logits, abl_logits)
                agree = compute_top1_agreement(baseline_logits, abl_logits)
                layer_kls.append(kl)
                layer_agree.append(agree)

            family_kls.append(layer_kls)
            family_agreement.append(layer_agree)

            if pi == 0:
                # Print first prompt's results
                top3 = sorted(range(NUM_LAYERS), key=lambda i: layer_kls[i], reverse=True)[:3]
                print(f"    Prompt[0] top3 layers: {[(f'L{l}({LAYER_TYPES[l][:1]})', round(layer_kls[l], 2)) for l in top3]}")

        # Average across prompts
        mean_kls = np.mean(family_kls, axis=0).tolist()
        mean_agree = np.mean(family_agreement, axis=0).tolist()
        results[family] = {
            "kl_per_layer": [round(x, 4) for x in mean_kls],
            "top1_agreement": [round(x, 4) for x in mean_agree],
        }

    # Summary across all families
    all_family_kls = np.array([results[f]["kl_per_layer"] for f in results])
    mean_across_families = np.mean(all_family_kls, axis=0)
    top_layers = sorted(range(NUM_LAYERS), key=lambda i: mean_across_families[i], reverse=True)

    output = {
        "experiment": "layer_zero_ablation",
        "model": "LiquidAI/LFM2.5-230M",
        "n_layers": NUM_LAYERS,
        "layer_types": LAYER_TYPES,
        "families": list(results.keys()),
        "per_family": results,
        "mean_kl_per_layer": [round(float(x), 4) for x in mean_across_families],
        "top_layers": top_layers[:7],
        "top_layer_info": [
            {"layer": l, "type": LAYER_TYPES[l], "mean_kl": round(float(mean_across_families[l]), 4)}
            for l in top_layers[:7]
        ],
    }

    print(f"\n  SUMMARY — Top hub layers by mean KL:")
    for info in output["top_layer_info"]:
        marker = " ***" if info["mean_kl"] > 5.0 else ""
        print(f"    L{info['layer']:2d} ({info['type']:5s}): KL={info['mean_kl']:.4f}{marker}")

    return output


# ============ EXPERIMENT 2: MLP ZERO ABLATION ============

def run_mlp_ablation(model, tokenizer, prompts_by_family, device):
    """Zero-ablate MLP (feed_forward) output at each of 14 layers."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: MLP ZERO ABLATION (all 14 layers)")
    print("=" * 60)

    results = {}

    for family, prompts in prompts_by_family.items():
        print(f"\n  Family: {family} ({len(prompts)} prompts)")
        family_kls = []

        for pi, p in enumerate(prompts):
            prompt = p["prompt"]
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                baseline_logits = model(ids).logits

            layer_kls = []
            for layer_idx in range(NUM_LAYERS):
                layer = model.model.layers[layer_idx]
                mlp = layer.feed_forward

                def ablate_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                handle = mlp.register_forward_hook(ablate_hook)
                with torch.no_grad():
                    abl_logits = model(ids).logits
                handle.remove()

                kl = compute_kl(baseline_logits, abl_logits)
                layer_kls.append(kl)

            family_kls.append(layer_kls)

            if pi == 0:
                top3 = sorted(range(NUM_LAYERS), key=lambda i: layer_kls[i], reverse=True)[:3]
                print(f"    Prompt[0] top3 MLP layers: {[(f'L{l}({LAYER_TYPES[l][:1]})', round(layer_kls[l], 2)) for l in top3]}")

        mean_kls = np.mean(family_kls, axis=0).tolist()
        results[family] = {"kl_per_layer": [round(x, 4) for x in mean_kls]}

    # Summary
    all_family_kls = np.array([results[f]["kl_per_layer"] for f in results])
    mean_across_families = np.mean(all_family_kls, axis=0)
    top_layers = sorted(range(NUM_LAYERS), key=lambda i: mean_across_families[i], reverse=True)

    output = {
        "experiment": "mlp_zero_ablation",
        "model": "LiquidAI/LFM2.5-230M",
        "n_layers": NUM_LAYERS,
        "layer_types": LAYER_TYPES,
        "families": list(results.keys()),
        "per_family": results,
        "mean_kl_per_layer": [round(float(x), 4) for x in mean_across_families],
        "top_layers": top_layers[:7],
        "top_layer_info": [
            {"layer": l, "type": LAYER_TYPES[l], "mean_kl": round(float(mean_across_families[l]), 4)}
            for l in top_layers[:7]
        ],
    }

    print(f"\n  SUMMARY — Top MLP hub layers by mean KL:")
    for info in output["top_layer_info"]:
        marker = " ***" if info["mean_kl"] > 5.0 else ""
        print(f"    L{info['layer']:2d} ({info['type']:5s}): KL={info['mean_kl']:.4f}{marker}")

    return output


# ============ EXPERIMENT 3: STEERING SWEEP ============

def run_steering_sweep(model, tokenizer, prompts_by_family, device, seed):
    """Steering vector experiments on ALL 14 layers.

    For each layer:
    - Compute steering vector from clean/corrupt pairs in factual_recall
    - Inject at multiple strengths across ALL families
    - Measure KL divergence from baseline
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: STEERING SWEEP (all 14 layers)")
    print("=" * 60)

    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    # Build clean/corrupt pairs from factual_recall
    factual = prompts_by_family.get("factual_recall", [])
    if len(factual) < 4:
        # Inline fallback
        positive_prompts = [
            "The capital of France is ",
            "The capital of Germany is ",
            "The capital of Japan is ",
            "The capital of Italy is ",
        ]
        negative_prompts = [
            "The capital of Australia is ",
            "The capital of Brazil is ",
            "The capital of Canada is ",
            "The capital of Mexico is ",
        ]
    else:
        # Use first half as positive, second half as negative (or corrupt if available)
        half = len(factual) // 2
        positive_prompts = [f["prompt"] for f in factual[:half]]
        negative_prompts = [f["prompt"] for f in factual[half:2*half]]

    print(f"\n  Steering vectors from {len(positive_prompts)} positive / {len(negative_prompts)} negative prompts")

    # Eval prompts from multiple families
    eval_prompts = []
    for family in ["factual_recall", "arithmetic", "json_schema", "copying", "code_syntax"]:
        if family in prompts_by_family:
            for p in prompts_by_family[family][:2]:
                eval_prompts.append({"prompt": p["prompt"], "family": family})

    if not eval_prompts:
        eval_prompts = [
            {"prompt": "The capital of France is ", "family": "factual_recall"},
            {"prompt": "2 + 3 = ", "family": "arithmetic"},
        ]

    # Compute steering vectors for each layer
    steering_vectors = {}
    for layer_idx in range(NUM_LAYERS):
        pos_acts = []
        neg_acts = []
        for prompt in positive_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx, position=-1)
            if act is not None:
                pos_acts.append(act.cpu())

        for prompt in negative_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx, position=-1)
            if act is not None:
                neg_acts.append(act.cpu())

        if pos_acts and neg_acts:
            sv = (torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)).squeeze(0)
            sv_norm = sv.norm().item()
            steering_vectors[layer_idx] = {"sv": sv, "norm": sv_norm}
            print(f"  L{layer_idx:2d} ({LAYER_TYPES[layer_idx]:5s}): sv_norm={sv_norm:.4f}")
        else:
            print(f"  L{layer_idx:2d} ({LAYER_TYPES[layer_idx]:5s}): no activations collected")

    # Run sweep
    print(f"\n  Running sweep across {len(eval_prompts)} eval prompts, {len(strengths)} strengths...")
    all_results = {}

    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in steering_vectors:
            continue

        sv = steering_vectors[layer_idx]["sv"]
        sv_norm = steering_vectors[layer_idx]["norm"]

        layer_results = {
            "layer": layer_idx,
            "type": LAYER_TYPES[layer_idx],
            "sv_norm": round(sv_norm, 6),
            "per_strength": {},
        }

        for strength in strengths:
            kl_values = []
            agree_values = []

            for ep in eval_prompts:
                ids = tokenizer(ep["prompt"], return_tensors="pt", truncation=True,
                                max_length=512)["input_ids"].to(device)

                # Baseline
                with torch.no_grad():
                    baseline_logits = model(ids).logits

                # Steered
                sv_scaled = sv.to(device).float() * strength

                def make_hook(sv_s):
                    def steer_hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                            hidden[:, -1, :] += sv_s
                            return (hidden,) + output[1:]
                        hidden = output.clone()
                        hidden[:, -1, :] += sv_s
                        return hidden
                    return steer_hook

                handle = model.model.layers[layer_idx].register_forward_hook(make_hook(sv_scaled))
                with torch.no_grad():
                    steered_logits = model(ids).logits
                handle.remove()

                kl = compute_kl(baseline_logits, steered_logits)
                agree = compute_top1_agreement(baseline_logits, steered_logits)
                kl_values.append(kl)
                agree_values.append(agree)

            layer_results["per_strength"][str(strength)] = {
                "mean_kl": round(float(np.mean(kl_values)), 6),
                "std_kl": round(float(np.std(kl_values)), 6),
                "mean_top1_agree": round(float(np.mean(agree_values)), 4),
                "n_evals": len(kl_values),
            }

        all_results[str(layer_idx)] = layer_results

        # Print summary for this layer
        max_kl = max(all_results[str(layer_idx)]["per_strength"].values(), key=lambda x: x["mean_kl"])
        print(f"  L{layer_idx:2d}: max_kl={max_kl['mean_kl']:.4f} (agree={max_kl['mean_top1_agree']:.2f})")

    # Find best steering configurations
    best_configs = []
    for lid, lr in all_results.items():
        for s, sd in lr["per_strength"].items():
            best_configs.append({
                "layer": int(lid),
                "type": lr["type"],
                "strength": float(s),
                "mean_kl": sd["mean_kl"],
                "top1_agree": sd["mean_top1_agree"],
            })
    best_configs.sort(key=lambda x: x["mean_kl"], reverse=True)

    output = {
        "experiment": "steering_sweep",
        "model": "LiquidAI/LFM2.5-230M",
        "n_layers": NUM_LAYERS,
        "layer_types": LAYER_TYPES,
        "strengths": strengths,
        "n_positive": len(positive_prompts),
        "n_negative": len(negative_prompts),
        "n_eval_prompts": len(eval_prompts),
        "seed": seed,
        "per_layer": all_results,
        "best_configs": best_configs[:10],
    }

    print(f"\n  SUMMARY — Top steering configurations:")
    for bc in best_configs[:7]:
        print(f"    L{bc['layer']:2d} s={bc['strength']:+.1f}: KL={bc['mean_kl']:.4f} (agree={bc['top1_agree']:.2f})")

    return output


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="LFM2.5-230M Layer Atlas")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.split("/")[-1]

    print(f"LFM2.5-230M Layer Atlas")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Timestamp: {timestamp}")
    print(f"Layer types: {LAYER_TYPES}")
    print(f"Conv layers: {CONV_LAYERS}")
    print(f"Attn layers: {ATTN_LAYERS}")

    # Check for existing results
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"lfm2_230m_layer_atlas_seed{args.seed}_{timestamp}.json"

    # Load model
    print(f"\nLoading model {args.model}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s on {device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Vocab: {model.config.vocab_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")

    # Verify layer structure
    print(f"\nVerifying layer structure...")
    for i in range(NUM_LAYERS):
        layer = model.model.layers[i]
        has_ff = hasattr(layer, "feed_forward")
        has_attn = hasattr(layer, "self_attn")
        has_conv = hasattr(layer, "conv")
        print(f"  L{i:2d}: feed_forward={has_ff}, self_attn={has_attn}, conv={has_conv}, type={LAYER_TYPES[i]}")

    # Quick sanity check
    test_prompt = "The capital of France is"
    test_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        test_logits = model(test_ids).logits
    top_token = tokenizer.decode(test_logits[0, -1, :].argmax())
    print(f"\n  Sanity: '{test_prompt}' -> '{top_token}'")

    # Set seed
    set_seed(args.seed)

    # Load tasks
    print("\nLoading tasks...")
    prompts_by_family = load_tasks(max_per_family=5)
    for family, prompts in prompts_by_family.items():
        print(f"  {family}: {len(prompts)} prompts")

    # Run experiments
    print("\n" + "#" * 60)
    print(f"Running all experiments (seed={args.seed})")
    print("#" * 60)

    t_start = time.time()

    # Experiment 1: Layer zero ablation
    set_seed(args.seed)
    layer_ablation_results = run_layer_zero_ablation(model, tokenizer, prompts_by_family, device)

    # Experiment 2: MLP ablation
    set_seed(args.seed)
    mlp_ablation_results = run_mlp_ablation(model, tokenizer, prompts_by_family, device)

    # Experiment 3: Steering sweep
    set_seed(args.seed)
    steering_results = run_steering_sweep(model, tokenizer, prompts_by_family, device, args.seed)

    total_time = time.time() - t_start

    # Compile final results
    final_results = {
        "experiment": "lfm2_230m_layer_atlas",
        "model": args.model,
        "seed": args.seed,
        "timestamp": now_iso(),
        "git_commit": git_commit_hash(),
        "total_time_seconds": round(total_time, 1),
        "architecture": {
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "layer_types": LAYER_TYPES,
            "conv_layers": CONV_LAYERS,
            "attn_layers": ATTN_LAYERS,
        },
        "tasks": {f: len(p) for f, p in prompts_by_family.items()},
        "layer_zero_ablation": layer_ablation_results,
        "mlp_ablation": mlp_ablation_results,
        "steering_sweep": steering_results,
    }

    save_json(final_results, out_path)

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: LFM2.5-230M LAYER ATLAS")
    print("=" * 60)

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Results: {out_path}")

    print("\n--- LAYER ZERO ABLATION HUBS ---")
    for info in layer_ablation_results["top_layer_info"]:
        marker = " ***HUB***" if info["mean_kl"] > 5.0 else ""
        print(f"  L{info['layer']:2d} ({info['type']:5s}): mean_KL={info['mean_kl']:.4f}{marker}")

    print("\n--- MLP ABLATION HUBS ---")
    for info in mlp_ablation_results["top_layer_info"]:
        marker = " ***HUB***" if info["mean_kl"] > 5.0 else ""
        print(f"  L{info['layer']:2d} ({info['type']:5s}): mean_KL={info['mean_kl']:.4f}{marker}")

    print("\n--- STEERING SWEEP (top configs) ---")
    for bc in steering_results["best_configs"][:7]:
        print(f"  L{bc['layer']:2d} s={bc['strength']:+.1f}: KL={bc['mean_kl']:.4f} agree={bc['top1_agree']:.2f}")

    # Cross-experiment hub identification
    print("\n--- CROSS-EXPERIMENT HUB IDENTIFICATION ---")
    layer_hub_scores = {}
    for lid in range(NUM_LAYERS):
        score = 0
        # Layer ablation rank
        if lid in layer_ablation_results["top_layers"][:5]:
            rank = layer_ablation_results["top_layers"].index(lid)
            score += (5 - rank)
        # MLP ablation rank
        if lid in mlp_ablation_results["top_layers"][:5]:
            rank = mlp_ablation_results["top_layers"].index(lid)
            score += (5 - rank)
        # Steering rank
        for bc in steering_results["best_configs"][:5]:
            if bc["layer"] == lid:
                score += 1
        layer_hub_scores[lid] = score

    hub_ranking = sorted(range(NUM_LAYERS), key=lambda i: layer_hub_scores[i], reverse=True)
    for rank, lid in enumerate(hub_ranking):
        ltype = LAYER_TYPES[lid]
        lay_kl = layer_ablation_results["mean_kl_per_layer"][lid]
        mlp_kl = mlp_ablation_results["mean_kl_per_layer"][lid]
        hub = "***HUB***" if layer_hub_scores[lid] >= 5 else ""
        print(f"  #{rank+1} L{lid:2d} ({ltype:5s}): score={layer_hub_scores[lid]}, "
              f"lay_KL={lay_kl:.2f}, mlp_KL={mlp_kl:.2f} {hub}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
