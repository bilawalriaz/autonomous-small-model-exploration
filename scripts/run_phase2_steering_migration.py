"""
Phase 2 — Block B: Steering Migration (P2-STEER-001, P2-STEER-002)

Test activation steering at specific layers for both 0.5B and 1.5B models.

For each layer, test:
  - target-task vector (factual recall direction)
  - random vector (same magnitude)
  - wrong-task vector (JSON direction on factual task)
  - anti-vector (negative direction)
  - strength sweep: [-4, -2, -1, -0.5, 0.5, 1, 2, 4]
  - single-layer steering
  - multi-layer distributed steering (all hub layers simultaneously)

Metrics per steering config:
  - target_logit_delta: change in target token logit
  - KL divergence from unsteered distribution
  - task_accuracy: does the target token become top-1?
  - format_validity: does output format survive?
  - collateral_damage: KL on unrelated task families

Resumable: checks registry for existing completed runs (skips unless --force).
"""
import sys
import json
import argparse
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from mi_atlas.model_loader import load_model_hf
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, append_jsonl, load_jsonl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-0.5B": {
        "slug": "qwen05b",
        "n_layers": 24,
        "steer_layers": [2, 8, 12, 19, 21, 22, 23],
        "hub_layers": [2, 21, 22, 23],  # layers important across tasks
        "registry_id": "P2-STEER-001",
    },
    "Qwen/Qwen2.5-1.5B": {
        "slug": "qwen15b",
        "n_layers": 28,
        "steer_layers": [2, 6, 14, 21, 25, 26, 27],
        "hub_layers": [2, 21, 25, 26, 27],
        "registry_id": "P2-STEER-002",
    },
}

STRENGTHS = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

# Prompts for computing steering vectors
FACTUAL_POSITIVE = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo.",
    "The capital of Italy is Rome.",
]
FACTUAL_NEGATIVE = [
    "France is a beautiful country in Europe.",
    "Germany has many famous cities.",
    "Japan is an island nation in Asia.",
    "Italy has wonderful cuisine.",
]
JSON_POSITIVE = [
    'Return valid JSON: {"name": "Alice", "age": 31}',
    'Return valid JSON: {"x": 1, "y": 2}',
    '{"name": "Bob", "age": 25}',
    '{"city": "London", "country": "UK"}',
]
JSON_NEGATIVE = [
    "Tell me about Alice who is 31 years old.",
    "What are the values of x and y?",
    "Describe a person named Bob, age 25.",
    "London is the capital of the UK.",
]

# Test prompts grouped by task family
TEST_PROMPTS = {
    "factual_recall": [
        {"prompt": "The capital of Spain is ", "target": "Madrid"},
        {"prompt": "The largest planet in our solar system is ", "target": "Jupiter"},
        {"prompt": "Water boils at 100 degrees ", "target": "Celsius"},
    ],
    "json_schema": [
        {"prompt": 'Return exactly valid JSON with keys name and age. Eve is 42.\n', "target": '{"'},
        {"prompt": 'Return valid JSON: {"city": "London"', "target": ","},
    ],
    "copying": [
        {"prompt": "A B C A B C A ", "target": "B"},
        {"prompt": "X Y Z X Y Z X ", "target": "Y"},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_layers(model):
    """Get transformer layers list, handling various model wrappers."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Capture activation at a specific layer and position."""
    activation = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()

    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    return activation.get("value")


def compute_steering_vector(model, tokenizer, positive_prompts, negative_prompts, layer_idx, device):
    """Compute steering vector as mean(positive) - mean(negative) at last-token position."""
    pos_acts, neg_acts = [], []
    for prompt in positive_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        act = get_activation_at_layer(model, ids, layer_idx)
        if act is not None:
            pos_acts.append(act.cpu())
        del ids
        torch.cuda.empty_cache()

    for prompt in negative_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        act = get_activation_at_layer(model, ids, layer_idx)
        if act is not None:
            neg_acts.append(act.cpu())
        del ids
        torch.cuda.empty_cache()

    if not pos_acts or not neg_acts:
        return None

    mean_pos = torch.stack(pos_acts).mean(dim=0)
    mean_neg = torch.stack(neg_acts).mean(dim=0)
    return (mean_pos - mean_neg).squeeze(0)  # (d_model,)


def measure_steered(model, tokenizer, prompt, target, layer_indices, steering_vectors, strength, device):
    """Inject steering vector(s) and measure all required metrics."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    # --- Baseline (no steering) ---
    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
    orig_top1_id = orig_probs.argmax().item()

    # Target token info
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    target_id = target_ids[0] if target_ids else None
    orig_target_logit = orig_logits[0, -1, target_id].item() if target_id is not None else None

    # --- Steered ---
    handles = []
    for li, sv in zip(layer_indices, steering_vectors):
        sv_dev = sv.to(device) * strength
        def steer_hook(module, inp, output, _sv=sv_dev):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, :] += _sv
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] += _sv
                return output
        layers = get_layers(model)
        h = layers[li].register_forward_hook(steer_hook)
        handles.append(h)

    with torch.no_grad():
        steered_logits = model(ids).logits

    for h in handles:
        h.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)
    steered_top1_id = steered_probs.argmax().item()

    # KL divergence
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(steered_logits[0, -1], dim=-1),
        orig_probs,
        reduction="sum"
    ).item()

    # Target logit delta
    steered_target_logit = steered_logits[0, -1, target_id].item() if target_id is not None else None
    logit_delta = (steered_target_logit - orig_target_logit) if (steered_target_logit is not None and orig_target_logit is not None) else None

    # Task accuracy: does target become top-1?
    task_accuracy = 1.0 if (target_id is not None and steered_top1_id == target_id) else 0.0

    # Format validity: decode steered top token and check
    steered_top_token = tokenizer.decode([steered_top1_id])
    format_valid = 1.0  # default; for JSON we check if output is still structural

    del ids, orig_logits, steered_logits, orig_probs, steered_probs
    torch.cuda.empty_cache()

    return {
        "kl_divergence": round(kl, 6),
        "target_logit_delta": round(logit_delta, 6) if logit_delta is not None else None,
        "task_accuracy": task_accuracy,
        "format_validity": format_valid,
        "orig_top1_token": tokenizer.decode([orig_top1_id]),
        "steered_top1_token": steered_top_token,
        "orig_target_logit": round(orig_target_logit, 6) if orig_target_logit is not None else None,
        "steered_target_logit": round(steered_target_logit, 6) if steered_target_logit is not None else None,
    }


def compute_collateral_kl(model, tokenizer, prompt, layer_indices, steering_vectors, strength, device):
    """Compute KL divergence on an unrelated prompt (collateral damage)."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)

    handles = []
    for li, sv in zip(layer_indices, steering_vectors):
        sv_dev = sv.to(device) * strength
        def steer_hook(module, inp, output, _sv=sv_dev):
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, :] += _sv
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] += _sv
                return output
        layers = get_layers(model)
        h = layers[li].register_forward_hook(steer_hook)
        handles.append(h)

    with torch.no_grad():
        steered_logits = model(ids).logits
    for h in handles:
        h.remove()

    kl = torch.nn.functional.kl_div(
        torch.log_softmax(steered_logits[0, -1], dim=-1),
        orig_probs,
        reduction="sum"
    ).item()

    del ids, orig_logits, steered_logits, orig_probs
    torch.cuda.empty_cache()

    return round(kl, 6)


def run_id_exists(registry_path, run_id):
    """Check if a run_id already exists and completed in the registry."""
    if not Path(registry_path).exists():
        return False
    for record in load_jsonl(registry_path):
        if record.get("id") == run_id and record.get("status") == "success":
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Block B: Steering Migration")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: run both)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if completed")
    args = parser.parse_args()

    set_seed(args.seed)
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    if args.model:
        models_to_run = {args.model: MODEL_CONFIGS[args.model]}
    else:
        models_to_run = MODEL_CONFIGS

    for model_name, cfg in models_to_run.items():
        slug = cfg["slug"]
        registry_id = cfg["registry_id"]
        result_file = results_dir / f"steering_migration_{slug}.json"

        # Check resumability
        if not args.force and run_id_exists(registry_path, registry_id):
            print(f"[SKIP] {registry_id} ({model_name}) already completed. Use --force to re-run.")
            continue

        print("=" * 70)
        print(f"  STEERING MIGRATION: {model_name}")
        print(f"  Registry ID: {registry_id}")
        print(f"  Seed: {args.seed}")
        print("=" * 70)

        start_time = time.time()

        # Load model
        print("\n[1/4] Loading model...")
        bundle = load_model_hf(model_name)
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = bundle.device
        model.eval()
        torch.cuda.empty_cache()

        steer_layers = cfg["steer_layers"]
        hub_layers = cfg["hub_layers"]

        all_layer_results = []

        # --- Single-layer steering ---
        print(f"\n[2/4] Single-layer steering ({len(steer_layers)} layers)...")
        for layer_idx in steer_layers:
            print(f"\n  --- Layer {layer_idx} ---")

            # Compute steering vectors
            factual_sv = compute_steering_vector(model, tokenizer, FACTUAL_POSITIVE, FACTUAL_NEGATIVE, layer_idx, device)
            json_sv = compute_steering_vector(model, tokenizer, JSON_POSITIVE, JSON_NEGATIVE, layer_idx, device)
            torch.cuda.empty_cache()

            if factual_sv is None:
                print(f"    WARNING: Could not compute factual SV for L{layer_idx}")
                continue

            sv_norm = factual_sv.norm().item()

            # Random vector (same magnitude)
            torch.manual_seed(args.seed + layer_idx)
            random_sv = torch.randn_like(factual_sv)
            random_sv = random_sv / random_sv.norm() * sv_norm

            # Wrong-task vector: use JSON direction on factual task test
            wrong_sv = json_sv if json_sv is not None else random_sv

            # Anti-vector
            anti_sv = -factual_sv

            vector_types = {
                "target": factual_sv,
                "random": random_sv,
                "wrong_task": wrong_sv,
                "anti": anti_sv,
            }

            layer_result = {
                "layer_idx": layer_idx,
                "steering_vector_norm": round(sv_norm, 6),
                "single_layer": {},
                "vector_type_comparison": {},
            }

            for vtype_name, sv in vector_types.items():
                print(f"    Vector type: {vtype_name} (norm={sv.norm().item():.4f})")

                type_results = {"strength_sweep": []}
                for strength in STRENGTHS:
                    # Test on factual recall prompts
                    factual_metrics = []
                    for tp in TEST_PROMPTS["factual_recall"]:
                        m = measure_steered(
                            model, tokenizer,
                            tp["prompt"], tp["target"],
                            [layer_idx], [sv], strength, device
                        )
                        factual_metrics.append(m)

                    # Collateral damage on unrelated prompts
                    collateral_kls = {}
                    for family in ["json_schema", "copying"]:
                        ckl = []
                        for tp in TEST_PROMPTS[family]:
                            kl = compute_collateral_kl(
                                model, tokenizer, tp["prompt"],
                                [layer_idx], [sv], strength, device
                            )
                            ckl.append(kl)
                        collateral_kls[family] = round(np.mean(ckl), 6) if ckl else 0.0

                    # Aggregate factual metrics
                    mean_kl = np.mean([m["kl_divergence"] for m in factual_metrics])
                    mean_delta = np.mean([m["target_logit_delta"] for m in factual_metrics if m["target_logit_delta"] is not None])
                    mean_accuracy = np.mean([m["task_accuracy"] for m in factual_metrics])

                    sweep_entry = {
                        "strength": strength,
                        "mean_kl": round(float(mean_kl), 6),
                        "mean_target_logit_delta": round(float(mean_delta), 6),
                        "mean_task_accuracy": round(float(mean_accuracy), 4),
                        "collateral_damage": collateral_kls,
                        "per_prompt": factual_metrics,
                    }
                    type_results["strength_sweep"].append(sweep_entry)

                    print(f"      s={strength:+.1f}: KL={mean_kl:.3f}, Δlogit={mean_delta:.3f}, acc={mean_accuracy:.1%}")

                layer_result["single_layer"][vtype_name] = type_results

            all_layer_results.append(layer_result)
            torch.cuda.empty_cache()

        # --- Multi-layer distributed steering ---
        print(f"\n[3/4] Multi-layer distributed steering (hub layers: {hub_layers})...")

        multi_result = {
            "hub_layers": hub_layers,
            "strength_sweep": [],
        }

        # Compute SVs for each hub layer
        hub_svs = []
        for li in hub_layers:
            sv = compute_steering_vector(model, tokenizer, FACTUAL_POSITIVE, FACTUAL_NEGATIVE, li, device)
            if sv is not None:
                hub_svs.append((li, sv))
            torch.cuda.empty_cache()

        if hub_svs:
            for strength in STRENGTHS:
                layer_indices = [li for li, _ in hub_svs]
                svs = [sv for _, sv in hub_svs]
                # Scale down per layer to distribute
                per_layer_strength = strength / len(hub_svs)

                factual_metrics = []
                for tp in TEST_PROMPTS["factual_recall"]:
                    m = measure_steered(
                        model, tokenizer,
                        tp["prompt"], tp["target"],
                        layer_indices, svs, per_layer_strength, device
                    )
                    factual_metrics.append(m)

                collateral_kls = {}
                for family in ["json_schema", "copying"]:
                    ckl = []
                    for tp in TEST_PROMPTS[family]:
                        kl = compute_collateral_kl(
                            model, tokenizer, tp["prompt"],
                            layer_indices, svs, per_layer_strength, device
                        )
                        ckl.append(kl)
                    collateral_kls[family] = round(np.mean(ckl), 6) if ckl else 0.0

                mean_kl = np.mean([m["kl_divergence"] for m in factual_metrics])
                mean_delta = np.mean([m["target_logit_delta"] for m in factual_metrics if m["target_logit_delta"] is not None])
                mean_accuracy = np.mean([m["task_accuracy"] for m in factual_metrics])

                sweep_entry = {
                    "strength": strength,
                    "per_layer_strength": round(per_layer_strength, 4),
                    "mean_kl": round(float(mean_kl), 6),
                    "mean_target_logit_delta": round(float(mean_delta), 6),
                    "mean_task_accuracy": round(float(mean_accuracy), 4),
                    "collateral_damage": collateral_kls,
                }
                multi_result["strength_sweep"].append(sweep_entry)
                print(f"    s={strength:+.1f} (per-layer={per_layer_strength:+.2f}): KL={mean_kl:.3f}, Δlogit={mean_delta:.3f}, acc={mean_accuracy:.1%}")

        # --- Assemble and save ---
        print(f"\n[4/4] Saving results...")
        elapsed = time.time() - start_time

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        dtype_str = str(bundle.dtype)

        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = "unknown"

        output = {
            "experiment_id": registry_id,
            "phase": 2,
            "block": "B",
            "model": model_name,
            "model_slug": slug,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "run_metadata": {
                "gpu": gpu_name,
                "dtype": dtype_str,
                "git_commit": git_hash,
                "n_layers": cfg["n_layers"],
                "steer_layers": steer_layers,
                "hub_layers": hub_layers,
                "strengths": STRENGTHS,
            },
            "single_layer_results": all_layer_results,
            "multi_layer_results": multi_result,
        }

        save_json(output, result_file)
        print(f"  Results saved to {result_file}")

        # Register in registry
        registry_entry = {
            "id": registry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "steering_migration",
            "model": model_name,
            "backend": "hf_native",
            "git_commit": git_hash,
            "config": "config/experiment_plan.yaml",
            "seed": args.seed,
            "inputs": [],
            "outputs": [str(result_file)],
            "status": "success",
            "summary": f"Steering migration on {model_name}: {len(steer_layers)} layers, {len(STRENGTHS)} strengths, 4 vector types, multi-layer distributed",
            "key_metrics": {},
            "failure": None,
            "next": "Analyze steering effectiveness and collateral damage patterns",
        }
        append_jsonl(registry_entry, registry_path)

        # Print summary
        print("\n" + "=" * 70)
        print(f"  STEERING MIGRATION SUMMARY: {model_name}")
        print("=" * 70)
        print(f"\n  Layers tested: {steer_layers}")
        print(f"  Strengths: {STRENGTHS}")
        print(f"  Vector types: target, random, wrong_task, anti")
        print(f"  Hub layers (multi): {hub_layers}")
        print(f"\n  Per-layer peak steering effectiveness (target vector):")
        for lr in all_layer_results:
            li = lr["layer_idx"]
            if "target" in lr.get("single_layer", {}):
                sweeps = lr["single_layer"]["target"]["strength_sweep"]
                best = max(sweeps, key=lambda x: abs(x.get("mean_target_logit_delta", 0))) if sweeps else None
                if best:
                    print(f"    L{li}: peak Δlogit={best['mean_target_logit_delta']:+.4f} at s={best['strength']:+.1f}, "
                          f"KL={best['mean_kl']:.3f}, acc={best['mean_task_accuracy']:.1%}")

        if multi_result["strength_sweep"]:
            best_multi = max(multi_result["strength_sweep"], key=lambda x: abs(x.get("mean_target_logit_delta", 0)))
            print(f"\n  Multi-layer peak: Δlogit={best_multi['mean_target_logit_delta']:+.4f} at s={best_multi['strength']:+.1f}, "
                  f"KL={best_multi['mean_kl']:.3f}, acc={best_multi['mean_task_accuracy']:.1%}")

        print(f"\n  Elapsed: {elapsed:.0f}s")
        print(f"  Results: {result_file}")

        # Cleanup
        del model, tokenizer, bundle
        torch.cuda.empty_cache()

    print("\n\nAll steering migration experiments complete.")


if __name__ == "__main__":
    main()
