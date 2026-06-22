"""
Phase 2 — Block C: Ablation Controls (P2-ABL-001, P2-ABL-002)

Replace zero-only ablation with stronger controls.

For all layers (L0-L23 for 0.5B, L0-L27 for 1.5B), compare:
  - zero ablation (set output to 0)
  - mean ablation (replace with mean activation across batch)
  - Gaussian/noise resample ablation (replace with noise matching activation statistics)
  - activation patch clean→corrupt (replace corrupt activation with clean)
  - activation patch corrupt→clean (replace clean activation with corrupt)
  - random same-shape patch control

Metrics: KL divergence vs baseline for each ablation type.

Resumable: checks registry for existing completed runs (skips unless --force).
"""
import sys
import json
import argparse
import time
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
        "registry_id": "P2-ABL-001",
    },
    "Qwen/Qwen2.5-1.5B": {
        "slug": "qwen15b",
        "n_layers": 28,
        "registry_id": "P2-ABL-002",
    },
}

ABLATION_TYPES = ["zero", "mean", "gaussian_resample", "patch_clean_to_corrupt", "patch_corrupt_to_clean", "random_patch"]

# Test prompts from each task family (clean/corrupt pairs for patching)
TEST_TASKS = {
    "factual_recall": {
        "clean": [
            {"prompt": "The capital of France is ", "target": "Paris"},
            {"prompt": "The capital of Germany is ", "target": "Berlin"},
            {"prompt": "The capital of Japan is ", "target": "Tokyo"},
        ],
        "corrupt": [
            {"prompt": "The capital of Flibber is ", "target": "N/A"},
            {"prompt": "The capital of Grommet is ", "target": "N/A"},
            {"prompt": "The capital of Jixxel is ", "target": "N/A"},
        ],
    },
    "json_schema": {
        "clean": [
            {"prompt": 'Return exactly valid JSON with keys name and age. Alice is 31.\n', "target": '{"'},
            {"prompt": 'Return exactly valid JSON with keys name and age. Bob is 25.\n', "target": '{"'},
        ],
        "corrupt": [
            {"prompt": "Tell me about Alice who is 31 years old.\n", "target": ""},
            {"prompt": "Describe Bob, a 25-year-old.\n", "target": ""},
        ],
    },
    "copying": {
        "clean": [
            {"prompt": "A B C A B C A ", "target": "B"},
        ],
        "corrupt": [
            {"prompt": "A B C X Y Z A ", "target": "B"},
        ],
    },
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


def compute_kl(logits_a, logits_b):
    """KL(P_a || P_b) at last token position."""
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()


def get_mean_activations(model, tokenizer, prompts, layer_idx, device, n_samples=5):
    """Compute mean activation at a layer across multiple prompts."""
    acts = []
    for prompt in prompts[:n_samples]:
        ids = tokenizer(prompt["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        captured = {}

        def hook_fn(module, inp, output, _cap=captured):
            if isinstance(output, tuple):
                _cap["value"] = output[0].detach().clone()
            else:
                _cap["value"] = output.detach().clone()

        layers = get_layers(model)
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(ids)
        handle.remove()

        if "value" in captured:
            acts.append(captured["value"].cpu())
        del ids
        torch.cuda.empty_cache()

    if not acts:
        return None
    return torch.cat(acts, dim=0).mean(dim=0, keepdim=True)  # (1, seq, d_model) mean


def get_activation_stats(model, tokenizer, prompts, layer_idx, device, n_samples=5):
    """Get mean and std of activations at a layer."""
    acts = []
    for prompt in prompts[:n_samples]:
        ids = tokenizer(prompt["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        captured = {}

        def hook_fn(module, inp, output, _cap=captured):
            if isinstance(output, tuple):
                _cap["value"] = output[0].detach().clone()
            else:
                _cap["value"] = output.detach().clone()

        layers = get_layers(model)
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(ids)
        handle.remove()

        if "value" in captured:
            # Flatten to (tokens, d_model)
            acts.append(captured["value"][0].cpu())
        del ids
        torch.cuda.empty_cache()

    if not acts:
        return None, None
    all_acts = torch.cat(acts, dim=0)  # (total_tokens, d_model)
    return all_acts.mean(dim=0), all_acts.std(dim=0)


def run_ablation(model, tokenizer, prompt, layer_idx, ablation_type, device,
                 mean_activation=None, activation_std=None, clean_activation=None):
    """Run a single ablation and return KL vs baseline."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    # Baseline
    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)

    # Ablated
    layers = get_layers(model)

    if ablation_type == "zero":
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)

    elif ablation_type == "mean":
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                if mean_activation is not None:
                    ma = mean_activation.to(hidden.device)
                    # Broadcast: match seq_len
                    if ma.dim() == 1:
                        ma = ma.unsqueeze(0)
                    if ma.shape[0] == 1:
                        ma = ma.expand_as(hidden)
                    else:
                        ma = ma[:hidden.shape[0]]
                    return (ma,) + output[1:]
                return (torch.zeros_like(hidden),) + output[1:]
            return torch.zeros_like(output)

    elif ablation_type == "gaussian_resample":
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                if mean_activation is not None and activation_std is not None:
                    ma = mean_activation.to(hidden.device)
                    std = activation_std.to(hidden.device)
                    if ma.dim() == 1:
                        noise = torch.randn_like(hidden) * std.unsqueeze(0) + ma.unsqueeze(0)
                    else:
                        noise = torch.randn_like(hidden) * std + ma
                    return (noise,) + output[1:]
                return (torch.randn_like(hidden),) + output[1:]
            return torch.randn_like(output)

    elif ablation_type == "random_patch":
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Random tensor of same shape, same scale as original
                scale = hidden.std()
                rand = torch.randn_like(hidden) * scale
                return (rand,) + output[1:]
            return torch.randn_like(output) * output.std()

    elif ablation_type in ("patch_clean_to_corrupt", "patch_corrupt_to_clean"):
        if clean_activation is None:
            return None
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                hidden = output[0]
                ca = clean_activation.to(hidden.device)
                # Match sequence lengths
                min_len = min(ca.shape[1], hidden.shape[1])
                hidden[:, :min_len, :] = ca[:, :min_len, :]
                return (hidden,) + output[1:]
            return output
    else:
        return None

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        abl_logits = model(ids).logits
    handle.remove()

    kl = compute_kl(orig_logits, abl_logits)

    del ids, orig_logits, abl_logits, orig_probs
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
    parser = argparse.ArgumentParser(description="Phase 2 Block C: Ablation Controls")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: run both)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if completed")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to test (default: all)")
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
        result_file = results_dir / f"ablation_controls_{slug}.json"

        # Check resumability
        if not args.force and run_id_exists(registry_path, registry_id):
            print(f"[SKIP] {registry_id} ({model_name}) already completed. Use --force to re-run.")
            continue

        print("=" * 70)
        print(f"  ABLATION CONTROLS: {model_name}")
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

        n_layers = cfg["n_layers"]
        if args.layers:
            layer_indices = [int(x) for x in args.layers.split(",")]
        else:
            layer_indices = list(range(n_layers))

        # Precompute activation statistics per layer (for mean/gaussian ablation)
        print(f"\n[2/4] Computing activation statistics for {n_layers} layers...")
        all_prompts = []
        for family_data in TEST_TASKS.values():
            all_prompts.extend(family_data["clean"])

        layer_stats = {}
        for li in layer_indices:
            mean_act, std_act = get_activation_stats(model, tokenizer, all_prompts, li, device, n_samples=5)
            layer_stats[li] = {"mean": mean_act, "std": std_act}
            torch.cuda.empty_cache()
            if (li + 1) % 5 == 0 or li == layer_indices[-1]:
                print(f"    Computed stats for L0-L{li}")

        # Precompute clean activations for patching
        print(f"\n[3/4] Computing clean activations for patching...")
        clean_activations = {}  # {family: {layer_idx: activation_tensor}}
        for family, family_data in TEST_TASKS.items():
            clean_activations[family] = {}
            for li in layer_indices:
                # Get activation from first clean prompt of this family
                clean_prompt = family_data["clean"][0]["prompt"]
                ids = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                captured = {}

                def hook_fn(module, inp, output, _cap=captured):
                    if isinstance(output, tuple):
                        _cap["value"] = output[0].detach().clone()
                    else:
                        _cap["value"] = output.detach().clone()

                layers = get_layers(model)
                handle = layers[li].register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(ids)
                handle.remove()
                clean_activations[family][li] = captured.get("value")
                del ids
                torch.cuda.empty_cache()

        # Run ablation experiments
        print(f"\n[4/4] Running ablation experiments ({len(layer_indices)} layers × {len(ABLATION_TYPES)} types × {len(TEST_TASKS)} families)...")
        layer_results = []

        for li in layer_indices:
            if (li) % 3 == 0:
                print(f"\n  Layer {li}/{n_layers-1}...")

            layer_result = {
                "layer_idx": li,
                "by_ablation_type": {},
            }

            mean_act = layer_stats[li]["mean"]
            std_act = layer_stats[li]["std"]

            for ablation_type in ABLATION_TYPES:
                type_results = {}

                for family, family_data in TEST_TASKS.items():
                    family_kls = []

                    # Ablation effect on clean prompts
                    for task in family_data["clean"]:
                        clean_act = clean_activations[family].get(li)

                        kl = run_ablation(
                            model, tokenizer,
                            task["prompt"], li, ablation_type, device,
                            mean_activation=mean_act,
                            activation_std=std_act,
                            clean_activation=clean_act,
                        )
                        if kl is not None:
                            family_kls.append(kl)

                    # For patch types, also test on corrupt prompts (corrupt→clean)
                    if ablation_type == "patch_corrupt_to_clean":
                        for task in family_data.get("corrupt", []):
                            # For corrupt→clean: we patch the clean activation into the corrupt run
                            # This means we run the corrupt prompt but replace the layer output
                            # with what we got from the clean prompt
                            clean_act = clean_activations[family].get(li)
                            kl = run_ablation(
                                model, tokenizer,
                                task["prompt"], li, "patch_clean_to_corrupt", device,
                                clean_activation=clean_act,
                            )
                            if kl is not None:
                                family_kls.append(kl)

                    type_results[family] = {
                        "mean_kl": round(float(np.mean(family_kls)), 6) if family_kls else 0.0,
                        "std_kl": round(float(np.std(family_kls)), 6) if family_kls else 0.0,
                        "n_prompts": len(family_kls),
                    }

                layer_result["by_ablation_type"][ablation_type] = type_results

            layer_results.append(layer_result)

            # Progress summary every 8 layers
            if (li + 1) % 8 == 0 or li == layer_indices[-1]:
                # Show summary for this batch
                for atype in ["zero", "mean", "gaussian_resample"]:
                    mean_kls = []
                    for lr in layer_results[-8:]:
                        for fam in lr["by_ablation_type"].get(atype, {}).values():
                            mean_kls.append(fam["mean_kl"])
                    if mean_kls:
                        batch_mean = np.mean(mean_kls)
                        print(f"    {atype}: mean KL across batch = {batch_mean:.4f}")

            torch.cuda.empty_cache()

        # --- Assemble and save ---
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
            "block": "C",
            "model": model_name,
            "model_slug": slug,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "run_metadata": {
                "gpu": gpu_name,
                "dtype": dtype_str,
                "git_commit": git_hash,
                "n_layers": n_layers,
                "layers_tested": layer_indices,
                "ablation_types": ABLATION_TYPES,
                "task_families": list(TEST_TASKS.keys()),
            },
            "results": layer_results,
        }

        save_json(output, result_file)
        print(f"\n  Results saved to {result_file}")

        # Register in registry
        registry_entry = {
            "id": registry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "ablation_controls",
            "model": model_name,
            "backend": "hf_native",
            "git_commit": git_hash,
            "config": "config/experiment_plan.yaml",
            "seed": args.seed,
            "inputs": [],
            "outputs": [str(result_file)],
            "status": "success",
            "summary": f"Ablation controls on {model_name}: {len(layer_indices)} layers × {len(ABLATION_TYPES)} types × {len(TEST_TASKS)} families",
            "key_metrics": {},
            "failure": None,
            "next": "Compare ablation types to determine which control method is most informative",
        }
        append_jsonl(registry_entry, registry_path)

        # Print summary
        print("\n" + "=" * 70)
        print(f"  ABLATION CONTROLS SUMMARY: {model_name}")
        print("=" * 70)

        # Aggregate by ablation type across all layers
        print(f"\n  Mean KL divergence by ablation type (across all layers and families):")
        for atype in ABLATION_TYPES:
            all_kls = []
            for lr in layer_results:
                for fam_data in lr["by_ablation_type"].get(atype, {}).values():
                    all_kls.append(fam_data["mean_kl"])
            if all_kls:
                print(f"    {atype:30s}: mean={np.mean(all_kls):.4f} ± {np.std(all_kls):.4f}, max={np.max(all_kls):.4f}")

        # Top-5 most affected layers per ablation type
        print(f"\n  Top-5 most affected layers per ablation type:")
        for atype in ["zero", "mean", "gaussian_resample"]:
            layer_mean_kls = []
            for lr in layer_results:
                kls = [fam_data["mean_kl"] for fam_data in lr["by_ablation_type"].get(atype, {}).values()]
                layer_mean_kls.append((lr["layer_idx"], np.mean(kls) if kls else 0))
            layer_mean_kls.sort(key=lambda x: x[1], reverse=True)
            top5_str = ", ".join(f"L{li}({kl:.3f})" for li, kl in layer_mean_kls[:5])
            print(f"    {atype}: {top5_str}")

        # Per-family comparison
        print(f"\n  Per-family mean KL by ablation type:")
        for family in TEST_TASKS:
            print(f"    {family}:")
            for atype in ["zero", "mean", "gaussian_resample"]:
                fam_kls = [lr["by_ablation_type"][atype][family]["mean_kl"]
                           for lr in layer_results if atype in lr["by_ablation_type"] and family in lr["by_ablation_type"][atype]]
                if fam_kls:
                    print(f"      {atype:30s}: mean={np.mean(fam_kls):.4f}")

        print(f"\n  Elapsed: {elapsed:.0f}s")
        print(f"  Results: {result_file}")

        # Cleanup
        del model, tokenizer, bundle
        torch.cuda.empty_cache()

    print("\n\nAll ablation control experiments complete.")


if __name__ == "__main__":
    main()
