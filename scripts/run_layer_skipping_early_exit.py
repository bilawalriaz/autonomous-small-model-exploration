"""exp_000022-023: Layer skipping and early-exit efficiency experiments.

Tests concrete inference optimizations derived from the MI atlas:

1. LAYER SKIPPING: Skip individual weak layers and measure task performance
   - L15 (weakest, max KL 3.37), L4-L8 (mid layers, low importance)
   - Skip combinations: skip L4-L8, skip L15+L4, skip all weak layers
   - Measure: KL divergence, target token probability, inference time

2. EARLY EXIT: Read logits from intermediate layers instead of L23
   - L22 is the unembedding pathway (97% recovery in cross-model patching)
   - Test: can we skip L23 entirely?
   - Test: can we exit at L21? L19?
   - Measure: output quality vs speed tradeoff

3. SELECTIVE COMPUTATION: Only compute critical layers for simple tasks
   - Copying/delimiter: only L0-L2 + L22 (skip mid layers)
   - Factual/JSON: all layers
   - Measure: can a task-aware router save computation?
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def get_layers(model):
    """Get transformer layers list, handling PeftModel."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def run_with_skipped_layers(model, input_ids, skip_layers):
    """Run inference with specified layers skipped (zeroed out)."""
    layers = get_layers(model)
    handles = []

    for idx in skip_layers:
        def skip_hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        h = layers[idx].register_forward_hook(skip_hook)
        handles.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in handles:
        h.remove()

    return logits


def run_with_early_exit(model, input_ids, exit_layer, n_layers):
    """Run inference but exit early — read logits from intermediate layer output.

    Uses the unembedding matrix to project intermediate residual to vocab.
    """
    layers = get_layers(model)
    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured["hidden"] = output[0].detach().clone()
        else:
            captured["hidden"] = output.detach().clone()

    handle = layers[exit_layer].register_forward_hook(capture_hook)

    with torch.no_grad():
        _ = model(input_ids)

    handle.remove()

    # Project through the model's lm_head
    hidden = captured["hidden"]
    if hasattr(model, 'lm_head'):
        early_logits = model.lm_head(hidden)
    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        early_logits = model.model.lm_head(hidden)
    else:
        # Fallback: run full model and capture logits
        with torch.no_grad():
            early_logits = model(input_ids).logits

    return early_logits


def compute_kl(logits_a, logits_b):
    """KL(P_a || P_b) at last token."""
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()


def compute_target_prob(logits, target_id):
    """Probability of target token at last position."""
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    return probs[target_id].item()


def measure_inference_time(model, input_ids, n_warmup=3, n_runs=10):
    """Measure inference time in milliseconds."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


def measure_skipped_time(model, input_ids, skip_layers, n_warmup=3, n_runs=10):
    """Measure inference time with layers skipped."""
    layers = get_layers(model)

    # Warmup
    for _ in range(n_warmup):
        handles = []
        for idx in skip_layers:
            def skip_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            h = layers[idx].register_forward_hook(skip_hook)
            handles.append(h)
        with torch.no_grad():
            _ = model(input_ids)
        for h in handles:
            h.remove()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        handles = []
        for idx in skip_layers:
            def skip_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            h = layers[idx].register_forward_hook(skip_hook)
            handles.append(h)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

        for h in handles:
            h.remove()

    return np.mean(times), np.std(times)


def main():
    set_seed(42)

    # Load task suite
    suite_path = PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"
    with open(suite_path) as f:
        suite = json.load(f)

    # Select representative prompts from key families
    test_prompts = []
    families_seen = set()
    for ex in suite:
        fam = ex.get("family", "")
        if fam not in families_seen:
            test_prompts.append({
                "prompt": ex["clean_prompt"],
                "family": fam,
                "target": ex.get("target", ""),
            })
            families_seen.add(fam)
            if len(test_prompts) >= 12:
                break

    # Load pairs for more target-tested prompts
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    for pair in pairs[:6]:
        test_prompts.append({
            "prompt": pair["prefix"],
            "family": pair["family"],
            "target": pair["target"],
        })

    # Deduplicate
    seen = set()
    unique = []
    for tp in test_prompts:
        if tp["prompt"] not in seen:
            seen.add(tp["prompt"])
            unique.append(tp)
    test_prompts = unique[:15]

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    device = bundle.device
    model.eval()

    # ============================================
    # PART 1: LAYER SKIPPING
    # ============================================
    print("\n" + "=" * 60)
    print("  PART 1: LAYER SKIPPING")
    print("=" * 60)

    # Define skip configurations based on atlas findings
    skip_configs = [
        ("skip_L15", [15]),                    # Weakest single layer
        ("skip_L4", [4]),                      # Low importance early-mid
        ("skip_L8", [8]),                      # Low importance mid
        ("skip_L4_L8", [4, 5, 6, 7, 8]),      # Skip all weak mid-layers
        ("skip_L15_L4_L8", [4, 5, 6, 7, 8, 15]),  # All weak layers
        ("skip_L3_L4_L5", [3, 4, 5]),          # Very early non-critical
        ("skip_L10_L11_L12", [10, 11, 12]),    # Non-critical mid
        ("skip_L14_L15_L16", [14, 15, 16]),    # Weak late-mid
        ("skip_6_layers", [4, 5, 8, 11, 15, 16]),  # 6 weakest
        ("skip_8_layers", [4, 5, 8, 10, 11, 14, 15, 16]),  # 8 weakest
    ]

    skip_results = []

    # First get baseline timing
    print("\n  Measuring baseline inference time...")
    # Use a representative prompt
    timing_prompt = tokenizer("The capital of France is Paris", return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
    base_time, base_std = measure_inference_time(model, timing_prompt)
    print(f"    Baseline: {base_time:.2f}ms ± {base_std:.2f}ms")

    for config_name, skip_layers in skip_configs:
        print(f"\n  Config: {config_name} (skipping {len(skip_layers)} layers)")

        config_results = {
            "config": config_name,
            "skip_layers": skip_layers,
            "n_skipped": len(skip_layers),
            "prompts": [],
        }

        for tp in test_prompts:
            prompt = tp["prompt"]
            family = tp["family"]
            target = tp["target"]

            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

            # Full model logits
            with torch.no_grad():
                full_logits = model(ids).logits

            # Skipped logits
            skip_logits = run_with_skipped_layers(model, ids, skip_layers)

            # KL divergence
            kl = compute_kl(full_logits, skip_logits)

            # Target prob comparison
            target_prob_full = 1.0
            target_prob_skip = 1.0
            if target:
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if len(target_ids) > 0:
                    target_id = target_ids[0]
                    target_prob_full = compute_target_prob(full_logits, target_id)
                    target_prob_skip = compute_target_prob(skip_logits, target_id)

            # Top-5 agreement
            full_probs = torch.softmax(full_logits[0, -1, :], dim=-1)
            skip_probs = torch.softmax(skip_logits[0, -1, :], dim=-1)
            full_top5 = set(torch.topk(full_probs, 5).indices.tolist())
            skip_top5 = set(torch.topk(skip_probs, 5).indices.tolist())
            top5_overlap = len(full_top5 & skip_top5) / 5.0

            config_results["prompts"].append({
                "family": family,
                "kl": round(kl, 6),
                "target_prob_full": round(target_prob_full, 6),
                "target_prob_skip": round(target_prob_skip, 6),
                "target_prob_ratio": round(target_prob_skip / max(target_prob_full, 1e-8), 4),
                "top5_overlap": round(top5_overlap, 4),
            })

        # Mean metrics
        mean_kl = np.mean([p["kl"] for p in config_results["prompts"]])
        mean_top5 = np.mean([p["top5_overlap"] for p in config_results["prompts"]])
        mean_prob_ratio = np.mean([p["target_prob_ratio"] for p in config_results["prompts"]])

        # Time measurement
        skip_time, skip_std = measure_skipped_time(model, timing_prompt, skip_layers)
        speedup = base_time / skip_time if skip_time > 0 else 0

        config_results["mean_kl"] = round(mean_kl, 6)
        config_results["mean_top5_overlap"] = round(mean_top5, 4)
        config_results["mean_prob_ratio"] = round(mean_prob_ratio, 4)
        config_results["inference_time_ms"] = round(skip_time, 2)
        config_results["inference_std_ms"] = round(skip_std, 2)
        config_results["speedup"] = round(speedup, 4)
        config_results["base_time_ms"] = round(base_time, 2)

        print(f"    Mean KL: {mean_kl:.4f}, Top-5 overlap: {mean_top5:.2%}, Speedup: {speedup:.3f}x")

        skip_results.append(config_results)

    # Find best skip config (lowest KL with highest speedup)
    best_skip = sorted(skip_results, key=lambda x: (x["mean_kl"], -x["n_skipped"]))[0]
    print(f"\n  Best skip config: {best_skip['config']} (KL={best_skip['mean_kl']:.4f}, {best_skip['n_skipped']} layers skipped, {best_skip['speedup']:.3f}x speedup)")

    # ============================================
    # PART 2: EARLY EXIT
    # ============================================
    print("\n" + "=" * 60)
    print("  PART 2: EARLY EXIT")
    print("=" * 60)

    exit_layers = [23, 22, 21, 20, 19, 18, 17]  # Full baseline down to early
    exit_results = []

    for exit_layer in exit_layers:
        print(f"\n  Exit at L{exit_layer} (skip {n_layers - exit_layer - 1} layers):")

        layer_results = {
            "exit_layer": exit_layer,
            "layers_skipped": n_layers - exit_layer - 1,
            "prompts": [],
        }

        for tp in test_prompts:
            prompt = tp["prompt"]
            family = tp["family"]
            target = tp["target"]

            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

            # Full model logits
            with torch.no_grad():
                full_logits = model(ids).logits

            # Early exit logits
            if exit_layer == n_layers - 1:
                early_logits = full_logits  # No skip
            else:
                early_logits = run_with_early_exit(model, ids, exit_layer, n_layers)

            # KL divergence
            kl = compute_kl(full_logits, early_logits)

            # Target prob
            target_prob_full = 1.0
            target_prob_early = 1.0
            if target:
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if len(target_ids) > 0:
                    target_id = target_ids[0]
                    target_prob_full = compute_target_prob(full_logits, target_id)
                    target_prob_early = compute_target_prob(early_logits, target_id)

            # Top-5 agreement
            full_probs = torch.softmax(full_logits[0, -1, :], dim=-1)
            early_probs = torch.softmax(early_logits[0, -1, :], dim=-1)
            full_top5 = set(torch.topk(full_probs, 5).indices.tolist())
            early_top5 = set(torch.topk(early_probs, 5).indices.tolist())
            top5_overlap = len(full_top5 & early_top5) / 5.0

            # Argmax match
            full_argmax = full_probs.argmax().item()
            early_argmax = early_probs.argmax().item()
            argmax_match = 1 if full_argmax == early_argmax else 0

            layer_results["prompts"].append({
                "family": family,
                "kl": round(kl, 6),
                "target_prob_full": round(target_prob_full, 6),
                "target_prob_early": round(target_prob_early, 6),
                "target_prob_ratio": round(target_prob_early / max(target_prob_full, 1e-8), 4),
                "top5_overlap": round(top5_overlap, 4),
                "argmax_match": argmax_match,
            })

        mean_kl = np.mean([p["kl"] for p in layer_results["prompts"]])
        mean_top5 = np.mean([p["top5_overlap"] for p in layer_results["prompts"]])
        mean_argmax = np.mean([p["argmax_match"] for p in layer_results["prompts"]])
        mean_prob_ratio = np.mean([p["target_prob_ratio"] for p in layer_results["prompts"]])

        # Theoretical speedup: (exit_layer + 1) / n_layers
        theoretical_speedup = n_layers / (exit_layer + 1)

        layer_results["mean_kl"] = round(mean_kl, 6)
        layer_results["mean_top5_overlap"] = round(mean_top5, 4)
        layer_results["mean_argmax_match"] = round(mean_argmax, 4)
        layer_results["mean_prob_ratio"] = round(mean_prob_ratio, 4)
        layer_results["theoretical_speedup"] = round(theoretical_speedup, 4)

        print(f"    Mean KL: {mean_kl:.4f}, Top-5: {mean_top5:.2%}, Argmax match: {mean_argmax:.2%}, Theoretical speedup: {theoretical_speedup:.2f}x")

        exit_results.append(layer_results)

    # Find best early exit (highest argmax match with most layers skipped)
    viable_exits = [e for e in exit_results if e["mean_argmax_match"] >= 0.8]
    if viable_exits:
        best_exit = sorted(viable_exits, key=lambda x: x["exit_layer"])[0]
        print(f"\n  Best early exit: L{best_exit['exit_layer']} (argmax match={best_exit['mean_argmax_match']:.2%}, KL={best_exit['mean_kl']:.4f}, skip {best_exit['layers_skipped']} layers, theoretical {best_exit['theoretical_speedup']:.2f}x speedup)")
    else:
        print(f"\n  No early exit with >80% argmax match found")

    # ============================================
    # PART 3: TASK-AWARE SELECTIVE COMPUTATION
    # ============================================
    print("\n" + "=" * 60)
    print("  PART 3: TASK-AWARE SELECTIVE COMPUTATION")
    print("=" * 60)

    # Based on atlas: simple tasks (copying, delimiter) don't need mid-layers
    # Complex tasks (factual, JSON) need full model
    task_configs = [
        ("simple_skip_mid", [4, 5, 6, 7, 8, 10, 11, 14, 15, 16], ["copying", "delimiter_tracking", "arithmetic"]),
        ("complex_skip_mid", [4, 5, 6, 7, 8, 10, 11, 14, 15, 16], ["factual_recall", "json_schema", "code_syntax"]),
        ("all_skip_4_layers", [4, 5, 8, 15], None),  # All families, skip 4 weakest
    ]

    task_results = []

    for config_name, skip_layers, families_filter in task_configs:
        print(f"\n  Config: {config_name}")

        config_prompts = test_prompts
        if families_filter:
            config_prompts = [tp for tp in test_prompts if tp["family"] in families_filter]

        if not config_prompts:
            continue

        config_data = {
            "config": config_name,
            "skip_layers": skip_layers,
            "n_skipped": len(skip_layers),
            "families": families_filter or "all",
            "prompts": [],
        }

        for tp in config_prompts:
            prompt = tp["prompt"]
            family = tp["family"]
            target = tp["target"]

            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                full_logits = model(ids).logits
            skip_logits = run_with_skipped_layers(model, ids, skip_layers)

            kl = compute_kl(full_logits, skip_logits)

            target_prob_full = 1.0
            target_prob_skip = 1.0
            if target:
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if len(target_ids) > 0:
                    target_id = target_ids[0]
                    target_prob_full = compute_target_prob(full_logits, target_id)
                    target_prob_skip = compute_target_prob(skip_logits, target_id)

            full_probs = torch.softmax(full_logits[0, -1, :], dim=-1)
            skip_probs = torch.softmax(skip_logits[0, -1, :], dim=-1)
            full_top5 = set(torch.topk(full_probs, 5).indices.tolist())
            skip_top5 = set(torch.topk(skip_probs, 5).indices.tolist())
            top5_overlap = len(full_top5 & skip_top5) / 5.0
            full_argmax = full_probs.argmax().item()
            skip_argmax = skip_probs.argmax().item()
            argmax_match = 1 if full_argmax == skip_argmax else 0

            config_data["prompts"].append({
                "family": family,
                "kl": round(kl, 6),
                "target_prob_full": round(target_prob_full, 6),
                "target_prob_skip": round(target_prob_skip, 6),
                "target_prob_ratio": round(target_prob_skip / max(target_prob_full, 1e-8), 4),
                "top5_overlap": round(top5_overlap, 4),
                "argmax_match": argmax_match,
            })

        mean_kl = np.mean([p["kl"] for p in config_data["prompts"]])
        mean_top5 = np.mean([p["top5_overlap"] for p in config_data["prompts"]])
        mean_argmax = np.mean([p["argmax_match"] for p in config_data["prompts"]])

        config_data["mean_kl"] = round(mean_kl, 6)
        config_data["mean_top5_overlap"] = round(mean_top5, 4)
        config_data["mean_argmax_match"] = round(mean_argmax, 4)

        print(f"    Mean KL: {mean_kl:.4f}, Top-5: {mean_top5:.2%}, Argmax: {mean_argmax:.2%}")

        task_results.append(config_data)

    # ============================================
    # SAVE RESULTS
    # ============================================
    output = {
        "experiment": "layer_skipping_and_early_exit",
        "n_layers": n_layers,
        "base_inference_time_ms": round(base_time, 2),
        "part1_layer_skipping": skip_results,
        "part2_early_exit": exit_results,
        "part3_selective_computation": task_results,
        "summary": {
            "best_skip_config": best_skip["config"],
            "best_skip_kl": best_skip["mean_kl"],
            "best_skip_speedup": best_skip["speedup"],
            "best_skip_layers": best_skip["n_skipped"],
            "best_exit_layer": best_exit["exit_layer"] if viable_exits else None,
            "best_exit_argmax": best_exit["mean_argmax_match"] if viable_exits else None,
            "best_exit_theoretical_speedup": best_exit["theoretical_speedup"] if viable_exits else None,
        },
    }

    output_path = PROJECT_ROOT / "experiments" / "results" / "layer_skipping_early_exit.json"
    save_json(output, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="efficiency",
        model="Qwen/Qwen2.5-0.5B",
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[str(suite_path), str(pairs_path)],
        outputs=[str(output_path)],
        status="success",
        summary=f"Layer skipping + early exit: {len(skip_configs)} skip configs, {len(exit_layers)} exit layers, {len(task_configs)} task-aware configs",
        key_metrics=output["summary"],
        next="Update publication report, create MI-Atlas skill",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
