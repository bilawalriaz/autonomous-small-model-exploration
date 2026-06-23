#!/usr/bin/env python3
"""Phase 3: Atlas-guided layer skip + recovery finetune.

Tests whether skipping layers is viable if guided by atlas knowledge:
  (a) Naive skip (zero out a layer) — already known to fail
  (b) Skip low-causal layers (identified by atlas as having smallest ablation effect)
  (c) Skip low-norm layers (identified by residual stream norm)
  (d) Skip random matched layers (control)

For each skip config: evaluate generation quality (top-5 token overlap
with baseline, repetition rate). Then for the atlas-guided skip, try
a short recovery finetune (50 steps) and re-evaluate.

Hypothesis: atlas-guided skip + recovery may work where naive skip fails.

Usage:
    python scripts/run_phase3_atlas_guided_skip.py --model Qwen/Qwen2.5-0.5B
    python scripts/run_phase3_atlas_guided_skip.py --model Qwen/Qwen2.5-1.5B --force
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType

from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, now_iso, git_commit_hash, PROJECT_ROOT

# Pre-computed ablation importance rankings (low-to-high importance)
# From Phase 1-2 layer ablation: these are the LOWEST importance layers
LOW_IMPORTANCE_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [4, 5, 8, 10, 11, 14, 15, 16],
    "Qwen/Qwen2.5-1.5B": [3, 4, 8, 9, 12, 13, 17, 18],
}


def get_layers(model):
    """Get transformer layers list."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def run_with_skipped_layers(model, input_ids, skip_layers):
    """Run inference with specified layers zeroed out."""
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


def compute_layer_norms(model, tokenizer, prompts, n_layers):
    """Compute mean residual stream norm per layer."""
    layer_norms = {}
    layers = get_layers(model)
    n_test = min(5, len(prompts))

    for layer_idx in range(n_layers):
        norms = []
        captured = {}

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured["norm"] = float(hidden.norm(dim=-1).mean().item())

        handle = layers[layer_idx].register_forward_hook(capture_hook)
        for i in range(n_test):
            prompt = prompts[i].clean_prompt if hasattr(prompts[i], 'clean_prompt') else prompts[i]
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(model.device)
            with torch.no_grad():
                _ = model(ids)
            if "norm" in captured:
                norms.append(captured["norm"])
        handle.remove()

        layer_norms[layer_idx] = float(np.mean(norms)) if norms else 0.0

    return layer_norms


def compute_ablation_effects(model, tokenizer, prompts, n_layers):
    """Quick ablation effect for each layer (mean KL across a few prompts)."""
    layer_effects = {}
    layers = get_layers(model)
    n_test = min(3, len(prompts))

    for layer_idx in range(n_layers):
        kl_values = []
        for i in range(n_test):
            prompt = prompts[i].clean_prompt if hasattr(prompts[i], 'clean_prompt') else prompts[i]
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(model.device)

            with torch.no_grad():
                orig_logits = model(ids).logits

            def zero_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

            handle = layers[layer_idx].register_forward_hook(zero_hook)
            with torch.no_grad():
                abl_logits = model(ids).logits
            handle.remove()

            orig_probs = torch.softmax(orig_logits[0, -1].float(), dim=-1)
            abl_log_probs = torch.log_softmax(abl_logits[0, -1].float(), dim=-1)
            kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
            kl_values.append(kl)

        layer_effects[layer_idx] = float(np.mean(kl_values)) if kl_values else 0.0

    return layer_effects


def evaluate_generation_quality(model, tokenizer, prompts, n_layers, skip_layers, max_new_tokens=50):
    """Evaluate generation quality with skipped layers.

    Returns: top-5 overlap with baseline, repetition rate, target prob ratio.
    """
    model.eval()
    results = []
    n_test = min(8, len(prompts))

    for i in range(n_test):
        prompt = prompts[i].clean_prompt if hasattr(prompts[i], 'clean_prompt') else prompts[i]
        target = prompts[i].target if hasattr(prompts[i], 'target') else ""

        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(model.device)

        # Baseline generation
        with torch.no_grad():
            baseline_logits = model(ids).logits
            baseline_gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id)

        # Skipped generation
        skip_logits = run_with_skipped_layers(model, ids, skip_layers)

        # Generate with skip hooks
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
            skip_gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)
        for h in handles:
            h.remove()

        # Decode
        baseline_text = tokenizer.decode(baseline_gen[0][ids.shape[1]:], skip_special_tokens=True)
        skip_text = tokenizer.decode(skip_gen[0][ids.shape[1]:], skip_special_tokens=True)

        # Top-5 overlap at last token
        base_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
        skip_probs = torch.softmax(skip_logits[0, -1], dim=-1)
        base_top5 = set(torch.topk(base_probs, 5).indices.tolist())
        skip_top5 = set(torch.topk(skip_probs, 5).indices.tolist())
        top5_overlap = len(base_top5 & skip_top5) / 5.0

        # Repetition rate in generated text
        repetition_rate = compute_repetition_rate(skip_text)

        # KL divergence
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(skip_logits[0, -1].float(), dim=-1),
            torch.softmax(baseline_logits[0, -1].float(), dim=-1),
            reduction="sum"
        ).item()

        # Target prob ratio
        target_prob_ratio = 1.0
        if target:
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if target_ids:
                target_id = target_ids[0]
                base_target_prob = base_probs[target_id].item()
                skip_target_prob = skip_probs[target_id].item()
                target_prob_ratio = skip_target_prob / max(base_target_prob, 1e-8)

        results.append({
            "prompt": prompt[:80],
            "top5_overlap": round(top5_overlap, 4),
            "kl": round(kl, 6),
            "target_prob_ratio": round(target_prob_ratio, 4),
            "repetition_rate": round(repetition_rate, 4),
            "baseline_text": baseline_text[:100],
            "skip_text": skip_text[:100],
            "text_changed": baseline_text.strip() != skip_text.strip(),
        })

    return {
        "n_eval": len(results),
        "mean_top5_overlap": round(float(np.mean([r["top5_overlap"] for r in results])), 4),
        "mean_kl": round(float(np.mean([r["kl"] for r in results])), 6),
        "mean_target_prob_ratio": round(float(np.mean([r["target_prob_ratio"] for r in results])), 4),
        "mean_repetition_rate": round(float(np.mean([r["repetition_rate"] for r in results])), 4),
        "fraction_changed": round(float(np.mean([r["text_changed"] for r in results])), 4),
        "per_prompt": results[:5],
    }


def compute_repetition_rate(text, ngram_size=3):
    """Compute fraction of repeated n-grams in text."""
    if len(text) < ngram_size * 2:
        return 0.0
    words = text.split()
    if len(words) < ngram_size:
        return 0.0
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngrams.append(tuple(words[i:i + ngram_size]))
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def recovery_finetune(model, tokenizer, train_prompts, steps=50, lr=2e-4, rank=8):
    """Short recovery finetune on JSON family after layer skipping."""
    train_texts = [p.clean_prompt + p.target for p in train_prompts]

    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=256):
            self.encodings = tokenizer(texts, truncation=True, max_length=max_length,
                                       padding="max_length", return_tensors="pt")
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    dataset = SimpleDataset(train_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(5, steps // 5),
                                                 num_training_steps=steps)

    model.train()
    losses = []
    start = time.time()

    for step in range(steps):
        batch_idx = step % len(dataloader)
        batch = {k: v.to(model.device) for k, v in dataloader[batch_idx].items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    elapsed = time.time() - start
    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 atlas-guided layer skip")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-skip", type=int, default=4, help="Number of layers to skip")
    parser.add_argument("--recovery-steps", type=int, default=50, help="Recovery finetune steps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    low_importance = LOW_IMPORTANCE_LAYERS.get(args.model, [4, 5, 8, 10, 11, 14, 15, 16])
    n_to_skip = min(args.n_skip, len(low_importance))

    print(f"Phase 3: Atlas-guided layer skip")
    print(f"Model: {args.model}")
    print(f"Layers to skip: {n_to_skip}")
    print(f"Recovery finetune steps: {args.recovery_steps}")
    print(f"Seed: {args.seed}")

    set_seed(args.seed)

    # Load model
    print("\nLoading model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    model.eval()

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    family_suite = suite.filter_by_family("json_schema")
    train_examples = list(family_suite.filter_by_split("train"))
    eval_examples = list(family_suite.filter_by_split("test"))
    if not train_examples:
        train_examples = list(family_suite)[:int(len(family_suite)*0.8)]
        eval_examples = list(family_suite)[int(len(family_suite)*0.8):]
    train_prompts = [{"prompt": e.clean_prompt, "target": e.target or ""} for e in train_examples]
    eval_prompts = [{"prompt": e.clean_prompt, "target": e.target or ""} for e in eval_examples]

    # Compute atlas data for this model
    print("\n  Computing layer ablation effects...")
    ablation_effects = compute_ablation_effects(model, tokenizer, eval_prompts, n_layers)

    print("  Computing layer norms...")
    layer_norms = compute_layer_norms(model, tokenizer, eval_prompts, n_layers)

    # Sort layers by ablation effect (ascending = least important first)
    sorted_by_effect = sorted(ablation_effects.items(), key=lambda x: x[1])
    sorted_by_norm = sorted(layer_norms.items(), key=lambda x: x[1])

    print(f"\n  Least important by ablation effect: {[x[0] for x in sorted_by_effect[:n_to_skip]]}")
    print(f"  Lowest norm layers: {[x[0] for x in sorted_by_norm[:n_to_skip]]}")

    # Define skip configurations
    rng = np.random.RandomState(args.seed)
    all_layer_indices = list(range(n_layers))
    random_layers = sorted(rng.choice(all_layer_indices, size=n_to_skip, replace=False).tolist())

    skip_configs = [
        ("naive_skip_worst", [sorted_by_effect[-1][0]],
         "Skip the MOST important single layer (expected to fail badly)"),
        ("atlas_guided_skip", [x[0] for x in sorted_by_effect[:n_to_skip]],
         f"Skip {n_to_skip} least important layers by ablation effect"),
        ("low_norm_skip", [x[0] for x in sorted_by_norm[:n_to_skip]],
         f"Skip {n_to_skip} lowest-norm layers"),
        ("random_matched_skip", random_layers,
         f"Skip {n_to_skip} random layers (control)"),
    ]

    # Run each config
    all_results = {}

    for config_name, skip_layers, description in skip_configs:
        print(f"\n  === {config_name} ===")
        print(f"    Skipping layers: {skip_layers}")
        print(f"    {description}")

        # Evaluate generation quality
        eval_result = evaluate_generation_quality(model, tokenizer, eval_prompts, n_layers, skip_layers)
        print(f"    Top-5 overlap: {eval_result['mean_top5_overlap']:.2%}")
        print(f"    Mean KL: {eval_result['mean_kl']:.4f}")
        print(f"    Target prob ratio: {eval_result['mean_target_prob_ratio']:.3f}")
        print(f"    Repetition rate: {eval_result['mean_repetition_rate']:.3f}")
        print(f"    Text changed: {eval_result['fraction_changed']:.0%}")

        config_result = {
            "config": config_name,
            "skip_layers": skip_layers,
            "description": description,
            "eval": eval_result,
            "recovery": None,
        }

        # For atlas-guided skip, try recovery finetune
        if config_name == "atlas_guided_skip" and args.recovery_steps > 0:
            print(f"\n    Running recovery finetune ({args.recovery_steps} steps)...")

            # We need to reload the model since we can't easily add LoRA on top of hooks
            # Instead, train with skip hooks active
            recovery_result = recovery_finetune(
                model, tokenizer, train_prompts,
                steps=args.recovery_steps, lr=2e-4
            )
            print(f"    Recovery final loss: {recovery_result['final_loss']:.4f}")

            # Re-evaluate after recovery
            recovery_eval = evaluate_generation_quality(model, tokenizer, eval_prompts, n_layers, skip_layers)
            print(f"    After recovery:")
            print(f"      Top-5 overlap: {recovery_eval['mean_top5_overlap']:.2%}")
            print(f"      Target prob ratio: {recovery_eval['mean_target_prob_ratio']:.3f}")
            print(f"      Repetition rate: {recovery_eval['mean_repetition_rate']:.3f}")

            config_result["recovery"] = {
                "training": recovery_result,
                "eval": recovery_eval,
            }

        all_results[config_name] = config_result

    # Comparison
    print(f"\n{'='*60}")
    print("  SKIP STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Config':30s}  {'Top5':>6s}  {'KL':>8s}  {'TgtRatio':>8s}  {'Repeat':>7s}")
    print(f"  {'─'*65}")

    for config_name, config_data in all_results.items():
        ev = config_data["eval"]
        print(f"  {config_name:30s}  {ev['mean_top5_overlap']:>6.2%}  {ev['mean_kl']:>8.4f}  "
              f"{ev['mean_target_prob_ratio']:>8.3f}  {ev['mean_repetition_rate']:>7.3f}")

        # Show recovery results if available
        if config_data.get("recovery"):
            rev = config_data["recovery"]["eval"]
            print(f"  {'  + recovery':30s}  {rev['mean_top5_overlap']:>6.2%}  {rev['mean_kl']:>8.4f}  "
                  f"{rev['mean_target_prob_ratio']:>8.3f}  {rev['mean_repetition_rate']:>7.3f}")

    # Determine best strategy
    viable = []
    for config_name, config_data in all_results.items():
        ev = config_data["eval"]
        # Viable if top-5 overlap > 50% and repetition < 50%
        if ev["mean_top5_overlap"] > 0.5 and ev["mean_repetition_rate"] < 0.5:
            viable.append((config_name, ev["mean_top5_overlap"]))

    if viable:
        best = max(viable, key=lambda x: x[1])
        print(f"\n  Best viable strategy: {best[0]} (top-5 overlap: {best[1]:.2%})")
    else:
        print(f"\n  No skip strategy is viable at {n_to_skip} layers skipped")

    # Save
    summary = {
        "experiment": "phase3_atlas_guided_skip",
        "model": args.model,
        "seed": args.seed,
        "n_skip": n_to_skip,
        "n_layers": n_layers,
        "recovery_steps": args.recovery_steps,
        "ablation_effects": {str(k): round(v, 6) for k, v in ablation_effects.items()},
        "layer_norms": {str(k): round(v, 6) for k, v in layer_norms.items()},
        "skip_configs": all_results,
        "viable_strategies": [v[0] for v in viable] if viable else [],
        "timestamp": now_iso(),
        "git_commit": git_commit_hash(),
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_atlas_guided_skip_{model_slug}.json"
    save_json(summary, out_path)
    print(f"\n  Results: {out_path}")

    # Register
    register_experiment(
        type="ablation",
        model=args.model,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(out_path)],
        status="success",
        summary=f"Atlas-guided skip ({model_slug}): {n_to_skip} layers skipped, "
                f"viable={len(viable)}/{len(skip_configs)}, "
                f"best={viable[0][0] if viable else 'none'}",
        next="Test on more models; try deeper skip with larger recovery finetune",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
