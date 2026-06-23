#!/usr/bin/env python3
"""Phase 3: Atlas-guided LoRA vs generic LoRA.

THE KEY EXPERIMENT: Tests whether atlas-identified critical layers produce better
LoRA adapters than generic all-linear targeting at equal or lower parameters.

Compares 3 strategies:
1. Atlas-guided: target only the layers identified as critical by the atlas
2. Random-matched: target the same number of random layers (control)
3. All-linear: target all linear layers (standard practice)

Measures: task accuracy (exact match, constraint satisfaction), not just ablation proxies.

Usage:
    python scripts/run_phase3_atlas_guided_lora.py --model Qwen/Qwen2.5-0.5B --family json_schema
    python scripts/run_phase3_atlas_guided_lora.py --model Qwen/Qwen2.5-0.5B --family factual_recall
    python scripts/run_phase3_atlas_guided_lora.py --model Qwen/Qwen2.5-0.5B --family code_semantics --force
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType

from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.metrics import exact_match_score, valid_json_score
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT

# Atlas-identified critical layers per model per family
# From Phase 1-2 dataset shard ablation
ATLAS_LAYERS = {
    "Qwen/Qwen2.5-0.5B": {
        "json_schema": [6, 12, 13],
        "factual_recall": [3, 16, 19],
        "code_semantics": [1, 10, 21],
        "copying": [2, 7, 9],  # dispersed, use core circuit
        "delimiter_tracking": [2, 7, 9],  # fully absorbed, use core circuit
    },
    "Qwen/Qwen2.5-1.5B": {
        "json_schema": [26, 27],  # hub layers
        "factual_recall": [26, 27],
        "code_semantics": [26, 27],
    },
}

# All linear modules in Qwen2.5
ALL_LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

# Atlas-recommended modules (from Phase 1 module sweep)
ATLAS_MODULES = ["o_proj"]  # Most efficient per Phase 1 findings


def get_lora_target_modules(model_name, strategy, n_layers, seed=42):
    """Get target modules based on strategy.

    Returns (target_modules, layers_to_transform, trainable_params_info).
    target_modules: short module names (e.g., ['o_proj', 'up_proj'])
    layers_to_transform: list of layer indices to apply LoRA to (None = all)
    """
    rng = random.Random(seed)

    if strategy == "atlas_guided":
        critical_layers = ATLAS_LAYERS.get(model_name, {}).get("atlas_guided", [2, 7, 9])
        target_modules = ["o_proj", "up_proj", "down_proj"]
        return target_modules, critical_layers, {
            "strategy": "atlas_guided", "layers": critical_layers,
            "modules": target_modules, "n_target_layers": len(critical_layers)
        }

    elif strategy == "random_matched":
        critical_layers = ATLAS_LAYERS.get(model_name, {}).get("json_schema", [2, 7, 9])
        n_target = len(critical_layers)
        all_layers = list(range(n_layers))
        random_layers = sorted(rng.sample(all_layers, min(n_target, n_layers)))
        target_modules = ["o_proj", "up_proj", "down_proj"]
        return target_modules, random_layers, {
            "strategy": "random_matched", "layers": random_layers,
            "modules": target_modules, "n_target_layers": len(random_layers)
        }

    elif strategy == "all_linear":
        target_modules = ALL_LINEAR_MODULES
        return target_modules, None, {
            "strategy": "all_linear", "layers": list(range(n_layers)),
            "modules": ALL_LINEAR_MODULES, "n_target_layers": n_layers
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_lora_adapter(model_name, strategy, family, train_data, n_layers, seed=42, r=8, alpha=16, steps=100):
    """Train a LoRA adapter with the given strategy and return the model + metrics."""
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

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

    # Load model
    bundle = load_model(model_name)
    model = bundle.model
    tokenizer = bundle.tokenizer

    # Get target modules
    target_modules, layers_to_transform, module_info = get_lora_target_modules(model_name, strategy, n_layers, seed)

    # Configure LoRA
    lora_kwargs = dict(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
    lora_config = LoraConfig(**lora_kwargs)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Prepare dataset
    dataset = SimpleDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=steps)

    model.train()
    losses = []
    start = time.time()
    data_iter = iter(dataloader)

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if (step + 1) % 25 == 0:
            print(f"  Step {step+1}/{steps}, loss={losses[-1]:.4f}")

    elapsed = time.time() - start

    return {
        "model": model,
        "bundle": bundle,
        "losses": losses,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "module_info": module_info,
        "elapsed_seconds": round(elapsed, 1),
        "final_loss": losses[-1] if losses else None,
    }


def evaluate_model(model, tokenizer, eval_prompts, family):
    """Evaluate model on task-specific metrics."""
    model.eval()
    results = []

    for prompt_info in eval_prompts:
        prompt = prompt_info["prompt"]
        target = prompt_info.get("target", "")
        constraints = prompt_info.get("constraints", [])

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Exact match
        exact = 1.0 if target and generated.strip().startswith(target.strip()) else 0.0

        # Constraint satisfaction
        constraint_score = 0.0
        if constraints:
            satisfied = sum(1 for c in constraints if c.lower() in generated.lower())
            constraint_score = satisfied / len(constraints)

        results.append({
            "prompt": prompt[:100],
            "target": target,
            "generated": generated[:200],
            "exact_match": exact,
            "constraint_satisfaction": constraint_score,
        })

    # Aggregate
    mean_exact = np.mean([r["exact_match"] for r in results]) if results else 0
    mean_constraint = np.mean([r["constraint_satisfaction"] for r in results]) if results else 0

    return {
        "n_eval": len(results),
        "mean_exact_match": float(mean_exact),
        "mean_constraint_satisfaction": float(mean_constraint),
        "per_prompt": results[:5],  # Keep first 5 for inspection
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 atlas-guided LoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--family", type=str, default="json_schema",
                       choices=["json_schema", "factual_recall", "code_semantics", "copying"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    strategies = ["atlas_guided", "random_matched", "all_linear"]

    print(f"Phase 3: Atlas-guided LoRA")
    print(f"Model: {args.model}")
    print(f"Family: {args.family}")
    print(f"Strategies: {strategies}")
    print(f"Seed: {args.seed}")

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    # Get training data for the family
    family_suite = suite.filter_by_family(args.family)
    train_examples = list(family_suite.filter_by_split("train"))
    eval_examples = list(family_suite.filter_by_split("test"))
    # Fallback: if no train/test split, use all examples
    if not train_examples:
        train_examples = list(family_suite)[:int(len(family_suite)*0.8)]
        eval_examples = list(family_suite)[int(len(family_suite)*0.8):]

    if not train_examples:
        print(f"No training data for {args.family}")
        return

    train_texts = [e.clean_prompt + (e.target or "") for e in train_examples]
    eval_prompts = [{"prompt": e.clean_prompt, "target": e.target or ""} for e in eval_examples]

    # Get model info
    temp_bundle = load_model(args.model)
    n_layers = temp_bundle.model.config.num_hidden_layers
    del temp_bundle
    torch.cuda.empty_cache()

    # Run each strategy
    all_results = {}
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy}")
        print(f"{'='*50}")

        # Set seed for reproducibility (use different offset for random_matched)
        seed = args.seed if strategy != "random_matched" else args.seed + 1000

        train_result = train_lora_adapter(
            args.model, strategy, args.family, train_texts,
            n_layers, seed=seed, r=args.rank, steps=args.steps
        )

        # Evaluate
        eval_result = evaluate_model(
            train_result["model"], train_result["bundle"].tokenizer,
            eval_prompts, args.family
        )

        all_results[strategy] = {
            "trainable_params": train_result["trainable_params"],
            "total_params": train_result["total_params"],
            "param_ratio": train_result["trainable_params"] / train_result["total_params"],
            "final_loss": train_result["final_loss"],
            "module_info": train_result["module_info"],
            "eval": eval_result,
            "elapsed_seconds": train_result["elapsed_seconds"],
        }

        print(f"  Trainable params: {train_result['trainable_params']:,}")
        print(f"  Final loss: {train_result['final_loss']:.4f}")
        print(f"  Exact match: {eval_result['mean_exact_match']:.3f}")
        print(f"  Constraint satisfaction: {eval_result['mean_constraint_satisfaction']:.3f}")

        # Clean up
        del train_result["model"], train_result["bundle"]
        torch.cuda.empty_cache()

    # Compare strategies
    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")

    comparison = {
        "experiment": "atlas_guided_lora",
        "model": args.model,
        "family": args.family,
        "seed": args.seed,
        "rank": args.rank,
        "steps": args.steps,
        "strategies": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Determine winner
    best_accuracy = max(all_results.items(), key=lambda x: x[1]["eval"]["mean_exact_match"])
    best_efficiency = max(all_results.items(),
                         key=lambda x: x[1]["eval"]["mean_exact_match"] / max(x[1]["param_ratio"], 1e-10))

    comparison["winner_accuracy"] = best_accuracy[0]
    comparison["winner_efficiency"] = best_efficiency[0]

    print(f"  Best accuracy: {best_accuracy[0]} ({best_accuracy[1]['eval']['mean_exact_match']:.3f})")
    print(f"  Best efficiency: {best_efficiency[0]}")

    for strategy in strategies:
        r = all_results[strategy]
        print(f"  {strategy:15s}: params={r['trainable_params']:>10,}  loss={r['final_loss']:.4f}  "
              f"exact={r['eval']['mean_exact_match']:.3f}  constraint={r['eval']['mean_constraint_satisfaction']:.3f}")

    # Save
    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_atlas_lora_{model_slug}_{args.family}.json"
    save_json(comparison, out_path)
    print(f"\n  Results: {out_path}")

    # Register
    register_experiment(
        type="training",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(out_path)],
        status="success",
        summary=f"Atlas-guided LoRA ({args.family}): winner={comparison['winner_accuracy']}, "
                f"atlas_exact={all_results['atlas_guided']['eval']['mean_exact_match']:.3f}, "
                f"all_linear_exact={all_results['all_linear']['eval']['mean_exact_match']:.3f}",
        next="Test on more families" if comparison["winner_accuracy"] == "atlas_guided" else "Investigate why atlas-guided underperforms",
    )


if __name__ == "__main__":
    main()
