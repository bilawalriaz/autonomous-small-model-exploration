"""SFT training on a single skill family, then comparison.

Train on JSON schema data and compare internals before/after.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset, split_dataset
from mi_atlas.training.sft import train_sft
from mi_atlas.eval_runner import evaluate_suite
from mi_atlas.backend import create_backend
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def main():
    set_seed(42)

    print("=" * 60)
    print("SFT TRAINING ON JSON SCHEMA FAMILY")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer

    # Prepare JSON-only dataset
    full_suite = build_default_suite()
    json_suite = full_suite.filter_by_family("json_schema")
    print(f"  JSON examples: {len(json_suite)}")

    # Also include some other families for interference measurement
    other_suite = TaskSuite([
        ex for ex in full_suite if ex.family in ["factual_recall", "arithmetic", "copying"]
    ])
    print(f"  Control examples: {len(other_suite)}")

    # Prepare SFT dataset
    sft_dataset = prepare_sft_dataset(json_suite)
    print(f"  SFT dataset: {len(sft_dataset)} examples")

    # Evaluate BEFORE training
    print("\n  Evaluating BEFORE training...")
    backend = create_backend(bundle)
    eval_before = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")
    print(f"  Before: overall mean = {eval_before['summary']['overall_mean']:.3f}")

    # Train
    output_dir = str(PROJECT_ROOT / "experiments" / "checkpoints" / "sft_json_schema")
    print(f"\n  Training SFT (max_steps=100, lr=2e-5)...")
    train_result = train_sft(
        model, tokenizer, sft_dataset, output_dir,
        config_override={"max_steps": 100, "learning_rate": 2e-5, "batch_size": 2}
    )
    print(f"  Training loss: {train_result['train_loss']:.4f}")

    # Evaluate AFTER training
    print("\n  Evaluating AFTER training...")
    eval_after = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")
    print(f"  After: overall mean = {eval_after['summary']['overall_mean']:.3f}")

    # Compare
    print("\n  COMPARISON:")
    before_scores = eval_before['summary']['primary_metric_by_family']
    after_scores = eval_after['summary']['primary_metric_by_family']
    for fam in sorted(set(list(before_scores.keys()) + list(after_scores.keys()))):
        b = before_scores.get(fam, 0)
        a = after_scores.get(fam, 0)
        delta = a - b
        marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"    {fam}: {b:.3f} → {a:.3f} ({marker}{abs(delta):.3f})")

    # Save results
    comparison = {
        "before": eval_before['summary'],
        "after": eval_after['summary'],
        "training": train_result,
        "per_family_delta": {
            fam: after_scores.get(fam, 0) - before_scores.get(fam, 0)
            for fam in set(list(before_scores.keys()) + list(after_scores.keys()))
        }
    }
    output_path = PROJECT_ROOT / "experiments" / "results" / "sft_json_comparison.json"
    save_json(comparison, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="training",
        model=bundle.model_name,
        backend="hf",
        config="config/training_plan.yaml",
        inputs=[],
        outputs=[output_dir, str(output_path)],
        status="success",
        summary=f"SFT JSON: loss={train_result['train_loss']:.4f}",
        next="Layer ablation on trained model to compare with base",
    )
    print("  SFT training complete!")


if __name__ == "__main__":
    main()
