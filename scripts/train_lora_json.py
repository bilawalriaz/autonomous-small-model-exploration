"""LoRA training with SFTTrainer for proper label handling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig, get_peft_model, TaskType
from mi_atlas.model_loader import load_model_hf
from mi_atlas.training.datasets import prepare_sft_dataset
from mi_atlas.task_suite import build_default_suite, TaskSuite
from mi_atlas.eval_runner import evaluate_suite
from mi_atlas.backend import create_backend
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT
from trl import SFTTrainer, SFTConfig


def main():
    set_seed(42)

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    model.gradient_checkpointing_enable()
    tokenizer = bundle.tokenizer

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare data
    full_suite = build_default_suite()
    json_suite = full_suite.filter_by_family("json_schema")
    ds = prepare_sft_dataset(json_suite)
    print(f"  SFT examples: {len(ds)}")

    # Eval BEFORE
    print("\nEvaluating BEFORE training...")
    backend = create_backend(bundle)
    eval_before = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")
    print(f"  Before: overall={eval_before['summary']['overall_mean']:.3f}")

    # LoRA SFT training
    output_dir = str(PROJECT_ROOT / "experiments" / "adapters" / "lora_json_r8")

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        report_to="none",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\nTraining LoRA SFT (r=8, 100 steps)...")
    result = trainer.train()
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # Save adapter
    adapter_path = Path(output_dir) / "adapter"
    model.save_pretrained(str(adapter_path))
    print(f"  Adapter saved to {adapter_path}")

    # Eval AFTER
    print("\nEvaluating AFTER training...")
    eval_after = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")
    print(f"  After: overall={eval_after['summary']['overall_mean']:.3f}")

    # Compare
    print("\n  PER-FAMILY COMPARISON:")
    before_scores = eval_before["summary"]["primary_metric_by_family"]
    after_scores = eval_after["summary"]["primary_metric_by_family"]
    for fam in sorted(set(list(before_scores.keys()) + list(after_scores.keys()))):
        b = before_scores.get(fam, 0)
        a = after_scores.get(fam, 0)
        delta = a - b
        marker = "+" if delta > 0 else "-" if delta < 0 else "="
        print(f"    {fam}: {b:.3f} -> {a:.3f} ({marker}{abs(delta):.3f})")

    # Save
    comparison = {
        "before": eval_before["summary"],
        "after": eval_after["summary"],
        "train_loss": result.training_loss,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "adapter_path": str(adapter_path),
        "lora_rank": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    save_json(comparison, str(PROJECT_ROOT / "experiments" / "results" / "lora_json_comparison.json"))

    register_experiment(
        type="training", model=bundle.model_name, backend="hf",
        config="config/training_plan.yaml", inputs=[],
        outputs=[str(adapter_path)], status="success",
        summary=f"LoRA r=8 JSON SFT: loss={result.training_loss:.4f}",
        next="Compare layer ablation before/after LoRA",
    )
    print("\nLoRA training complete!")


if __name__ == "__main__":
    main()
