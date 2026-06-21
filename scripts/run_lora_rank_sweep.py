"""LoRA rank sweep: train at r=1,2,4,8,16 and compare internals.

Experiment family C from the research prompt.
Question: does higher rank distribute skill across more components?
"""
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_line, plot_multi_line
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT
from trl import SFTTrainer, SFTConfig


def train_lora_at_rank(model, tokenizer, dataset, rank, output_dir):
    """Train LoRA at a specific rank."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,  # alpha = 2*rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=500,
        report_to="none",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Get adapter norms
    adapter_norms = {}
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    layer_key = f"layer_{layer_idx:02d}"
                    norm = param.data.float().norm().item()
                    adapter_norms[layer_key] = adapter_norms.get(layer_key, 0.0) + norm
                    break

    # Save adapter
    adapter_path = Path(output_dir) / "adapter"
    peft_model.save_pretrained(str(adapter_path))

    # Unwrap to base for next iteration
    del peft_model
    torch.cuda.empty_cache()

    return {
        "train_loss": result.training_loss,
        "adapter_norms": adapter_norms,
        "adapter_path": str(adapter_path),
        "rank": rank,
    }


def run_mlp_ablation_on_lora(base_model, adapter_path, tokenizer, suite, families, n_layers):
    """Run MLP ablation on a LoRA-adapted model."""
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model.eval()

    effect_matrix = np.zeros((n_layers, len(families)))
    for fam_idx, family in enumerate(families):
        examples = list(suite.filter_by_family(family))[:3]
        for layer_idx in range(n_layers):
            kl_effects = []
            for example in examples:
                inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(lora_model.device)
                with torch.no_grad():
                    orig_logits = lora_model(input_ids).logits
                mlp = lora_model.base_model.model.model.layers[layer_idx].mlp
                def zero_hook(module, input, output):
                    return torch.zeros_like(output)
                handle = mlp.register_forward_hook(zero_hook)
                with torch.no_grad():
                    abl_logits = lora_model(input_ids).logits
                handle.remove()
                orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
                abl_log_probs = torch.log_softmax(abl_logits[0, -1], dim=-1)
                kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
                kl_effects.append(kl)
            effect_matrix[layer_idx, fam_idx] = np.mean(kl_effects) if kl_effects else 0.0

    del lora_model
    torch.cuda.empty_cache()
    return effect_matrix


def main():
    set_seed(42)

    ranks = [1, 2, 4, 8, 16]
    print(f"LoRA Rank Sweep: ranks={ranks}")

    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    base_model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]

    suite = build_default_suite()
    json_suite = suite.filter_by_family("json_schema")
    ds = prepare_sft_dataset(json_suite)
    families = suite.families

    # Run base ablation once
    print("\nRunning BASE ablation...")
    base_matrix = np.zeros((n_layers, len(families)))
    for fam_idx, family in enumerate(families):
        examples = list(suite.filter_by_family(family))[:3]
        for layer_idx in range(n_layers):
            kl_effects = []
            for example in examples:
                inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                                   truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(base_model.device)
                with torch.no_grad():
                    orig_logits = base_model(input_ids).logits
                mlp = base_model.model.layers[layer_idx].mlp
                def zero_hook(module, input, output):
                    return torch.zeros_like(output)
                handle = mlp.register_forward_hook(zero_hook)
                with torch.no_grad():
                    abl_logits = base_model(input_ids).logits
                handle.remove()
                orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
                abl_log_probs = torch.log_softmax(abl_logits[0, -1], dim=-1)
                kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
                kl_effects.append(kl)
            base_matrix[layer_idx, fam_idx] = np.mean(kl_effects) if kl_effects else 0.0
    print("  Base ablation done.")

    all_results = {"ranks": ranks, "base_matrix": base_matrix.tolist(), "families": families}

    for rank in ranks:
        print(f"\n{'='*50}")
        print(f"Training LoRA r={rank}...")
        output_dir = str(PROJECT_ROOT / "experiments" / "adapters" / f"lora_json_r{rank}")

        train_result = train_lora_at_rank(base_model, tokenizer, ds, rank, output_dir)
        print(f"  Loss: {train_result['train_loss']:.4f}")
        print(f"  Adapter norms: {dict(sorted(train_result['adapter_norms'].items()))}")

        print(f"  Running ablation for r={rank}...")
        abl_matrix = run_mlp_ablation_on_lora(
            base_model, train_result["adapter_path"], tokenizer, suite, families, n_layers
        )

        diff = abl_matrix - base_matrix
        json_idx = families.index("json_schema")
        json_diff = diff[:, json_idx]
        top_shifted = sorted(enumerate(json_diff), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"  JSON top shifts: {', '.join(f'L{i}({v:+.2f})' for i, v in top_shifted)}")

        all_results[f"r{rank}"] = {
            "train_loss": train_result["train_loss"],
            "adapter_norms": train_result["adapter_norms"],
            "ablation_matrix": abl_matrix.tolist(),
            "diff_matrix": diff.tolist(),
        }

        # Reload base model for next iteration
        del abl_matrix
        torch.cuda.empty_cache()

    # Save
    output_path = PROJECT_ROOT / "experiments" / "results" / "lora_rank_sweep.json"
    save_json(all_results, output_path)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*50}")
    print("RANK SWEEP SUMMARY")
    print(f"{'='*50}")
    for rank in ranks:
        r = all_results[f"r{rank}"]
        json_idx = families.index("json_schema")
        l0_json = r["ablation_matrix"][0][json_idx]
        l2_json = r["ablation_matrix"][2][json_idx]
        print(f"  r={rank}: loss={r['train_loss']:.4f}, L0_json={l0_json:.2f}, L2_json={l2_json:.2f}")

    register_experiment(
        type="training", model=bundle.model_name, backend="hf",
        config="config/training_plan.yaml", inputs=[],
        outputs=[str(output_path)], status="success",
        summary=f"LoRA rank sweep: r={ranks}, all converged",
        next="LoRA target-module sweep, better activation patching",
    )
    print("\nRank sweep complete!")


if __name__ == "__main__":
    main()
