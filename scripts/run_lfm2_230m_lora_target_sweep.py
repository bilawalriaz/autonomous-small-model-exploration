#!/usr/bin/env python3
"""
LoRA Target Module Sweep for LFM2.5-230M.

Trains adapters on different target module configurations and compares effectiveness.

Configs:
  1. All-linear (q_proj, k_proj, v_proj, out_proj, gate_proj, up_proj, down_proj) on ALL layers
  2. Attn-only (q_proj, k_proj, v_proj, out_proj) on attention layers L2,4,6,8,10,12
  3. MLP-only (gate_proj, up_proj, down_proj) on ALL layers
  4. Conv-only (in_proj, out_proj) on conv layers L0,1,3,5,7,9,11,13
  5. Atlas-guided (o_proj + MLP) on hub layers L0,L2,L4,L5
  6. Atlas-guided (full) on hub layers L0,L2,L4,L5
"""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
import time
import gc
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_ID = "LiquidAI/LFM2.5-230M"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "adapters" / "lfm2_230m"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "lfm2_230m_lora_training.json"

# Architecture constants
NUM_LAYERS = 14
LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv", "full_attention", "conv",
]
CONV_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "conv"]
ATTN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]

# Short module names for PEFT layers_to_transform
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
CONV_MODULES = ["in_proj", "out_proj"]
ALL_LINEAR_MODULES = ATTN_MODULES + MLP_MODULES


def log(msg):
    """Print with flush."""
    print(msg, flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_training_data():
    """Load arithmetic task data."""
    data_path = PROJECT_ROOT / "data" / "tasks" / "canonical_short" / "arithmetic.json"
    with open(data_path) as f:
        data = json.load(f)
    examples = data["examples"]
    train_examples = examples[:100]
    val_examples = examples[100:160]
    test_examples = examples[160:]
    return train_examples, val_examples, test_examples


def format_for_sft(examples, tokenizer, max_length=128):
    """Format examples for SFT: concatenate prompt+target, mask prompt in labels."""
    formatted = []
    for ex in examples:
        prompt = ex["prompt"]
        target = ex["target"]
        full_text = prompt + target

        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_length, add_special_tokens=False)["input_ids"][0]
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=max_length, add_special_tokens=False)["input_ids"][0]

        labels = full_ids.clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100

        # Pad
        pad_len = max_length - len(full_ids)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        if pad_len > 0:
            attention_mask[-pad_len:] = 0

        formatted.append({
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return formatted


def compute_kl_divergence(model, tokenizer, test_prompts, device):
    """Compute KL divergence on test prompts (last-token distribution)."""
    model.eval()
    kl_values = []
    top1_correct = 0
    total = 0

    for ex in test_prompts:
        prompt = ex["prompt"]
        target = ex["target"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=128, add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if len(target_ids) == 0:
            continue
        target_token_id = target_ids[0].item()
        top1 = logits.argmax().item()
        if top1 == target_token_id:
            top1_correct += 1
        total += 1
        uniform = torch.ones_like(probs) / len(probs)
        kl = torch.nn.functional.kl_div(torch.log(probs + 1e-10), uniform, reduction="sum").item()
        kl_values.append(kl)

    return {
        "mean_kl_from_uniform": round(float(np.mean(kl_values)), 4) if kl_values else 0.0,
        "top1_accuracy": round(top1_correct / total, 4) if total > 0 else 0.0,
        "n_evaluated": total,
    }


def compute_kl_between_models(base_logits, lora_logits):
    """KL divergence between base and LoRA model on same inputs."""
    kl_values = []
    for i in range(min(len(base_logits), len(lora_logits))):
        probs_base = torch.softmax(base_logits[i].float(), dim=-1)
        probs_lora = torch.softmax(lora_logits[i].float(), dim=-1)
        kl = torch.nn.functional.kl_div(torch.log(probs_lora + 1e-10), probs_base, reduction="sum").item()
        kl_values.append(kl)
    return round(float(np.mean(kl_values)), 4) if kl_values else 0.0


def count_lora_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_base_logits(model, tokenizer, prompts, device):
    """Get base model logits for KL comparison."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p["prompt"], return_tensors="pt", truncation=True,
                            max_length=128, add_special_tokens=False)["input_ids"].to(device)
            logits = model(ids).logits[0, -1, :].detach().cpu()
            all_logits.append(logits)
    return all_logits


# ============ CONFIGURATIONS ============

CONFIGS = {
    "all_linear_all_layers": {
        "description": "All-linear (q,k,v,out,gate,up,down) on ALL 14 layers",
        "target_modules": ALL_LINEAR_MODULES,
        "layers_to_transform": None,
    },
    "attn_only_attn_layers": {
        "description": "Attention-only (q,k,v,out) on attention layers L2,4,6,8,10,12",
        "target_modules": ATTN_MODULES,
        "layers_to_transform": ATTN_LAYERS,
    },
    "mlp_only_all_layers": {
        "description": "MLP-only (gate,up,down) on ALL 14 layers",
        "target_modules": MLP_MODULES,
        "layers_to_transform": None,
    },
    "conv_only_conv_layers": {
        "description": "Conv-only (in_proj,out_proj) on conv layers L0,1,3,5,7,9,11,13",
        "target_modules": CONV_MODULES,
        "layers_to_transform": CONV_LAYERS,
    },
    "atlas_o_proj_mlp_hubs": {
        "description": "Atlas-guided (out_proj + MLP) on hub layers L0,L2,L4,L5",
        "target_modules": ["out_proj"] + MLP_MODULES,
        "layers_to_transform": [0, 2, 4, 5],
    },
    "atlas_full_hubs": {
        "description": "Atlas-guided (full linear) on hub layers L0,L2,L4,L5",
        "target_modules": ALL_LINEAR_MODULES,
        "layers_to_transform": [0, 2, 4, 5],
    },
}


def train_single_config(config_name, config, model, tokenizer, train_data, test_data,
                        device, use_bf16, base_logits):
    """Train a single LoRA configuration."""
    log(f"\n{'='*60}")
    log(f"Training: {config_name}")
    log(f"  {config['description']}")
    log(f"{'='*60}")

    adapter_dir = OUTPUT_DIR / config_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Build LoraConfig
    lora_kwargs = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": config["target_modules"],
        "lora_dropout": 0.05,
        "task_type": TaskType.CAUSAL_LM,
        "bias": "none",
    }
    if config["layers_to_transform"] is not None:
        lora_kwargs["layers_to_transform"] = config["layers_to_transform"]

    peft_config = LoraConfig(**lora_kwargs)

    log("  Applying LoRA adapter...")
    try:
        peft_model = get_peft_model(model, peft_config)
    except Exception as e:
        log(f"  ERROR applying LoRA: {e}")
        traceback.print_exc()
        return {
            "config_name": config_name,
            "description": config["description"],
            "status": "failed_lora_apply",
            "error": str(e),
        }

    trainable, total = count_lora_params(peft_model)
    log(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Format training data
    log("  Tokenizing training data...")
    train_formatted = format_for_sft(train_data, tokenizer, max_length=128)
    dataset = Dataset.from_list(train_formatted)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training
    log("  Starting training (100 steps)...")
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        max_steps=100,
        warmup_steps=10,
        save_steps=100,
        logging_steps=10,
        fp16=False,
        bf16=use_bf16,
        seed=42,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=(device.type == "cpu"),
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    try:
        train_result = trainer.train()
    except Exception as e:
        log(f"  ERROR during training: {e}")
        traceback.print_exc()
        return {
            "config_name": config_name,
            "description": config["description"],
            "status": "failed_training",
            "error": str(e),
        }
    train_time = time.time() - t0

    # Extract loss history
    loss_history = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_history.append({"step": entry.get("step", 0), "loss": round(entry["loss"], 4)})

    # Save adapter
    adapter_path = adapter_dir / "adapter"
    peft_model.save_pretrained(str(adapter_path))
    log(f"  Saved adapter to {adapter_path}")

    # Evaluate KL divergence
    log("  Evaluating KL divergence...")
    with torch.no_grad():
        lora_logits = []
        for p in test_data[:20]:
            ids = tokenizer(p["prompt"], return_tensors="pt", truncation=True,
                            max_length=128, add_special_tokens=False)["input_ids"].to(device)
            logits = peft_model(ids).logits[0, -1, :].detach().cpu()
            lora_logits.append(logits)

    kl_after = compute_kl_between_models(base_logits, lora_logits)
    full_eval = compute_kl_divergence(peft_model, tokenizer, test_data, device)

    # Clean up - restore base model
    peft_model.disable_adapter_layers()
    del peft_model
    gc.collect()

    result = {
        "config_name": config_name,
        "description": config["description"],
        "status": "success",
        "target_modules": config["target_modules"],
        "layers_to_transform": config["layers_to_transform"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "train_time_s": round(train_time, 1),
        "final_loss": round(train_result.training_loss, 4),
        "loss_history": loss_history,
        "kl_base_vs_lora": kl_after,
        "post_train_eval": full_eval,
    }

    log(f"  Done in {train_time:.1f}s | Final loss: {train_result.training_loss:.4f} | KL: {kl_after:.4f}")
    log(f"  Test accuracy: {full_eval['top1_accuracy']:.2%}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None, help="Config names to run")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    log("=" * 60)
    log("LFM2.5-230M LoRA Target Module Sweep")
    log("=" * 60)

    # Check existing results
    if RESULTS_PATH.exists() and not args.force:
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
        completed = {r["config_name"] for r in existing.get("results", []) if r.get("status") == "success"}
        log(f"  Found existing results for: {completed}")
    else:
        existing = None
        completed = set()

    # Load model
    log(f"\nLoading {MODEL_ID}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    log(f"  Model loaded on {device} (dtype={dtype}, bf16={use_bf16})")
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")

    # Load data
    log("\nLoading training data...")
    train_data, val_data, test_data = load_training_data()
    log(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Evaluate base model
    log("\nEvaluating base model on test set...")
    base_eval = compute_kl_divergence(model, tokenizer, test_data, device)
    log(f"  Base model test accuracy: {base_eval['top1_accuracy']:.2%}")

    # Pre-compute base logits for KL comparison
    log("  Pre-computing base logits for KL comparison...")
    base_logits = get_base_logits(model, tokenizer, test_data[:20], device)

    # Determine configs to run
    configs_to_run = args.configs or list(CONFIGS.keys())
    configs_to_run = [c for c in configs_to_run if c not in completed or args.force]
    log(f"\nConfigs to run: {configs_to_run}")

    results_list = existing["results"] if existing else []

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            log(f"  WARNING: Unknown config '{config_name}', skipping")
            continue

        result = train_single_config(
            config_name, CONFIGS[config_name],
            model, tokenizer, train_data, test_data,
            device, use_bf16, base_logits,
        )
        results_list.append(result)

        # Save intermediate results
        save_results(results_list, base_eval)

    # Print summary
    log("\n" + "=" * 80)
    log("SUMMARY TABLE")
    log("=" * 80)
    log(f"{'Config':<28} {'Params':>10} {'%':>6} {'Loss':>8} {'KL':>8} {'Acc':>8} {'Time':>8}")
    log("-" * 80)
    for r in results_list:
        if r.get("status") != "success":
            log(f"{r['config_name']:<28} FAILED: {r.get('error', 'unknown')}")
            continue
        log(f"{r['config_name']:<28} {r['trainable_params']:>10,} {r['trainable_pct']:>5.2f}% "
            f"{r['final_loss']:>8.4f} {r['kl_base_vs_lora']:>8.4f} "
            f"{r['post_train_eval']['top1_accuracy']:>7.2%} {r['train_time_s']:>7.1f}s")

    log(f"\nBase model test accuracy: {base_eval['top1_accuracy']:.2%}")
    log(f"\nResults saved to: {RESULTS_PATH}")
    save_results(results_list, base_eval)
    log("Done.")


def save_results(results_list, base_eval):
    output = {
        "experiment": "lfm2_230m_lora_target_sweep",
        "model": MODEL_ID,
        "timestamp": now_iso(),
        "training_config": {
            "r": 8, "alpha": 16, "dropout": 0.05,
            "max_steps": 100, "batch_size": 1, "lr": 2e-4, "seed": 42,
            "gradient_checkpointing": True,
        },
        "architecture": {
            "n_layers": NUM_LAYERS,
            "layer_types": LAYER_TYPES,
            "conv_layers": CONV_LAYERS,
            "attn_layers": ATTN_LAYERS,
        },
        "base_eval": base_eval,
        "results": results_list,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
