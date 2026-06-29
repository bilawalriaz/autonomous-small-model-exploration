#!/usr/bin/env python3
"""LFM2.5-230M LoRA Target Module Sweep — train adapters on different module configs."""
import json, torch, sys, os
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_230m"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

# Architecture constants
CONV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13]
ATTN_LAYERS = [2, 4, 6, 8, 10, 12]
ALL_LAYERS = list(range(14))
HUB_LAYERS = [0, 2, 4, 5]  # From atlas: L0 (hub), L2, L4, L5

# LoRA configs to test
CONFIGS = {
    "all_linear": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "in_proj"],
        "layers_to_transform": None,  # All layers
        "description": "All linear modules, all layers"
    },
    "attn_only": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "layers_to_transform": ATTN_LAYERS,
        "description": "Attention modules on attn layers only"
    },
    "mlp_only": {
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "layers_to_transform": None,
        "description": "MLP modules on all layers"
    },
    "conv_proj_only": {
        "target_modules": ["in_proj", "out_proj"],
        "layers_to_transform": CONV_LAYERS,
        "description": "Conv projections on conv layers only"
    },
    "attn_proj_only": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "layers_to_transform": ALL_LAYERS,
        "description": "Attention projections on all layers (including where they don't exist - PEFT skips)"
    },
    "atlas_guided": {
        "target_modules": ["out_proj", "gate_proj", "up_proj", "down_proj"],
        "layers_to_transform": HUB_LAYERS,
        "description": "Atlas-guided: o_proj + MLP on hub layers L0,L2,L4,L5"
    },
    "atlas_full": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "in_proj"],
        "layers_to_transform": HUB_LAYERS,
        "description": "Atlas-guided: all linear on hub layers L0,L2,L4,L5"
    },
}


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length",
                            return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            self.examples.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_training_data(tokenizer, n=80):
    """Load arithmetic training data."""
    data_path = PROJECT / "data" / "tasks" / "canonical_short" / "arithmetic.json"
    with open(data_path) as f:
        data = json.load(f)
    examples = data.get("examples", [])
    train_ex = [e for e in examples if e.get("metadata", {}).get("split") == "train"][:n]
    if len(train_ex) < n:
        train_ex = examples[:n]
    # Format as prompt + target
    texts = [f"{e['prompt']}{e['target']}" for e in train_ex]
    return texts


def compute_eval_kl(model, tokenizer, prompts, baseline_logits):
    """Compute KL divergence between model and baseline on eval prompts."""
    kls = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits.float(), -1),
            torch.softmax(baseline_logits.to(logits.device).float(), -1),
            reduction='batchmean'
        ).item()
        kls.append(kl)
    return np.mean(kls)


def train_lora_config(config_name, config, model, tokenizer, train_texts, eval_prompts, baseline_logits, seed=42):
    """Train a LoRA adapter with given config and return results."""
    print(f"\n  Training: {config_name} ({config['description']})")

    # Reload fresh model for each config
    model_fresh = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=config["target_modules"],
        layers_to_transform=config["layers_to_transform"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    try:
        peft_model = get_peft_model(model_fresh, lora_config)
    except Exception as e:
        print(f"    FAILED to create PEFT model: {e}")
        del model_fresh
        return {"config": config_name, "status": "failed", "error": str(e)}

    # Count trainable params
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # Pre-training KL
    pre_kl = compute_eval_kl(peft_model, tokenizer, eval_prompts, baseline_logits)
    print(f"    Pre-train KL: {pre_kl:.4f}")

    # Train
    dataset = SimpleDataset(train_texts, tokenizer)
    output_dir = str(ADAPTERS_DIR / config_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        logging_steps=20,
        save_steps=50,
        save_total_limit=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
    )

    try:
        train_result = trainer.train()
        final_loss = train_result.training_loss
        print(f"    Final loss: {final_loss:.4f}")
    except Exception as e:
        print(f"    Training FAILED: {e}")
        del peft_model, model_fresh
        return {"config": config_name, "status": "failed", "error": str(e)}

    # Post-training KL
    post_kl = compute_eval_kl(peft_model, tokenizer, eval_prompts, baseline_logits)
    kl_change = post_kl - pre_kl
    print(f"    Post-train KL: {post_kl:.4f} (change: {kl_change:+.4f})")

    # Save adapter
    try:
        peft_model.save_pretrained(output_dir)
        print(f"    Saved adapter to {output_dir}")
    except Exception as e:
        print(f"    Could not save adapter: {e}")

    # Cleanup
    del peft_model, model_fresh
    torch.cuda.empty_cache()

    return {
        "config": config_name,
        "description": config["description"],
        "status": "success",
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "final_loss": round(final_loss, 4),
        "pre_train_kl": round(pre_kl, 4),
        "post_train_kl": round(post_kl, 4),
        "kl_change": round(kl_change, 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=None, help="Subset of configs to run")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    # Load data
    train_texts = load_training_data(tokenizer, n=80)
    print(f"Training examples: {len(train_texts)}")

    eval_prompts = [
        "The capital of France is",
        "2 + 2 =",
        '{"key":',
        "def hello():",
        "The opposite of hot is",
    ]

    # Compute baseline logits
    print("Computing baseline logits...")
    baseline_logits = {}
    for prompt in eval_prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            baseline_logits[prompt] = model(ids).logits.cpu()

    # Use first prompt's logits as reference
    ref_logits = list(baseline_logits.values())[0]

    # Run sweeps
    configs_to_run = args.configs or list(CONFIGS.keys())
    all_results = []

    print(f"\n{'='*60}")
    print(f"LoRA TARGET MODULE SWEEP ({len(configs_to_run)} configs)")
    print(f"{'='*60}")

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"  Unknown config: {config_name}")
            continue
        result = train_lora_config(
            config_name, CONFIGS[config_name],
            model, tokenizer, train_texts, eval_prompts, ref_logits, args.seed
        )
        all_results.append(result)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"lfm2_230m_lora_sweep_seed{args.seed}_{ts}.json"
    output = {
        "model": MODEL, "seed": args.seed, "timestamp": ts,
        "n_training_examples": len(train_texts),
        "max_steps": 100, "lr": 2e-4, "lora_r": 8, "lora_alpha": 16,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Params':>10} {'Loss':>8} {'Pre-KL':>8} {'Post-KL':>8} {'KL Δ':>8}")
    print("-" * 65)
    for r in all_results:
        if r["status"] == "success":
            print(f"{r['config']:<20} {r['trainable_params']:>10,} {r['final_loss']:>8.4f} "
                  f"{r['pre_train_kl']:>8.4f} {r['post_train_kl']:>8.4f} {r['kl_change']:>+8.4f}")
        else:
            print(f"{r['config']:<20} {'FAILED':>10}")

    # Best config
    successful = [r for r in all_results if r["status"] == "success"]
    if successful:
        best = min(successful, key=lambda x: x["kl_change"])
        print(f"\nBest config: {best['config']} (KL change: {best['kl_change']:+.4f})")
        most_efficient = min(successful, key=lambda x: x["trainable_params"])
        print(f"Most efficient: {most_efficient['config']} ({most_efficient['trainable_params']:,} params)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
