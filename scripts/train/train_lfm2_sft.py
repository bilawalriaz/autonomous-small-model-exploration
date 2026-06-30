#!/usr/bin/env python3
"""Central SFT training script for LFM2.5-230M with LoRA.

CLI:
    python scripts/train/train_lfm2_sft.py \
        --config configs/sft/baseline_lfm2_230m_quality.yaml \
        --dataset data/sft/format_ablation/multi_turn_concise.jsonl \
        --run-id lfm2_230m_quality_format_ablation_multi_turn_concise
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def detect_format(records: list[dict]) -> str:
    """Detect dataset format: 'chat' (messages) or 'alpaca_flat' (instruction/input/output)."""
    if not records:
        raise ValueError("Empty dataset")
    sample = records[0]
    if "messages" in sample:
        return "chat"
    if "text" in sample:
        return "text"
    if "instruction" in sample or "output" in sample:
        return "alpaca_flat"
    raise ValueError(f"Unknown dataset format. Keys: {list(sample.keys())}")


def convert_alpaca_to_chat(records: list[dict]) -> list[dict]:
    """Convert alpaca_flat format to chat messages format."""
    converted = []
    for rec in records:
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        output = rec.get("output", "")

        user_msg = instruction
        if inp:
            user_msg = f"{instruction}\n\n{inp}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
        converted.append({"messages": messages})
    return converted


def convert_alpaca_to_text(records: list[dict], tokenizer) -> list[dict]:
    """Convert alpaca_flat format to text using chat template."""
    converted = []
    for rec in records:
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        output = rec.get("output", "")

        user_msg = instruction
        if inp:
            user_msg = f"{instruction}\n\n{inp}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        converted.append({"text": text})
    return converted


def main():
    parser = argparse.ArgumentParser(description="SFT training for LFM2.5-230M with LoRA")
    parser.add_argument("--config", required=True, help="SFT config YAML path")
    parser.add_argument("--dataset", required=True, help="Training dataset JSONL path")
    parser.add_argument("--run-id", required=True, help="Run identifier for output")
    parser.add_argument("--force", action="store_true", help="Overwrite existing adapter")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]
    dtype = get_dtype(config["model"].get("dtype", "bf16"))
    adapter_cfg = config.get("adapter", {})
    training_cfg = config.get("training", {})

    # Output directory
    adapter_dir = PROJECT_ROOT / "adapters" / args.run_id
    if adapter_dir.exists() and not args.force:
        log.warning(f"Adapter dir exists: {adapter_dir}. Use --force to overwrite.")
        sys.exit(1)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_path = PROJECT_ROOT / args.dataset
    if not dataset_path.exists():
        log.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    records = load_jsonl(dataset_path)
    log.info(f"Loaded {len(records)} records from {dataset_path}")

    # Detect format and convert
    fmt = detect_format(records)
    log.info(f"Detected dataset format: {fmt}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert to appropriate format
    if fmt == "alpaca_flat":
        # Convert to text field using chat template
        text_records = convert_alpaca_to_text(records, tokenizer)
        dataset = Dataset.from_list(text_records)
        use_chat_format = False
    elif fmt == "chat":
        dataset = Dataset.from_list(records)
        use_chat_format = True
    elif fmt == "text":
        dataset = Dataset.from_list(records)
        use_chat_format = False
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Train/eval split
    split = dataset.train_test_split(
        test_size=config.get("data", {}).get("eval_split", 0.1),
        seed=training_cfg.get("seed", 42),
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model
    log.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=adapter_cfg.get("r", 8),
        lora_alpha=adapter_cfg.get("alpha", 16),
        lora_dropout=adapter_cfg.get("dropout", 0.05),
        target_modules=adapter_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=adapter_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments (use SFTConfig for TRL 1.7+ compatibility)
    max_seq_length = training_cfg.get("max_seq_length", 1024)
    training_args = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=training_cfg.get("batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation", 4),
        max_steps=training_cfg.get("max_steps", 300),
        warmup_steps=training_cfg.get("warmup_steps", 10),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        save_steps=training_cfg.get("save_steps", 150),
        eval_steps=training_cfg.get("eval_steps", 50),
        logging_steps=training_cfg.get("logging_steps", 10),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        seed=training_cfg.get("seed", 42),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        report_to=training_cfg.get("report_to", "none"),
        optim="adafactor",
        lr_scheduler_type=training_cfg.get("lr_scheduler", "constant"),
        eval_strategy="steps",
        load_best_model_at_end=False,
        max_length=max_seq_length,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    log.info("Starting training...")
    result = trainer.train()
    log.info(f"Training complete. Loss: {result.training_loss:.4f}, Steps: {result.global_step}")

    # Save adapter
    model.save_pretrained(str(adapter_dir / "adapter"))
    tokenizer.save_pretrained(str(adapter_dir / "adapter"))
    log.info(f"Saved adapter to {adapter_dir / 'adapter'}")

    # Save metadata
    metadata = {
        "run_id": args.run_id,
        "model": model_name,
        "config": config,
        "dataset": str(dataset_path),
        "dataset_format": fmt,
        "dataset_size": len(records),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "final_loss": result.training_loss,
        "global_step": result.global_step,
        "training_metrics": result.metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter_config": {
            "r": adapter_cfg.get("r", 8),
            "alpha": adapter_cfg.get("alpha", 16),
            "target_modules": adapter_cfg.get("target_modules"),
        },
    }
    with open(adapter_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved metadata to {adapter_dir / 'metadata.json'}")

    # Save config snapshot
    with open(adapter_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log.info(f"Done. Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
