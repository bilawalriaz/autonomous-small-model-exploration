#!/usr/bin/env python3
"""Unsloth QLoRA SFT for LiquidAI/LFM2.5-8B-A1B on an 8GB RTX 2070.

Default recipe is intentionally conservative:
- 4-bit base weights
- fp16 training
- micro-batch 1 with gradient accumulation
- rank-8 LoRA
- 1024-token context

Example:
    python scripts/train/train_lfm25_8b_unsloth_qlora.py \
        --dataset /home/billz/scored/exports/sft_strict_q8_response.jsonl \
        --output-dir /home/billz/results/lfm25_8b_sft_q8_strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Unsloth patches Transformers/TRL at import time; keep it before TRL imports.
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

log = logging.getLogger("lfm25_8b_unsloth_qlora")


DEFAULT_MODEL = "LiquidAI/LFM2.5-8B-A1B"
DEFAULT_TARGET_CANDIDATES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", required=True, help="JSONL with text, messages, or prompt/response fields")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="auto",
        help="Comma-separated modules, or auto. Auto keeps known LFM/Qwen-style projection names present in the model.",
    )
    parser.add_argument("--optim", default="adamw_8bit", help="Use adamw_8bit by default for QLoRA; adafactor is allowed.")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--packing", action="store_true", help="Pack short samples together. Off by default for strict output tasks.")
    parser.add_argument(
        "--padding-free",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable by default because aero's 2070/Unsloth stack can stall at step 0 on padding-free training.",
    )
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Unsloth load-time GPU memory target. Default raises Unsloth's conservative 0.5 for 8GB cards.",
    )
    parser.add_argument(
        "--offload-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload embeddings if 4-bit model placement still does not fit.",
    )
    parser.add_argument(
        "--device-map",
        choices=["sequential", "auto", "cuda"],
        default="cuda",
        help="Use cuda to force all quantized modules onto GPU; sequential/auto may dispatch to CPU and fail for bnb 4-bit.",
    )
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Load and format data, then exit before loading the model.")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        return [
            {"role": str(msg["role"]), "content": str(msg["content"])}
            for msg in record["messages"]
            if isinstance(msg, dict) and "role" in msg and "content" in msg
        ]

    prompt = record.get("prompt") or record.get("instruction") or record.get("input")
    response = (
        record.get("assistant_response")
        or record.get("winner_response")
        or record.get("response")
        or record.get("output")
        or record.get("completion")
    )
    if prompt is None or response is None:
        raise ValueError(f"Cannot infer prompt/response fields from keys: {sorted(record.keys())}")

    return [
        {"role": "user", "content": str(prompt).strip()},
        {"role": "assistant", "content": str(response).strip()},
    ]


def fallback_chat_text(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()
        if role == "user":
            chunks.append(f"<|user|>\n{content}")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}")
        else:
            chunks.append(f"<|{role}|>\n{content}")
    return "\n".join(chunks).strip()


def records_to_dataset(records: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    rows: list[dict[str, str]] = []
    for record in records:
        if isinstance(record.get("text"), str):
            text = record["text"].strip()
        else:
            messages = normalize_messages(record)
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).strip()
            else:
                text = fallback_chat_text(messages)
        if text:
            rows.append({"text": text})

    if not rows:
        raise ValueError("No usable training rows after conversion")
    return Dataset.from_list(rows)


def resolve_target_modules(model: Any, requested: str) -> list[str]:
    if requested != "auto":
        return [item.strip() for item in requested.split(",") if item.strip()]

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in DEFAULT_TARGET_CANDIDATES if name in module_names]
    if not targets:
        raise ValueError(
            "Could not auto-detect LoRA target modules. "
            "Re-run with --target-modules q_proj,k_proj,v_proj,o_proj,gate_up_proj,down_proj"
        )
    return targets


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.force and not args.dry_run:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --force to continue.")

    run_id = args.run_id or f"lfm25_8b_a1b_unsloth_qlora_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log.info("Run id: %s", run_id)
    log.info("Dataset: %s", dataset_path)
    log.info("Output: %s", output_dir)

    records = load_jsonl(dataset_path)
    log.info("Loaded %d JSONL records", len(records))

    if args.dry_run:
        log.info("Dry run requested; validating record shape without loading model.")
        for record in records[:5]:
            if "text" not in record:
                normalize_messages(record)
        log.info("Dry run OK")
        return 0

    device_map: str | dict[str, int]
    device_map = {"": 0} if args.device_map == "cuda" else args.device_map

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        offload_embedding=args.offload_embedding,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = records_to_dataset(records, tokenizer)
    split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed) if args.eval_split else None
    train_dataset = split["train"] if split else dataset
    eval_dataset = split["test"] if split else None
    log.info("Train rows: %d", len(train_dataset))
    if eval_dataset is not None:
        log.info("Eval rows: %d", len(eval_dataset))

    target_modules = resolve_target_modules(model, args.target_modules)
    log.info("LoRA target modules: %s", ",".join(target_modules))

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
    )

    bf16 = bool(is_bfloat16_supported())
    fp16 = not bf16
    log.info("Precision: bf16=%s fp16=%s", bf16, fp16)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "seed": args.seed,
        "fp16": fp16,
        "bf16": bf16,
        "optim": args.optim,
        "packing": args.packing,
        "padding_free": args.padding_free,
        "dataset_num_proc": args.dataset_num_proc,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataset_text_field": "text",
        "max_length": args.max_seq_length,
    }
    if args.num_train_epochs is not None:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
    else:
        training_kwargs["max_steps"] = args.max_steps
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**training_kwargs),
    )

    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset": str(dataset_path),
        "dataset_rows": len(records),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "output_dir": str(output_dir),
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "offload_embedding": args.offload_embedding,
        "device_map": args.device_map,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "use_rslora": args.use_rslora,
        },
        "training": {
            "max_steps": args.max_steps if args.num_train_epochs is None else None,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "optim": args.optim,
            "fp16": fp16,
            "bf16": bf16,
            "packing": args.packing,
            "padding_free": args.padding_free,
            "dataset_num_proc": args.dataset_num_proc,
            "dataloader_num_workers": args.dataloader_num_workers,
        },
        "metrics": result.metrics,
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    log.info("Saved adapter: %s", adapter_dir)
    log.info("Saved metadata: %s", output_dir / "metadata.json")
    if torch.cuda.is_available():
        log.info("Peak allocated VRAM: %.2f GiB", torch.cuda.max_memory_allocated() / 1024**3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
