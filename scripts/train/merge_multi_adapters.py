#!/usr/bin/env python3
"""Merge multiple PEFT adapters into a base model with custom scaling weights and export to GGUF.

Example:
    python scripts/train/merge_multi_adapters.py \
        --base-model LiquidAI/LFM2.5-1.2B-Instruct \
        --math-adapter /home/billz/results/lfm25_12b_math_adapter/adapter \
        --format-adapter /home/billz/results/lfm25_12b_instruct_sft_q8_strict/checkpoints/checkpoint-200 \
        --output-dir /home/billz/results/lfm25_12b_multi_merge_m1.0_f0.7 \
        --math-weight 1.0 \
        --format-weight 0.7 \
        --quantization q4_k_m
"""

import argparse
import sys
from pathlib import Path

# Unsloth patches import sequence
from unsloth import FastLanguageModel
import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base model Hugging Face path")
    parser.add_argument("--math-adapter", required=True, help="Path to math adapter directory")
    parser.add_argument("--format-adapter", required=True, help="Path to formatting adapter directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save GGUF files")
    parser.add_argument("--math-weight", type=float, default=1.0, help="Weight for math adapter")
    parser.add_argument("--format-weight", type=float, default=0.7, help="Weight for format adapter")
    parser.add_argument("--quantization", default="q4_k_m", help="Quantization method (e.g., q4_k_m, q8_0, f16)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
        trust_remote_code=True,
    )

    from peft import PeftModel
    print(f"Loading math adapter: {args.math_adapter} as 'math'")
    model = PeftModel.from_pretrained(model, args.math_adapter, adapter_name="math")

    print(f"Loading format adapter: {args.format_adapter} as 'format'")
    model.load_adapter(args.format_adapter, adapter_name="format")

    # Combine using weighted sum
    print(f"Combining adapters: math (weight={args.math_weight}) + format (weight={args.format_weight})")
    model.add_weighted_adapter(
        adapters=["math", "format"],
        weights=[args.math_weight, args.format_weight],
        adapter_name="merged",
        combination_type="linear"
    )

    # Set as active
    print("Activating merged adapter 'merged'...")
    model.set_adapter("merged")

    print("Merging adapters and unloading PEFT wrapper...")
    model = model.merge_and_unload()

    print(f"Exporting merged model to GGUF in {output_dir} using quantization method: {args.quantization}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    print(f"Export complete. Saved GGUF in {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
