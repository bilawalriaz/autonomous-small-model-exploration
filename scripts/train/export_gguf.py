#!/usr/bin/env python3
"""Export an Unsloth model or adapter to GGUF format.

Example:
    python scripts/train/export_gguf.py \
        --model-name /home/billz/results/lfm25_12b_instruct_sft_q8_strict/adapter \
        --output-dir /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf \
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
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or path to local PEFT adapter directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save GGUF files")
    parser.add_argument("--quantization", default="q4_k_m", help="Quantization method (e.g., q4_k_m, q8_0, f16)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    # Load in 16-bit to preserve quality during GGUF conversion
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
        trust_remote_code=True,
    )

    print(f"Exporting to GGUF in {output_dir} using quantization method: {args.quantization}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    print(f"Export complete. Files saved in {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
