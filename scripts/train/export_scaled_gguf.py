#!/usr/bin/env python3
"""Export an Unsloth model or adapter to GGUF format with custom weight scaling.

Example:
    python scripts/train/export_scaled_gguf.py \
        --model-name /home/billz/results/lfm25_12b_instruct_sft_q8_strict/checkpoints/checkpoint-200 \
        --output-dir /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_scaled_0.3 \
        --scale 0.3 \
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
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA weight scale multiplier (e.g. 0.3)")
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

    if args.scale != 1.0:
        print(f"Scaling LoRA adapter weights by factor of {args.scale}...")
        scaled_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "scaling"):
                if isinstance(module.scaling, dict) and "default" in module.scaling:
                    module.scaling["default"] *= args.scale
                    scaled_count += 1
                elif isinstance(module.scaling, (float, int)):
                    module.scaling *= args.scale
                    scaled_count += 1
        print(f"Scaled {scaled_count} LoRA modules successfully.")

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
