#!/usr/bin/env python3
"""Merge multiple LoRA adapters directly into the base model parameters.

Example:
    python scripts/train/merge_lora_direct.py \
        --base-model LiquidAI/LFM2.5-1.2B-Instruct \
        --math-adapter /home/billz/results/lfm25_12b_math_adapter/adapter \
        --format-adapter /home/billz/results/lfm25_12b_instruct_sft_q8_strict/checkpoints/checkpoint-200 \
        --output-dir /home/billz/results/lfm25_12b_direct_merge_m1.0_f0.7 \
        --math-weight 1.0 \
        --format-weight 0.7
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model ID or path")
    parser.add_argument("--math-adapter", required=True, help="Path to math PEFT adapter directory")
    parser.add_argument("--format-adapter", required=True, help="Path to format PEFT adapter directory")
    parser.add_argument("--output-dir", required=True, help="Output directory to save merged model")
    parser.add_argument("--math-weight", type=float, default=1.0, help="Weight scaling for math adapter")
    parser.add_argument("--format-weight", type=float, default=0.7, help="Weight scaling for format adapter")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading adapter safetensors...")
    math_weights = load_file(Path(args.math_adapter) / "adapter_model.safetensors")
    format_weights = load_file(Path(args.format_adapter) / "adapter_model.safetensors")

    # Load and calculate scaling factors from config configs
    math_config = json.load(open(Path(args.math_adapter) / "adapter_config.json"))
    math_scale = math_config["lora_alpha"] / math_config["r"]

    format_config = json.load(open(Path(args.format_adapter) / "adapter_config.json"))
    format_scale = format_config["lora_alpha"] / format_config["r"]

    print(f"Loaded config parameters:")
    print(f"  Math adapter: r={math_config['r']}, alpha={math_config['lora_alpha']}, scale={math_scale}, weight={args.math_weight}")
    print(f"  Format adapter: r={format_config['r']}, alpha={format_config['lora_alpha']}, scale={format_scale}, weight={args.format_weight}")

    state_dict = model.state_dict()
    updated_keys = []

    # Iterate over adapter keys
    for key in math_weights.keys():
        if "lora_A" in key:
            # e.g., base_model.model.model.layers.10.self_attn.k_proj.lora_A.weight
            # Map to model.layers.10.self_attn.k_proj.weight
            base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            
            if base_key not in state_dict:
                print(f"Warning: mapped key {base_key!r} not found in base model")
                continue

            lora_A_math = math_weights[key]
            lora_B_key = key.replace("lora_A", "lora_B")
            lora_B_math = math_weights[lora_B_key]

            # Compute math delta = (B * A) * scale * weight
            delta_math = torch.matmul(lora_B_math.float(), lora_A_math.float()) * math_scale * args.math_weight

            # Compute format delta
            delta_format = torch.zeros_like(delta_math)
            if key in format_weights:
                lora_A_format = format_weights[key]
                lora_B_format = format_weights[lora_B_key]
                delta_format = torch.matmul(lora_B_format.float(), lora_A_format.float()) * format_scale * args.format_weight

            # Update base weight directly
            orig_weight = state_dict[base_key].float()
            new_weight = orig_weight + delta_math + delta_format

            state_dict[base_key].copy_(new_weight.to(state_dict[base_key].dtype))
            updated_keys.append(base_key)

    print(f"Successfully merged {len(updated_keys)} projection layer weights.")

    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(str(output_dir))

    print(f"Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    print("Model merged successfully and saved.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
