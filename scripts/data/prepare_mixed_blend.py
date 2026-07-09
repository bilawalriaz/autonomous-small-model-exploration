#!/usr/bin/env python3
"""Prepare mixed dataset blend for LFM SFT validation.

Combines:
  - 413 formatting examples from our strict SFT dataset (half of 827)
  - 2,000 coding examples from Magicoder-Evol-Instruct-110K
  - 1,600 math reasoning examples from GSM8K (train split)

Output:
  - data/sft/mixed_blend_4k.jsonl
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_mixed_blend")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-sft-path",
        default="/home/billz/scored/exports/sft_strict_q8_response.jsonl",
        help="Path to the strict formatting SFT dataset"
    )
    parser.add_argument(
        "--output",
        default="data/sft/mixed_blend_4k.jsonl",
        help="Output path for the mixed jsonl dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # 1. Load strict SFT formatting examples
    strict_sft_path = Path(args.strict_sft_path)
    if not strict_sft_path.exists():
        # Try relative to project root
        strict_sft_path = Path(__file__).resolve().parents[2] / "scored/exports/sft_strict_q8_response.jsonl"
        if not strict_sft_path.exists():
            # Try mac path as fallback
            strict_sft_path = Path("/Users/bilawalriaz/scored/exports/sft_strict_q8_response.jsonl")
            
    if not strict_sft_path.exists():
        log.error(f"Strict SFT dataset not found. Checked: {args.strict_sft_path} and fallbacks.")
        return 1

    formatting_records = load_jsonl(strict_sft_path)
    log.info(f"Loaded {len(formatting_records)} formatting examples")
    
    # Sample 413 examples
    formatting_sample = random.sample(formatting_records, min(413, len(formatting_records)))
    log.info(f"Sampled {len(formatting_sample)} formatting examples")

    # Format formatting examples
    mixed_data = []
    for record in formatting_sample:
        # Check messages format or instruct format
        if "messages" in record and isinstance(record["messages"], list):
            mixed_data.append({"messages": record["messages"]})
        else:
            # Reconstruct message format
            prompt = record.get("prompt") or record.get("instruction")
            response = record.get("response") or record.get("assistant_response")
            mixed_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
            })

    # 2. Download and sample from Magicoder-Evol-Instruct-110K
    log.info("Loading ise-uiuc/Magicoder-Evol-Instruct-110K dataset from Hugging Face...")
    try:
        magicoder_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        log.info(f"Loaded {len(magicoder_dataset)} Magicoder examples")
        
        # Shuffle and sample 2000
        magicoder_indices = list(range(len(magicoder_dataset)))
        random.shuffle(magicoder_indices)
        magicoder_sample_indices = magicoder_indices[:2000]
        
        for idx in magicoder_sample_indices:
            example = magicoder_dataset[idx]
            mixed_data.append({
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["response"]}
                ]
            })
        log.info("Successfully added 2,000 Magicoder examples to the blend")
    except Exception as e:
        log.error(f"Failed to load/sample Magicoder dataset: {e}")
        return 1

    # 3. Download and sample from GSM8K train split
    log.info("Loading gsm8k dataset from Hugging Face...")
    try:
        gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")
        log.info(f"Loaded {len(gsm8k_dataset)} GSM8K training examples")
        
        gsm8k_indices = list(range(len(gsm8k_dataset)))
        random.shuffle(gsm8k_indices)
        gsm8k_sample_indices = gsm8k_indices[:1600]
        
        for idx in gsm8k_sample_indices:
            example = gsm8k_dataset[idx]
            mixed_data.append({
                "messages": [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]}
                ]
            })
        log.info("Successfully added 1,600 GSM8K examples to the blend")
    except Exception as e:
        log.error(f"Failed to load/sample GSM8K dataset: {e}")
        return 1

    # Shuffle final mix
    random.shuffle(mixed_data)
    log.info(f"Total mixed dataset size: {len(mixed_data)} examples")

    # Save to output
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in mixed_data:
            f.write(json.dumps(record) + "\n")

    log.info(f"Saved final mixed dataset to {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
