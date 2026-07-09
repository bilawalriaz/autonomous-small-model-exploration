#!/usr/bin/env python3
"""Format the Hugging Face GSM8K training dataset into chat templates and save to JSONL."""

import json
import logging
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_gsm8k_train")

def main():
    dataset_dir = PROJECT_ROOT / "data" / "sft"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "gsm8k_train_formatted.jsonl"

    log.info("Loading GSM8K train split from HF datasets...")
    # gsm8k contains main and socratic configs
    ds = load_dataset("openai/gsm8k", "main", split="train")
    log.info(f"Loaded {len(ds)} training examples.")

    import re
    log.info(f"Formatting examples to chat messages (stripping calculator tags)...")
    formatted_count = 0
    with open(output_path, "w") as f:
        for item in ds:
            # Strip calculator tags like <<16-3-4=9>> from answer
            cleaned_answer = re.sub(r"<<.*?>>", "", item["answer"].strip())
            messages = [
                {"role": "user", "content": item["question"].strip()},
                {"role": "assistant", "content": cleaned_answer}
            ]
            f.write(json.dumps({"messages": messages}) + "\n")
            formatted_count += 1

    log.info(f"Successfully formatted and saved {formatted_count} examples to {output_path}")

if __name__ == "__main__":
    main()
