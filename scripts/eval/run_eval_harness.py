#!/usr/bin/env python3
"""Run eval harness: generate responses from base model + adapters on the eval set.

CLI:
    python scripts/eval/run_eval_harness.py \
        --config configs/eval/lfm2_small_model_eval.yaml \
        --adapter adapters/lfm2_230m_quality_format_ablation_multi_turn_concise \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629

    # Base model only:
    python scripts/eval/run_eval_harness.py \
        --config configs/eval/lfm2_small_model_eval.yaml \
        --base-only \
        --run-id lfm2_230m_base_20260629
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def load_model(model_name: str, dtype: torch.dtype, adapter_path: str | None = None):
    """Load base model optionally with PEFT adapter."""
    log.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_path:
        log.info(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_prompt(record: dict, tokenizer) -> str:
    """Build prompt text from eval record, handling chat and flat formats."""
    messages = record.get("messages") or record.get("prompt_messages")
    if messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Flat prompt
    return record.get("prompt") or record.get("instruction", "")


def generate_response(model, tokenizer, prompt: str, gen_config: dict) -> tuple[str, float, int]:
    """Generate a single response. Returns (text, elapsed_seconds, token_count)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 512),
            temperature=gen_config.get("temperature", 0.2),
            top_p=gen_config.get("top_p", 0.9),
            repetition_penalty=gen_config.get("repetition_penalty", 1.1),
            do_sample=gen_config.get("do_sample", True),
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, elapsed, len(new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Run eval harness for model/adapter evaluation")
    parser.add_argument("--config", required=True, help="Eval config YAML path")
    parser.add_argument("--adapter", default=None, help="PEFT adapter path")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--base-only", action="store_true", help="Evaluate base model only (no adapter)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    model_name = config["model"]["name"]
    dtype = get_dtype(config["model"].get("dtype", "bf16"))
    gen_config = config.get("generation", {})
    eval_set_path = PROJECT_ROOT / config["eval_set"]["path"]

    # Load eval data
    if not eval_set_path.exists():
        log.error(f"Eval set not found: {eval_set_path}")
        sys.exit(1)
    eval_data = load_jsonl(eval_set_path)
    log.info(f"Loaded {len(eval_data)} eval prompts from {eval_set_path}")

    # Determine adapter
    adapter_path = None if args.base_only else args.adapter
    if adapter_path:
        adapter_path = str(PROJECT_ROOT / adapter_path)
        if not Path(adapter_path).exists():
            log.error(f"Adapter path not found: {adapter_path}")
            sys.exit(1)

    # Load model
    model, tokenizer = load_model(model_name, dtype, adapter_path)

    # Output directory
    output_dir = PROJECT_ROOT / "results" / "evals" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "outputs.jsonl"
    metadata_path = output_dir / "metadata.json"

    # Generate responses
    log.info(f"Generating {len(eval_data)} responses...")
    results = []
    for i, record in enumerate(eval_data):
        prompt = build_prompt(record, tokenizer)
        eval_id = record.get("eval_id", record.get("id", f"eval_{i:04d}"))
        category = record.get("category", "unknown")

        response, gen_time, tokens_generated = generate_response(model, tokenizer, prompt, gen_config)

        result = {
            "eval_id": eval_id,
            "category": category,
            "prompt": prompt,
            "generated_response": response,
            "generation_time": round(gen_time, 3),
            "tokens_generated": tokens_generated,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            log.info(f"  Generated {i + 1}/{len(eval_data)}")

    # Write outputs
    with open(outputs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    log.info(f"Wrote {len(results)} outputs to {outputs_path}")

    # Write metadata
    metadata = {
        "run_id": args.run_id,
        "model": model_name,
        "adapter_path": adapter_path,
        "base_only": args.base_only,
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_count": len(results),
        "seed": args.seed,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Wrote metadata to {metadata_path}")

    # Summary
    avg_time = sum(r["generation_time"] for r in results) / len(results)
    avg_tokens = sum(r["tokens_generated"] for r in results) / len(results)
    log.info(f"Done. Avg gen time: {avg_time:.2f}s, Avg tokens: {avg_tokens:.1f}")


if __name__ == "__main__":
    main()
