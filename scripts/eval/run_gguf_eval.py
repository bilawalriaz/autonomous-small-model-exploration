#!/usr/bin/env python3
"""Run evaluation on a GGUF model using llama-completion.

Example:
    python scripts/eval/run_gguf_eval.py \
        --model-gguf /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf \
        --run-id lfm25_12b_sft_gguf \
        --eval-set data/eval/small_model_eval_v1.jsonl
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_gguf_eval")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-gguf", required=True, help="Path to the GGUF model file on aero")
    parser.add_argument("--run-id", required=True, help="Run identifier for output files")
    parser.add_argument("--eval-set", default="data/eval/small_model_eval_v1.jsonl", help="Evaluation JSONL path")
    parser.add_argument("--tokenizer-name", default="LiquidAI/LFM2.5-1.2B-Instruct", help="Tokenizer name or path")
    parser.add_argument("--llama-path", default="/home/billz/llama.cpp/build/bin/llama-completion", help="Path to llama-completion binary")
    parser.add_argument("--gpu-layers", type=int, default=99, help="Number of layers to offload to GPU")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to run (for quick testing)")
    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_prompt(record: dict, tokenizer) -> str:
    messages = record.get("messages") or record.get("prompt_messages")
    if messages:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return record.get("prompt") or record.get("instruction", "")

def run_completion(
    llama_path: str,
    gguf_path: str,
    prompt: str,
    gpu_layers: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float
) -> tuple[str, float, int]:
    # Write prompt to temp file to avoid CLI argument length limit or shell parsing issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        temp_prompt_path = f.name

    cmd = [
        llama_path,
        "-m", gguf_path,
        "-f", temp_prompt_path,
        "-n", str(max_new_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--repeat-penalty", str(repetition_penalty),
        "-ngl", str(gpu_layers),
        "-no-cnv"
    ]

    t0 = time.time()
    # Redirect stderr to devnull to keep stdout clean, capture output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    elapsed = time.time() - t0

    try:
        os.remove(temp_prompt_path)
    except Exception:
        pass

    stdout = result.stdout
    
    # Post-process output: remove prompt from beginning if echoed
    response = stdout
    if response.startswith(prompt):
        response = response[len(prompt):]
    
    # Strip whitespaces
    response = response.strip()
    
    # Estimate token count (1 token roughly 4 characters)
    tokens_generated = len(response.split())
    return response, elapsed, tokens_generated

def main() -> int:
    args = parse_args()
    
    eval_set_path = Path(args.eval_set).resolve()
    if not eval_set_path.exists():
        log.error(f"Evaluation set not found: {eval_set_path}")
        return 1

    records = load_jsonl(eval_set_path)
    if args.limit:
        records = records[:args.limit]
    log.info(f"Loaded {len(records)} evaluation prompts")

    log.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    output_dir = Path("results/evals") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "outputs.jsonl"
    metadata_path = output_dir / "metadata.json"

    log.info(f"Starting evaluation of GGUF model: {args.model_gguf}")
    results = []
    
    for i, record in enumerate(records):
        prompt = build_prompt(record, tokenizer)
        eval_id = record.get("eval_id", record.get("id", f"eval_{i:04d}"))
        category = record.get("category", "unknown")
        
        response, elapsed, tokens = run_completion(
            llama_path=args.llama_path,
            gguf_path=args.model_gguf,
            prompt=prompt,
            gpu_layers=args.gpu_layers,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        result = {
            "eval_id": eval_id,
            "category": category,
            "prompt": prompt,
            "generated_response": response,
            "generation_time": round(elapsed, 3),
            "tokens_generated": tokens
        }
        results.append(result)
        
        # Intermediate logging
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            log.info(f"  Processed {i + 1}/{len(records)} prompts")

    # Write output JSONL
    with open(outputs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"Saved evaluation results to {outputs_path}")

    # Write metadata JSON
    metadata = {
        "run_id": args.run_id,
        "model_gguf": args.model_gguf,
        "eval_set": str(eval_set_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_count": len(results),
        "config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
