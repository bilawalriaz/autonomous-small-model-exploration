#!/usr/bin/env python3
"""Run GSM8K evaluation on GGUF models.

Example:
    python scripts/eval/run_gsm8k_eval.py \
        --model-gguf /home/billz/results/lfm25_12b_instruct_sft_q8_strict/gguf_gguf/LFM2.5-1.2B-Instruct.Q4_K_M.gguf \
        --output results/evals/gsm8k_sft_results.jsonl \
        --limit 100
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_gsm8k_eval")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-set", default="data/gsm8k-test.jsonl")
    parser.add_argument("--tokenizer-name", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--llama-path", default="/home/billz/llama.cpp/build/bin/llama-completion")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def extract_numerical_answer(text: str) -> str | None:
    """Extract final answer from text, looking for #### prefix or last number."""
    # Try looking for #### <number>
    match = re.search(r'####\s*(-?\d+(?:[\.,]\d+)?)', text)
    if match:
        val = match.group(1).replace(',', '')
        # Convert to float/int if possible
        try:
            return str(int(float(val)))
        except ValueError:
            return val.strip()

    # Fallback to the last number in the text
    numbers = re.findall(r'-?\d+(?:[\.,]\d+)?', text)
    if numbers:
        val = numbers[-1].replace(',', '')
        try:
            return str(int(float(val)))
        except ValueError:
            return val.strip()
            
    return None

def extract_gold_answer(text: str) -> str | None:
    match = re.search(r'####\s*(-?\d+)', text)
    if match:
        return match.group(1).strip()
    return None

def run_completion(
    llama_path: str,
    gguf_path: str,
    prompt: str,
    gpu_layers: int,
    temperature: float,
    top_p: float
) -> tuple[str, float]:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        temp_prompt_path = f.name

    cmd = [
        llama_path,
        "-m", gguf_path,
        "-f", temp_prompt_path,
        "-n", "512", # Max generation tokens for math steps
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "-ngl", str(gpu_layers),
        "-no-cnv"
    ]

    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    elapsed = time.time() - t0

    try:
        os.remove(temp_prompt_path)
    except Exception:
        pass

    stdout = result.stdout
    response = stdout
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response.strip(), elapsed

def main() -> int:
    args = parse_args()

    eval_set_path = Path(args.eval_set).resolve()
    if not eval_set_path.exists():
        log.error(f"GSM8K test set not found at {eval_set_path}")
        return 1

    records = load_jsonl(eval_set_path)
    if args.limit:
        records = records[:args.limit]
    log.info(f"Loaded {len(records)} GSM8K test prompts to run")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    correct_count = 0
    results = []

    for i, record in enumerate(records):
        # Wrap prompt in instruct template
        messages = [{"role": "user", "content": record["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response, elapsed = run_completion(
            llama_path=args.llama_path,
            gguf_path=args.model_gguf,
            prompt=formatted_prompt,
            gpu_layers=args.gpu_layers,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        gold_text = record.get("gold_answer", "")
        gold_ans = extract_gold_answer(gold_text)
        pred_ans = extract_numerical_answer(response)
        
        is_correct = (gold_ans == pred_ans) if (gold_ans and pred_ans) else False
        if is_correct:
            correct_count += 1
            
        result = {
            "index": record.get("_idx", i),
            "prompt": record["prompt"],
            "gold_answer_text": gold_text,
            "gold_answer_value": gold_ans,
            "response": response,
            "predicted_value": pred_ans,
            "is_correct": is_correct,
            "time_seconds": round(elapsed, 3)
        }
        results.append(result)
        
        with open(output_path, "a" if i > 0 else "w") as f:
            f.write(json.dumps(result) + "\n")
            
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            accuracy = (correct_count / (i + 1)) * 100
            log.info(f"  Processed {i + 1}/{len(records)} prompts | Current Accuracy: {accuracy:.2f}% ({correct_count}/{i+1})")

    final_accuracy = (correct_count / len(records)) * 100
    log.info(f"Completed GSM8K evaluation. Final Accuracy: {final_accuracy:.2f}% ({correct_count}/{len(records)})")
    
    # Save a small summary file
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model_gguf": args.model_gguf,
            "eval_count": len(records),
            "correct_count": correct_count,
            "accuracy": final_accuracy
        }, f, indent=2)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
