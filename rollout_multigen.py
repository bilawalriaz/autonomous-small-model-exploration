#!/usr/bin/env python3
"""
Multi-generation rollout script.
Generates N rollouts per prompt at different temperatures.
Saves everything: prompt, response, reasoning, logprobs, gold_answer, metadata.
Resumable via completed.jsonl (tracks prompt+temp combo).
"""
import json, time, requests, sys, os, hashlib
from pathlib import Path
from datetime import datetime, timezone
import random

# Config via env vars
SERVER_URL = os.environ.get("STUDENT_URL", "http://localhost:8080")
SHARD_FILE = os.environ.get("SHARD_FILE", "/home/billz/mixed-shard.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/billz/rollouts")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "65536"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "6"))
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))

# Temperature schedule: spread across range for diversity
TEMPERATURES = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_completed():
    """Load set of (prompt_hash, temperature) pairs already done."""
    completed = set()
    completed_file = Path(OUTPUT_DIR) / "completed.jsonl"
    if completed_file.exists():
        with open(completed_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed.add((d.get("prompt_hash", ""), d.get("temperature", 0)))
    return completed

def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def load_shard():
    prompts = []
    with open(SHARD_FILE) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def generate(prompt_text, temperature, max_tokens=MAX_TOKENS):
    """Generate a single rollout at given temperature."""
    messages = [{"role": "user", "content": prompt_text}]
    
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json={
            "model": "model",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 80,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "logprobs": True,
            "top_logprobs": 3,
        }, timeout=TIMEOUT)
        r.raise_for_status()
        elapsed = time.monotonic() - t0
        
        data = r.json()
        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        reasoning = choice["message"].get("reasoning_content", "")
        
        # Extract logprobs
        logprobs_data = []
        if choice.get("logprobs") and "content" in choice["logprobs"]:
            for lp in choice["logprobs"]["content"]:
                logprobs_data.append({
                    "token": lp.get("token", ""),
                    "logprob": lp.get("logprob", 0),
                    "top_logprobs": lp.get("top_logprobs", []),
                })
        
        usage = data.get("usage", {})
        timings = data.get("timings", {})
        
        return {
            "status": "ok",
            "response": content,
            "reasoning": reasoning,
            "logprobs": logprobs_data,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "tokens_per_second": timings.get("predicted_per_second", 0),
            "elapsed_seconds": round(elapsed, 2),
            "finish_reason": choice.get("finish_reason", ""),
            "model": data.get("model", ""),
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "status": "error",
            "error": str(e),
            "response": "",
            "reasoning": "",
            "logprobs": [],
            "elapsed_seconds": round(elapsed, 2),
        }

def main():
    ensure_output_dir()
    completed = load_completed()
    shard = load_shard()
    
    # Filter to remaining (prompt+temp combos)
    remaining = []
    for p in shard:
        prompt_text = p.get("prompt", "")
        ph = prompt_hash(prompt_text)
        for temp in TEMPERATURES[:NUM_GENERATIONS]:
            if (ph, temp) not in completed:
                remaining.append((p, temp, ph))
    
    total_expected = len(shard) * NUM_GENERATIONS
    total_done = total_expected - len(remaining)
    
    print(f"{'='*60}")
    print(f"MULTI-GEN ROLLOUT GENERATOR")
    print(f"{'='*60}")
    print(f"  Server: {SERVER_URL}")
    print(f"  Shard: {len(shard)} prompts")
    print(f"  Generations per prompt: {NUM_GENERATIONS}")
    print(f"  Temperatures: {TEMPERATURES[:NUM_GENERATIONS]}")
    print(f"  Expected total: {total_expected}")
    print(f"  Already done: {total_done}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    
    # Health check
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.status_code != 200:
            print("ERROR: Server not healthy")
            sys.exit(1)
        print("  Server: healthy ✅")
    except Exception as e:
        print(f"  Server: UNREACHABLE - {e}")
        sys.exit(1)
    
    completed_file = Path(OUTPUT_DIR) / "completed.jsonl"
    rollouts_file = Path(OUTPUT_DIR) / "rollouts.jsonl"
    
    # Group remaining by prompt for better batching
    by_prompt = {}
    for p, temp, ph in remaining:
        if ph not in by_prompt:
            by_prompt[ph] = (p, [])
        by_prompt[ph][1].append(temp)
    
    gen_num = 0
    total_remaining = len(remaining)
    
    for ph, (prompt_rec, temps) in by_prompt.items():
        prompt_text = prompt_rec.get("prompt", "")
        
        for temp in temps:
            gen_num += 1
            dataset = prompt_rec.get('_dataset', prompt_rec.get('prompt_meta', {}).get('_dataset', 'unknown'))
            
            print(f"\n[{gen_num}/{total_remaining}] [{dataset}] temp={temp} {prompt_text[:60]}...", flush=True)
            
            result = generate(prompt_text, temp)
            
            rollout = {
                "prompt_hash": ph,
                "prompt": prompt_text,
                "temperature": temp,
                "generation_index": TEMPERATURES[:NUM_GENERATIONS].index(temp),
                "_dataset": dataset,
                "prompt_meta": {
                    "doc_id": prompt_rec.get("doc_id"),
                    "entity_type": prompt_rec.get("entity_type"),
                    "output_format": prompt_rec.get("output_format"),
                    "top_level_count": prompt_rec.get("top_level_count"),
                    "json_schema": prompt_rec.get("json_schema"),
                    "gold_answer": prompt_rec.get("gold_answer"),
                    "category": prompt_rec.get("category"),
                    "subject": prompt_rec.get("subject"),
                },
                "response": result.get("response", ""),
                "reasoning": result.get("reasoning", ""),
                "logprobs": result.get("logprobs", []),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "tokens_per_second": result.get("tokens_per_second", 0),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "finish_reason": result.get("finish_reason", ""),
                "model": result.get("model", ""),
                "status": result.get("status", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_url": SERVER_URL,
                "max_tokens": MAX_TOKENS,
            }
            
            # Save rollout
            with open(rollouts_file, "a") as f:
                f.write(json.dumps(rollout, default=str) + "\n")
            
            # Mark completed
            with open(completed_file, "a") as f:
                f.write(json.dumps({"prompt_hash": ph, "temperature": temp}) + "\n")
            
            status = result.get("status", "")
            if status == "ok":
                toks = result.get("completion_tokens", 0)
                tps = result.get("tokens_per_second", 0)
                reason_len = len(result.get("reasoning", ""))
                resp_len = len(result.get("response", ""))
                print(f"  ✅ {toks} tok, {tps:.1f} t/s, {result['elapsed_seconds']:.0f}s, reasoning={reason_len}ch resp={resp_len}ch")
            else:
                print(f"  ❌ {result.get('error', 'unknown')}")
            
            time.sleep(0.3)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"COMPLETE: {gen_num} generations produced")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
