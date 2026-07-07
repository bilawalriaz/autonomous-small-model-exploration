#!/usr/bin/env python3
"""
Rollout generator for ifstruct-v1.0 dataset.
Runs on each machine against its local student llama-server.
Saves: prompt, full messages, response, logprobs, timing, metadata.
Resumes from completed.jsonl if interrupted.
"""
import json, time, requests, sys, os, hashlib
from pathlib import Path
from datetime import datetime, timezone

# Config - override via env vars
SERVER_URL = os.environ.get("STUDENT_URL", "http://localhost:8080")
SHARD_FILE = os.environ.get("SHARD_FILE", "/home/billz/ifstruct-shard.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/billz/rollouts")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "65536"))  # 64k output
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))  # 10 min per request

def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_completed():
    completed = set()
    completed_file = Path(OUTPUT_DIR) / "completed.jsonl"
    if completed_file.exists():
        with open(completed_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    # Use hash of prompt as unique key
                    completed.add(d.get("prompt_hash", ""))
    return completed

def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def load_shard():
    prompts = []
    with open(SHARD_FILE) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                prompts.append(d)
    return prompts

def generate(prompt_text, prompt_meta):
    """Generate a rollout for a single prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SERVER_URL}/v1/chat/completions", json={
            "model": "model",
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": 0.95,
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
    
    # Filter to remaining
    remaining = [p for p in shard if prompt_hash(p.get("prompt", "")) not in completed]
    
    print(f"{'='*60}")
    print(f"ROLLOUT GENERATOR")
    print(f"{'='*60}")
    print(f"  Server: {SERVER_URL}")
    print(f"  Shard: {len(shard)} prompts")
    print(f"  Already done: {len(completed)}")
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
    
    # Process prompts
    completed_file = Path(OUTPUT_DIR) / "completed.jsonl"
    rollouts_file = Path(OUTPUT_DIR) / "rollouts.jsonl"
    
    for i, prompt_rec in enumerate(remaining):
        prompt_text = prompt_rec.get("prompt", "")
        ph = prompt_hash(prompt_text)
        
        print(f"\n[{i+1}/{len(remaining)}] {prompt_text[:80]}...", flush=True)
        
        result = generate(prompt_text, prompt_rec)
        
        # Build the rollout record
        rollout = {
            "prompt_hash": ph,
            "prompt": prompt_text,
            "prompt_meta": {
                "doc_id": prompt_rec.get("doc_id"),
                "entity_type": prompt_rec.get("entity_type"),
                "output_format": prompt_rec.get("output_format"),
                "top_level_count": prompt_rec.get("top_level_count"),
                "json_schema": prompt_rec.get("json_schema"),
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
            "temperature": TEMPERATURE,
        }
        
        # Save to rollouts file (append)
        with open(rollouts_file, "a") as f:
            f.write(json.dumps(rollout, default=str) + "\n")
        
        # Save to completed file (for resume)
        with open(completed_file, "a") as f:
            f.write(json.dumps({"prompt_hash": ph, "prompt": prompt_text[:100]}) + "\n")
        
        status = result.get("status", "")
        if status == "ok":
            toks = result.get("completion_tokens", 0)
            tps = result.get("tokens_per_second", 0)
            reason_len = len(result.get("reasoning", ""))
            resp_len = len(result.get("response", ""))
            print(f"  ✅ {toks} tokens, {tps:.1f} t/s, {result['elapsed_seconds']:.0f}s, reasoning={reason_len}ch, response={resp_len}ch")
        else:
            print(f"  ❌ {result.get('error', 'unknown')}")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Summary
    total = len(shard)
    done = len(completed) + len(remaining)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {done}/{total} prompts processed")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
