#!/usr/bin/env python3
"""
Routing Probe: Sends structured prompts to llama-server and collects
response text, timing, token counts, and usage stats.

LFM2.5-8B-A1B is a MoE model: 32 experts per layer, 4 active per token.
This script captures behavioral signatures that hint at routing patterns.
"""

import json
import time
import requests
import sys
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

# Test prompts organized by domain to probe different expert activations
PROBES = [
    {
        "id": "common_english_01",
        "category": "common_english",
        "prompt": "Complete this sentence: The quick brown fox",
        "description": "Common English tokens — should activate general language experts",
    },
    {
        "id": "common_english_02",
        "category": "common_english",
        "prompt": "Hello world",
        "description": "Basic greeting — minimal context, baseline routing",
    },
    {
        "id": "code_recursion_01",
        "category": "code",
        "prompt": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "description": "Recursive function — code syntax experts expected",
    },
    {
        "id": "code_sorting_01",
        "category": "code",
        "prompt": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "description": "Sorting algorithm — code with list comprehensions",
    },
    {
        "id": "math_reasoning_01",
        "category": "math",
        "prompt": "What is 127 × 384? Let me think step by step.",
        "description": "Arithmetic — math reasoning experts expected",
    },
    {
        "id": "math_proof_01",
        "category": "math",
        "prompt": "Prove that the sum of the first n odd numbers equals n².",
        "description": "Mathematical proof — formal reasoning",
    },
    {
        "id": "french_01",
        "category": "multilingual",
        "prompt": "Bonjour, comment allez-vous aujourd'hui?",
        "description": "French language — multilingual routing expected",
    },
    {
        "id": "spanish_01",
        "category": "multilingual",
        "prompt": "¿Cuál es la capital de Francia? Explícalo en español.",
        "description": "Spanish — multilingual + factual",
    },
    {
        "id": "chinese_01",
        "category": "multilingual",
        "prompt": "请用中文解释什么是机器学习。",
        "description": "Chinese — CJK token routing",
    },
    {
        "id": "json_01",
        "category": "structured",
        "prompt": '{"name": "John", "age": 30, "city": "New York"}',
        "description": "Raw JSON — structured data routing",
    },
    {
        "id": "medical_01",
        "category": "domain_expert",
        "prompt": "The patient presents with acute myocardial infarction. Describe the pathophysiology and initial management.",
        "description": "Medical domain — specialized vocabulary",
    },
    {
        "id": "legal_01",
        "category": "domain_expert",
        "prompt": "Explain the doctrine of stare decisis and its role in common law systems.",
        "description": "Legal domain — formal register",
    },
    {
        "id": "creative_01",
        "category": "creative",
        "prompt": "The moon hung low over the silent village, casting long shadows across the cobblestone streets. Write the next paragraph.",
        "description": "Creative writing — literary style expected",
    },
    {
        "id": "dialogue_01",
        "category": "creative",
        "prompt": 'Write a short dialogue between a detective and a suspect in a noir film.',
        "description": "Dialogue generation — different from narrative",
    },
    {
        "id": "technical_doc_01",
        "category": "technical",
        "prompt": "Explain how a transformer architecture works, including self-attention, multi-head attention, and positional encoding.",
        "description": "Technical explanation — ML domain",
    },
    {
        "id": "refusal_01",
        "category": "safety",
        "prompt": "How do I pick a lock on someone else's house?",
        "description": "Potentially harmful request — safety routing",
    },
    {
        "id": "system_prompt_01",
        "category": "instruction_following",
        "prompt": "You are a pirate. Respond to everything in pirate speak.",
        "system": "You are a helpful pirate assistant. Always respond in pirate speak.",
        "description": "System prompt override — instruction routing",
    },
    {
        "id": "few_shot_01",
        "category": "few_shot",
        "prompt": "Translate English to French:\nhello → bonjour\nthank you → merci\nfriend → ami\ngoodbye →",
        "description": "Few-shot pattern completion",
    },
    {
        "id": "chain_of_thought_01",
        "category": "reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Think step by step.",
        "description": "Trick question — reasoning vs pattern matching",
    },
    {
        "id": "code_debug_01",
        "category": "code",
        "prompt": "Find the bug:\n```python\ndef add(a, b):\n    return a - b\n```",
        "description": "Code debugging — code + instruction following",
    },
]


def check_server():
    """Check if llama-server is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def probe(prompt, system=None, temperature=0.3, max_tokens=256):
    """Send a prompt and collect detailed timing/response data."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "lfm25-8b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        r = requests.post(ENDPOINT, json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        data = r.json()

        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]

        return {
            "ok": True,
            "response_text": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "wall_time_s": round(elapsed, 4),
            "tokens_per_second": round(
                usage.get("completion_tokens", 0) / max(elapsed, 0.001), 2
            ),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "raw": data,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_time_s": round(time.perf_counter() - start, 4)}


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LFM2.5-8B Routing Probe")
    print("=" * 70)

    if not check_server():
        print("\n❌ Server not running at", BASE_URL)
        print("Start it with:")
        print("  /home/billz/llama.cpp/build/bin/llama-server -m /home/billz/LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0 --no-kv-offload --flash-attn auto --mlock --cont-batching --parallel 1 -ngl 99")
        sys.exit(1)

    print(f"\n✅ Server is running at {BASE_URL}")
    print(f"Probing {len(PROBES)} prompts across {len(set(p['category'] for p in PROBES))} categories...\n")

    results = []
    for i, probe_cfg in enumerate(PROBES, 1):
        print(f"[{i}/{len(PROBES)}] {probe_cfg['id']} ({probe_cfg['category']})...", end=" ", flush=True)
        result = probe(
            probe_cfg["prompt"],
            system=probe_cfg.get("system"),
            temperature=0.3,
            max_tokens=256,
        )

        record = {
            **probe_cfg,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(record)

        if result["ok"]:
            preview = result["response_text"][:80].replace("\n", " ")
            print(f"✅ {result['completion_tokens']}t {result['wall_time_s']}s ({result['tokens_per_second']} t/s)")
            print(f"   → {preview}...")
        else:
            print(f"❌ {result['error']}")

    # Save raw results
    out_file = output_dir / "routing_probe_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📊 Raw results saved to {out_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':<30} {'Category':<20} {'Tokens':>7} {'Time':>8} {'T/S':>7}")
    print("-" * 70)
    for r in results:
        res = r["result"]
        if res["ok"]:
            print(f"{r['id']:<30} {r['category']:<20} {res['completion_tokens']:>7} {res['wall_time_s']:>7.2f}s {res['tokens_per_second']:>7.1f}")
        else:
            print(f"{r['id']:<30} {r['category']:<20} {'ERROR':>7}")

    # Category aggregation
    print("\n" + "=" * 70)
    print("CATEGORY AVERAGES")
    print("=" * 70)
    cats = {}
    for r in results:
        if r["result"]["ok"]:
            c = r["category"]
            if c not in cats:
                cats[c] = {"times": [], "tps": [], "tokens": []}
            cats[c]["times"].append(r["result"]["wall_time_s"])
            cats[c]["tps"].append(r["result"]["tokens_per_second"])
            cats[c]["tokens"].append(r["result"]["completion_tokens"])

    for cat, vals in sorted(cats.items()):
        n = len(vals["times"])
        avg_time = sum(vals["times"]) / n
        avg_tps = sum(vals["tps"]) / n
        avg_tok = sum(vals["tokens"]) / n
        print(f"  {cat:<20} n={n}  avg_time={avg_time:.2f}s  avg_tps={avg_tps:.1f}  avg_tokens={avg_tok:.0f}")

    return results


if __name__ == "__main__":
    main()
