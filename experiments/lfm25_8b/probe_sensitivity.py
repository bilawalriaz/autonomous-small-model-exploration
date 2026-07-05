#!/usr/bin/env python3
"""
Sensitivity Probe: Tests model sensitivity to perturbations — temperature,
token removal, character swaps — to understand routing stability.

The hypothesis: if different experts fire for different token sequences,
perturbations that change the input tokens should cause non-linear
changes in output distribution and timing.
"""

import json
import time
import random
import requests
import sys
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8080"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"

BASE_PROMPT = "The quick brown fox jumps over the lazy dog"

PERTURBATIONS = {
    "temperature_sweep": {
        "description": "Same prompt at different temperatures",
        "prompts": [BASE_PROMPT],
        "temperatures": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    },
    "character_swap": {
        "description": "Single character swaps in prompt",
        "perturbations": [
            ("original", BASE_PROMPT),
            ("swap_0", BASE_PROMPT.replace("fox", "fax", 1)),
            ("swap_1", BASE_PROMPT.replace("quick", "qick", 1)),
            ("swap_2", BASE_PROMPT.replace("brown", "brwn", 1)),
            ("swap_3", BASE_PROMPT.replace("lazy", "lzy", 1)),
            ("swap_4", BASE_PROMPT.replace("dog", "dig", 1)),
            ("swap_all_chars", "Th3 qu!ck br0wn f0x jumps 0v3r th3 l@zy d0g"),
            ("case_lower", BASE_PROMPT.lower()),
            ("case_upper", BASE_PROMPT.upper()),
            ("case_toggle", "tHe QuIcK bRoWn FoX jUmPs OvEr ThE lAzY dOg"),
        ],
    },
    "token_removal": {
        "description": "Removing tokens from the prompt",
        "perturbations": [
            ("full", "The quick brown fox jumps over the lazy dog"),
            ("remove_1", "quick brown fox jumps over the lazy dog"),
            ("remove_2", "brown fox jumps over the lazy dog"),
            ("remove_3", "fox jumps over the lazy dog"),
            ("remove_4", "jumps over the lazy dog"),
            ("remove_5", "over the lazy dog"),
            ("remove_6", "the lazy dog"),
            ("remove_7", "lazy dog"),
            ("remove_8", "dog"),
            ("remove_words", "fox dog jumps lazy"),
        ],
    },
    "token_addition": {
        "description": "Adding tokens to the prompt",
        "perturbations": [
            ("base", BASE_PROMPT),
            ("add_1", BASE_PROMPT + " slowly"),
            ("add_2", BASE_PROMPT + " quickly"),
            ("add_3", BASE_PROMPT + " while the sun set"),
            ("add_contradict", BASE_PROMPT + " but the fox was fast"),
            ("add_code", BASE_PROMPT + " print('hello')"),
            ("add_french", BASE_PROMPT + " Bonjour"),
        ],
    },
    "repetition": {
        "description": "Repeating the same token sequences",
        "perturbations": [
            ("1x", "The "),
            ("2x", "The The "),
            ("3x", "The The The "),
            ("4x", "The The The The "),
            ("5x", "The The The The The "),
            ("10x", " ".join(["The"] * 10)),
            ("20x", " ".join(["The"] * 20)),
        ],
    },
    "context_length": {
        "description": "Varying context length around the target",
        "perturbations": [
            ("short", "fox"),
            ("medium", "The quick brown fox jumps over"),
            ("full", "The quick brown fox jumps over the lazy dog"),
            ("verbose", "In a peaceful meadow, a quick brown fox was seen gracefully jumping over a very lazy dog that was sleeping in the warm afternoon sun."),
            ("very_verbose", "Once upon a time, in a distant land where the hills rolled gently under a golden sky, there lived a quick brown fox. This fox was known throughout the land for its agility and cunning nature. One day, while exploring the meadows near the old castle, the fox encountered a lazy dog lying by the stream. Without hesitation, the quick brown fox jumped gracefully over the lazy dog."),
        ],
    },
    "semantic_perturbation": {
        "description": "Same meaning, different words",
        "perturbations": [
            ("original", BASE_PROMPT),
            ("synonym", "The swift auburn fox leaps above the idle canine"),
            ("paraphrase", "A fast brown-colored fox vaulted over a sleeping dog"),
            ("reorder", "Over the lazy dog jumps the quick brown fox"),
            ("passive", "The lazy dog is jumped over by the quick brown fox"),
            ("negation", "The fox did not fail to jump over the dog"),
        ],
    },
}


def check_server():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def probe(prompt, temperature=0.3, max_tokens=100, system=None):
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
            "response": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "wall_time_s": round(elapsed, 4),
            "tokens_per_second": round(
                usage.get("completion_tokens", 0) / max(elapsed, 0.001), 2
            ),
            "temperature": temperature,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_time_s": round(time.perf_counter() - start, 4)}


def run_temperature_sweep(cfg):
    """Test output variance across temperatures."""
    print("\n🌡️  Temperature Sweep")
    print("-" * 60)
    results = []
    for temp in cfg["temperatures"]:
        for prompt in cfg["prompts"]:
            print(f"  temp={temp:.1f} prompt='{prompt[:40]}...'...", end=" ", flush=True)
            r = probe(prompt, temperature=temp, max_tokens=100)
            record = {"perturbation": f"temp_{temp}", "prompt": prompt, "result": r}
            results.append(record)
            if r["ok"]:
                preview = r["response"][:60].replace("\n", " ")
                print(f"✅ {r['tokens_per_second']:.1f} t/s → {preview}...")
            else:
                print(f"❌ {r['error']}")
    return results


def run_perturbation_test(cfg, name):
    """Test sensitivity to a category of perturbations."""
    print(f"\n🔬 {name.replace('_', ' ').title()}")
    print("-" * 60)
    results = []
    for label, prompt in cfg["perturbations"]:
        print(f"  [{label}] '{prompt[:50]}...'...", end=" ", flush=True)
        r = probe(prompt, temperature=0.0, max_tokens=100)
        record = {"perturbation": label, "prompt": prompt, "result": r}
        results.append(record)
        if r["ok"]:
            preview = r["response"][:60].replace("\n", " ")
            print(f"✅ {r['wall_time_s']:.2f}s → {preview}...")
        else:
            print(f"❌ {r['error']}")
    return results


def compute_sensitivity_metrics(results):
    """Compute metrics from perturbation results."""
    metrics = {"total_tests": 0, "successful": 0, "failed": 0}

    for category, tests in results.items():
        cat_metrics = {"tests": 0, "times": [], "tps": [], "responses": []}
        for t in tests:
            r = t.get("result", {})
            metrics["total_tests"] += 1
            if r.get("ok"):
                metrics["successful"] += 1
                cat_metrics["tests"] += 1
                cat_metrics["times"].append(r["wall_time_s"])
                cat_metrics["tps"].append(r["tokens_per_second"])
                cat_metrics["responses"].append(r["response"][:200])
            else:
                metrics["failed"] += 1

        if cat_metrics["tests"] > 0:
            n = cat_metrics["tests"]
            cat_metrics["avg_time"] = round(sum(cat_metrics["times"]) / n, 4)
            cat_metrics["std_time"] = round(
                (sum((t - cat_metrics["avg_time"]) ** 2 for t in cat_metrics["times"]) / n) ** 0.5, 4
            )
            cat_metrics["avg_tps"] = round(sum(cat_metrics["tps"]) / n, 2)
            # Response diversity — how many unique responses
            unique = set(r[:100] for r in cat_metrics["responses"])
            cat_metrics["response_diversity"] = len(unique) / n if n > 0 else 0

        metrics[category] = cat_metrics

    return metrics


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LFM2.5-8B Sensitivity Probe")
    print("=" * 70)

    if not check_server():
        print("\n❌ Server not running at", BASE_URL)
        print("Start it with:")
        print("  /home/billz/llama.cpp/build/bin/llama-server -m /home/billz/LFM2.5-8B-A1B-Uncensored-Gaston-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --ctx-size 131072 --cache-type-k q4_0 --cache-type-v q4_0 --no-kv-offload --flash-attn auto --mlock --cont-batching --parallel 1 -ngl 99")
        sys.exit(1)

    print(f"\n✅ Server running at {BASE_URL}\n")

    all_results = {}

    # Temperature sweep
    all_results["temperature_sweep"] = run_temperature_sweep(PERTURBATIONS["temperature_sweep"])

    # Perturbation tests
    for name in ["character_swap", "token_removal", "token_addition", "repetition", "context_length", "semantic_perturbation"]:
        all_results[name] = run_perturbation_test(PERTURBATIONS[name], name)

    # Compute metrics
    metrics = compute_sensitivity_metrics(all_results)

    # Save
    out_file = output_dir / "sensitivity_probe_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "metrics": metrics}, f, indent=2, default=str)
    print(f"\n📊 Results saved to {out_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SENSITIVITY METRICS")
    print("=" * 70)
    print(f"Total tests: {metrics['total_tests']}  Successful: {metrics['successful']}  Failed: {metrics['failed']}")
    print()
    for cat, m in sorted(metrics.items()):
        if cat in ("total_tests", "successful", "failed"):
            continue
        if isinstance(m, dict) and "avg_time" in m:
            print(f"  {cat:<25} avg_time={m['avg_time']:.3f}s σ={m['std_time']:.3f}s  "
                  f"avg_tps={m['avg_tps']:.1f}  diversity={m.get('response_diversity', 0):.2f}")

    print("\n💡 Interpretation:")
    print("  - High diversity (>0.8) → routing is sensitive to perturbation")
    print("  - Low diversity (<0.3) → routing is robust/stable")
    print("  - High σ in timing → perturbations change compute path significantly")
    print("  - Low σ in timing → compute path is similar regardless of perturbation")


if __name__ == "__main__":
    main()
