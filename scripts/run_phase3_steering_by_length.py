#!/usr/bin/env python3
"""Phase 3: Steering effectiveness vs prompt length.

Tests whether steering vector injection degrades with longer prompts. Creates
short (5-15 tokens), medium (30-60 tokens), and long (100-256 tokens) variants
of factual recall and JSON prompts.

For each length variant:
  - Compute steering vector from clean/corrupt pairs
  - Inject at hub layers with multiple strengths
  - Measure KL divergence and target logit delta

Reports whether steering effectiveness degrades with prompt length.

Usage:
    python scripts/run_phase3_steering_by_length.py --model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend
from mi_atlas.steering import compute_steering_vector, inject_steering_vector
from mi_atlas.metrics import kl_divergence
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, PROJECT_ROOT, set_seed, now_iso


# ── Hub layers by model ────────────────────────────────────────────────
HUB_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [2, 21, 22, 23],
    "Qwen/Qwen2.5-1.5B": [2, 21, 25, 26, 27],
}


# ── Prompt templates by length bucket ──────────────────────────────────

# Short: 5-15 tokens
SHORT_FACTUAL = [
    "The capital of France is ",
    "The capital of Japan is ",
    "The capital of Germany is ",
    "The capital of Italy is ",
    "The largest planet is ",
    "Water boils at 100 ",
    "The symbol for gold is ",
    "The fastest land animal is ",
]

SHORT_JSON = [
    "Return valid JSON: name=Alice, age=30\n",
    "Return valid JSON: color=red, count=5\n",
    "JSON: city=London, pop=9\n",
    "JSON with name and age: Bob, 25\n",
    "JSON: x=1, y=2\n",
    "Return JSON: title=test, val=42\n",
    "JSON: fruit=apple, qty=10\n",
    "JSON: lang=Python, ver=3\n",
]

# Medium: 30-60 tokens
MEDIUM_FACTUAL = [
    "Please answer the following factual question accurately. What is the capital city of France? The answer is ",
    "I need to know the capital of Japan for my geography homework. The capital of Japan is ",
    "For a trivia game, tell me: what is the capital of Germany? The answer is ",
    "In the context of European geography, the capital city of Italy is ",
    "When considering the solar system, the largest planet is definitely ",
    "According to basic chemistry, water boils at 100 degrees on the Celsius scale. In Fahrenheit, that would be ",
    "The chemical element with the symbol Au on the periodic table is commonly known as ",
    "Among all land animals, the fastest runner is the ",
]

MEDIUM_JSON = [
    "Convert the following information into a valid JSON object with keys 'name' and 'age': The person's name is Alice and she is 30 years old.\n",
    "Create a JSON response with exactly two keys 'color' and 'count'. The color is red and the count is 5.\n",
    "I need you to return a properly formatted JSON object. The keys should be 'city' and 'population'. City is London and population is 9 million.\n",
    "Please format this data as valid JSON with 'name' and 'age' fields. The name is Bob and he is 25 years old.\n",
    "Return a JSON object with keys 'x' and 'y' where x equals 1 and y equals 2. Make sure it is valid JSON.\n",
    "Generate a JSON structure containing 'title' set to 'test' and 'value' set to 42. Return only valid JSON.\n",
    "I want a JSON response with 'fruit' set to 'apple' and 'quantity' set to 10. Return valid JSON output.\n",
    "Format as JSON: the programming language is Python and the version number is 3. Use keys 'language' and 'version'.\n",
]

# Long: 100-256 tokens
LONG_FACTUAL = [
    "I'm preparing a comprehensive geography quiz for my university students and I need accurate answers. "
    "One of the questions asks about the capital city of France, which is a major European country known for "
    "its rich history, art, and cuisine. The city I'm looking for is situated along the Seine River and is "
    "famous for landmarks like the Eiffel Tower and the Louvre Museum. What is the capital of France? The answer is ",

    "For my research paper on Asian capitals, I need to verify the capital of Japan. This island nation in "
    "East Asia has a constitutional monarchy and is known for its unique blend of ancient traditions and "
    "modern technology. The capital city is the most populous metropolitan area in the world and hosted "
    "the Olympic Games. The capital of Japan is ",

    "In my European history class, we're studying major capital cities. Germany, officially the Federal "
    "Republic of Germany, is a country in Central Europe. Its capital was famously divided during the Cold War "
    "and then reunified in 1990. This city is known for its history, culture, and the famous Brandenburg Gate. "
    "The capital of Germany is ",

    "I'm writing a travel guide about Mediterranean countries. Italy, officially the Italian Republic, is a "
    "boot-shaped peninsula extending into the Mediterranean Sea. Its capital city is located in the central-western "
    "portion of the Italian Peninsula and is home to Vatican City. This ancient city was the center of the Roman Empire. "
    "The capital of Italy is ",

    "For an astronomy presentation, I'm listing facts about our solar system. When considering all the planets "
    "orbiting our Sun, from Mercury to Neptune, one stands out due to its massive size. This gas giant has a "
    "mass more than twice that of all other planets combined and is famous for its Great Red Spot. "
    "The largest planet in our solar system is ",

    "In a chemistry lecture about phase transitions, we discussed how pure water at standard atmospheric pressure "
    "undergoes a phase change from liquid to gas at a specific temperature on the Celsius scale. This temperature "
    "is commonly referenced in everyday life and scientific experiments. Water boils at 100 degrees ",

    "While studying the periodic table of elements for my chemistry exam, I came across element number 79. "
    "This precious metal has been valued throughout human history for its beauty and resistance to corrosion. "
    "Its chemical symbol comes from the Latin word 'aurum'. The chemical symbol for gold is ",

    "In our biology class discussing animal locomotion, we analyzed the running speeds of various land animals. "
    "The cheetah, found primarily in Africa, holds the record as the fastest land animal, capable of reaching "
    "speeds up to 70 mph in short bursts. The fastest land animal is the ",
]

LONG_JSON = [
    "I need you to help me with a data formatting task. Please take the following information and convert it "
    "into a properly structured JSON object. The object should have exactly two keys: 'name' and 'age'. "
    "The person's name is Alice and she is 30 years old. Make sure to return only valid JSON with no extra text "
    "or explanation. The expected format is: {\"name\": \"Alice\", \"age\": 30}. Now please return the JSON:\n",

    "For a REST API response, I need to format some data as JSON. The response should contain exactly two fields: "
    "'color' and 'count'. The color value should be 'red' and the count value should be the number 5. "
    "Please ensure the JSON is valid and properly formatted with correct types - strings should be quoted "
    "and numbers should not be quoted. Return only the JSON object:\n",

    "I'm building a city information database and need to store data in JSON format. Each entry has 'city' "
    "and 'population' as keys. For this entry, the city name is 'London' and the population is approximately "
    "9 million people. Please create a valid JSON object with these two keys and their values. "
    "Remember that the population should be a number, not a string. Return the JSON:\n",

    "Please help me create a user profile in JSON format. The profile needs exactly two fields: 'name' and 'age'. "
    "The user's name is Bob and he is 25 years old. I need this in proper JSON format where the name is a string "
    "and the age is an integer. Do not include any markdown formatting or code blocks. Just return the raw JSON:\n",

    "I'm writing a configuration file that needs to be in valid JSON format. The configuration has two parameters: "
    "'x' representing the horizontal coordinate and 'y' representing the vertical coordinate. Both values are "
    "integers - x is 1 and y is 2. Please return a valid JSON object with these two keys and their integer values. "
    "No explanation needed, just the JSON:\n",

    "For a test automation framework, I need to generate test data in JSON format. The test case requires an object "
    "with 'title' set to the string 'test' and 'value' set to the integer 42. Please create a properly formatted "
    "JSON object with these two fields. Ensure proper comma placement and bracket matching. Return only the JSON:\n",

    "I'm updating a fruit inventory system and need to add a new entry. The entry should be a JSON object with "
    "two fields: 'fruit' which should be 'apple' and 'quantity' which should be the number 10. "
    "Make sure the JSON is valid - strings must be in double quotes, numbers must not be quoted, "
    "and there should be no trailing commas. Return the JSON object:\n",

    "For a programming language reference card, I need to store information in JSON format. The data includes "
    "'language' set to 'Python' and 'version' set to 3. Create a valid JSON object with these two string keys. "
    "Both values should be strings in double quotes. The output should be parseable by any standard JSON parser. "
    "Return only the JSON without any additional text:\n",
]


def get_prompts_by_bucket(family):
    """Return (short, medium, long) prompt lists for a family."""
    if family == "factual_recall":
        return SHORT_FACTUAL, MEDIUM_FACTUAL, LONG_FACTUAL
    elif family == "json_schema":
        return SHORT_JSON, MEDIUM_JSON, LONG_JSON
    else:
        # Fallback to factual
        return SHORT_FACTUAL, MEDIUM_FACTUAL, LONG_FACTUAL


def get_corrupt_prompts(family):
    """Return (short, medium, long) corrupt prompt lists."""
    if family == "factual_recall":
        short_corrupt = [
            "The capital of Germany is ",
            "The capital of Italy is ",
            "The capital of Spain is ",
            "The capital of France is ",
            "The smallest planet is ",
            "Water freezes at 100 ",
            "The symbol for silver is ",
            "The slowest land animal is ",
        ]
        medium_corrupt = [
            "Please answer the following factual question accurately. What is the capital city of Germany? The answer is ",
            "I need to know the capital of Italy for my geography homework. The capital of Italy is ",
            "For a trivia game, tell me: what is the capital of Spain? The answer is ",
            "In the context of European geography, the capital city of France is ",
            "When considering the solar system, the smallest planet is definitely ",
            "According to basic chemistry, water freezes at 100 degrees on the Celsius scale. In Fahrenheit, that would be ",
            "The chemical element with the symbol Ag on the periodic table is commonly known as ",
            "Among all land animals, the slowest runner is the ",
        ]
        long_corrupt = [
            "I'm preparing a comprehensive geography quiz for my university students and I need accurate answers. "
            "One of the questions asks about the capital city of Germany. What is the capital of Germany? The answer is ",
            "For my research paper on Asian capitals, I need to verify the capital of Italy. The capital of Italy is ",
            "In my European history class, we're studying major capital cities. The capital of Spain is ",
            "I'm writing a travel guide about Mediterranean countries. The capital of France is ",
            "For an astronomy presentation, the smallest planet in our solar system is ",
            "In a chemistry lecture, water freezes at 100 degrees ",
            "While studying the periodic table, the chemical symbol for silver is ",
            "In our biology class, the slowest land animal is the ",
        ]
        return short_corrupt, medium_corrupt, long_corrupt
    else:
        # For JSON, use factual corrupt as fallback
        short_corrupt = [
            "The capital of Germany is ",
            "The capital of Italy is ",
            "The capital of Spain is ",
            "The capital of France is ",
            "The smallest planet is ",
            "Water freezes at 100 ",
            "The symbol for silver is ",
            "The slowest land animal is ",
        ]
        return short_corrupt, short_corrupt, short_corrupt


def compute_kl_effect(backend, prompt, sv, layer_name, strengths):
    """Compute KL divergence at each steering strength."""
    results = []
    for strength in strengths:
        try:
            output = inject_steering_vector(
                backend, prompt, layer_name, sv, strength
            )
            if output.get("status") != "success":
                continue

            baseline_logits = output["original_logits"][0, -1, :]
            steered_logits = output["steered_logits"][0, -1, :]

            kl = kl_divergence(
                baseline_logits.unsqueeze(0), steered_logits.unsqueeze(0)
            )

            # Target logit delta (compare top-1 change)
            baseline_top = baseline_logits.argmax().item()
            steered_top = steered_logits.argmax().item()

            results.append({
                "strength": strength,
                "kl": float(kl),
                "baseline_top_token": baseline_top,
                "steered_top_token": steered_top,
                "top1_changed": baseline_top != steered_top,
            })
        except Exception:
            pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 steering by length")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    model_slug = args.model.split("/")[-1]
    hub_layers = HUB_LAYERS.get(args.model, [2, 12, 22])
    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]

    print(f"Phase 3: Steering by prompt length")
    print(f"Model: {args.model}")
    print(f"Hub layers: {hub_layers}")
    print(f"Strengths: {strengths}")

    # Load model
    print("\nLoading model...")
    start = time.time()
    bundle = load_model(args.model)
    backend = create_backend(bundle)
    print(f"  Loaded in {time.time() - start:.1f}s, {backend.n_layers} layers")

    # Load pairs for steering vector computation
    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    pairs = []
    if pairs_path.exists():
        pairs = load_json(pairs_path)
        print(f"  Loaded {len(pairs)} clean/corrupt pairs")

    families = ["factual_recall", "json_schema"]
    length_buckets = ["short", "medium", "long"]
    all_results = {}

    for family in families:
        print(f"\n{'='*60}")
        print(f"Family: {family}")
        family_pairs = [p for p in pairs if p.get("family") == family]

        short_prompts, med_prompts, long_prompts = get_prompts_by_bucket(family)
        short_corrupt, med_corrupt, long_corrupt = get_corrupt_prompts(family)

        bucket_data = {
            "short": {"clean": short_prompts, "corrupt": short_corrupt, "token_range": "5-15"},
            "medium": {"clean": med_prompts, "corrupt": med_corrupt, "token_range": "30-60"},
            "long": {"clean": long_prompts, "corrupt": long_corrupt, "token_range": "100-256"},
        }

        family_result = {}

        for bucket_name, bucket_info in bucket_data.items():
            clean_prompts = bucket_info["clean"]
            corrupt_prompts = bucket_info["corrupt"]
            token_range = bucket_info["token_range"]

            print(f"\n  Bucket: {bucket_name} ({token_range} tokens, {len(clean_prompts)} prompts)")

            # Measure actual token lengths
            token_lengths = []
            for p in clean_prompts[:3]:
                inputs = backend.tokenize(p)
                token_lengths.append(inputs["input_ids"].shape[1])
            avg_tokens = float(np.mean(token_lengths)) if token_lengths else 0

            bucket_results = {
                "token_range": token_range,
                "avg_tokens_measured": avg_tokens,
                "n_prompts": len(clean_prompts),
                "per_layer": {},
            }

            for layer_idx in hub_layers:
                layer_name = f"layer_{layer_idx:02d}"
                print(f"    Layer {layer_idx}...", end="", flush=True)

                # Compute steering vector from clean/corrupt pairs
                if family_pairs:
                    pair = family_pairs[0]
                    pos_prompts = [pair["clean"]] if isinstance(pair["clean"], str) else [pair["clean"]]
                    neg_prompts = [pair["corrupt"]] if isinstance(pair["corrupt"], str) else [pair["corrupt"]]
                else:
                    pos_prompts = clean_prompts[:3]
                    neg_prompts = corrupt_prompts[:3]

                try:
                    sv = compute_steering_vector(
                        backend, pos_prompts, neg_prompts, layer_name
                    )
                    sv_norm = float(torch.norm(sv).item())
                except Exception as e:
                    print(f" SV failed: {e}")
                    continue

                # Run steering on clean prompts
                kl_by_strength = {s: [] for s in strengths}
                logit_delta_by_strength = {s: [] for s in strengths}

                for prompt in clean_prompts:
                    per_prompt = compute_kl_effect(
                        backend, prompt, sv, layer_name, strengths
                    )
                    for r in per_prompt:
                        s = r["strength"]
                        kl_by_strength[s].append(r["kl"])
                        logit_delta_by_strength[s].append(1.0 if r["top1_changed"] else 0.0)

                layer_summary = []
                for s in strengths:
                    kls = kl_by_strength[s]
                    deltas = logit_delta_by_strength[s]
                    if kls:
                        layer_summary.append({
                            "strength": s,
                            "mean_kl": float(np.mean(kls)),
                            "std_kl": float(np.std(kls)),
                            "top1_change_rate": float(np.mean(deltas)) if deltas else 0,
                            "n": len(kls),
                        })

                bucket_results["per_layer"][str(layer_idx)] = {
                    "steering_vector_norm": sv_norm,
                    "by_strength": layer_summary,
                }

                # Print best strength for this layer
                if layer_summary:
                    best = max(layer_summary, key=lambda x: x["mean_kl"])
                    print(f" best KL={best['mean_kl']:.3f} at s={best['strength']}")
                else:
                    print(" no results")

            family_result[bucket_name] = bucket_results

        all_results[family] = family_result

    # ── Cross-length comparison ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CROSS-LENGTH COMPARISON")

    comparisons = {}
    for family in families:
        if family not in all_results:
            continue

        # Find best KL per bucket (across all layers and strengths)
        bucket_peaks = {}
        for bucket_name in length_buckets:
            if bucket_name not in all_results[family]:
                continue
            bucket = all_results[family][bucket_name]
            peak_kl = 0
            peak_layer = None
            peak_strength = None
            for layer_key, layer_data in bucket["per_layer"].items():
                for s_data in layer_data["by_strength"]:
                    if s_data["mean_kl"] > peak_kl:
                        peak_kl = s_data["mean_kl"]
                        peak_layer = int(layer_key)
                        peak_strength = s_data["strength"]
            bucket_peaks[bucket_name] = {
                "peak_kl": peak_kl,
                "peak_layer": peak_layer,
                "peak_strength": peak_strength,
            }

        # Compute degradation ratio
        short_peak = bucket_peaks.get("short", {}).get("peak_kl", 0)
        long_peak = bucket_peaks.get("long", {}).get("peak_kl", 0)
        degradation_ratio = long_peak / max(short_peak, 1e-10)

        comparisons[family] = {
            "bucket_peaks": bucket_peaks,
            "short_peak_kl": short_peak,
            "long_peak_kl": long_peak,
            "degradation_ratio": degradation_ratio,
            "degrades_significantly": degradation_ratio < 0.5,
        }

        print(f"\n  {family}:")
        for bn in length_buckets:
            if bn in bucket_peaks:
                bp = bucket_peaks[bn]
                print(f"    {bn:8s}: peak KL={bp['peak_kl']:.3f} at L{bp['peak_layer']} s={bp['peak_strength']}")
        print(f"    Degradation ratio (long/short): {degradation_ratio:.2f}")
        print(f"    Significant degradation: {'YES' if degradation_ratio < 0.5 else 'NO'}")

    # Save results
    results = {
        "experiment": "steering_by_length",
        "phase": 3,
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "n_layers": backend.n_layers,
        "hub_layers": hub_layers,
        "strengths": strengths,
        "timestamp": now_iso(),
        "families": all_results,
        "comparisons": comparisons,
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_steering_by_length_{model_slug}.json"
    save_json(results, out_path)
    print(f"\nResults saved: {out_path}")

    # Register
    any_degrades = any(c.get("degrades_significantly") for c in comparisons.values())
    register_experiment(
        type="robustness",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[str(pairs_path)],
        outputs=[str(out_path)],
        status="success",
        summary=(
            f"Steering by length: "
            + ", ".join(f"{fam} degradation={c['degradation_ratio']:.2f}"
                        for fam, c in comparisons.items())
            + f". Significant degradation: {any_degrades}"
        ),
        key_metrics={
            "degradation_ratios": {fam: c["degradation_ratio"] for fam, c in comparisons.items()},
            "degrades_significantly": any_degrades,
        },
        next=(
            "If steering degrades with length, hub mechanism may be position-dependent. "
            "If stable, steering vectors transfer across prompt complexity."
        ),
    )


if __name__ == "__main__":
    main()
