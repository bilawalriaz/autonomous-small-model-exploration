#!/usr/bin/env python3
"""
Phase 13 Qualitative Judge Evaluation

Uses mimo-v2.5 via OpenCode Go to judge:
1. Pointwise scoring of baseline outputs
2. Pointwise scoring of best noisy rollout outputs
3. Pairwise comparison (baseline vs noisy)

Usage:
    python judge_phase13.py --api-url https://opencode.ai/zen/go/v1 --api-key $OPENCODE_GO_API_KEY
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DIMENSIONS = [
    "correctness", "instruction_following", "output_format",
    "concision", "usefulness", "overall",
]

POINTWISE_SYSTEM = (
    "You are an expert judge evaluating language model outputs for structured extraction tasks. "
    "The task is: given a natural language description, extract the requested fields as structured data. "
    "Score each dimension from 1 (worst) to 5 (best). "
    "Respond ONLY with valid JSON. Do not include reasoning or explanation outside the JSON."
)

PAIRWISE_SYSTEM = (
    "You are an expert judge comparing two model outputs for a structured extraction task. "
    "The task is: given a natural language description, extract the requested fields as structured data. "
    "One output is labeled A, the other B. Choose the winner or declare a tie. "
    "Respond ONLY with valid JSON."
)


def call_judge(api_url: str, api_key: str, model: str, system: str, user: str, max_tokens: int = 2048) -> dict:
    """Call the judge API."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content")
    
    # Thinking models may put output in reasoning
    if not content and msg.get("reasoning"):
        reasoning = msg["reasoning"]
        json_match = re.search(r'\{[^{}]*\}', reasoning, re.DOTALL)
        if json_match:
            content = json_match.group(0)
    if not content:
        raise ValueError("No content in API response")
    
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def judge_pointwise(api_url: str, api_key: str, model: str, prompt: str, response: str) -> dict:
    """Score a single response."""
    if len(response) > 3000:
        response = response[:3000] + "\n... [truncated for judging]"
    
    score_dims = ", ".join(f"{d} (1-5)" for d in DIMENSIONS)
    user_msg = (
        f"## Task Prompt\n{prompt}\n\n"
        f"## Model Response\n{response}\n\n"
        f"Score these dimensions: {score_dims}\n"
        f'Return JSON: {{"scores": {{"{DIMENSIONS[0]}": N, ...}}, "reasoning": "one sentence"}}'
    )
    return call_judge(api_url, api_key, model, POINTWISE_SYSTEM, user_msg)


def judge_pairwise(api_url: str, api_key: str, model: str, prompt: str,
                   resp_a: str, resp_b: str, label_a: str = "baseline", label_b: str = "noisy") -> dict:
    """Compare two responses."""
    if len(resp_a) > 2000:
        resp_a = resp_a[:2000] + "\n... [truncated]"
    if len(resp_b) > 2000:
        resp_b = resp_b[:2000] + "\n... [truncated]"
    
    user_msg = (
        f"## Task Prompt\n{prompt}\n\n"
        f"## Response A ({label_a})\n{resp_a}\n\n"
        f"## Response B ({label_b})\n{resp_b}\n\n"
        "Which response better extracts the requested information? "
        'Return JSON: {"winner": "A|B|tie", "reason": "one sentence", '
        '"scores_a": {"correctness": N, "overall": N}, '
        '"scores_b": {"correctness": N, "overall": N}}'
    )
    return call_judge(api_url, api_key, model, PAIRWISE_SYSTEM, user_msg)


def load_phase13_data(results_dir: Path) -> list[dict]:
    """Load and merge data from 13A (baseline) and 13E (rollouts)."""
    # Load 13E for rollout data
    e_data = json.load(open(results_dir / "13E_seed42.json"))
    e_results = e_results = e_data["results"]
    
    # Load 13D for baseline data (has per-prompt baseline outputs)
    # Actually, 13E has baseline_correct flag but not the actual text
    # Let me load the raw 13E data which has cluster samples
    
    prompts_data = []
    
    # Get the eval prompts
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ptrm"))
    from run_phase13 import EVAL_PROMPTS, format_extraction_prompt
    
    for key, val in e_results.items():
        if key.startswith("_"):
            continue
        idx = val["prompt_idx"]
        if idx >= len(EVAL_PROMPTS):
            continue
        
        item = EVAL_PROMPTS[idx]
        prompt_text = format_extraction_prompt(item)
        
        clusters = val.get("clusters", [])
        best_cluster = max(clusters, key=lambda c: c["avg_field_recall"]) if clusters else None
        
        prompts_data.append({
            "prompt_idx": idx,
            "task_prompt": item["prompt"],
            "expected": item["expected"],
            "full_prompt": prompt_text,
            "baseline_correct": bool(val["baseline_correct"]),
            "n_clusters": val["n_clusters"],
            "correct_fraction": val["correct_fraction"],
            "best_cluster_sample": best_cluster["sample"] if best_cluster else "",
            "best_cluster_recall": best_cluster["avg_field_recall"] if best_cluster else 0,
            "best_cluster_size": best_cluster["size"] if best_cluster else 0,
        })
    
    return prompts_data


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Qualitative Judge")
    parser.add_argument("--api-url", default=os.environ.get("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENCODE_GO_API_KEY", ""))
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "phase13"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "phase13" / "judge_qualitative.json"))
    parser.add_argument("--max-prompts", type=int, default=15, help="Max prompts to judge")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    
    # Load data
    print("Loading Phase 13 data...")
    prompts_data = load_phase13_data(results_dir)
    print(f"Loaded {len(prompts_data)} prompts")
    
    # Filter to wrong-baseline prompts (most interesting)
    wrong_prompts = [p for p in prompts_data if not p["baseline_correct"]]
    right_prompts = [p for p in prompts_data if p["baseline_correct"]]
    
    # Sample: mix of wrong and right
    judge_prompts = wrong_prompts[:args.max_prompts - len(right_prompts)] + right_prompts
    judge_prompts = judge_prompts[:args.max_prompts]
    
    print(f"Judging {len(judge_prompts)} prompts ({len([p for p in judge_prompts if not p['baseline_correct']])} wrong baseline, {len([p for p in judge_prompts if p['baseline_correct']])} right baseline)")
    
    # We need baseline text. Since 13E doesn't store it, regenerate it.
    # Import the generation function
    from run_phase13 import load_model, generate_baseline, EVAL_PROMPTS, format_extraction_prompt
    
    print("Loading model for baseline generation...")
    model, tokenizer = load_model("cuda:0")
    
    all_results = []
    
    for i, pdata in enumerate(judge_prompts):
        idx = pdata["prompt_idx"]
        item = EVAL_PROMPTS[idx]
        prompt_text = format_extraction_prompt(item)
        
        print(f"\n--- Prompt {idx}: {item['prompt'][:60]}... ---")
        
        # Generate baseline
        baseline = generate_baseline(model, tokenizer, prompt_text)
        baseline_text = baseline["text"]
        
        best_noisy_text = pdata["best_cluster_sample"]
        
        print(f"  Baseline: {baseline_text[:80]}...")
        print(f"  Best noisy: {best_noisy_text[:80]}...")
        
        # Pointwise: baseline
        try:
            baseline_scores = judge_pointwise(args.api_url, args.api_key, args.model, 
                                               pdata["task_prompt"], baseline_text)
            print(f"  Baseline scores: {baseline_scores.get('scores', {})}")
        except Exception as e:
            print(f"  Baseline judge failed: {e}")
            baseline_scores = {"scores": {}, "error": str(e)}
        
        time.sleep(1)  # Rate limit
        
        # Pointwise: best noisy
        try:
            noisy_scores = judge_pointwise(args.api_url, args.api_key, args.model,
                                            pdata["task_prompt"], best_noisy_text)
            print(f"  Noisy scores: {noisy_scores.get('scores', {})}")
        except Exception as e:
            print(f"  Noisy judge failed: {e}")
            noisy_scores = {"scores": {}, "error": str(e)}
        
        time.sleep(1)
        
        # Pairwise
        try:
            pairwise = judge_pairwise(args.api_url, args.api_key, args.model,
                                       pdata["task_prompt"], baseline_text, best_noisy_text)
            print(f"  Pairwise winner: {pairwise.get('winner', '?')} — {pairwise.get('reason', '')}")
        except Exception as e:
            print(f"  Pairwise judge failed: {e}")
            pairwise = {"error": str(e)}
        
        time.sleep(1)
        
        all_results.append({
            "prompt_idx": idx,
            "task_prompt": pdata["task_prompt"],
            "expected": pdata["expected"],
            "baseline_text": baseline_text,
            "noisy_text": best_noisy_text,
            "baseline_correct": pdata["baseline_correct"],
            "n_clusters": pdata["n_clusters"],
            "correct_fraction": pdata["correct_fraction"],
            "baseline_scores": baseline_scores,
            "noisy_scores": noisy_scores,
            "pairwise": pairwise,
        })
    
    # Save results
    output = {
        "experiment": "phase13_qualitative_judge",
        "judge_model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(all_results),
        "results": all_results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    baseline_wins = sum(1 for r in all_results if r.get("pairwise", {}).get("winner") == "A")
    noisy_wins = sum(1 for r in all_results if r.get("pairwise", {}).get("winner") == "B")
    ties = sum(1 for r in all_results if r.get("pairwise", {}).get("winner") == "tie")
    
    baseline_avg = {}
    noisy_avg = {}
    for dim in DIMENSIONS:
        b_scores = [r["baseline_scores"]["scores"][dim] for r in all_results 
                    if "scores" in r.get("baseline_scores", {}) and dim in r["baseline_scores"].get("scores", {})]
        n_scores = [r["noisy_scores"]["scores"][dim] for r in all_results
                    if "scores" in r.get("noisy_scores", {}) and dim in r["noisy_scores"].get("scores", {})]
        if b_scores:
            baseline_avg[dim] = sum(b_scores) / len(b_scores)
        if n_scores:
            noisy_avg[dim] = sum(n_scores) / len(n_scores)
    
    print(f"Pairwise: baseline wins {baseline_wins}, noisy wins {noisy_wins}, ties {ties}")
    print(f"Baseline avg scores: {baseline_avg}")
    print(f"Noisy avg scores: {noisy_avg}")
    
    # Unload model
    del model, tokenizer
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
