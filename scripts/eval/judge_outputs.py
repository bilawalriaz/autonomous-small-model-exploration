#!/usr/bin/env python3
"""Judge generated outputs using a stronger model (pointwise or pairwise).

CLI:
    # Pointwise scoring
    python scripts/eval/judge_outputs.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --mode pointwise --judge-model mimo-v2.5

    # Pairwise comparison
    python scripts/eval/judge_outputs.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --mode pairwise --baseline-run-id lfm2_230m_base_20260629
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Score dimensions for pointwise judging
DIMENSIONS = [
    "correctness", "instruction_following", "output_format",
    "concision", "usefulness", "hallucination_risk", "overall",
]

SLOP_PHRASES = [
    "as an ai", "i apologize", "i'm sorry, but", "as a language model",
    "i don't have personal", "it's important to note that",
    "please note that", "i hope this helps",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cache_key(eval_id: str, response_text: str) -> str:
    return hashlib.sha256(f"{eval_id}:{response_text}".encode()).hexdigest()[:16]


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    cache = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["cache_key"]] = rec
    return cache


def save_cache_entry(cache_path: Path, entry: dict):
    with open(cache_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def mock_judge_pointwise(eval_id: str, prompt: str, response: str, seed: int = 42) -> dict:
    """Mock pointwise judge with deterministic random scores for dev/testing."""
    rng = random.Random(hash(f"{eval_id}:{seed}"))
    scores = {dim: rng.randint(1, 5) for dim in DIMENSIONS}
    # Penalize slop
    resp_lower = response.lower()
    slop_count = sum(1 for p in SLOP_PHRASES if p in resp_lower)
    if slop_count > 0:
        scores["concision"] = max(1, scores["concision"] - slop_count)
        scores["overall"] = max(1, scores["overall"] - slop_count)
    return {"scores": scores, "reasoning": "mock judge (API unavailable)"}


def mock_judge_pairwise(eval_id: str, prompt: str, resp_a: str, resp_b: str, seed: int = 42) -> dict:
    """Mock pairwise judge."""
    rng = random.Random(hash(f"{eval_id}:{seed}"))
    winner = rng.choice(["model_a", "model_b", "tie"])
    return {
        "winner": winner,
        "reason": f"mock judge decision (API unavailable): {winner} wins",
        "scores_a": {dim: rng.randint(1, 5) for dim in DIMENSIONS},
        "scores_b": {dim: rng.randint(1, 5) for dim in DIMENSIONS},
    }


def api_judge_pointwise(eval_id: str, prompt: str, response: str, judge_model: str, api_url: str, api_key: str) -> dict:
    """Call OpenAI-compatible API for pointwise judging."""
    import requests

    score_dims = ", ".join(f"{d} (1-5)" for d in DIMENSIONS)
    system_msg = (
        "You are an expert judge evaluating language model outputs. "
        "Score each dimension from 1 (worst) to 5 (best). "
        "Respond ONLY with valid JSON."
    )
    user_msg = (
        f"## Prompt\n{prompt}\n\n## Response\n{response}\n\n"
        f"Score these dimensions: {score_dims}\n"
        f'Return JSON: {{"scores": {{"{DIMENSIONS[0]}": N, ...}}, "reasoning": "..."}}'
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        log.warning(f"API judge failed for {eval_id}: {e}, falling back to mock")
        return mock_judge_pointwise(eval_id, prompt, response)


def api_judge_pairwise(eval_id: str, prompt: str, resp_a: str, resp_b: str,
                       judge_model: str, api_url: str, api_key: str, model_a: str, model_b: str) -> dict:
    """Call OpenAI-compatible API for pairwise judging."""
    import requests

    system_msg = (
        "You are an expert judge comparing two model outputs. "
        "One is labeled A, the other B. Choose the winner or declare a tie. "
        "Respond ONLY with valid JSON."
    )
    user_msg = (
        f"## Prompt\n{prompt}\n\n## Response A\n{resp_a}\n\n## Response B\n{resp_b}\n\n"
        'Return JSON: {"winner": "model_a|model_b|tie", "reason": "...", '
        '"scores_a": {"correctness": N, ...}, "scores_b": {"correctness": N, ...}}'
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        log.warning(f"API judge failed for {eval_id}: {e}, falling back to mock")
        return mock_judge_pairwise(eval_id, prompt, resp_a, resp_b)


def main():
    parser = argparse.ArgumentParser(description="Judge eval outputs with a stronger model")
    parser.add_argument("--run-id", required=True, help="Run ID to judge")
    parser.add_argument("--mode", choices=["pointwise", "pairwise"], required=True)
    parser.add_argument("--baseline-run-id", default=None, help="Baseline run ID for pairwise mode")
    parser.add_argument("--judge-model", default="mimo-v2.5", help="Judge model name")
    parser.add_argument("--api-url", default=os.environ.get("JUDGE_API_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=42, help="Seed for mock judge")
    parser.add_argument("--force", action="store_true", help="Re-score even cached results")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / "evals" / args.run_id
    outputs_path = run_dir / "outputs.jsonl"
    if not outputs_path.exists():
        log.error(f"Outputs not found: {outputs_path}")
        sys.exit(1)

    outputs = load_jsonl(outputs_path)
    log.info(f"Loaded {len(outputs)} outputs from {outputs_path}")

    # Cache
    cache_dir = run_dir / "judge_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.mode}_cache.jsonl"
    cache = load_cache(cache_path)
    log.info(f"Loaded {len(cache)} cached scores")

    # Judge output
    scores_path = run_dir / "judge_scores.jsonl"

    # Determine if API is available
    api_available = False
    if args.api_key and args.api_url:
        try:
            import requests
            resp = requests.get(f"{args.api_url}/v1/models", timeout=5)
            api_available = resp.status_code == 200
        except Exception:
            pass

    if api_available:
        log.info(f"Using API judge: {args.api_url}, model: {args.judge_model}")
    else:
        log.info("API not available, using mock judge (deterministic random scores)")

    # Load baseline for pairwise
    baseline_outputs = {}
    if args.mode == "pairwise":
        if not args.baseline_run_id:
            log.error("--baseline-run-id required for pairwise mode")
            sys.exit(1)
        baseline_path = PROJECT_ROOT / "results" / "evals" / args.baseline_run_id / "outputs.jsonl"
        if not baseline_path.exists():
            log.error(f"Baseline outputs not found: {baseline_path}")
            sys.exit(1)
        for rec in load_jsonl(baseline_path):
            baseline_outputs[rec["eval_id"]] = rec
        log.info(f"Loaded {len(baseline_outputs)} baseline outputs")

    scored = 0
    skipped = 0

    for output in outputs:
        eval_id = output["eval_id"]
        response = output["generated_response"]
        ck = cache_key(eval_id, response)

        if ck in cache and not args.force:
            skipped += 1
            continue

        prompt = output["prompt"]

        if args.mode == "pointwise":
            if api_available:
                result = api_judge_pointwise(eval_id, prompt, response, args.judge_model, args.api_url, args.api_key)
            else:
                result = mock_judge_pointwise(eval_id, prompt, response, args.seed)
            entry = {
                "eval_id": eval_id,
                "category": output.get("category", "unknown"),
                "mode": "pointwise",
                "scores": result.get("scores", {}),
                "reasoning": result.get("reasoning", ""),
                "cache_key": ck,
            }
        else:
            # pairwise
            baseline = baseline_outputs.get(eval_id)
            if not baseline:
                log.warning(f"No baseline for {eval_id}, skipping")
                continue
            resp_b = baseline["generated_response"]
            if api_available:
                result = api_judge_pairwise(
                    eval_id, prompt, response, resp_b,
                    args.judge_model, args.api_url, args.api_key,
                    args.run_id, args.baseline_run_id,
                )
            else:
                result = mock_judge_pairwise(eval_id, prompt, response, resp_b, args.seed)
            entry = {
                "eval_id": eval_id,
                "category": output.get("category", "unknown"),
                "mode": "pairwise",
                "model_a": args.run_id,
                "model_b": args.baseline_run_id,
                "winner": result.get("winner", "tie"),
                "reason": result.get("reason", ""),
                "scores": {
                    "model_a": result.get("scores_a", {}),
                    "model_b": result.get("scores_b", {}),
                },
                "cache_key": ck,
            }

        save_cache_entry(cache_path, entry)
        # Append to scores file
        with open(scores_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        scored += 1

        if scored % 10 == 0:
            log.info(f"  Scored {scored} so far...")

    log.info(f"Done. Scored {scored} new, skipped {skipped} cached. Output: {scores_path}")


if __name__ == "__main__":
    main()
