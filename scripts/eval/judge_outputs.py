#!/usr/bin/env python3
"""Judge generated outputs using a stronger model (pointwise or pairwise).

CLI:
    # Pointwise scoring (API judge — will FAIL if API unavailable)
    python scripts/eval/judge_outputs.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --mode pointwise --judge-model mimo-v2.5

    # Pairwise comparison
    python scripts/eval/judge_outputs.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --mode pairwise --baseline-run-id lfm2_230m_base_20260629

    # Mock judging (explicit opt-in required for scientific integrity)
    python scripts/eval/judge_outputs.py \
        --run-id lfm2_230m_format_ablation_multi_turn_concise_20260629 \
        --mode pointwise --mock
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


def stable_seed_hash(eval_id: str, seed: int) -> int:
    """Deterministic seed from eval_id + seed, using sha256 (not Python hash())."""
    digest = hashlib.sha256(f"{eval_id}:{seed}".encode()).hexdigest()
    return int(digest[:16], 16)


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
    """Mock pointwise judge with deterministic random scores for dev/testing.

    Uses hashlib.sha256 for seeding — deterministic across Python versions and platforms.
    """
    rng = random.Random(stable_seed_hash(eval_id, seed))
    scores = {dim: rng.randint(1, 5) for dim in DIMENSIONS}
    # Penalize slop
    resp_lower = response.lower()
    slop_count = sum(1 for p in SLOP_PHRASES if p in resp_lower)
    if slop_count > 0:
        scores["concision"] = max(1, scores["concision"] - slop_count)
        scores["overall"] = max(1, scores["overall"] - slop_count)
    return {"scores": scores, "reasoning": "mock judge (deterministic — NOT real API scores)"}


def mock_judge_pairwise(eval_id: str, prompt: str, resp_a: str, resp_b: str, seed: int = 42) -> dict:
    """Mock pairwise judge with deterministic seeding."""
    rng = random.Random(stable_seed_hash(eval_id, seed))
    winner = rng.choice(["model_a", "model_b", "tie"])
    return {
        "winner": winner,
        "reason": f"mock judge decision (deterministic): {winner} wins",
        "scores_a": {dim: rng.randint(1, 5) for dim in DIMENSIONS},
        "scores_b": {dim: rng.randint(1, 5) for dim in DIMENSIONS},
    }


def api_judge_pointwise(eval_id: str, prompt: str, response: str,
                        judge_model: str, api_url: str, api_key: str) -> dict:
    """Call OpenAI-compatible API for pointwise judging.

    Raises on failure — caller must handle (no silent fallback).
    """
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


def api_judge_pairwise(eval_id: str, prompt: str, resp_a: str, resp_b: str,
                       judge_model: str, api_url: str, api_key: str,
                       model_a: str, model_b: str) -> dict:
    """Call OpenAI-compatible API for pairwise judging.

    Raises on failure — caller must handle (no silent fallback).
    """
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

    resp = requests.post(f"{api_url}/v1/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def check_api_available(api_url: str, api_key: str) -> bool:
    """Check if the judge API is reachable. Returns False on any failure."""
    try:
        import requests
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(f"{api_url}/v1/models", headers=headers, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Judge eval outputs with a stronger model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "  --mock is REQUIRED for mock judging. Without it, the script will\n"
            "  fail if the API is unavailable (no silent fallback).\n"
            "  Every score entry includes a 'judge_source' field: 'api' or 'mock'.\n"
        ),
    )
    parser.add_argument("--run-id", required=True, help="Run ID to judge")
    parser.add_argument("--mode", choices=["pointwise", "pairwise"], required=True)
    parser.add_argument("--baseline-run-id", default=None, help="Baseline run ID for pairwise mode")
    parser.add_argument("--judge-model", default="mimo-v2.5", help="Judge model name")
    parser.add_argument("--api-url", default=os.environ.get("JUDGE_API_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", ""))
    parser.add_argument("--seed", type=int, default=42, help="Seed for mock judge")
    parser.add_argument("--force", action="store_true", help="Re-score even cached results")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock (deterministic random) judging instead of API. "
             "All scores will be marked judge_source='mock'.",
    )
    parser.add_argument(
        "--strict-report-mode", action="store_true",
        help="Fail if judge API is unavailable AND --mock is not set. "
             "Prevents any use of mock scores in strict reporting.",
    )
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

    # Determine API availability
    api_available = False
    if not args.mock:
        api_available = check_api_available(args.api_url, args.api_key)

    # Enforce strict report mode
    if args.strict_report_mode and not api_available and not args.mock:
        log.error(
            "STRICT REPORT MODE: Judge API is not available and --mock was not set.\n"
            f"  API URL: {args.api_url}\n"
            f"  To proceed with mock judging, add --mock (scores will be marked judge_source='mock').\n"
            f"  To use a real API, ensure the judge server is running and set --api-url / JUDGE_API_URL."
        )
        sys.exit(1)

    # If not mock and API is unavailable, fail hard (no silent fallback)
    if not args.mock and not api_available:
        log.error(
            "Judge API is not available and --mock was not set. Refusing to proceed.\n"
            f"  API URL: {args.api_url}\n"
            f"  Set --mock to use deterministic mock scores (will be marked judge_source='mock'),\n"
            f"  or ensure the judge server is running."
        )
        sys.exit(1)

    # Determine judge source
    judge_source = "mock" if args.mock else "api"

    if judge_source == "api":
        log.info(f"Using API judge: {args.api_url}, model: {args.judge_model}")
    else:
        log.warning(
            "Using MOCK judge (deterministic random scores). "
            "All scores will be marked judge_source='mock'."
        )

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
    api_count = 0
    mock_count = 0

    for output in outputs:
        eval_id = output["eval_id"]
        response = output["generated_response"]
        ck = cache_key(eval_id, response)

        if ck in cache and not args.force:
            skipped += 1
            # Count judge_source from cached entries
            cached_entry = cache[ck]
            if cached_entry.get("judge_source") == "mock":
                mock_count += 1
            else:
                api_count += 1
            continue

        prompt = output["prompt"]

        if args.mode == "pointwise":
            if judge_source == "api":
                try:
                    result = api_judge_pointwise(
                        eval_id, prompt, response,
                        args.judge_model, args.api_url, args.api_key,
                    )
                except Exception as e:
                    log.error(f"API judge failed for {eval_id}: {e}")
                    sys.exit(1)
            else:
                result = mock_judge_pointwise(eval_id, prompt, response, args.seed)
            entry = {
                "eval_id": eval_id,
                "category": output.get("category", "unknown"),
                "mode": "pointwise",
                "judge_source": judge_source,
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
            if judge_source == "api":
                try:
                    result = api_judge_pairwise(
                        eval_id, prompt, response, resp_b,
                        args.judge_model, args.api_url, args.api_key,
                        args.run_id, args.baseline_run_id,
                    )
                except Exception as e:
                    log.error(f"API judge failed for {eval_id}: {e}")
                    sys.exit(1)
            else:
                result = mock_judge_pairwise(eval_id, prompt, response, resp_b, args.seed)
            entry = {
                "eval_id": eval_id,
                "category": output.get("category", "unknown"),
                "mode": "pairwise",
                "judge_source": judge_source,
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
        if judge_source == "mock":
            mock_count += 1
        else:
            api_count += 1

        if scored % 10 == 0:
            log.info(f"  Scored {scored} so far...")

    # Final summary
    total = scored + skipped
    print(f"\n{'='*60}")
    print(f"  JUDGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total scored:    {total}")
    print(f"  New this run:    {scored}")
    print(f"  Cached (skipped): {skipped}")
    print(f"  Judge source breakdown:")
    print(f"    api:  {api_count}")
    print(f"    mock: {mock_count}")
    if mock_count > 0:
        print(f"\n  ⚠ WARNING: {mock_count} score(s) used mock judging (judge_source='mock').")
        print(f"    These are NOT real API scores. Do not report them as real evaluation results.")
    print(f"{'='*60}\n")

    log.info(f"Done. Scored {scored} new, skipped {skipped} cached. Output: {scores_path}")


if __name__ == "__main__":
    main()
