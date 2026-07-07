#!/usr/bin/env python3
"""
Teacher Scoring & Reasoning Condensation via Gemma4.

For each unique prompt (with N generations at different temperatures):
1. Show all generations to Gemma4
2. Rank them by quality (correctness, format adherence, reasoning clarity)
3. Pick the winner
4. Condense the winner's reasoning into a concise thinking chain

Resumable via scored.jsonl (tracks prompt_hash).

Output: scored_rollouts.jsonl — one entry per prompt with best generation + condensed reasoning.
"""
import json, time, sys, os, hashlib, requests, random, textwrap
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ───────────────────────────────────────────────────────────────────
TEACHER_URL = os.environ.get("TEACHER_URL", "http://100.100.61.28:1234")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL",
    "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp@q4_k_m")
INPUT_DIR = os.environ.get("INPUT_DIR", "/Users/bilawalriaz/rollouts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/Users/bilawalriaz/scored")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry


# ─── Prompts ──────────────────────────────────────────────────────────────────

RANKING_PROMPT = textwrap.dedent("""\
You are a training data quality judge for a language model distillation pipeline.

A small model generated {n_gens} responses to the same prompt at different temperatures.
Your job is to:
1. Rank all generations by quality
2. Pick the single best one
3. Rewrite its thinking chain to be concise and direct

### Scoring criteria:
- **Correctness**: Does the answer match the gold answer (if provided)?
- **Format adherence**: Does the output match the requested format (JSON schema, etc)?
- **Reasoning clarity**: Is the thinking chain clear and logical, not full of dead ends?
- **Completeness**: Does it fully address the prompt?

### Prompt:
{prompt}

### Gold answer:
{gold_answer}

### Schema:
{json_schema}

### Generations:

{generations_block}

### Output format (strict JSON):
{{
  "ranking": [<indices ranked best to worst, 0-indexed>],
  "winner_index": <0-indexed best generation>,
  "quality_score": <1-10>,
  "correctness": <"correct"|"incorrect"|"partial"|"na">,
  "format_valid": <true|false>,
  "winner_reason": "<one sentence why this generation won>",
  "condensed_reasoning": "<edit the winner's thinking chain to be shorter — DELETE dead ends, 'oh wait' moments, repeated attempts, verbose restating. Do NOT rephrase or rewrite. Keep the model's exact words, just remove the fat. If already concise, keep as-is.>"
}}
""")

CONDENSE_ONLY_PROMPT = textwrap.dedent("""\
You are editing a thinking chain from a language model to make it shorter.

RULES:
- Edit the EXISTING text — do not rewrite it. Keep the model's exact words and phrasing.
- ONLY delete lines/blocks that are: dead ends, "oh wait" moments, repeated attempts at the same thing, self-corrections that went nowhere, or verbose restating of the same step.
- Do NOT rephrase, summarize, or restructure. The goal is the same thinking, just shorter.
- Do NOT add explanations or connective tissue that wasn't there.
- If it's already concise, return it unchanged.
- Output ONLY the edited reasoning, nothing else.

Original reasoning ({n_chars} chars):
{reasoning}
""")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_rollouts(input_dir):
    """Load all rollouts and group by prompt_hash."""
    groups = {}
    for f in Path(input_dir).glob("*.jsonl"):
        if f.name.startswith("completed"):
            continue
        print(f"  Loading {f.name}...", flush=True)
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ph = rec.get("prompt_hash", "")
                if ph not in groups:
                    groups[ph] = {
                        "prompt": rec.get("prompt", ""),
                        "prompt_meta": rec.get("prompt_meta", {}),
                        "generations": [],
                    }
                groups[ph]["generations"].append(rec)
    return groups


def load_scored(output_dir):
    """Load set of already-scored prompt_hashes."""
    done = set()
    scored_file = Path(output_dir) / "scored.jsonl"
    if scored_file.exists():
        with open(scored_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add(d.get("prompt_hash", ""))
    return done


def call_teacher(prompt_text, max_retries=MAX_RETRIES):
    """Call Gemma4 with retry logic."""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{TEACHER_URL}/v1/chat/completions",
                json={
                    "model": TEACHER_MODEL,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 8192,
                    "temperature": 0.1,
                },
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            choice = r.json()["choices"][0]
            msg = choice["message"]
            # LM Studio returns reasoning_content separately when thinking is on
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            # If content is empty but reasoning exists, the model hit token limit while thinking
            if not content.strip() and reasoning:
                return f"[THINKING ONLY — model hit token limit]\n{reasoning}"
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"    Retry {attempt+1}/{max_retries} after {delay}s: {e}", flush=True)
                time.sleep(delay)
            else:
                raise


def parse_teacher_response(text):
    """Extract JSON from teacher response, handling markdown code blocks."""
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove first and last lines
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def format_generations(gens):
    """Format generations block for the ranking prompt."""
    parts = []
    for i, g in enumerate(gens):
        resp = g.get("response", "")[:2000]  # truncate long responses
        reasoning = g.get("reasoning", "")[:3000]  # truncate reasoning too
        temp = g.get("temperature", "?")
        parts.append(
            f"--- Generation {i} (temp={temp}) ---\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Response:\n{resp}\n"
        )
    return "\n".join(parts)


def check_correctness(gold_answer, response):
    """Quick heuristic correctness check for math/science answers."""
    if not gold_answer:
        return "na"
    import re
    gold_nums = set(re.findall(r'-?\d+\.?\d*', str(gold_answer)))
    resp_nums = set(re.findall(r'-?\d+\.?\d*', response))
    if not gold_nums:
        return "na"
    overlap = gold_nums & resp_nums
    if len(overlap) == len(gold_nums):
        return "correct"
    if len(overlap) > 0:
        return "partial"
    return "incorrect"


def score_prompt(ph, group):
    """Score a single prompt: rank generations + condense reasoning."""
    meta = group["prompt_meta"]
    prompt = group["prompt"]
    gens = group["generations"]

    gold_answer = meta.get("gold_answer") or "N/A"
    json_schema = meta.get("json_schema") or "N/A"
    dataset = meta.get("_dataset") or meta.get("category") or "unknown"

    # Build the ranking prompt
    gen_block = format_generations(gens)
    full_prompt = RANKING_PROMPT.format(
        n_gens=len(gens),
        prompt=prompt,
        gold_answer=gold_answer,
        json_schema=json_schema,
        generations_block=gen_block,
    )

    # Call teacher
    t0 = time.monotonic()
    response_text = call_teacher(full_prompt)
    elapsed = time.monotonic() - t0

    # Parse
    result = parse_teacher_response(response_text)
    if result is None:
        # Fallback: use first generation
        print(f"    ⚠ Parse failed, using generation 0 as fallback", flush=True)
        result = {
            "ranking": list(range(len(gens))),
            "winner_index": 0,
            "quality_score": 3,
            "correctness": check_correctness(str(gold_answer), gens[0]["response"]),
            "format_valid": False,
            "winner_reason": "parse_failed_fallback",
            "condensed_reasoning": gens[0].get("reasoning", "")[:2000],
        }

    winner_idx = result.get("winner_index", 0)
    if winner_idx >= len(gens):
        winner_idx = 0

    winner = gens[winner_idx]

    # Build scored record
    scored = {
        "prompt_hash": ph,
        "prompt": prompt,
        "prompt_meta": meta,
        "dataset": dataset,
        "n_generations": len(gens),
        "winner_index": winner_idx,
        "winner_temperature": winner.get("temperature"),
        "winner_response": winner.get("response", ""),
        "condensed_reasoning": result.get("condensed_reasoning", ""),
        "original_reasoning_len": len(winner.get("reasoning", "")),
        "condensed_reasoning_len": len(result.get("condensed_reasoning", "")),
        "quality_score": result.get("quality_score", 0),
        "correctness": result.get("correctness", "na"),
        "format_valid": result.get("format_valid", False),
        "ranking": result.get("ranking", []),
        "winner_reason": result.get("winner_reason", ""),
        "gold_answer": gold_answer,
        "scoring_time_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return scored


def save_scored(scored, output_dir):
    """Append a scored record to the output file."""
    with open(Path(output_dir) / "scored.jsonl", "a") as f:
        f.write(json.dumps(scored, default=str) + "\n")


def print_stats(scored_list):
    """Print summary statistics."""
    if not scored_list:
        return
    total = len(scored_list)
    avg_quality = sum(s["quality_score"] for s in scored_list) / total
    correct = sum(1 for s in scored_list if s["correctness"] == "correct")
    partial = sum(1 for s in scored_list if s["correctness"] == "partial")
    incorrect = sum(1 for s in scored_list if s["correctness"] == "incorrect")
    na = sum(1 for s in scored_list if s["correctness"] == "na")
    avg_savings = 0
    savings_count = 0
    for s in scored_list:
        orig = s["original_reasoning_len"]
        comp = s["condensed_reasoning_len"]
        if orig > 0:
            avg_savings += (1 - comp / orig)
            savings_count += 1
    if savings_count:
        avg_savings /= savings_count

    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total prompts: {total}")
    print(f"  Avg quality:   {avg_quality:.1f}/10")
    print(f"  Correctness:   {correct} correct, {partial} partial, {incorrect} incorrect, {na} N/A")
    print(f"  Avg reasoning compression: {avg_savings*100:.0f}%")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    if cmd == "stats":
        # Just print stats from existing scored file
        scored_file = Path(OUTPUT_DIR) / "scored.jsonl"
        if not scored_file.exists():
            print("No scored file found.")
            return
        scored = []
        with open(scored_file) as f:
            for line in f:
                if line.strip():
                    scored.append(json.loads(line))
        print_stats(scored)
        return

    if cmd == "score-one":
        # Score a single prompt for testing
        test_input = sys.argv[2] if len(sys.argv) > 2 else INPUT_DIR
        groups = load_rollouts(test_input)
        if not groups:
            print("No rollouts found.")
            return
        ph = list(groups.keys())[0]
        print(f"Scoring test prompt {ph}...")
        scored = score_prompt(ph, groups[ph])
        print(json.dumps(scored, indent=2, default=str))
        return

    # Full scoring
    print(f"{'='*60}")
    print(f"TEACHER SCORING & REASONING CONDENSATION")
    print(f"{'='*60}")
    print(f"  Teacher:  {TEACHER_URL} ({TEACHER_MODEL})")
    print(f"  Input:    {INPUT_DIR}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Workers:  {MAX_WORKERS}")
    print()

    # Health check
    try:
        r = requests.get(f"{TEACHER_URL}/v1/models", timeout=10)
        r.raise_for_status()
        print("  Teacher: reachable ✅")
    except Exception as e:
        print(f"  Teacher: UNREACHABLE — {e}")
        sys.exit(1)

    # Load rollouts
    print("\nLoading rollouts...", flush=True)
    groups = load_rollouts(INPUT_DIR)
    print(f"  Unique prompts: {len(groups)}")

    # Load already scored
    done = load_scored(OUTPUT_DIR)
    remaining = {ph: g for ph, g in groups.items() if ph not in done}
    print(f"  Already scored: {len(done)}")
    print(f"  Remaining:      {len(remaining)}")

    if not remaining:
        print("\nNothing to score.")
        print_stats([])
        return

    # Score in order (for reproducibility), but with progress tracking
    scored_list = []
    total = len(remaining)

    for i, (ph, group) in enumerate(remaining.items()):
        ds = group["prompt_meta"].get("_dataset", "?")
        n = len(group["generations"])
        print(f"\n[{i+1}/{total}] [{ds}] {n} gens — {group['prompt'][:60]}...", flush=True)

        try:
            scored = score_prompt(ph, group)
            save_scored(scored, OUTPUT_DIR)
            scored_list.append(scored)

            q = scored["quality_score"]
            c = scored["correctness"]
            cr = scored["condensed_reasoning_len"]
            orr = scored["original_reasoning_len"]
            savings = f"{(1-cr/max(orr,1))*100:.0f}%" if orr > 0 else "N/A"
            print(f"  ✅ q={q}/10, {c}, reasoning {savings} compressed, {scored['scoring_time_seconds']:.1f}s", flush=True)

            # Progress every 50
            if (i + 1) % 50 == 0:
                print_stats(scored_list)

        except Exception as e:
            print(f"  ❌ {e}", flush=True)
            # Save with error so we don't retry forever
            error_rec = {
                "prompt_hash": ph,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(Path(OUTPUT_DIR) / "errors.jsonl", "a") as f:
                f.write(json.dumps(error_rec, default=str) + "\n")

        # Rate limiting: small delay between calls
        time.sleep(0.2)

    print_stats(scored_list)
    print(f"\nOutput: {OUTPUT_DIR}/scored.jsonl")


if __name__ == "__main__":
    main()
