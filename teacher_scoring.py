#!/usr/bin/env python3
"""
Teacher Scoring & Reasoning Condensation via Gemma4.

For each unique prompt (with N generations at different temperatures):
1. Show all generations to Gemma4 in a SINGLE call
2. Rank them by quality (correctness, format adherence, reasoning clarity)
3. Pick the winner
4. Condense the winner's reasoning in the same call (delete dead ends, keep voice)

Resumable via scored.jsonl (tracks prompt_hash).
Output: scored_rollouts.jsonl — one entry per prompt with best generation + condensed reasoning.

Token budget per call (Gemma4 Q4_K_M, 128k context):
- System prompt: ~300 tokens
- Prompt + gold + schema: ~200-2000 tokens
- Per generation: reasoning (capped 3000 chars ≈ 750 tok) + response (capped 2000 chars ≈ 500 tok)
- 6 gens: ~7,500 tokens
- Output: ~2,000 tokens
- Total median: ~5,000 tokens | p95: ~20,000 tokens | max safe: ~100,000 tokens
"""
import json, time, sys, os, hashlib, requests, random, textwrap
from pathlib import Path
from datetime import datetime, timezone

# ─── Config ───────────────────────────────────────────────────────────────────
TEACHER_URL = os.environ.get("TEACHER_URL", "http://100.100.61.28:8080")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gemma4")  # llama.cpp ignores model name, uses whatever is loaded
INPUT_DIR = os.environ.get("INPUT_DIR", "/Users/bilawalriaz/rollouts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/Users/bilawalriaz/scored")
TIMEOUT = int(os.environ.get("TIMEOUT", "180"))
MAX_RETRIES = 3
RETRY_DELAY = 3

# Truncation limits — keep per-gen token count manageable
MAX_REASONING_CHARS = int(os.environ.get("MAX_REASONING_CHARS", "3000"))  # ~750 tokens
MAX_RESPONSE_CHARS = int(os.environ.get("MAX_RESPONSE_CHARS", "2000"))   # ~500 tokens
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "4000"))       # ~1000 tokens


# ─── Prompts ──────────────────────────────────────────────────────────────────

RANKING_PROMPT = textwrap.dedent("""\
You are a training data quality judge for a language model distillation pipeline.

A small model generated {n_gens} responses to the same prompt at different temperatures.
Your job is to:
1. Rank all generations by quality
2. Pick the single best one
3. Edit its thinking chain to be shorter

### Scoring criteria (applied strictly):
- **Correctness**: Does the answer match the gold answer (if provided)?
- **Format adherence (STRICT)**: The prompt specifies an exact output format. Check:
  - If the prompt says "Respond in JSON" → output MUST be valid JSON, not YAML or prose
  - If the prompt says "Provide the output in YAML" → output MUST be valid YAML, not JSON
  - If the prompt says "Output the raw JSON directly without code block fencing" → no ```json fences allowed
  - If the prompt specifies a schema → the output MUST match that schema's structure
  - If the prompt says "Just the JSON, no commentary" → no preamble or explanation before the output
  - A response with the RIGHT content but WRONG format gets penalized. Format violations are a disqualifying flaw.
- **Reasoning clarity**: Is the thinking chain clear and logical, not full of dead ends?
- **Completeness**: Does it fully address the prompt?

### What the prompt asked for:
{prompt}

### Gold answer:
{gold_answer}

### Generations:

{generations_block}

### Output format (strict JSON — no markdown fences):
{{
  "ranking": [<indices ranked best to worst, 0-indexed>],
  "winner_index": <0-indexed best generation>,
  "quality_score": <1-10>,
  "correctness": <"correct"|"incorrect"|"partial"|"na">,
  "format_valid": <true|false — DID THE OUTPUT MATCH THE REQUESTED FORMAT EXACTLY?>,
  "winner_reason": "<one sentence why this generation won>",
  "format_violations": "<list any format issues across all generations, e.g. 'Gen 0 output JSON when YAML was requested'>",
  "condensed_reasoning": "<edit the winner's thinking chain to be shorter — DELETE dead ends, 'oh wait' moments, repeated attempts, verbose restating. Do NOT rephrase or rewrite. Keep the model's exact words, just remove the fat. If already concise, keep as-is.>"
}}
""")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def truncate(text, max_chars):
    """Truncate text to max_chars, cutting at last newline if possible."""
    if not text or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind('\n')
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut] + f"\n[...truncated from {len(text)} chars]"


def load_rollouts(input_dir):
    """Load all rollouts and group by prompt_hash."""
    groups = {}
    input_path = Path(input_dir)

    # If input_dir is a file, use it directly
    if input_path.is_file() and input_path.suffix == '.jsonl':
        files = [input_path]
    else:
        files = list(input_path.glob("*.jsonl"))
        files = [f for f in files if f.name != "completed.jsonl"]

    for f in files:
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
    """Call Gemma4 with retry logic. Returns content string."""
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
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            # If content is empty but reasoning exists, model hit token limit while thinking
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
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def format_generations(gens):
    """Format generations block for the ranking prompt, with truncation."""
    parts = []
    for i, g in enumerate(gens):
        reasoning = truncate(g.get("reasoning", ""), MAX_REASONING_CHARS)
        response = truncate(g.get("response", ""), MAX_RESPONSE_CHARS)
        temp = g.get("temperature", "?")
        parts.append(
            f"--- Generation {i} (temp={temp}) ---\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Response:\n{response}\n"
        )
    return "\n".join(parts)


def score_prompt(ph, group):
    """Score a single prompt: rank generations + condense reasoning, all in one call."""
    meta = group["prompt_meta"]
    prompt = truncate(group["prompt"], MAX_PROMPT_CHARS)
    gens = group["generations"]

    gold_answer = meta.get("gold_answer") or "N/A"
    dataset = meta.get("_dataset") or meta.get("category") or "unknown"

    # Build the ranking prompt
    gen_block = format_generations(gens)
    full_prompt = RANKING_PROMPT.format(
        n_gens=len(gens),
        prompt=prompt,
        gold_answer=gold_answer,
        generations_block=gen_block,
    )

    # Single teacher call: rank + condense in one shot
    t0 = time.monotonic()
    response_text = call_teacher(full_prompt)
    elapsed = time.monotonic() - t0

    # Parse
    result = parse_teacher_response(response_text)
    if result is None:
        print(f"    ⚠ Parse failed, using generation 0 as fallback", flush=True)
        result = {
            "ranking": list(range(len(gens))),
            "winner_index": 0,
            "quality_score": 3,
            "correctness": "na",
            "format_valid": False,
            "winner_reason": "parse_failed_fallback",
            "condensed_reasoning": gens[0].get("reasoning", "")[:MAX_REASONING_CHARS],
        }

    winner_idx = result.get("winner_index", 0)
    if winner_idx >= len(gens):
        winner_idx = 0
    winner = gens[winner_idx]

    # Build scored record
    orig_reasoning = winner.get("reasoning", "")
    cond_reasoning = result.get("condensed_reasoning", "")

    scored = {
        "prompt_hash": ph,
        "prompt": group["prompt"],
        "prompt_meta": meta,
        "dataset": dataset,
        "n_generations": len(gens),
        "winner_index": winner_idx,
        "winner_temperature": winner.get("temperature"),
        "winner_response": winner.get("response", ""),
        "condensed_reasoning": cond_reasoning,
        "original_reasoning_len": len(orig_reasoning),
        "condensed_reasoning_len": len(cond_reasoning),
        "quality_score": result.get("quality_score", 0),
        "correctness": result.get("correctness", "na"),
        "format_valid": result.get("format_valid", False),
        "ranking": result.get("ranking", []),
        "winner_reason": result.get("winner_reason", ""),
        "gold_answer": str(gold_answer)[:500],
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

    avg_time = sum(s["scoring_time_seconds"] for s in scored_list) / total

    print(f"\n{'='*60}")
    print(f"SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total prompts: {total}")
    print(f"  Avg quality:   {avg_quality:.1f}/10")
    print(f"  Correctness:   {correct} correct, {partial} partial, {incorrect} incorrect, {na} N/A")
    print(f"  Avg reasoning compression: {avg_savings*100:.0f}%")
    print(f"  Avg time per prompt: {avg_time:.1f}s")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    # If first arg looks like a directory/file, treat it as input for full run
    input_override = None
    if cmd not in ("stats", "score-one", "dry-run", "full") and os.path.isdir(os.path.expanduser(cmd)):
        input_override = os.path.expanduser(cmd)
        cmd = "full"

    if cmd == "stats":
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

    if cmd == "dry-run":
        # Score 5 prompts without saving — for validation
        test_input = sys.argv[2] if len(sys.argv) > 2 else INPUT_DIR
        groups = load_rollouts(test_input)
        if not groups:
            print("No rollouts found.")
            return
        scored_list = []
        for i, (ph, group) in enumerate(groups.items()):
            if i >= 5:
                break
            ds = group["prompt_meta"].get("_dataset", "?")
            n = len(group["generations"])
            print(f"\n[{i+1}/5] [{ds}] {n} gens — {group['prompt'][:60]}...", flush=True)
            scored = score_prompt(ph, group)
            scored_list.append(scored)
            q = scored["quality_score"]
            c = scored["correctness"]
            cr = scored["condensed_reasoning_len"]
            orr = scored["original_reasoning_len"]
            savings = f"{(1-cr/max(orr,1))*100:.0f}%" if orr > 0 else "N/A"
            print(f"  ✅ q={q}/10, {c}, reasoning {savings} compressed, {scored['scoring_time_seconds']:.1f}s", flush=True)
        print_stats(scored_list)
        return

    # Full scoring
    if input_override:
        INPUT_DIR_VAL = input_override
    else:
        INPUT_DIR_VAL = INPUT_DIR
    print(f"{'='*60}")
    print(f"TEACHER SCORING & REASONING CONDENSATION")
    print(f"{'='*60}")
    print(f"  Teacher:  {TEACHER_URL} ({TEACHER_MODEL})")
    print(f"  Input:    {INPUT_DIR_VAL}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Limits:   reasoning={MAX_REASONING_CHARS}ch, response={MAX_RESPONSE_CHARS}ch, prompt={MAX_PROMPT_CHARS}ch")
    print()

    # Health check
    try:
        r = requests.get(f"{TEACHER_URL}/health", timeout=10)
        r.raise_for_status()
        print("  Teacher: reachable ✅")
    except Exception as e:
        print(f"  Teacher: UNREACHABLE — {e}")
        sys.exit(1)

    # Load rollouts
    print("\nLoading rollouts...", flush=True)
    groups = load_rollouts(INPUT_DIR_VAL)
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

    # Estimate time
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
                elapsed = sum(s["scoring_time_seconds"] for s in scored_list)
                remaining_count = total - i - 1
                avg_time = elapsed / len(scored_list)
                eta_hours = (remaining_count * avg_time) / 3600
                print(f"  ⏱️ ETA: {eta_hours:.1f}h remaining ({remaining_count} prompts)", flush=True)

        except Exception as e:
            print(f"  ❌ {e}", flush=True)
            error_rec = {
                "prompt_hash": ph,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(Path(OUTPUT_DIR) / "errors.jsonl", "a") as f:
                f.write(json.dumps(error_rec, default=str) + "\n")

        # Small delay between calls to avoid overwhelming the teacher
        time.sleep(0.2)

    print_stats(scored_list)
    print(f"\nOutput: {OUTPUT_DIR}/scored.jsonl")


if __name__ == "__main__":
    main()
