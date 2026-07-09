#!/usr/bin/env python3
"""
Teacher Scoring & Reasoning Condensation via an OpenAI-compatible teacher.

For each unique prompt (with N generations at different temperatures):
1. Show all generations to the teacher in a SINGLE call
2. Rank them by quality (correctness, format adherence, reasoning clarity)
3. Pick the winner
4. Condense the winner's reasoning in the same call (delete dead ends, keep voice)

Resumable via scored.jsonl (tracks prompt_hash).
Output: scored_rollouts.jsonl — one entry per prompt with best generation + condensed reasoning.

Token budget per call:
- System prompt: ~300 tokens
- Prompt + gold + schema: ~200-2000 tokens
- Per generation: reasoning + response. Tencent HY3/OpenRouter defaults are untruncated with a 262k context guard.
- 6 gens are sent in one prompt-hash group and one API call.
- Output: ~2,000 tokens
- Total median: ~5,000 tokens | p95: ~20,000 tokens | max safe: ~100,000 tokens
"""
import json, time, sys, os, requests, textwrap
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

# ─── Config ───────────────────────────────────────────────────────────────────
TEACHER_PROVIDER = os.environ.get("TEACHER_PROVIDER", "").strip().lower()
if not TEACHER_PROVIDER and os.environ.get("OPENROUTER_API_KEY"):
    TEACHER_PROVIDER = "openrouter"
TEACHER_PROVIDER = TEACHER_PROVIDER or "local"
MIXED_TEACHERS = TEACHER_PROVIDER in ("mixed", "hy3_opencode", "openrouter_opencode") or os.environ.get("MIXED_TEACHERS", "0") == "1"

DEFAULT_TEACHER_URL = "https://openrouter.ai/api/v1" if TEACHER_PROVIDER == "openrouter" else "http://100.100.61.28:8080"
DEFAULT_TEACHER_MODEL = "tencent/hy3:free" if TEACHER_PROVIDER == "openrouter" else "gemma4"

TEACHER_URL = os.environ.get("TEACHER_URL", DEFAULT_TEACHER_URL)
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", DEFAULT_TEACHER_MODEL)
TEACHER_API_KEY = os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://github.com/bilawalriaz/autonomous-small-model-exploration")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "autonomous-small-model-exploration")
INPUT_DIR = os.environ.get("INPUT_DIR", "/Users/bilawalriaz/rollouts")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/Users/bilawalriaz/scored")
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
STOP_ON_PROVIDER_ERROR = os.environ.get("STOP_ON_PROVIDER_ERROR", "1") != "0"
STOP_ON_RESPONSE_ERROR = os.environ.get("STOP_ON_RESPONSE_ERROR", "1") != "0"
ALLOW_PARSE_FALLBACK = os.environ.get("ALLOW_PARSE_FALLBACK", "0") == "1"
DEFAULT_RESPONSE_FORMAT = "json_schema" if TEACHER_PROVIDER == "openrouter" and TEACHER_MODEL == "tencent/hy3:free" else "json_object"
RESPONSE_FORMAT = os.environ.get("RESPONSE_FORMAT", DEFAULT_RESPONSE_FORMAT).strip().lower()
REQUEST_JSON_RESPONSE = os.environ.get("REQUEST_JSON_RESPONSE", "1") != "0"
SAVE_BAD_RESPONSES = os.environ.get("SAVE_BAD_RESPONSES", "1") != "0"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6" if TEACHER_PROVIDER == "openrouter" else "1"))
TEACHER_CONTEXT_TOKENS = int(os.environ.get(
    "TEACHER_CONTEXT_TOKENS",
    "262000" if TEACHER_PROVIDER == "openrouter" and TEACHER_MODEL == "tencent/hy3:free" else "100000",
))

LONG_CONTEXT_TEACHER = TEACHER_PROVIDER == "openrouter" and TEACHER_MODEL == "tencent/hy3:free"

# Truncation limits. 0 means no truncation. For HY3/OpenRouter, defaults are
# untruncated so the scorer compares full rollout generations in one request.
MAX_REASONING_CHARS = int(os.environ.get("MAX_REASONING_CHARS", "0" if LONG_CONTEXT_TEACHER else "3000"))
MAX_RESPONSE_CHARS = int(os.environ.get("MAX_RESPONSE_CHARS", "0" if LONG_CONTEXT_TEACHER else "2000"))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "0" if LONG_CONTEXT_TEACHER else "4000"))
MAX_GENERATIONS_PER_CALL = int(os.environ.get("MAX_GENERATIONS_PER_CALL", "0"))  # 0 = include all generations for the prompt
EXPECTED_GENERATIONS = int(os.environ.get("EXPECTED_GENERATIONS", "6"))

OPENCODE_URL = os.environ.get("OPENCODE_URL", "https://opencode.ai/zen/go/v1/chat/completions")
OPENCODE_MODEL = os.environ.get("OPENCODE_MODEL", "mimo-v2.5")
OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_CONTEXT_TOKENS = int(os.environ.get("OPENCODE_CONTEXT_TOKENS", "100000"))
OPENCODE_RESPONSE_FORMAT = os.environ.get("OPENCODE_RESPONSE_FORMAT", "none").strip().lower()

SCORING_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
        },
        "winner_index": {"type": "integer"},
        "quality_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "correctness": {"type": "string", "enum": ["correct", "incorrect", "partial", "na"]},
        "format_valid": {"type": "boolean"},
        "winner_reason": {"type": "string"},
        "format_violations": {"type": "string"},
        "condensed_reasoning": {"type": "string"},
    },
    "required": [
        "ranking",
        "winner_index",
        "quality_score",
        "correctness",
        "format_valid",
        "winner_reason",
        "format_violations",
        "condensed_reasoning",
    ],
}


class ProviderExhausted(RuntimeError):
    """Raised when a provider cannot continue because of auth, credits, or limits."""


class TeacherResponseError(RuntimeError):
    """Raised when the teacher response cannot be used as a durable label."""

    def __init__(self, message, raw_response=None, response_text=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.response_text = response_text


# ─── Prompts ──────────────────────────────────────────────────────────────────

RANKING_PROMPT = textwrap.dedent("""\
You are a strict validation-first training data judge for a language model distillation pipeline.

A small model generated {n_gens} responses to the same prompt at different temperatures.
Your job is to:
1. Validate every generation against the user's requested format, schema, constraints, and answer.
2. Rank all generations by training-data quality.
3. Pick the best valid generation. If none are valid, pick the least-bad generation but mark it invalid/partial.
4. Edit only the winner's thinking chain to be shorter.

### Mandatory validation pass
Before ranking, inspect each generation for hard failures. A generation has a hard failure if any of these are true:
  - If the prompt says "Respond in JSON" → output MUST be valid JSON, not YAML or prose
  - If the prompt says "Provide the output in YAML" → output MUST be valid YAML, not JSON
  - If the prompt says "Output the raw JSON directly without code block fencing" → no ```json fences allowed
  - If the prompt requires a markdown code fence → the response MUST include the requested fence type
  - If the prompt requires a bare array/list at top level → the response MUST NOT wrap it in an object
  - If the prompt requires an object with a specific top-level key → the response MUST use that key
  - If the prompt specifies a schema → the output MUST match that schema's structure
  - If the prompt says "Just the JSON, no commentary" → no preamble or explanation before the output
  - No extra invented fields anywhere if the prompt says exact keys/field set
  - Required fields must be present
  - Enum values must be exactly from the allowed set
  - Numeric min/max/range constraints must be satisfied
  - Count constraints must be satisfied (for example exactly 4 items, 2-3 items, min/max items)
  - Cross-field arithmetic must be internally consistent when requested
  - The final answer must match the gold answer when a gold answer is provided

### Ranking rules
- Prefer a fully valid generation over any invalid generation, even if the invalid one is more fluent.
- If multiple generations are fully valid, rank by correctness, completeness, realism, and concise reasoning.
- If no generation is fully valid, choose the least-bad generation, but:
  - set "format_valid": false if it violates output format, schema shape, field set, enum, range, or count constraints
  - set "correctness": "partial" or "incorrect" if it fails the answer, schema, or arithmetic constraints
  - cap "quality_score" at 6 for any hard failure
  - cap "quality_score" at 4 for wrong top-level format, unparseable JSON/YAML, or wrong final answer with gold available

### Score calibration
- 10: fully valid, correct, complete, clean format, no material issues
- 8-9: valid and correct with only minor style/realism weaknesses
- 6-7: usable but has minor omissions or weak reasoning; no hard format/schema failure
- 4-5: partially useful but has a hard failure or important constraint issue
- 1-3: unparseable, wrong answer, wrong top-level format, or mostly unusable

### Reasoning condensation
- Condense only the winner's reasoning.
- Delete dead ends, repeated attempts, "oh wait" moments, and verbose restating.
- Do NOT invent new reasoning, fix the answer, or rewrite in a new voice.
- Keep the model's exact words where possible; remove only unnecessary text.

### Output requirements for your JSON
- "ranking" must include every generation index exactly once.
- "winner_index" must be the first element of "ranking".
- "format_valid" is about the winner only.
- "format_violations" must explicitly mention the winner's violations if any, plus notable violations in other generations.
- "winner_reason" must mention why the winner is valid, or if none are valid, why it is the least bad.
- Do not give a high score to a generation while also saying it has a schema/format/range/count/arithmetic violation.

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
    if max_chars <= 0:
        return text
    if not text or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind('\n')
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut] + f"\n[...truncated from {len(text)} chars]"


def build_teacher_configs():
    """Return one or two provider configs for the current run."""
    if MIXED_TEACHERS:
        return [
            {
                "provider": "openrouter",
                "url": os.environ.get("OPENROUTER_TEACHER_URL", "https://openrouter.ai/api/v1"),
                "model": os.environ.get("OPENROUTER_TEACHER_MODEL", "tencent/hy3:free"),
                "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                "response_format": os.environ.get("OPENROUTER_RESPONSE_FORMAT", "json_schema").strip().lower(),
                "request_json_response": True,
                "context_tokens": int(os.environ.get("OPENROUTER_CONTEXT_TOKENS", os.environ.get("TEACHER_CONTEXT_TOKENS", "262000"))),
                "name": "openrouter-hy3",
            },
            {
                "provider": "opencode",
                "url": OPENCODE_URL,
                "model": OPENCODE_MODEL,
                "api_key": OPENCODE_API_KEY,
                "response_format": OPENCODE_RESPONSE_FORMAT,
                "request_json_response": OPENCODE_RESPONSE_FORMAT not in ("", "none", "off"),
                "context_tokens": OPENCODE_CONTEXT_TOKENS,
                "name": "opencode-go",
            },
        ]

    return [
        {
            "provider": TEACHER_PROVIDER,
            "url": TEACHER_URL,
            "model": TEACHER_MODEL,
            "api_key": TEACHER_API_KEY,
            "response_format": RESPONSE_FORMAT,
            "request_json_response": REQUEST_JSON_RESPONSE,
            "context_tokens": TEACHER_CONTEXT_TOKENS,
            "name": TEACHER_PROVIDER,
        }
    ]


TEACHER_CONFIGS = build_teacher_configs()


def api_base_url(config=None):
    """Return the OpenAI-compatible API base URL, ending at /v1."""
    config = config or TEACHER_CONFIGS[0]
    url = config["url"].rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def chat_completions_url(config=None):
    config = config or TEACHER_CONFIGS[0]
    url = config["url"].rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{api_base_url(config)}/chat/completions"


def health_base_url(config=None):
    """Return the host/root URL for local health checks."""
    config = config or TEACHER_CONFIGS[0]
    parsed = urlparse(config["url"])
    if not parsed.scheme or not parsed.netloc:
        return config["url"].rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


def request_headers(config=None):
    config = config or TEACHER_CONFIGS[0]
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"):
        headers["Authorization"] = f"Bearer {config['api_key']}"
    if config["provider"] == "openrouter":
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        headers["X-Title"] = OPENROUTER_APP_NAME
    return headers


def response_error_message(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    error = payload.get("error", payload)
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)[:500]
    return str(error)[:500]


def extract_message_text(message):
    """Extract assistant text from common OpenAI-compatible message shapes."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def save_bad_response(prompt_hash, error, raw_response=None, response_text=None, config=None):
    """Persist unusable provider responses for diagnosis without storing secrets."""
    if not SAVE_BAD_RESPONSES:
        return None
    config = config or TEACHER_CONFIGS[0]
    bad_dir = Path(OUTPUT_DIR) / "bad_teacher_responses"
    bad_dir.mkdir(parents=True, exist_ok=True)
    path = bad_dir / f"{prompt_hash}_{int(time.time())}.json"
    record = {
        "prompt_hash": prompt_hash,
        "error": str(error),
        "error_type": type(error).__name__,
        "teacher_provider": config["provider"],
        "teacher_model": config["model"],
        "teacher_api_base": api_base_url(config),
        "teacher_name": config["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw_response,
        "response_text": response_text,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def check_teacher_reachable(config=None):
    """Validate the configured teacher without assuming local /health exists."""
    config = config or TEACHER_CONFIGS[0]
    if config["provider"] == "opencode":
        if not config.get("api_key"):
            raise ProviderExhausted("opencode-go API key is missing; set OPENCODE_API_KEY")
        return "configured"
    if config["provider"] != "openrouter":
        try:
            r = requests.get(f"{health_base_url(config)}/health", timeout=10)
            r.raise_for_status()
            return "health"
        except Exception:
            pass

    r = requests.get(f"{api_base_url(config)}/models", headers=request_headers(config), timeout=20)
    if r.status_code in (401, 402, 403, 429):
        raise ProviderExhausted(f"teacher unavailable ({r.status_code}): {response_error_message(r)}")
    r.raise_for_status()
    return "models"


def response_format_payload(config=None):
    config = config or TEACHER_CONFIGS[0]
    response_format = config.get("response_format", "")
    if not config.get("request_json_response", True) or response_format in ("", "none", "off"):
        return None
    if response_format == "json_object":
        return {"type": "json_object"}
    if response_format == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "rollout_teacher_score",
                "strict": True,
                "schema": SCORING_RESPONSE_SCHEMA,
            },
        }
    raise ValueError(f"Unsupported response_format={response_format!r}; use json_schema, json_object, or none")


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


def call_teacher(prompt_text, config=None, max_retries=MAX_RETRIES):
    """Call the configured OpenAI-compatible teacher with retry logic."""
    config = config or TEACHER_CONFIGS[0]
    for attempt in range(max_retries):
        try:
            payload = {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
            fmt = response_format_payload(config)
            if fmt:
                payload["response_format"] = fmt
            r = requests.post(
                chat_completions_url(config),
                headers=request_headers(config),
                json=payload,
                timeout=TIMEOUT,
            )
            if r.status_code in (401, 402, 403, 429):
                raise ProviderExhausted(f"provider stopped ({r.status_code}): {response_error_message(r)}")
            if r.status_code in (400, 404, 422):
                try:
                    raw = r.json()
                except ValueError:
                    raw = {"text": r.text[:2000]}
                raise TeacherResponseError(
                    f"provider rejected request ({r.status_code}): {response_error_message(r)}",
                    raw_response=raw,
                    response_text=r.text[:2000],
                )
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            msg = choice["message"]
            content = extract_message_text(msg)
            reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
            finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
            if not content.strip():
                reason = f"teacher returned empty content; finish_reason={finish_reason}, usage={data.get('usage')}"
                if reasoning:
                    reason += ", reasoning was present without final JSON content"
                raise TeacherResponseError(reason, raw_response=data, response_text=content)
            return content, data
        except ProviderExhausted:
            raise
        except TeacherResponseError:
            raise
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
    """Format one prompt's generations for a single ranking call."""
    parts = []
    selected_gens = gens if MAX_GENERATIONS_PER_CALL <= 0 else gens[:MAX_GENERATIONS_PER_CALL]
    for i, g in enumerate(selected_gens):
        reasoning = truncate(g.get("reasoning", ""), MAX_REASONING_CHARS)
        response = truncate(g.get("response", ""), MAX_RESPONSE_CHARS)
        temp = g.get("temperature", "?")
        parts.append(
            f"--- Generation {i} (temp={temp}) ---\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Response:\n{response}\n"
        )
    return "\n".join(parts)


def score_prompt(ph, group, config=None):
    """Score a single prompt: rank generations + condense reasoning, all in one call."""
    config = config or TEACHER_CONFIGS[0]
    meta = group["prompt_meta"]
    prompt = truncate(group["prompt"], MAX_PROMPT_CHARS)
    gens = group["generations"]
    if len(gens) != EXPECTED_GENERATIONS:
        print(f"    warning: expected {EXPECTED_GENERATIONS} generations, found {len(gens)}", flush=True)

    gold_answer = meta.get("gold_answer") or "N/A"
    dataset = meta.get("_dataset") or meta.get("category") or "unknown"

    # Build the ranking prompt
    print(f"    building prompt...", end="", flush=True)
    gen_block = format_generations(gens)
    gen_count_for_call = len(gens) if MAX_GENERATIONS_PER_CALL <= 0 else min(len(gens), MAX_GENERATIONS_PER_CALL)
    print(f" {len(gen_block):,}ch gen block from {gen_count_for_call}/{len(gens)} generations", end="", flush=True)
    full_prompt = RANKING_PROMPT.format(
        n_gens=len(gens),
        prompt=prompt,
        gold_answer=gold_answer,
        generations_block=gen_block,
    )
    print(f" → {len(full_prompt):,}ch total", end="", flush=True)

    # Single teacher call: rank + condense in one shot
    est_tokens = len(full_prompt) // 4
    if est_tokens > config["context_tokens"]:
        raise TeacherResponseError(
            f"estimated prompt size {est_tokens:,} tokens exceeds configured context {config['context_tokens']:,}; "
            "raise truncation only if the provider supports it, otherwise reduce per-generation limits"
        )
    print(f" (~{est_tokens:,} tok) → calling {config['name']} {api_base_url(config)}...", flush=True)
    t0 = time.monotonic()
    response_text, raw_response = call_teacher(full_prompt, config)
    elapsed = time.monotonic() - t0
    print(f"    ✅ {elapsed:.1f}s, {len(response_text):,} chars back", flush=True)

    # Parse
    result = parse_teacher_response(response_text)
    if result is None:
        if not ALLOW_PARSE_FALLBACK:
            raise TeacherResponseError(
                "teacher response was not parseable JSON; not marking prompt as scored",
                raw_response=raw_response,
                response_text=response_text,
            )
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
        "n_generations_sent_to_teacher": gen_count_for_call,
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
        "teacher_provider": config["provider"],
        "teacher_model": config["model"],
        "teacher_api_base": api_base_url(config),
        "teacher_name": config["name"],
        "teacher_context_tokens": config["context_tokens"],
        "scoring_time_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return scored


def save_scored(scored, output_dir):
    """Append a scored record to the output file."""
    with open(Path(output_dir) / "scored.jsonl", "a") as f:
        f.write(json.dumps(scored, default=str) + "\n")
        f.flush()


def save_error(prompt_hash, error, error_type=None, config=None):
    config = config or TEACHER_CONFIGS[0]
    error_rec = {
        "prompt_hash": prompt_hash,
        "error": str(error),
        "error_type": error_type or type(error).__name__,
        "teacher_provider": config["provider"],
        "teacher_model": config["model"],
        "teacher_api_base": api_base_url(config),
        "teacher_name": config["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(Path(OUTPUT_DIR) / "errors.jsonl", "a") as f:
        f.write(json.dumps(error_rec, default=str) + "\n")
        f.flush()


def log_scored(scored):
    q = scored["quality_score"]
    c = scored["correctness"]
    cr = scored["condensed_reasoning_len"]
    orr = scored["original_reasoning_len"]
    savings = f"{(1-cr/max(orr,1))*100:.0f}%" if orr > 0 else "N/A"
    teacher = scored.get("teacher_name") or scored.get("teacher_provider", "?")
    print(f"  ✅ {teacher} q={q}/10, {c}, reasoning {savings} compressed, {scored['scoring_time_seconds']:.1f}s", flush=True)


def score_indexed(item):
    idx, total, ph, group, config = item
    ds = group["prompt_meta"].get("_dataset", "?")
    n = len(group["generations"])
    print(f"\n[{idx}/{total}] [{config['name']}] [{ds}] {n} gens — {group['prompt'][:60]}...", flush=True)
    scored = score_prompt(ph, group, config)
    return idx, ph, scored


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


def provider_worker_plan():
    """Map provider names to prompt-group worker slots."""
    if len(TEACHER_CONFIGS) == 1:
        return [(TEACHER_CONFIGS[0], max(1, MAX_WORKERS))]
    first = MAX_WORKERS // 2
    second = MAX_WORKERS - first
    return [(TEACHER_CONFIGS[0], max(1, first)), (TEACHER_CONFIGS[1], max(1, second))]


def assign_provider_configs(batch):
    """Assign configs to one batch according to the worker split."""
    plan = provider_worker_plan()
    assigned = []
    batch_pos = 0
    while batch_pos < len(batch):
        for config, slots in plan:
            for _ in range(slots):
                if batch_pos >= len(batch):
                    break
                idx, total, ph, group = batch[batch_pos]
                assigned.append((idx, total, ph, group, config))
                batch_pos += 1
            if batch_pos >= len(batch):
                break
    return assigned


def provider_config_for_item_number(item_number):
    """Return provider config for a 0-based item position using the worker split."""
    plan = provider_worker_plan()
    cycle = sum(slots for _, slots in plan)
    slot = item_number % max(1, cycle)
    offset = 0
    for config, slots in plan:
        if slot < offset + slots:
            return config
        offset += slots
    return plan[-1][0]


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
        scored = score_prompt(ph, groups[ph], TEACHER_CONFIGS[0])
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
            config = TEACHER_CONFIGS[i % len(TEACHER_CONFIGS)]
            scored = score_prompt(ph, group, config)
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
    print(f"  Provider: {'mixed' if MIXED_TEACHERS else TEACHER_PROVIDER}")
    for config, slots in provider_worker_plan():
        print(f"  Teacher:  {config['name']} × {slots} workers → {api_base_url(config)} ({config['model']})")
    print(f"  Input:    {INPUT_DIR_VAL}")
    print(f"  Output:   {OUTPUT_DIR}")
    context_summary = ", ".join(f"{c['name']}={c['context_tokens']:,}" for c in TEACHER_CONFIGS)
    json_summary = ", ".join(
        f"{c['name']}={c['response_format'] if c.get('request_json_response') else 'off'}"
        for c in TEACHER_CONFIGS
    )
    print(f"  Context:  {context_summary} tokens configured")
    print(f"  Limits:   reasoning={MAX_REASONING_CHARS}ch, response={MAX_RESPONSE_CHARS}ch, prompt={MAX_PROMPT_CHARS}ch, max_tokens={MAX_TOKENS}")
    print(f"  Calls:    one API call per prompt_hash; generations per call={'all' if MAX_GENERATIONS_PER_CALL <= 0 else MAX_GENERATIONS_PER_CALL}")
    print(f"  Workers:  {MAX_WORKERS}")
    print(f"  JSON:     {json_summary}")
    print(f"  Resume:   completed prompt_hashes in scored.jsonl are skipped; failed prompts stay pending")
    print()

    # Health check
    for config, _ in provider_worker_plan():
        try:
            check = check_teacher_reachable(config)
            print(f"  Teacher {config['name']}: reachable via /{check} ✅")
        except ProviderExhausted as e:
            print(f"  Teacher {config['name']}: STOPPED — {e}")
            sys.exit(1)
        except Exception as e:
            print(f"  Teacher {config['name']}: UNREACHABLE — {e}")
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

    scored_list = []
    total = len(remaining)
    items = [(i, total, ph, group) for i, (ph, group) in enumerate(remaining.items(), start=1)]

    stop = False
    next_item = 0
    active = {}

    def submit_next(executor):
        nonlocal next_item
        if next_item >= len(items):
            return False
        idx, total, ph, group = items[next_item]
        assigned = (idx, total, ph, group, provider_config_for_item_number(next_item))
        future = executor.submit(score_indexed, assigned)
        active[future] = assigned
        next_item += 1
        return True

    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        for _ in range(min(max(1, MAX_WORKERS), len(items))):
            submit_next(executor)

        while active:
            done_futures, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done_futures:
                idx, _, ph, _, config = active.pop(future)
                try:
                    _, _, scored = future.result()
                    save_scored(scored, OUTPUT_DIR)
                    scored_list.append(scored)
                    log_scored(scored)

                    if len(scored_list) % 50 == 0:
                        print_stats(scored_list)
                        remaining_count = total - idx
                        avg_time = sum(s["scoring_time_seconds"] for s in scored_list) / len(scored_list)
                        eta_hours = (remaining_count * avg_time / max(1, MAX_WORKERS)) / 3600
                        print(f"  ETA: {eta_hours:.1f}h remaining ({remaining_count} prompts, {MAX_WORKERS} workers)", flush=True)

                except ProviderExhausted as e:
                    print(f"  Provider stopped: {e}", flush=True)
                    save_error(ph, e, "provider_exhausted", config)
                    if STOP_ON_PROVIDER_ERROR:
                        print("  Stopping after current in-flight requests; scored prompts were saved, failed prompt remains pending.", flush=True)
                        stop = True
                except TeacherResponseError as e:
                    print(f"  Teacher response error: {e}", flush=True)
                    save_error(ph, e, config=config)
                    bad_path = save_bad_response(ph, e, e.raw_response, e.response_text, config)
                    if bad_path:
                        print(f"  Saved bad teacher response: {bad_path}", flush=True)
                    if STOP_ON_RESPONSE_ERROR:
                        print("  Stopping after current in-flight requests so response handling can be fixed before more calls are spent.", flush=True)
                        stop = True
                except Exception as e:
                    print(f"  ❌ {e}", flush=True)
                    save_error(ph, e, config=config)

                if not stop:
                    submit_next(executor)

            if stop and not active:
                break

    print_stats(scored_list)
    print(f"\nOutput: {OUTPUT_DIR}/scored.jsonl")


if __name__ == "__main__":
    main()
