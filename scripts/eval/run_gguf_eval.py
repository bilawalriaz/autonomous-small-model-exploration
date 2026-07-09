#!/usr/bin/env python3
"""Deterministically evaluate a GGUF model with an exact HF chat template.

The evaluator deliberately asks llama.cpp to *not* display the prompt.  It
stores raw stdout and extracts only the generated completion at explicit
MiniCPM stop markers.  Run ``--validate-fixtures`` before using it for a new
model family or llama.cpp build.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_gguf_eval")

# The current MiniCPM GGUF tokenizer renders its EOS as ``[end of text]`` on
# stdout, while HF exposes it as ``</s>``.  Stop on both representations.
MINICPM_STOP_MARKERS = ("<|im_end|>", "</s>", "[end of text]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-gguf", help="Path to the GGUF model file on the execution host")
    parser.add_argument("--run-id", help="Run identifier for output files")
    parser.add_argument("--eval-set", default="data/eval/small_model_eval_v1.jsonl")
    parser.add_argument("--tokenizer-name", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--llama-path", default="/home/billz/llama.cpp/build/bin/llama-completion")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop", action="append", default=None,
                        help="Stop marker; repeat to add markers (default: MiniCPM im_end and EOS).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-fixtures", action="store_true",
                        help="Validate output extraction fixtures and exit without model inference.")
    parser.add_argument("--fixtures", default="tests/fixtures/minicpm5_gguf_output_extraction.json")
    args = parser.parse_args()
    if not args.validate_fixtures and (not args.model_gguf or not args.run_id):
        parser.error("--model-gguf and --run-id are required unless --validate-fixtures is used")
    if args.temperature != 0.0:
        parser.error("This paired evaluator is deterministic: --temperature must be 0.0")
    if args.top_p != 1.0 or args.top_k != 0 or args.repetition_penalty != 1.0:
        parser.error("This paired evaluator requires --top-p 1.0, --top-k 0, and --repetition-penalty 1.0")
    return args


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(record: dict, tokenizer) -> str:
    """Render exactly the tokenizer-provided template; never approximate it."""
    messages = record.get("messages") or record.get("prompt_messages")
    if messages:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Tokenizer has no chat template; refusing to substitute a generic template")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = record.get("prompt") or record.get("instruction")
    if not prompt:
        raise ValueError(f"Record has neither messages nor prompt: {record!r}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def extract_generated_text(stdout: str, prompt: str, stop_markers: tuple[str, ...]) -> tuple[str, str | None, bool]:
    """Return completion, stop marker, and whether an unexpected prompt echo was removed.

    Prompt stripping remains only a defensive compatibility path for historical
    llama.cpp builds; normal calls use ``--no-display-prompt`` and therefore
    never exercise it.  Stops are cut at the *first* marker occurrence.
    """
    text = stdout.replace("\r\n", "\n")
    normalized_prompt = prompt.replace("\r\n", "\n")
    prompt_echo_removed = text.startswith(normalized_prompt)
    if prompt_echo_removed:
        text = text[len(normalized_prompt):]
    positions = [(text.find(marker), marker) for marker in stop_markers if text.find(marker) >= 0]
    stop_marker = None
    if positions:
        position, stop_marker = min(positions, key=lambda item: item[0])
        text = text[:position]
    return text.strip(), stop_marker, prompt_echo_removed


def validate_fixtures(path: Path) -> bool:
    fixtures = json.loads(path.read_text())
    failures = []
    for fixture in fixtures:
        output, stop, _ = extract_generated_text(
            fixture["stdout"], fixture["prompt"], MINICPM_STOP_MARKERS
        )
        if output != fixture["expected"] or stop != fixture["expected_stop"]:
            failures.append({"name": fixture["name"], "actual": output, "stop": stop})
    if failures:
        log.error("Output extraction fixtures failed: %s", json.dumps(failures))
        return False
    log.info("Output extraction fixtures passed: %d/%d", len(fixtures), len(fixtures))
    return True


def run_completion(args: argparse.Namespace, prompt: str, stop_markers: tuple[str, ...]) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = handle.name
    cmd = [
        args.llama_path, "-m", args.model_gguf, "-f", prompt_path,
        "-n", str(args.max_new_tokens), "--temp", "0", "--top-p", "1",
        "--top-k", "0", "--repeat-penalty", "1", "--seed", str(args.seed),
        "-ngl", str(args.gpu_layers), "-no-cnv", "--no-display-prompt",
        "--no-perf", "--override-kv", "tokenizer.ggml.add_bos_token=bool:false",
    ]
    for marker in stop_markers:
        cmd.extend(["-r", marker])
    try:
        start = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        elapsed = time.time() - start
    finally:
        os.unlink(prompt_path)
    if result.returncode:
        raise RuntimeError(f"llama.cpp failed ({result.returncode}): {result.stderr.strip()}")
    generated, stop_marker, echoed = extract_generated_text(result.stdout, prompt, stop_markers)
    return {
        "generated_response": generated,
        "raw_stdout": result.stdout,
        "stop_marker": stop_marker,
        "prompt_echo_removed": echoed,
        "generation_time": round(elapsed, 3),
        "returncode": result.returncode,
        "command": ["<prompt-file>" if item == prompt_path else item for item in cmd],
    }


def main() -> int:
    args = parse_args()
    fixtures_path = Path(args.fixtures)
    if not validate_fixtures(fixtures_path):
        return 1
    if args.validate_fixtures:
        return 0

    eval_set_path = Path(args.eval_set).resolve()
    records = load_jsonl(eval_set_path)
    if args.limit:
        records = records[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    template = tokenizer.chat_template
    template_sha256 = hashlib.sha256(template.encode()).hexdigest()
    stop_markers = tuple(args.stop or MINICPM_STOP_MARKERS)
    output_dir = Path("results/evals") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, record in enumerate(records):
        prompt = build_prompt(record, tokenizer)
        completion = run_completion(args, prompt, stop_markers)
        completion["tokens_generated"] = len(tokenizer.encode(completion["generated_response"], add_special_tokens=False))
        completion.update({
            "eval_id": record.get("eval_id", record.get("id", f"eval_{index:04d}")),
            "category": record.get("category", "unknown"),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "template_residue_detected": any(marker in completion["generated_response"] for marker in MINICPM_STOP_MARKERS),
        })
        results.append(completion)
        if (index + 1) % 10 == 0 or index + 1 == len(records):
            log.info("Processed %d/%d prompts", index + 1, len(records))

    outputs_path = output_dir / "outputs.jsonl"
    with outputs_path.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    metadata = {
        "run_id": args.run_id, "model_gguf": args.model_gguf,
        "eval_set": str(eval_set_path), "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_count": len(results), "tokenizer_name": args.tokenizer_name,
        "chat_template_sha256": template_sha256, "stop_markers": stop_markers,
        "config": {"temperature": 0.0, "top_p": 1.0, "top_k": 0,
                   "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0,
                   "seed": args.seed, "no_display_prompt": True},
        "fixture_validation": {"path": str(fixtures_path), "passed": True},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Saved %d deterministic outputs to %s", len(results), outputs_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
