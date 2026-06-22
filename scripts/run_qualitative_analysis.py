"""run_qualitative_analysis.py — Qualitative LLM analysis using the prompt library.

Tests the "vibe" of a model across dimensions:
- Prose quality (coherence, fluency, style)
- Creativity (divergent thinking, originality)
- Correctness (factual, logical, code)
- Malleability (instruction following, constraint adherence, format compliance)
- Inference speed (tokens/sec)

Uses prompts from the mi-prompt-library submodule.
Works with both HF Transformers and llama.cpp (GGUF) backends.

Usage:
    # HF native
    python scripts/run_qualitative_analysis.py --model Qwen/Qwen2.5-0.5B --suffix 0.5b --backend hf

    # llama.cpp GGUF
    python scripts/run_qualitative_analysis.py --model /path/to/model.gguf --suffix 0.5b_q8 --backend llama
"""
import sys
import json
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts" / "prompts"


def load_all_prompts():
    """Load all prompts from the prompt library."""
    all_prompts = []
    for prompt_file in sorted(PROMPTS_DIR.glob("*.json")):
        with open(prompt_file) as f:
            prompts = json.load(f)
            all_prompts.extend(prompts)
    return all_prompts


# ============ HF BACKEND ============

def run_hf_inference(model, tokenizer, prompt, device, max_new_tokens=200):
    """Run inference with HF Transformers and measure speed."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    # Timed generation
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)
    tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0

    return {
        "text": generated_text,
        "n_tokens": n_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
    }


# ============ LLAMA.CPP BACKEND ============

def run_llama_inference(model_path, prompt, max_new_tokens=200, n_gpu_layers=99):
    """Run inference with llama.cpp and measure speed."""
    llama_cli = Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli"

    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_new_tokens),
        "--temp", "0",
        "--no-display-prompt",
        "-ngl", str(n_gpu_layers),
        "--simple-io",
        "--no-conversation",
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start

    # llama-cli outputs the generated text after the prompt
    output_text = result.stdout.strip()

    # Estimate tokens (rough: chars/4 for English)
    n_tokens = len(output_text) // 4
    tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0

    return {
        "text": output_text,
        "n_tokens": n_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
    }


# ============ QUALITATIVE SCORING ============

def score_response(prompt_obj, response_text, tokens_per_sec):
    """Generate automated scores for a response based on simple heuristics.

    Note: These are automated heuristic scores, not human evaluations.
    They capture structural properties (length, constraint adherence) but
    not nuanced quality. Human review is needed for final scoring.
    """
    text = response_text.strip()
    scores = {}

    # Constraint adherence checks
    constraints = prompt_obj.get("constraints", [])
    constraint_checks = []

    for c in constraints:
        c_lower = c.lower()
        if "exactly 3 sentences" in c_lower or "exactly 3 bullet" in c_lower:
            # Count sentences
            sentences = [s for s in text.replace("\n", " ").split(".") if s.strip()]
            n = len(sentences)
            constraint_checks.append({"constraint": c, "met": n == 3, "value": n, "expected": 3})

        elif "exactly 4 lines" in c_lower:
            lines = [l for l in text.split("\n") if l.strip()]
            n = len(lines)
            constraint_checks.append({"constraint": c, "met": n == 4, "value": n, "expected": 4})

        elif "exactly 5" in c_lower:
            items = [l for l in text.split("\n") if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-") or l.strip().startswith("*"))]
            n = len(items)
            constraint_checks.append({"constraint": c, "met": n == 5, "value": n, "expected": 5})

        elif "json only" in c_lower or "output only the json" in c_lower:
            is_json = text.startswith("{") and text.endswith("}") and "\n\n" not in text[:10]
            constraint_checks.append({"constraint": c, "met": is_json, "value": is_json})

        elif "no other text" in c_lower or "json only" in c_lower:
            has_extra = any(marker in text for marker in ["```", "Here is", "Here's", "Sure,", "The answer"])
            constraint_checks.append({"constraint": c, "met": not has_extra, "value": not has_extra})

        elif "280 chars" in c_lower or "280 characters" in c_lower:
            # Find the tweet section
            constraint_checks.append({"constraint": c, "met": len(text) <= 500, "value": len(text)})  # Rough check

    # Basic quality metrics
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = max(1, len([s for s in text.replace("\n", " ").split(".") if s.strip()]))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Repetition detection (simple n-gram check)
    words = text.lower().split()
    if len(words) > 10:
        bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams))
        repetition_ratio = 1 - (unique_bigrams / len(bigrams))
    else:
        repetition_ratio = 0

    # Empty/garbage detection
    is_empty = len(text.strip()) == 0
    is_repetitive = repetition_ratio > 0.3
    is_garbage = any(ord(c) > 0x4E00 for c in text[:20]) and not any(c.isalpha() and ord(c) < 0x200 for c in text[:20])

    scores = {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "repetition_ratio": round(repetition_ratio, 4),
        "is_empty": is_empty,
        "is_repetitive": is_repetitive,
        "is_garbage": is_garbage,
        "tokens_per_second": tokens_per_sec,
        "constraint_checks": constraint_checks,
        "constraints_met": sum(1 for c in constraint_checks if c.get("met", False)),
        "constraints_total": len(constraint_checks),
        "constraint_adherence_rate": round(sum(1 for c in constraint_checks if c.get("met", False)) / max(len(constraint_checks), 1), 4),
    }

    return scores


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Run qualitative analysis on a model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix for output files")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "llama"], help="Inference backend")
    parser.add_argument("--quant", type=str, default="none", choices=["none", "8bit", "4bit"], help="Quantization level (HF backend only)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens per prompt")
    args = parser.parse_args()

    set_seed(42)

    print("=" * 60)
    print(f"  QUALITATIVE ANALYSIS")
    print(f"  Model: {args.model}")
    print(f"  Suffix: {args.suffix}")
    print(f"  Backend: {args.backend}")
    print(f"  Quantization: {args.quant}")
    print("=" * 60)

    # Load prompts
    all_prompts = load_all_prompts()
    print(f"  Loaded {len(all_prompts)} prompts from prompt library")

    # Group by category
    categories = {}
    for p in all_prompts:
        cat = p["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(p)

    print(f"  Categories: {list(categories.keys())}")

    # Initialize backend
    model = None
    tokenizer = None
    device = None

    if args.backend == "hf":
        from mi_atlas.model_loader import load_model_hf
        dtype = torch.bfloat16
        if args.quant == "8bit":
            dtype = torch.float16  # bitsandbytes needs fp16
        elif args.quant == "4bit":
            dtype = torch.float16
        bundle = load_model_hf(args.model, dtype=dtype)
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = bundle.device
        model.eval()
        arch = bundle.architecture

        # Apply bitsandbytes quantization if requested
        if args.quant in ("8bit", "4bit"):
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            del model
            del bundle
            torch.cuda.empty_cache()
            quant_config_kwargs = {
                "load_in_8bit": (args.quant == "8bit"),
                "load_in_4bit": (args.quant == "4bit"),
                "bnb_4bit_compute_dtype": torch.float16,
            }
            if args.quant == "4bit":
                quant_config_kwargs["bnb_4bit_quant_type"] = "nf4"
            quant_config = BitsAndBytesConfig(**quant_config_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=quant_config, device_map="auto"
            )
            model.eval()
            print(f"  Loaded with {args.quant} quantization")

        print(f"  Architecture: {arch['n_layers']}L, {arch['n_heads']}H, d={arch['d_model']}")
    else:
        # llama.cpp — model is a path to GGUF file
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"  ERROR: Model file not found: {model_path}")
            return
        arch = {"model_path": str(model_path), "file_size_mb": round(model_path.stat().st_size / 1e6, 2)}
        print(f"  GGUF file: {arch['model_path']} ({arch['file_size_mb']}MB)")

    # Run all prompts
    results = []
    all_speeds = []

    for prompt_obj in all_prompts:
        prompt_id = prompt_obj["id"]
        category = prompt_obj["category"]
        prompt_text = prompt_obj["prompt"]

        print(f"\n  [{prompt_id}] ({category}) ", end="", flush=True)

        try:
            if args.backend == "hf":
                result = run_hf_inference(model, tokenizer, prompt_text, device, args.max_tokens)
            else:
                result = run_llama_inference(model_path, prompt_text, args.max_tokens)

            scores = score_response(prompt_obj, result["text"], result["tokens_per_second"])

            entry = {
                "id": prompt_id,
                "category": category,
                "prompt": prompt_text,
                "constraints": prompt_obj.get("constraints", []),
                "response": result["text"],
                "n_tokens": result["n_tokens"],
                "elapsed_seconds": result["elapsed_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "scores": scores,
            }

            results.append(entry)
            all_speeds.append(result["tokens_per_second"])

            status = "OK"
            if scores["is_empty"]:
                status = "EMPTY"
            elif scores["is_garbage"]:
                status = "GARBAGE"
            elif scores["is_repetitive"]:
                status = "REPETITIVE"

            print(f"→ {status} ({result['tokens_per_second']:.1f} tok/s, {result['n_tokens']} tok, {scores['constraint_adherence_rate']:.0%} constraints)")

        except Exception as e:
            print(f"→ ERROR: {e}")
            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt_text,
                "response": f"ERROR: {str(e)}",
                "n_tokens": 0,
                "elapsed_seconds": 0,
                "tokens_per_second": 0,
                "scores": {"is_empty": True, "is_garbage": False, "constraint_adherence_rate": 0},
            })

    # Compute summary statistics
    category_summary = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        category_summary[cat] = {
            "n_prompts": len(cat_results),
            "mean_tokens_per_sec": round(np.mean([r["tokens_per_second"] for r in cat_results]), 2),
            "mean_word_count": round(np.mean([r["scores"]["word_count"] for r in cat_results if "word_count" in r.get("scores", {})]), 2),
            "mean_constraint_adherence": round(np.mean([r["scores"].get("constraint_adherence_rate", 0) for r in cat_results]), 4),
            "empty_count": sum(1 for r in cat_results if r["scores"].get("is_empty", False)),
            "garbage_count": sum(1 for r in cat_results if r["scores"].get("is_garbage", False)),
            "repetitive_count": sum(1 for r in cat_results if r["scores"].get("is_repetitive", False)),
        }

    overall_summary = {
        "model": args.model,
        "backend": args.backend,
        "n_prompts": len(results),
        "mean_tokens_per_sec": round(np.mean(all_speeds), 2) if all_speeds else 0,
        "std_tokens_per_sec": round(np.std(all_speeds), 2) if all_speeds else 0,
        "total_empty": sum(1 for r in results if r["scores"].get("is_empty", False)),
        "total_garbage": sum(1 for r in results if r["scores"].get("is_garbage", False)),
        "total_repetitive": sum(1 for r in results if r["scores"].get("is_repetitive", False)),
        "mean_constraint_adherence": round(np.mean([r["scores"].get("constraint_adherence_rate", 0) for r in results]), 4),
        "mean_repetition_ratio": round(np.mean([r["scores"].get("repetition_ratio", 0) for r in results]), 4),
        "category_summary": category_summary,
        "architecture": arch if args.backend == "hf" else None,
    }

    output = {
        "experiment": "qualitative_analysis",
        "model": args.model,
        "backend": args.backend,
        "suffix": args.suffix,
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(results),
        "results": results,
        "summary": overall_summary,
    }

    output_path = PROJECT_ROOT / "experiments" / "results" / f"qualitative_{args.suffix}.json"
    save_json(output, output_path)
    print(f"\n  Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  QUALITATIVE ANALYSIS COMPLETE: {args.model}")
    print(f"{'='*60}")
    print(f"  Mean speed: {overall_summary['mean_tokens_per_sec']:.1f} tok/s")
    print(f"  Constraint adherence: {overall_summary['mean_constraint_adherence']:.1%}")
    print(f"  Empty/garbage/repetitive: {overall_summary['total_empty']}/{overall_summary['total_garbage']}/{overall_summary['total_repetitive']}")
    print(f"  Categories:")
    for cat, stats in category_summary.items():
        print(f"    {cat}: {stats['mean_tokens_per_sec']:.1f} tok/s, {stats['mean_constraint_adherence']:.0%} constraints, {stats['empty_count']} empty, {stats['repetitive_count']} repetitive")


if __name__ == "__main__":
    main()
