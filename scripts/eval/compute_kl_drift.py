#!/usr/bin/env python3
"""Compute KL drift between base model and adapter (full or proxy mode).

CLI:
    # Proxy mode (cheap, default)
    python scripts/eval/compute_kl_drift.py \
        --base-model LiquidAI/LFM2.5-230M \
        --adapter adapters/lfm2_230m_quality_format_ablation_multi_turn_concise \
        --prompts data/eval/kl_drift_prompts.jsonl \
        --run-id lfm2_230m_format_ablation_multi_turn_concise

    # Full KL (expensive)
    python scripts/eval/compute_kl_drift.py \
        --base-model LiquidAI/LFM2.5-230M \
        --adapter adapters/lfm2_230m_quality_format_ablation_multi_turn_concise \
        --prompts data/eval/kl_drift_prompts.jsonl \
        --run-id lfm2_230m_format_ablation_multi_turn_concise \
        --mode full
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

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


def load_model_and_tokenizer(model_name: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_adapter_model(model_name: str, adapter_path: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.2, top_p=0.9, do_sample=True,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def compute_token_kl(logits_base: torch.Tensor, logits_adapter: torch.Tensor) -> list[float]:
    """Compute per-token KL divergence between base and adapter logits."""
    # logits shape: (1, seq_len, vocab_size)
    log_probs_base = torch.log_softmax(logits_base.float(), dim=-1)
    log_probs_adapter = torch.log_softmax(logits_adapter.float(), dim=-1)
    probs_base = torch.exp(log_probs_base)

    # KL(P_base || P_adapter) = sum P_base * (log P_base - log P_adapter)
    kl = (probs_base * (log_probs_base - log_probs_adapter)).sum(dim=-1)
    return kl[0].cpu().tolist()


def compute_full_kl(base_model, base_tok, adapter_model, adapter_tok, prompt: str, max_tokens: int = 256) -> dict:
    """Compute full token-level KL divergence."""
    inputs = base_tok(prompt, return_tensors="pt").to(base_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        base_out = base_model(**inputs)
        adapter_out = adapter_model(**inputs)

    # Only compare logits on the input tokens (first pass)
    base_logits = base_out.logits[:, :-1, :]  # shift for next-token prediction
    adapter_logits = adapter_out.logits[:, :-1, :]

    kl_values = compute_token_kl(base_logits, adapter_logits)

    # Also generate text from both for proxy metrics
    base_text = generate_text(base_model, base_tok, prompt, max_tokens)
    adapter_text = generate_text(adapter_model, adapter_tok, prompt, max_tokens)

    return {
        "avg_kl": round(sum(kl_values) / len(kl_values), 6) if kl_values else 0,
        "median_kl": round(sorted(kl_values)[len(kl_values) // 2], 6) if kl_values else 0,
        "max_kl": round(max(kl_values), 6) if kl_values else 0,
        "token_count": len(kl_values),
        "base_length": len(base_text),
        "adapter_length": len(adapter_text),
        "length_drift": len(adapter_text) - len(base_text),
    }


def compute_proxy_kl(base_model, base_tok, adapter_model, adapter_tok, prompt: str, max_tokens: int = 256) -> dict:
    """Cheap proxy: compare generated text properties."""
    base_text = generate_text(base_model, base_tok, prompt, max_tokens)
    adapter_text = generate_text(adapter_model, adapter_tok, prompt, max_tokens)

    base_slop = sum(1 for p in SLOP_PHRASES if p in base_text.lower())
    adapter_slop = sum(1 for p in SLOP_PHRASES if p in adapter_text.lower())

    # Repetition rate: fraction of repeated n-grams
    def repetition_rate(text: str, n: int = 3) -> float:
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        seen = {}
        repeats = 0
        for ng in ngrams:
            seen[ng] = seen.get(ng, 0) + 1
            if seen[ng] > 1:
                repeats += 1
        return round(repeats / len(ngrams), 4)

    return {
        "base_length": len(base_text),
        "adapter_length": len(adapter_text),
        "length_drift": len(adapter_text) - len(base_text),
        "length_drift_ratio": round(len(adapter_text) / max(len(base_text), 1), 3),
        "base_slop_count": base_slop,
        "adapter_slop_count": adapter_slop,
        "slop_drift": adapter_slop - base_slop,
        "base_repetition_rate": repetition_rate(base_text),
        "adapter_repetition_rate": repetition_rate(adapter_text),
        "repetition_drift": round(repetition_rate(adapter_text) - repetition_rate(base_text), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute KL drift between base model and adapter")
    parser.add_argument("--base-model", required=True, help="Base model name/path")
    parser.add_argument("--adapter", required=True, help="Adapter path")
    parser.add_argument("--prompts", required=True, help="Prompts JSONL for drift computation")
    parser.add_argument("--run-id", required=True, help="Run identifier for output")
    parser.add_argument("--mode", choices=["proxy", "full"], default="proxy", help="Drift computation mode")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    prompts_path = PROJECT_ROOT / args.prompts
    if not prompts_path.exists():
        log.error(f"Prompts file not found: {prompts_path}")
        sys.exit(1)

    prompts = load_jsonl(prompts_path)
    log.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    adapter_path = str(PROJECT_ROOT / args.adapter)
    if not Path(adapter_path).exists():
        log.error(f"Adapter path not found: {adapter_path}")
        sys.exit(1)

    log.info(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_model_and_tokenizer(args.base_model)

    log.info(f"Loading adapter model from: {adapter_path}")
    adapter_model, adapter_tok = load_adapter_model(args.base_model, adapter_path)

    results = []
    by_category = defaultdict(list)

    for i, prompt_rec in enumerate(prompts):
        prompt = prompt_rec.get("prompt") or prompt_rec.get("text", "")
        category = prompt_rec.get("category", "unknown")

        if args.mode == "full":
            result = compute_full_kl(base_model, base_tok, adapter_model, adapter_tok, prompt, args.max_tokens)
        else:
            result = compute_proxy_kl(base_model, base_tok, adapter_model, adapter_tok, prompt, args.max_tokens)

        result["prompt_id"] = prompt_rec.get("prompt_id", prompt_rec.get("id", f"p_{i:04d}"))
        result["category"] = category
        results.append(result)
        by_category[category].append(result)

        if (i + 1) % 10 == 0:
            log.info(f"  Processed {i + 1}/{len(prompts)}")

    # Aggregate
    if args.mode == "full":
        all_kl = [r["avg_kl"] for r in results]
        avg_kl = round(sum(all_kl) / len(all_kl), 6) if all_kl else 0
        median_kl = round(sorted(all_kl)[len(all_kl) // 2], 6) if all_kl else 0
    else:
        avg_kl = None
        median_kl = None

    length_drifts = [r["length_drift"] for r in results]
    avg_length_drift = round(sum(length_drifts) / len(length_drifts), 2) if length_drifts else 0

    refusal_rate = round(
        sum(1 for r in results if r.get("adapter_slop_count", 0) > 0) / len(results), 3
    ) if results else 0

    rep_rates = [r.get("adapter_repetition_rate", 0) for r in results]
    avg_repetition = round(sum(rep_rates) / len(rep_rates), 4) if rep_rates else 0

    # Category breakdown
    cat_breakdown = {}
    for cat, cat_results in by_category.items():
        cat_lengths = [r["length_drift"] for r in cat_results]
        cat_breakdown[cat] = {
            "count": len(cat_results),
            "avg_length_drift": round(sum(cat_lengths) / len(cat_lengths), 2) if cat_lengths else 0,
        }
        if args.mode == "full":
            cat_kls = [r["avg_kl"] for r in cat_results]
            cat_breakdown[cat]["avg_kl"] = round(sum(cat_kls) / len(cat_kls), 6) if cat_kls else 0

    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "prompt_count": len(results),
        "average_kl": avg_kl,
        "median_kl": median_kl,
        "output_length_drift": avg_length_drift,
        "refusal_rate": refusal_rate,
        "repetition_rate": avg_repetition,
        "category_breakdown": cat_breakdown,
        "per_prompt": results,
    }

    # Save
    output_dir = PROJECT_ROOT / "results" / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}_kl.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Saved KL drift results to {output_path}")
    log.info(f"Summary: avg_kl={avg_kl}, length_drift={avg_length_drift}, refusal_rate={refusal_rate}, rep_rate={avg_repetition}")


if __name__ == "__main__":
    main()
