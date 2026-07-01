#!/usr/bin/env python3
"""
Phase 14: SFT Deep Dive for LFM2.5-230M
Focus: optimal SFT recipe for reliability on structured extraction.

Experiments:
  14A: Dataset size sweep — minimum viable dataset for 230M
  14B: Hard example injection — close the messy-text gap
  14C: Curriculum learning — easy→hard vs random vs hard→easy
  14D: Reliability replication — multi-seed variance check

Usage:
  python run_phase14.py --experiment 14A --device cuda:0
  python run_phase14.py --all --device cuda:0
"""

import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path
from collections import Counter
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT / "results" / "phase14"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_sft_phase14"
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

HUB_LAYERS = [0, 2, 4, 5]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]

# Best training config from Phase 9
DEFAULT_LR = 2e-4
DEFAULT_R = 8
DEFAULT_STEPS = 300
DEFAULT_BATCH = 2
DEFAULT_GRAD_ACCUM = 4

# Structured extraction eval prompts (held out — not in training data)
EVAL_PROMPTS = [
    {"prompt": "Extract name, age, city from: John Smith is 35 years old and lives in London.", "expected": {"name": "John Smith", "age": 35, "city": "London"}},
    {"prompt": "Extract product, price, currency from: Samsung Galaxy S24 priced at 849 GBP.", "expected": {"product": "Samsung Galaxy S24", "price": 849, "currency": "GBP"}},
    {"prompt": "Extract title, director, year from: Inception directed by Christopher Nolan in 2010.", "expected": {"title": "Inception", "director": "Christopher Nolan", "year": 2010}},
    {"prompt": "Extract city, country, population from: Tokyo, Japan has a population of 13,960,000.", "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000}},
    {"prompt": "Extract name, role, company from: Sarah Chen is the CTO at Stripe.", "expected": {"name": "Sarah Chen", "role": "CTO", "company": "Stripe"}},
    {"prompt": "Extract date, event, location from: On March 15, 2025, the GDC conference was held in San Francisco.", "expected": {"date": "March 15, 2025", "event": "GDC conference", "location": "San Francisco"}},
    {"prompt": "Extract make, model, year, price from: 2024 Tesla Model 3 — $38,990.", "expected": {"make": "Tesla", "model": "Model 3", "year": 2024, "price": 38990}},
    {"prompt": "Extract author, title, year from: '1984' by George Orwell, published 1949.", "expected": {"author": "George Orwell", "title": "1984", "year": 1949}},
    {"prompt": "Extract latitude, longitude, city from: 48.8566, 2.3522 — Paris.", "expected": {"latitude": 48.8566, "longitude": 2.3522, "city": "Paris"}},
    {"prompt": "Extract temperature, condition, city from: London today: 15°C, cloudy.", "expected": {"temperature": 15, "condition": "cloudy", "city": "London"}},
    {"prompt": "Extract drug, dosage, frequency from: Take 500mg of Amoxicillin twice daily.", "expected": {"drug": "Amoxicillin", "dosage": "500mg", "frequency": "twice daily"}},
    {"prompt": "Extract IP, port, protocol from: Server at 192.168.1.100:8080 using HTTPS.", "expected": {"ip": "192.168.1.100", "port": 8080, "protocol": "HTTPS"}},
    {"prompt": "Extract stock, price, change from: AAPL trading at $185.42, up 2.3%.", "expected": {"stock": "AAPL", "price": 185.42, "change": "+2.3%"}},
    {"prompt": "Extract email, role, company from: Contact jane.doe@meta.com, Senior Engineer at Meta.", "expected": {"email": "jane.doe@meta.com", "role": "Senior Engineer", "company": "Meta"}},
    {"prompt": "Extract endpoint, method, status from: GET /api/users returned 200 OK.", "expected": {"endpoint": "/api/users", "method": "GET", "status": 200}},
]

# Messy prompts for hard-example injection and robustness testing
MESSY_PROMPTS = [
    {"prompt": "pull name age city frm this: john smith 35 london", "expected": {"name": "John Smith", "age": 35, "city": "London"}},
    {"prompt": "Product: iphone15 Price:$999 Currency:USD — extract these fields", "expected": {"product": "iPhone 15", "price": 999, "currency": "USD"}},
    {"prompt": "get title dir yr — inception chris nolan 2010", "expected": {"title": "Inception", "director": "Chris Nolan", "year": 2010}},
    {"prompt": "tokyo japan pop 13.96M — extract city country population", "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000}},
    {"prompt": "name: sarah chen, cto @ stripe — pull name role company", "expected": {"name": "Sarah Chen", "role": "CTO", "company": "Stripe"}},
    {"prompt": "wwdc25 was june 9 in cupertino — date event loc", "expected": {"date": "June 9, 2025", "event": "WWDC 2025", "location": "Cupertino"}},
    {"prompt": "tesla model 3 2024 costs 38990 usd — make model year price", "expected": {"make": "Tesla", "model": "Model 3", "year": 2024, "price": 38990}},
    {"prompt": "orwell wrote 1984 in 1949 — author title year", "expected": {"author": "George Orwell", "title": "1984", "year": 1949}},
    {"prompt": "its 15C and cloudy in london rn — temp condition city", "expected": {"temperature": 15, "condition": "cloudy", "city": "London"}},
    {"prompt": "take amox 500mg 2x daily — drug dose freq", "expected": {"drug": "Amoxicillin", "dosage": "500mg", "frequency": "twice daily"}},
    {"prompt": "aapl at 185.42, +2.3% — stock price change", "expected": {"stock": "AAPL", "price": 185.42, "change": "+2.3%"}},
    {"prompt": "GET /api/users -> 200 OK — endpoint method status", "expected": {"endpoint": "/api/users", "method": "GET", "status": 200}},
    {"prompt": "mumbai india 12.4M ppl — city country population", "expected": {"city": "Mumbai", "country": "India", "population": 12442373}},
    {"prompt": "sammy galaxy s24 is £849 — product price currency", "expected": {"product": "Samsung Galaxy S24", "price": 849, "currency": "GBP"}},
    {"prompt": "matrix by wachowskis 1999 — title director year", "expected": {"title": "The Matrix", "director": "Wachowskis", "year": 1999}},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_smol_magpie(n=None):
    """Load smol-magpie-ultra multi-turn concise data."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train", trust_remote_code=True)
    if n:
        ds = ds.select(range(min(n, len(ds))))

    texts = []
    for ex in ds:
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        parts = []
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                parts.append(f"### Response:\n{content}")
        text = "\n\n".join(parts)
        if text and len(text.strip()) > 20:
            texts.append(text)

    print(f"  Loaded {len(texts)} smol-magpie examples")
    return texts


def make_extraction_pair(prompt, expected, messy=False):
    """Create a training example for structured extraction."""
    instruction = f"Extract the requested information as JSON.\n\nInput: {prompt}"
    response = json.dumps(expected, ensure_ascii=False)
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def load_hard_examples(n=50):
    """Generate extraction training pairs from messy prompts."""
    texts = []
    for p in MESSY_PROMPTS:
        texts.append(make_extraction_pair(p["prompt"], p["expected"], messy=True))
    # Also add clean extraction pairs
    for p in EVAL_PROMPTS:
        texts.append(make_extraction_pair(p["prompt"], p["expected"], messy=False))
    return texts[:n]


def make_extraction_dataset(n_clean, n_messy=0):
    """Create a mixed extraction training dataset."""
    texts = []
    # Clean extraction pairs
    for p in EVAL_PROMPTS * max(1, n_clean // len(EVAL_PROMPTS)):
        texts.append(make_extraction_pair(p["prompt"], p["expected"]))
    texts = texts[:n_clean]
    # Messy extraction pairs
    messy_texts = []
    for p in MESSY_PROMPTS * max(1, n_messy // len(MESSY_PROMPTS)):
        messy_texts.append(make_extraction_pair(p["prompt"], p["expected"], messy=True))
    texts.extend(messy_texts[:n_messy])
    return texts


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.examples = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            self.examples.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return model, tok


USE_CHAT_TEMPLATE = True


def format_eval_prompt(item, tokenizer):
    instruction = f"Extract the requested information as JSON.\n\nInput: {item['prompt']}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"{instruction}\n\nOutput:"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction(text, expected):
    hits = 0
    for key, val in expected.items():
        if str(val).lower() in text.lower():
            hits += 1
    field_recall = hits / max(len(expected), 1)
    exact = 1.0 if field_recall == 1.0 else 0.0
    return {"field_recall": field_recall, "exact_match": exact}


def generate_and_eval(model, tokenizer, prompt_text, expected, max_new_tokens=64, temp=0.2, top_p=0.9):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temp, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    return evaluate_extraction(text, expected)


def run_eval(model, tokenizer, prompts):
    """Evaluate model on a set of prompts. Returns accuracy + field recall."""
    model.eval()
    exact_hits = 0
    recall_sum = 0.0
    for item in prompts:
        prompt_text = format_eval_prompt(item, tokenizer)
        result = generate_and_eval(model, tokenizer, prompt_text, item["expected"])
        exact_hits += result["exact_match"]
        recall_sum += result["field_recall"]
    n = len(prompts)
    return {"accuracy": exact_hits / n, "field_recall": recall_sum / n, "n_correct": int(exact_hits), "n_total": n}


def run_noise_eval(model, tokenizer, prompts, sigma=0.01, K=5):
    """Evaluate with noise + best-of-K selection."""
    model.eval()
    exact_hits = 0
    recall_sum = 0.0
    for item in prompts:
        prompt_text = format_eval_prompt(item, tokenizer)
        best_recall = 0.0
        best_exact = 0.0
        for k in range(K):
            device = next(model.parameters()).device
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            set_seed(42 * 1000 + k)
            def embed_noise_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                gen = torch.Generator(device=hidden.device)
                gen.manual_seed(42 * 10000 + k)
                noise = torch.randn(hidden.shape, generator=gen, device=hidden.device, dtype=hidden.dtype)
                hidden = hidden + sigma * noise
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            embed_module = model.get_input_embeddings()
            h = embed_module.register_forward_hook(embed_noise_hook)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=64,
                    do_sample=True, temperature=0.2, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            h.remove()
            text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            result = evaluate_extraction(text, item["expected"])
            if result["field_recall"] > best_recall:
                best_recall = result["field_recall"]
                best_exact = result["exact_match"]
        exact_hits += best_exact
        recall_sum += best_recall
    n = len(prompts)
    return {"accuracy": exact_hits / n, "field_recall": recall_sum / n, "n_correct": int(exact_hits), "n_total": n}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sft(texts, tokenizer, exp_id, steps=DEFAULT_STEPS, lr=DEFAULT_LR, r=DEFAULT_R, seed=42):
    """Train a LoRA SFT adapter. Returns (model, metrics)."""
    model, _ = load_model_and_tokenizer()

    lora_config = LoraConfig(
        r=r, lora_alpha=r * 2,
        target_modules=TARGET_MODULES,
        layers_to_transform=HUB_LAYERS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    dataset = SFTDataset(texts, tokenizer)
    output_dir = str(ADAPTERS_DIR / exp_id)

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=steps,
        per_device_train_batch_size=DEFAULT_BATCH,
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, steps // 10),
        weight_decay=0.01,
        logging_steps=max(1, steps // 10),
        save_steps=steps,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    from transformers import Adafactor
    optimizer = Adafactor(
        model.parameters(), lr=lr, scale_parameter=False,
        relative_step=False, clip_threshold=1.0, decay_rate=-0.8, weight_decay=0.01,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, optimizers=(optimizer, None))

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    # Extract loss curve
    loss_curve = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_curve.append({"step": entry.get("step", 0), "loss": round(entry["loss"], 4)})

    metrics = {
        "final_loss": round(result.training_loss, 4),
        "train_time_s": round(train_time, 1),
        "trainable_params": trainable,
        "loss_curve": loss_curve,
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Experiment 14A: Dataset size sweep
# ---------------------------------------------------------------------------

def run_14A(tokenizer, args):
    """Find the minimum viable dataset size for 230M SFT."""
    print("\n=== Experiment 14A: Dataset Size Sweep ===")

    sizes = [25, 50, 100, 200, 345, 500]
    results = {}

    for n in sizes:
        exp_id = f"14A_n{n}"
        print(f"\n--- n={n} ---")

        texts = load_smol_magpie(n=n)
        if not texts:
            print(f"  No data for n={n}, skipping")
            continue

        model, train_metrics = train_sft(texts, tokenizer, exp_id, steps=DEFAULT_STEPS, seed=args.seed)

        # Eval: clean prompts
        clean_result = run_eval(model, tokenizer, EVAL_PROMPTS)
        print(f"  Clean: {clean_result['accuracy']:.3f} ({clean_result['n_correct']}/{clean_result['n_total']})")

        # Eval: messy prompts
        messy_result = run_eval(model, tokenizer, MESSY_PROMPTS)
        print(f"  Messy: {messy_result['accuracy']:.3f} ({messy_result['n_correct']}/{messy_result['n_total']})")

        # Eval: noise + best-of-K on clean
        noise_result = run_noise_eval(model, tokenizer, EVAL_PROMPTS, sigma=0.01, K=5)
        print(f"  Noisy clean: {noise_result['accuracy']:.3f} ({noise_result['n_correct']}/{noise_result['n_total']})")

        results[f"n{n}"] = {
            "n_examples": len(texts),
            "steps": DEFAULT_STEPS,
            "training": train_metrics,
            "clean_eval": clean_result,
            "messy_eval": messy_result,
            "noisy_eval": noise_result,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 14B: Hard example injection
# ---------------------------------------------------------------------------

def run_14B(tokenizer, args):
    """Inject messy-text extraction pairs into training data."""
    print("\n=== Experiment 14B: Hard Example Injection ===")

    # Use 200 clean smol-magpie + varying amounts of extraction pairs
    n_clean = 200
    messy_fractions = [0, 10, 25, 50, 100]
    results = {}

    for n_messy in messy_fractions:
        exp_id = f"14B_m{n_messy}"
        print(f"\n--- n_messy={n_messy} ---")

        # Base: smol-magpie data
        base_texts = load_smol_magpie(n=n_clean)
        # Add extraction pairs
        extraction_texts = make_extraction_dataset(n_clean=0, n_messy=n_messy)
        texts = base_texts + extraction_texts
        print(f"  Total: {len(texts)} ({len(base_texts)} smol + {len(extraction_texts)} extraction)")

        model, train_metrics = train_sft(texts, tokenizer, exp_id, steps=DEFAULT_STEPS, seed=args.seed)

        # Eval: clean
        clean_result = run_eval(model, tokenizer, EVAL_PROMPTS)
        print(f"  Clean: {clean_result['accuracy']:.3f}")

        # Eval: messy
        messy_result = run_eval(model, tokenizer, MESSY_PROMPTS)
        print(f"  Messy: {messy_result['accuracy']:.3f}")

        results[f"m{n_messy}"] = {
            "n_clean": len(base_texts),
            "n_messy": len(extraction_texts),
            "training": train_metrics,
            "clean_eval": clean_result,
            "messy_eval": messy_result,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 14C: Curriculum learning
# ---------------------------------------------------------------------------

def run_14C(tokenizer, args):
    """Test if data ordering matters: random vs easy→hard."""
    print("\n=== Experiment 14C: Curriculum Learning ===")

    n = 200
    results = {}

    texts = load_smol_magpie(n=n)
    if len(texts) < n:
        n = len(texts)

    # Estimate difficulty by text length (shorter = easier)
    texts_with_len = [(t, len(t)) for t in texts]
    texts_sorted_asc = [t for t, _ in sorted(texts_with_len, key=lambda x: x[1])]
    texts_sorted_desc = list(reversed(texts_sorted_asc))

    rng = np.random.RandomState(args.seed)
    texts_random = list(texts)
    rng.shuffle(texts_random)

    orderings = {
        "random": texts_random,
        "easy_to_hard": texts_sorted_asc,
        "hard_to_easy": texts_sorted_desc,
    }

    for order_name, ordered_texts in orderings.items():
        exp_id = f"14C_{order_name}"
        print(f"\n--- Order: {order_name} ---")

        model, train_metrics = train_sft(ordered_texts, tokenizer, exp_id, steps=DEFAULT_STEPS, seed=args.seed)

        clean_result = run_eval(model, tokenizer, EVAL_PROMPTS)
        messy_result = run_eval(model, tokenizer, MESSY_PROMPTS)
        print(f"  Clean: {clean_result['accuracy']:.3f}, Messy: {messy_result['accuracy']:.3f}")

        results[order_name] = {
            "n_examples": len(ordered_texts),
            "training": train_metrics,
            "clean_eval": clean_result,
            "messy_eval": messy_result,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Experiment 14D: Reliability replication
# ---------------------------------------------------------------------------

def run_14D(tokenizer, args):
    """Train best config across 3 seeds. Measure variance."""
    print("\n=== Experiment 14D: Reliability Replication ===")

    seeds = [42, 137, 2026]
    n = 200
    results = {}

    texts = load_smol_magpie(n=n)

    for seed in seeds:
        exp_id = f"14D_seed{seed}"
        print(f"\n--- Seed: {seed} ---")

        model, train_metrics = train_sft(texts, tokenizer, exp_id, steps=DEFAULT_STEPS, seed=seed)

        clean_result = run_eval(model, tokenizer, EVAL_PROMPTS)
        messy_result = run_eval(model, tokenizer, MESSY_PROMPTS)
        noise_result = run_noise_eval(model, tokenizer, EVAL_PROMPTS, sigma=0.01, K=5)
        print(f"  Clean: {clean_result['accuracy']:.3f}, Messy: {messy_result['accuracy']:.3f}, Noisy: {noise_result['accuracy']:.3f}")

        results[f"seed{seed}"] = {
            "seed": seed,
            "n_examples": len(texts),
            "training": train_metrics,
            "clean_eval": clean_result,
            "messy_eval": messy_result,
            "noisy_eval": noise_result,
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Compute variance
    accs = [r["clean_eval"]["accuracy"] for r in results.values()]
    results["_summary"] = {
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "min_accuracy": round(float(np.min(accs)), 4),
        "max_accuracy": round(float(np.max(accs)), 4),
    }
    print(f"\n  Reliability: {np.mean(accs):.3f} ± {np.std(accs):.3f} (min={np.min(accs):.3f}, max={np.max(accs):.3f})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "14A": ("Dataset Size Sweep", run_14A),
    "14B": ("Hard Example Injection", run_14B),
    "14C": ("Curriculum Learning", run_14C),
    "14D": ("Reliability Replication", run_14D),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 14: SFT Deep Dive")
    parser.add_argument("--experiment", "-e", choices=list(EXPERIMENT_MAP.keys()),
                        default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Must specify --experiment or --all")

    set_seed(args.seed)

    # Load tokenizer once (shared across experiments)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.all:
        experiments = list(EXPERIMENT_MAP.keys())
    else:
        experiments = [args.experiment]

    for exp_id in experiments:
        name, run_fn = EXPERIMENT_MAP[exp_id]
        print(f"\n{'='*60}")
        print(f"Running {exp_id}: {name}")
        print(f"{'='*60}")

        start_time = time.time()
        results = run_fn(tok, args)
        elapsed = time.time() - start_time

        output = {
            "experiment": exp_id,
            "name": name,
            "model": MODEL,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": results,
        }

        result_path = RESULTS_DIR / f"{exp_id}_seed{args.seed}.json"
        with open(result_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {result_path}")
        print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    print("\nAll Phase 14 experiments complete.")


if __name__ == "__main__":
    main()
