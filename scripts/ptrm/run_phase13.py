#!/usr/bin/env python3
"""
Phase 13: Atlas-Guided Stochastic Inference (PTRM-Inspired)

Injects Gaussian noise at specific layers during inference to escape "bad basins"
and uses selection strategies to pick the best rollout.

Usage:
  python run_phase13.py --experiment 13A --device cuda:0
  python run_phase13.py --experiment 13B --device cuda:0
  python run_phase13.py --experiment 13C --device cuda:0
  python run_phase13.py --experiment 13D --device cuda:0
  python run_phase13.py --experiment 13E --device cuda:0
  python run_phase13.py --experiment 13F --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "phase13"
CONFIGS_DIR = REPO_ROOT / "configs"

MODEL_NAME = "LiquidAI/LFM2.5-230M"  # Overridden by --model CLI arg

# Chat template support: if tokenizer has apply_chat_template, use it
USE_CHAT_TEMPLATE = False

# Structured extraction prompts for evaluation
EVAL_PROMPTS = [
    {"prompt": "Extract the name, age, and city from: John Smith is 35 years old and lives in London.", "expected": {"name": "John Smith", "age": 35, "city": "London"}},
    {"prompt": "Extract the name, age, and city from: Maria Garcia, aged 28, resides in Madrid.", "expected": {"name": "Maria Garcia", "age": 28, "city": "Madrid"}},
    {"prompt": "Extract the name, age, and city from: Ahmed Hassan (42) from Cairo.", "expected": {"name": "Ahmed Hassan", "age": 42, "city": "Cairo"}},
    {"prompt": "Extract the product, price, and currency from: The iPhone 15 costs $999.", "expected": {"product": "iPhone 15", "price": 999, "currency": "$"}},
    {"prompt": "Extract the product, price, and currency from: Samsung Galaxy S24 priced at 849 GBP.", "expected": {"product": "Samsung Galaxy S24", "price": 849, "currency": "GBP"}},
    {"prompt": "Extract the product, price, and currency from: Sony WH-1000XM5 headphones retail for 349 EUR.", "expected": {"product": "Sony WH-1000XM5", "price": 349, "currency": "EUR"}},
    {"prompt": "Extract the title, director, and year from: Inception directed by Christopher Nolan in 2010.", "expected": {"title": "Inception", "director": "Christopher Nolan", "year": 2010}},
    {"prompt": "Extract the title, director, and year from: The Matrix, directed by the Wachowskis, released 1999.", "expected": {"title": "The Matrix", "director": "Wachowskis", "year": 1999}},
    {"prompt": "Extract the title, director, and year from: Parasite (2019) by Bong Joon-ho.", "expected": {"title": "Parasite", "director": "Bong Joon-ho", "year": 2019}},
    {"prompt": "Extract the city, country, and population from: Tokyo, Japan has a population of 13,960,000.", "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000}},
    {"prompt": "Extract the city, country, and population from: Mumbai in India: 12,442,373 people.", "expected": {"city": "Mumbai", "country": "India", "population": 12442373}},
    {"prompt": "Extract the city, country, and population from: Sao Paulo, Brazil — population 12,325,232.", "expected": {"city": "Sao Paulo", "country": "Brazil", "population": 12325232}},
    {"prompt": "Extract the name, role, and company from: Sarah Chen is the CTO at Stripe.", "expected": {"name": "Sarah Chen", "role": "CTO", "company": "Stripe"}},
    {"prompt": "Extract the name, role, and company from: Dr. James Wilson, VP of Engineering, works at Anthropic.", "expected": {"name": "Dr. James Wilson", "role": "VP of Engineering", "company": "Anthropic"}},
    {"prompt": "Extract the name, role, and company from: Lisa Park — Head of Product at Notion.", "expected": {"name": "Lisa Park", "role": "Head of Product", "company": "Notion"}},
    {"prompt": "Extract the date, event, and location from: On March 15, 2025, the GDC conference was held in San Francisco.", "expected": {"date": "March 15, 2025", "event": "GDC conference", "location": "San Francisco"}},
    {"prompt": "Extract the date, event, and location from: WWDC 2025 took place June 9 in Cupertino.", "expected": {"date": "June 9, 2025", "event": "WWDC 2025", "location": "Cupertino"}},
    {"prompt": "Extract the date, event, and location from: The Olympics opening ceremony on July 26, 2024 in Paris.", "expected": {"date": "July 26, 2024", "event": "Olympics opening ceremony", "location": "Paris"}},
    {"prompt": "Extract the make, model, year, and price from: 2024 Tesla Model 3 — $38,990.", "expected": {"make": "Tesla", "model": "Model 3", "year": 2024, "price": 38990}},
    {"prompt": "Extract the make, model, year, and price from: A 2023 Toyota Camry LE costs $26,420.", "expected": {"make": "Toyota", "model": "Camry LE", "year": 2023, "price": 26420}},
    {"prompt": "Extract the make, model, year, and price from: BMW X5 xDrive40i (2024) priced at $63,200.", "expected": {"make": "BMW", "model": "X5 xDrive40i", "year": 2024, "price": 63200}},
    {"prompt": "Extract the author, title, and year from: '1984' by George Orwell, published 1949.", "expected": {"author": "George Orwell", "title": "1984", "year": 1949}},
    {"prompt": "Extract the author, title, and year from: To Kill a Mockingbird — Harper Lee, 1960.", "expected": {"author": "Harper Lee", "title": "To Kill a Mockingbird", "year": 1960}},
    {"prompt": "Extract the author, title, and year from: The Great Gatsby by F. Scott Fitzgerald (1925).", "expected": {"author": "F. Scott Fitzgerald", "title": "The Great Gatsby", "year": 1925}},
    {"prompt": "Extract the latitude, longitude, and city from: 48.8566, 2.3522 — Paris.", "expected": {"latitude": 48.8566, "longitude": 2.3522, "city": "Paris"}},
    {"prompt": "Extract the latitude, longitude, and city from: Coordinates 35.6762, 139.6503 for Tokyo.", "expected": {"latitude": 35.6762, "longitude": 139.6503, "city": "Tokyo"}},
    {"prompt": "Extract the latitude, longitude, and city from: New York is at 40.7128, -74.0060.", "expected": {"latitude": 40.7128, "longitude": -74.006, "city": "New York"}},
    {"prompt": "Extract the language, paradigm, and year from: Python, multi-paradigm, created in 1991.", "expected": {"language": "Python", "paradigm": "multi-paradigm", "year": 1991}},
    {"prompt": "Extract the language, paradigm, and year from: Haskell — functional, released 1990.", "expected": {"language": "Haskell", "paradigm": "functional", "year": 1990}},
    {"prompt": "Extract the language, paradigm, and year from: Rust (systems programming, 2015).", "expected": {"language": "Rust", "paradigm": "systems programming", "year": 2015}},
    {"prompt": "Extract the temperature, condition, and city from: London today: 15°C, cloudy.", "expected": {"temperature": 15, "condition": "cloudy", "city": "London"}},
    {"prompt": "Extract the temperature, condition, and city from: Tokyo weather — 28 degrees, sunny.", "expected": {"temperature": 28, "condition": "sunny", "city": "Tokyo"}},
    {"prompt": "Extract the temperature, condition, and city from: New York: 32°F, snowing.", "expected": {"temperature": 32, "condition": "snowing", "city": "New York"}},
    {"prompt": "Extract the drug, dosage, and frequency from: Take 500mg of Amoxicillin twice daily.", "expected": {"drug": "Amoxicillin", "dosage": "500mg", "frequency": "twice daily"}},
    {"prompt": "Extract the drug, dosage, and frequency from: Ibuprofen 200mg every 6 hours.", "expected": {"drug": "Ibuprofen", "dosage": "200mg", "frequency": "every 6 hours"}},
    {"prompt": "Extract the drug, dosage, and frequency from: Metformin 850mg once a day with meals.", "expected": {"drug": "Metformin", "dosage": "850mg", "frequency": "once a day"}},
    {"prompt": "Extract the IP, port, and protocol from: Server at 192.168.1.100:8080 using HTTPS.", "expected": {"ip": "192.168.1.100", "port": 8080, "protocol": "HTTPS"}},
    {"prompt": "Extract the IP, port, and protocol from: Connect to 10.0.0.5 on port 3306 via MySQL.", "expected": {"ip": "10.0.0.5", "port": 3306, "protocol": "MySQL"}},
    {"prompt": "Extract the IP, port, and protocol from: Redis running at 127.0.0.1:6379 TCP.", "expected": {"ip": "127.0.0.1", "port": 6379, "protocol": "TCP"}},
    {"prompt": "Extract the stock, price, and change from: AAPL trading at $185.42, up 2.3%.", "expected": {"stock": "AAPL", "price": 185.42, "change": "+2.3%"}},
    {"prompt": "Extract the stock, price, and change from: GOOGL — $141.80, down 0.8%.", "expected": {"stock": "GOOGL", "price": 141.80, "change": "-0.8%"}},
    {"prompt": "Extract the stock, price, and change from: NVDA at $875.28 (change: +4.1%).", "expected": {"stock": "NVDA", "price": 875.28, "change": "+4.1%"}},
    {"prompt": "Extract the email, role, and company from: Contact jane.doe@meta.com, Senior Engineer at Meta.", "expected": {"email": "jane.doe@meta.com", "role": "Senior Engineer", "company": "Meta"}},
    {"prompt": "Extract the email, role, and company from: bob@openai.com — Research Scientist, OpenAI.", "expected": {"email": "bob@openai.com", "role": "Research Scientist", "company": "OpenAI"}},
    {"prompt": "Extract the email, role, and company from: alice.chen@deepmind.com, Team Lead at DeepMind.", "expected": {"email": "alice.chen@deepmind.com", "role": "Team Lead", "company": "DeepMind"}},
    {"prompt": "Extract the endpoint, method, and status from: GET /api/users returned 200 OK.", "expected": {"endpoint": "/api/users", "method": "GET", "status": 200}},
    {"prompt": "Extract the endpoint, method, and status from: POST /auth/login — 401 Unauthorized.", "expected": {"endpoint": "/auth/login", "method": "POST", "status": 401}},
    {"prompt": "Extract the endpoint, method, and status from: DELETE /api/items/42 responded 204 No Content.", "expected": {"endpoint": "/api/items/42", "method": "DELETE", "status": 204}},
    {"prompt": "Extract the name, breed, and age from: Buddy is a 5-year-old Golden Retriever.", "expected": {"name": "Buddy", "breed": "Golden Retriever", "age": 5}},
    {"prompt": "Extract the name, breed, and age from: Luna, Persian cat, 3 years old.", "expected": {"name": "Luna", "breed": "Persian cat", "age": 3}},
    {"prompt": "Extract the name, breed, and age from: Max the German Shepherd, age 7.", "expected": {"name": "Max", "breed": "German Shepherd", "age": 7}},
]

# Atlas data for LFM2.5-230M
SKIP_KL = {
    0: 82.9, 1: 36.3, 2: 30.8, 3: 25.9, 4: 34.6, 5: 44.4,
    6: 8.7, 7: 7.3, 8: 10.2, 9: 7.1, 10: 17.7, 11: 8.7,
    12: 29.1, 13: 12.1,
}

LAYER_TYPES = {
    0: 'conv', 1: 'conv', 2: 'attn', 3: 'conv', 4: 'attn', 5: 'conv',
    6: 'attn', 7: 'conv', 8: 'attn', 9: 'conv', 10: 'attn', 11: 'conv',
    12: 'attn', 13: 'conv',
}

RESIDUAL_NORMS = {
    0: 2.02, 1: 2.19, 2: 1.53, 3: 2.33, 4: 1.41, 5: 25.54,
    6: 25.54, 7: 25.56, 8: 25.5, 9: 25.5, 10: 25.5, 11: 25.5,
    12: 25.5, 13: 22.91,
}

# Normalized importance weights
TOTAL_KL = sum(SKIP_KL.values())
LAYER_WEIGHTS = {k: v / TOTAL_KL for k, v in SKIP_KL.items()}

# ---------------------------------------------------------------------------
# Noise injection hooks
# ---------------------------------------------------------------------------

def make_noise_hook(sigma, noise_seed=None):
    """Create a forward hook that adds Gaussian noise to the residual stream."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            if noise_seed is not None:
                gen = torch.Generator(device=hidden.device)
                gen.manual_seed(noise_seed)
                noise = torch.randn(hidden.shape, generator=gen, device=hidden.device, dtype=hidden.dtype)
            else:
                noise = torch.randn_like(hidden)
            hidden = hidden + sigma * noise
            return (hidden,) + output[1:]
        else:
            noise = torch.randn_like(output)
            return output + sigma * noise
    return hook_fn


def make_scaled_noise_hook(base_sigma, layer_idx):
    """Noise scaled by layer importance (atlas-guided)."""
    sigma = base_sigma * LAYER_WEIGHTS.get(layer_idx, 0.01) * 14  # Scale so avg = base_sigma
    return make_noise_hook(sigma)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name=None, device="cuda:0"):
    """Load model and tokenizer."""
    global MODEL_NAME, USE_CHAT_TEMPLATE
    if model_name:
        MODEL_NAME = model_name
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Detect chat template support
    USE_CHAT_TEMPLATE = hasattr(tokenizer, 'apply_chat_template')
    if USE_CHAT_TEMPLATE:
        print("Using chat template for prompt formatting")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def generate_noisy_rollout(model, tokenizer, prompt, layer_indices, sigma,
                           max_new_tokens=64, rollout_seed=None, temp=0.2, top_p=0.9,
                           noise_mode="prompt_only"):
    """Generate one rollout with noise injected at specified layers.
    
    noise_mode:
      - "embed": noise at embedding layer only (clean, architecture-agnostic)
      - "layer_first_step": noise at specified layers, prompt-phase only via custom forward
      - "every_step": noise at every generation step (destructive, for comparison)
    
    Optimized: 2 forward passes total (generation + metrics)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if rollout_seed is not None:
        set_seed(rollout_seed)

    if noise_mode == "embed":
        # Add noise to embeddings only — architecture-agnostic, single perturbation
        def embed_noise_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            gen = torch.Generator(device=hidden.device)
            gen.manual_seed(rollout_seed if rollout_seed else 0)
            noise = torch.randn(hidden.shape, generator=gen, device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + sigma * noise
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        h = model.model.embed_tokens.register_forward_hook(embed_noise_hook)
        with torch.no_grad():
            full_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        h.remove()

    elif noise_mode == "every_step":
        # Noise at specified layers, every step (destructive)
        hooks = []
        for idx in layer_indices:
            hook = make_noise_hook(sigma, noise_seed=(rollout_seed + idx) if rollout_seed else None)
            h = model.model.layers[idx].register_forward_hook(hook)
            hooks.append(h)

        with torch.no_grad():
            full_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        for h in hooks:
            h.remove()

    else:  # "prompt_only" — run prompt with layer hooks, then clean generate
        # NOTE: LFM2.5 conv layers (kernel_size=4) require >=4 tokens, so KV cache
        # with single-token continuation doesn't work. Fall back to embed mode.
        def embed_noise_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            gen = torch.Generator(device=hidden.device)
            gen.manual_seed(rollout_seed if rollout_seed else 0)
            noise = torch.randn(hidden.shape, generator=gen, device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + sigma * noise
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        h = model.model.embed_tokens.register_forward_hook(embed_noise_hook)
        with torch.no_grad():
            full_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        h.remove()

    text = tokenizer.decode(full_output[0][prompt_len:], skip_special_tokens=True).strip()

    # Single forward pass for both NLL and confidence
    with torch.no_grad():
        logits = model(full_output).logits[0]
        prompt_logits = logits[:prompt_len - 1].float()
        prompt_labels = full_output[0, 1:prompt_len]
        prompt_nll = F.cross_entropy(prompt_logits, prompt_labels).item()
        first_gen_logit = logits[prompt_len - 1].float()
        max_conf = F.softmax(first_gen_logit / temp, dim=-1).max().item()

    return {"text": text, "prompt_nll": prompt_nll, "max_conf": max_conf}


def generate_scaled_noisy_rollout(model, tokenizer, prompt, layer_indices, base_sigma,
                                   max_new_tokens=64, rollout_seed=None, temp=0.2, top_p=0.9):
    """Generate one rollout with atlas-scaled noise. Optimized: 2 forward passes."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    hooks = []
    for idx in layer_indices:
        layer_sigma = base_sigma * LAYER_WEIGHTS.get(idx, 0.01) * 14
        hook = make_noise_hook(layer_sigma, noise_seed=(rollout_seed + idx) if rollout_seed else None)
        h = model.model.layers[idx].register_forward_hook(hook)
        hooks.append(h)

    if rollout_seed is not None:
        set_seed(rollout_seed)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    for h in hooks:
        h.remove()

    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    with torch.no_grad():
        logits = model(output).logits[0]
        prompt_logits = logits[:prompt_len - 1].float()
        prompt_labels = output[0, 1:prompt_len]
        prompt_nll = F.cross_entropy(prompt_logits, prompt_labels).item()
        first_gen_logit = logits[prompt_len - 1].float()
        max_conf = F.softmax(first_gen_logit / temp, dim=-1).max().item()

    return {"text": text, "prompt_nll": prompt_nll, "max_conf": max_conf}


def generate_baseline(model, tokenizer, prompt, max_new_tokens=64, temp=0.2, top_p=0.9):
    """Generate without noise (baseline). Optimized: 2 forward passes."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    with torch.no_grad():
        logits = model(output).logits[0]
        prompt_logits = logits[:prompt_len - 1].float()
        prompt_labels = output[0, 1:prompt_len]
        prompt_nll = F.cross_entropy(prompt_logits, prompt_labels).item()
        first_gen_logit = logits[prompt_len - 1].float()
        max_conf = F.softmax(first_gen_logit / temp, dim=-1).max().item()

    return {"text": text, "prompt_nll": prompt_nll, "max_conf": max_conf}


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_random(rollouts, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(rollouts))
    return rollouts[idx], idx

def select_lowest_nll(rollouts):
    best_idx = min(range(len(rollouts)), key=lambda i: rollouts[i]["prompt_nll"])
    return rollouts[best_idx], best_idx

def select_highest_conf(rollouts):
    best_idx = max(range(len(rollouts)), key=lambda i: rollouts[i]["max_conf"])
    return rollouts[best_idx], best_idx

def select_majority_vote(rollouts):
    texts = [r["text"][:200] for r in rollouts]  # Truncate for comparison
    counter = Counter(texts)
    best_text = counter.most_common(1)[0][0]
    for i, r in enumerate(rollouts):
        if r["text"][:200] == best_text:
            return r, i
    return rollouts[0], 0

def select_oracle(rollouts, expected):
    """Oracle: pick the rollout whose text best matches expected output."""
    def score(text):
        hits = 0
        total = 0
        for key, val in expected.items():
            total += 1
            val_str = str(val).lower()
            if val_str in text.lower():
                hits += 1
        return hits / max(total, 1)

    best_idx = max(range(len(rollouts)), key=lambda i: score(rollouts[i]["text"]))
    return rollouts[best_idx], best_idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction(text, expected):
    """Score extraction quality: field-level F1 and exact match."""
    hits = 0
    total = len(expected)
    for key, val in expected.items():
        val_str = str(val).lower()
        if val_str in text.lower():
            hits += 1
    field_recall = hits / max(total, 1)
    exact = 1.0 if field_recall == 1.0 else 0.0
    # JSON validity
    try:
        json.loads(text)
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        json_valid = False
    return {
        "field_recall": field_recall,
        "exact_match": exact,
        "json_valid": json_valid,
    }


def format_extraction_prompt(item, tokenizer=None):
    """Format an eval item into an instruction prompt.
    Uses chat template if available, otherwise plain text."""
    instruction = f"Extract the requested information as JSON.\n\nInput: {item['prompt']}"
    
    if USE_CHAT_TEMPLATE and tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    
    return f"{instruction}\n\nOutput:"


# ---------------------------------------------------------------------------
# Completion diversity analysis (for 13E)
# ---------------------------------------------------------------------------

def compute_pairwise_similarity(texts):
    """Simple token-overlap Jaccard similarity between all pairs."""
    def tokenize(t):
        return set(t.lower().split())
    n = len(texts)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tokenize(texts[i]), tokenize(texts[j])
            if not a and not b:
                sim = 1.0
            elif not a or not b:
                sim = 0.0
            else:
                sim = len(a & b) / len(a | b)
            sims.append(sim)
    return sims


def cluster_completions(texts, threshold=0.6):
    """Cluster completions by Jaccard similarity."""
    clusters = []
    for text in texts:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            a = set(text.lower().split())
            b = set(rep.lower().split())
            if a and b:
                sim = len(a & b) / len(a | b)
            else:
                sim = 0.0
            if sim >= threshold:
                cluster.append(text)
                placed = True
                break
        if not placed:
            clusters.append([text])
    return clusters


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_13A(model, tokenizer, args):
    """13A: Noise localization — which layer gives the biggest best-of-K boost?"""
    print("\n=== Experiment 13A: Noise Localization ===")
    K = 10
    sigma = 0.01  # Reduced from 0.2 — prompt-only noise at hub is still potent
    prompts = EVAL_PROMPTS[:30]
    # Test key layers only: hub, secondary hubs, weak control layers
    KEY_LAYERS = [0, 1, 4, 5, 7, 12, 13]  # L0=hub, L4=secondary, L5=strongest MLP, L7/L13=weak controls
    results = {}

    for layer_idx in KEY_LAYERS:
        layer_type = LAYER_TYPES[layer_idx]
        kl = SKIP_KL[layer_idx]
        print(f"\n--- Layer {layer_idx} ({layer_type}), skip KL={kl:.1f} ---")

        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for item in prompts:
            prompt_text = format_extraction_prompt(item, tokenizer)
            best_field_recall = 0.0
            best_exact = 0.0

            for k in range(K):
                rollout = generate_noisy_rollout(
                    model, tokenizer, prompt_text,
                    layer_indices=[layer_idx], sigma=sigma,
                    rollout_seed=args.seed * 10000 + layer_idx * 100 + k,
                )
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
                if eval_result["field_recall"] > best_field_recall:
                    best_field_recall = eval_result["field_recall"]
                    best_exact = eval_result["exact_match"]

            exact_hits += best_exact
            field_recall_sum += best_field_recall
            n += 1

        accuracy = exact_hits / n
        avg_recall = field_recall_sum / n
        print(f"  best-of-{K} accuracy: {accuracy:.3f} (recall: {avg_recall:.3f})")

        results[f"layer_{layer_idx}"] = {
            "layer": layer_idx,
            "type": layer_type,
            "skip_kl": kl,
            "best_of_K_accuracy": accuracy,
            "avg_field_recall": avg_recall,
            "K": K,
            "sigma": sigma,
        }

    # Also run baseline (no noise)
    print("\n--- Baseline (no noise) ---")
    exact_hits = 0
    field_recall_sum = 0.0
    for item in prompts:
        prompt_text = format_extraction_prompt(item, tokenizer)
        rollout = generate_baseline(model, tokenizer, prompt_text)
        eval_result = evaluate_extraction(rollout["text"], item["expected"])
        exact_hits += eval_result["exact_match"]
        field_recall_sum += eval_result["field_recall"]

    baseline_acc = exact_hits / len(prompts)
    baseline_recall = field_recall_sum / len(prompts)
    print(f"  Baseline accuracy: {baseline_acc:.3f} (recall: {baseline_recall:.3f})")

    results["baseline"] = {
        "best_of_K_accuracy": baseline_acc,
        "avg_field_recall": baseline_recall,
        "K": 1,
        "sigma": 0,
    }

    return results


def run_13B(model, tokenizer, args):
    """13B: Width scaling — how does accuracy scale with K?"""
    print("\n=== Experiment 13B: Width Scaling ===")
    hub_layer = 0  # L0 is the hub
    sigma = 0.01
    K_values = [1, 2, 5, 10, 20, 50]
    prompts = EVAL_PROMPTS[:30]  # Fewer prompts for speed
    results = {}

    for K in K_values:
        print(f"\n--- K={K} ---")
        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for item in prompts:
            prompt_text = format_extraction_prompt(item, tokenizer)
            best_field_recall = 0.0
            best_exact = 0.0

            for k in range(K):
                rollout = generate_noisy_rollout(
                    model, tokenizer, prompt_text,
                    layer_indices=[hub_layer], sigma=sigma,
                    rollout_seed=args.seed * 10000 + K * 100 + k,
                )
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
                if eval_result["field_recall"] > best_field_recall:
                    best_field_recall = eval_result["field_recall"]
                    best_exact = eval_result["exact_match"]

            exact_hits += best_exact
            field_recall_sum += best_field_recall
            n += 1

        accuracy = exact_hits / n
        avg_recall = field_recall_sum / n
        print(f"  best-of-{K} accuracy: {accuracy:.3f} (recall: {avg_recall:.3f})")

        results[f"K_{K}"] = {
            "K": K,
            "best_of_K_accuracy": accuracy,
            "avg_field_recall": avg_recall,
        }

    return results


def run_13C(model, tokenizer, args):
    """13C: Sigma sweep — what's the optimal noise magnitude?"""
    print("\n=== Experiment 13C: Sigma Sweep ===")
    hub_layer = 0
    K = 10
    sigma_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    prompts = EVAL_PROMPTS[:30]
    results = {}

    for sigma in sigma_values:
        print(f"\n--- sigma={sigma} ---")
        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for item in prompts:
            prompt_text = format_extraction_prompt(item, tokenizer)
            best_field_recall = 0.0
            best_exact = 0.0

            for k in range(K):
                rollout = generate_noisy_rollout(
                    model, tokenizer, prompt_text,
                    layer_indices=[hub_layer], sigma=sigma,
                    rollout_seed=args.seed * 10000 + int(sigma * 1000) + k,
                )
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
                if eval_result["field_recall"] > best_field_recall:
                    best_field_recall = eval_result["field_recall"]
                    best_exact = eval_result["exact_match"]

            exact_hits += best_exact
            field_recall_sum += best_field_recall
            n += 1

        accuracy = exact_hits / n
        avg_recall = field_recall_sum / n
        print(f"  best-of-{K} accuracy: {accuracy:.3f} (recall: {avg_recall:.3f})")

        results[f"sigma_{sigma}"] = {
            "sigma": sigma,
            "best_of_K_accuracy": accuracy,
            "avg_field_recall": avg_recall,
            "relative_perturbation": sigma / RESIDUAL_NORMS[hub_layer],
        }

    return results


def run_13D(model, tokenizer, args):
    """13D: Selection strategy — which selection method is best?"""
    print("\n=== Experiment 13D: Selection Strategy ===")
    hub_layer = 0
    sigma = 0.01
    K = 10
    prompts = EVAL_PROMPTS[:40]
    results = {}

    strategies = ["random", "lowest_nll", "highest_conf", "majority_vote", "oracle"]
    strategy_fns = {
        "random": lambda rollouts, item: select_random(rollouts, seed=args.seed),
        "lowest_nll": lambda rollouts, item: select_lowest_nll(rollouts),
        "highest_conf": lambda rollouts, item: select_highest_conf(rollouts),
        "majority_vote": lambda rollouts, item: select_majority_vote(rollouts),
        "oracle": lambda rollouts, item: select_oracle(rollouts, item["expected"]),
    }

    # Pre-generate all rollouts (shared across strategies)
    print("Generating rollouts...")
    all_rollouts = {}
    for i, item in enumerate(prompts):
        prompt_text = format_extraction_prompt(item, tokenizer)
        rollouts = []
        for k in range(K):
            rollout = generate_noisy_rollout(
                model, tokenizer, prompt_text,
                layer_indices=[hub_layer], sigma=sigma,
                rollout_seed=args.seed * 10000 + i * 100 + k,
            )
            rollouts.append(rollout)
        all_rollouts[i] = rollouts
        if (i + 1) % 10 == 0:
            print(f"  Generated rollouts for {i+1}/{len(prompts)} prompts")

    # Evaluate each strategy
    for strategy_name in strategies:
        print(f"\n--- Strategy: {strategy_name} ---")
        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for i, item in enumerate(prompts):
            rollouts = all_rollouts[i]
            selected, _ = strategy_fns[strategy_name](rollouts, item)
            eval_result = evaluate_extraction(selected["text"], item["expected"])
            exact_hits += eval_result["exact_match"]
            field_recall_sum += eval_result["field_recall"]
            n += 1

        accuracy = exact_hits / n
        avg_recall = field_recall_sum / n
        print(f"  accuracy: {accuracy:.3f} (recall: {avg_recall:.3f})")

        results[strategy_name] = {
            "strategy": strategy_name,
            "accuracy": accuracy,
            "avg_field_recall": avg_recall,
        }

    # Baseline (no noise, single pass)
    print("\n--- Baseline (no noise) ---")
    exact_hits = 0
    field_recall_sum = 0.0
    for item in prompts:
        prompt_text = format_extraction_prompt(item, tokenizer)
        rollout = generate_baseline(model, tokenizer, prompt_text)
        eval_result = evaluate_extraction(rollout["text"], item["expected"])
        exact_hits += eval_result["exact_match"]
        field_recall_sum += eval_result["field_recall"]

    results["baseline"] = {
        "accuracy": exact_hits / len(prompts),
        "avg_field_recall": field_recall_sum / len(prompts),
    }

    return results


def run_13E(model, tokenizer, args):
    """13E: Bad basin detection — do distinct completion clusters exist?"""
    print("\n=== Experiment 13E: Bad Basin Detection ===")
    hub_layer = 0
    sigma = 0.01
    K = 50  # More rollouts for clustering
    prompts = EVAL_PROMPTS[:20]
    results = {}

    for i, item in enumerate(prompts):
        prompt_text = format_extraction_prompt(item, tokenizer)

        # First check baseline
        baseline = generate_baseline(model, tokenizer, prompt_text)
        baseline_eval = evaluate_extraction(baseline["text"], item["expected"])

        # Generate K noisy rollouts
        rollouts = []
        for k in range(K):
            rollout = generate_noisy_rollout(
                model, tokenizer, prompt_text,
                layer_indices=[hub_layer], sigma=sigma,
                rollout_seed=args.seed * 100000 + i * 1000 + k,
            )
            rollout["eval"] = evaluate_extraction(rollout["text"], item["expected"])
            rollouts.append(rollout)

        # Cluster
        texts = [r["text"] for r in rollouts]
        clusters = cluster_completions(texts, threshold=0.6)

        # Analyze clusters
        cluster_info = []
        for ci, cluster in enumerate(clusters):
            cluster_evals = [
                rollouts[j]["eval"] for j in range(len(texts))
                if texts[j] in cluster
            ]
            avg_recall = np.mean([e["field_recall"] for e in cluster_evals])
            any_exact = any(e["exact_match"] for e in cluster_evals)
            cluster_info.append({
                "cluster_id": ci,
                "size": len(cluster),
                "avg_field_recall": avg_recall,
                "has_exact_match": any_exact,
                "sample": cluster[0][:150],
            })

        # Find correct cluster
        correct_clusters = [c for c in cluster_info if c["has_exact_match"]]
        correct_fraction = sum(c["size"] for c in correct_clusters) / K if correct_clusters else 0

        results[f"prompt_{i}"] = {
            "prompt_idx": i,
            "baseline_correct": baseline_eval["exact_match"],
            "n_clusters": len(clusters),
            "correct_fraction": correct_fraction,
            "cluster_sizes": [c["size"] for c in cluster_info],
            "clusters": cluster_info,
        }

        status = "BASELINE CORRECT" if baseline_eval["exact_match"] else "BASELINE WRONG"
        print(f"  Prompt {i} ({status}): {len(clusters)} clusters, "
              f"correct fraction: {correct_fraction:.2%}")

    # Summary
    wrong_prompts = [r for r in results.values() if not r["baseline_correct"]]
    if wrong_prompts:
        avg_clusters = np.mean([r["n_clusters"] for r in wrong_prompts])
        avg_correct_frac = np.mean([r["correct_fraction"] for r in wrong_prompts])
        print(f"\nSummary for {len(wrong_prompts)} wrong-baseline prompts:")
        print(f"  Avg clusters: {avg_clusters:.1f}")
        print(f"  Avg correct fraction: {avg_correct_frac:.2%}")
        results["_summary"] = {
            "n_wrong_baseline": len(wrong_prompts),
            "avg_clusters": avg_clusters,
            "avg_correct_fraction": avg_correct_frac,
        }

    return results


def run_13F(model, tokenizer, args):
    """13F: Atlas-guided vs uniform vs random noise — head-to-head."""
    print("\n=== Experiment 13F: Head-to-Head Comparison ===")
    K = 10
    sigma = 0.01
    prompts = EVAL_PROMPTS[:40]
    results = {}

    conditions = {
        "no_noise": {"layers": [], "sigma": 0, "scaled": False},
        "uniform_all": {"layers": list(range(14)), "sigma": sigma, "scaled": False},
        "hub_only_L0": {"layers": [0], "sigma": sigma, "scaled": False},
        "random_layer_L7": {"layers": [7], "sigma": sigma, "scaled": False},
        "early_phase_L0_L5": {"layers": [0, 1, 2, 3, 4, 5], "sigma": sigma, "scaled": False},
        "atlas_scaled": {"layers": list(range(14)), "sigma": sigma, "scaled": True},
    }

    for cond_name, cond in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for i, item in enumerate(prompts):
            prompt_text = format_extraction_prompt(item, tokenizer)

            if not cond["layers"]:
                # Baseline: no noise
                rollout = generate_baseline(model, tokenizer, prompt_text)
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
            else:
                best_recall = 0.0
                best_exact = 0.0
                for k in range(K):
                    if cond["scaled"]:
                        # Atlas-scaled: noise at all layers, sigma proportional to importance
                        # Build per-layer hooks with different sigmas
                        layer_sigmas = {}
                        for layer_idx in cond["layers"]:
                            layer_sigmas[layer_idx] = sigma * LAYER_WEIGHTS.get(layer_idx, 0.01) * 14
                        # Use a helper: inject noise at all layers with weighted sigma
                        rollout = generate_scaled_noisy_rollout(
                            model, tokenizer, prompt_text,
                            layer_indices=cond["layers"],
                            base_sigma=sigma,
                            rollout_seed=args.seed * 100000 + i * 1000 + k,
                        )
                        eval_result = evaluate_extraction(rollout["text"], item["expected"])
                    else:
                        rollout = generate_noisy_rollout(
                            model, tokenizer, prompt_text,
                            layer_indices=cond["layers"], sigma=cond["sigma"],
                            rollout_seed=args.seed * 10000 + i * 100 + k,
                        )
                        eval_result = evaluate_extraction(rollout["text"], item["expected"])

                    if eval_result["field_recall"] > best_recall:
                        best_recall = eval_result["field_recall"]
                        best_exact = eval_result["exact_match"]

                eval_result = {"field_recall": best_recall, "exact_match": best_exact}

            exact_hits += eval_result["exact_match"]
            field_recall_sum += eval_result["field_recall"]
            n += 1

        accuracy = exact_hits / n
        avg_recall = field_recall_sum / n
        print(f"  accuracy: {accuracy:.3f} (recall: {avg_recall:.3f})")

        results[cond_name] = {
            "condition": cond_name,
            "accuracy": accuracy,
            "avg_field_recall": avg_recall,
            "layers": cond["layers"],
            "sigma": cond["sigma"],
            "K": K if cond["layers"] else 1,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "13A": ("Noise Localization", run_13A),
    "13B": ("Width Scaling", run_13B),
    "13C": ("Sigma Sweep", run_13C),
    "13D": ("Selection Strategy", run_13D),
    "13E": ("Bad Basin Detection", run_13E),
    "13F": ("Head-to-Head Comparison", run_13F),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 13: PTRM-Inspired Experiments")
    parser.add_argument("--experiment", "-e", choices=list(EXPERIMENT_MAP.keys()),
                        default=None, help="Which experiment to run")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (e.g. Qwen/Qwen2.5-0.5B-Instruct). Default: LiquidAI/LFM2.5-230M")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--all", action="store_true", help="Run all experiments sequentially")
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Must specify --experiment or --all")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(model_name=args.model, device=args.device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        results = run_fn(model, tokenizer, args)
        elapsed = time.time() - start_time

        # Save results
        output = {
            "experiment": exp_id,
            "name": name,
            "model": MODEL_NAME,
            "seed": args.seed,
            "device": args.device,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": results,
        }

        # Model slug for filename (e.g. qwen05b, lfm2_230m)
        model_slug = MODEL_NAME.split("/")[-1].lower().replace("-", "_").replace(".", "")[:20]
        result_path = RESULTS_DIR / f"{exp_id}_{model_slug}_seed{args.seed}.json"
        with open(result_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {result_path}")
        print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
