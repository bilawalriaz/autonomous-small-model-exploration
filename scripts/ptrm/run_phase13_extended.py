#!/usr/bin/env python3
"""
Phase 13 Extended: Steps 3-6 from the Next Steps table.

13G: Train Q-head — lightweight MLP to close gap to oracle selection
13H: Layer-specific noise — atlas-guided noise at weak layers with higher σ
13I: LoRA adapter sensitivity — does training change noise response?
13J: Real-world extraction — messy/informal text extraction

Usage:
  python run_phase13_extended.py --experiment 13G --model LiquidAI/LFM2.5-230M --device cuda:0
  python run_phase13_extended.py --experiment 13G --model Qwen/Qwen2.5-0.5B-Instruct --device cuda:0
  python run_phase13_extended.py --all --model LiquidAI/LFM2.5-230M --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "phase13"
ADAPTERS_DIR = REPO_ROOT / "experiments" / "adapters"

MODEL_NAME = "LiquidAI/LFM2.5-230M"
USE_CHAT_TEMPLATE = False

# LFM2.5-230M atlas data
LFM_SKIP_KL = {
    0: 82.9, 1: 36.3, 2: 30.8, 3: 25.9, 4: 34.6, 5: 44.4,
    6: 8.7, 7: 7.3, 8: 10.2, 9: 7.1, 10: 17.7, 11: 8.7,
    12: 29.1, 13: 12.1,
}
LFM_LAYER_TYPES = {
    0: 'conv', 1: 'conv', 2: 'attn', 3: 'conv', 4: 'attn', 5: 'conv',
    6: 'attn', 7: 'conv', 8: 'attn', 9: 'conv', 10: 'attn', 11: 'conv',
    12: 'attn', 13: 'conv',
}
LFM_TOTAL_KL = sum(LFM_SKIP_KL.values())
LFM_WEIGHTS = {k: v / LFM_TOTAL_KL for k, v in LFM_SKIP_KL.items()}
LFM_RESIDUAL_NORMS = {
    0: 2.02, 1: 2.19, 2: 1.53, 3: 2.33, 4: 1.41, 5: 25.54,
    6: 25.54, 7: 25.56, 8: 25.5, 9: 25.5, 10: 25.5, 11: 25.5,
    12: 25.5, 13: 22.91,
}

# Qwen2.5-0.5B atlas data (from Phase 1)
QWEN_SKIP_KL = {i: 10.0 for i in range(24)}  # Placeholder — uniform for now
QWEN_LAYER_TYPES = {i: 'attn' for i in range(24)}

# Structured extraction prompts
EVAL_PROMPTS = [
    {"prompt": "Extract the name, age, and city from: John Smith is 35 years old and lives in London.", "expected": {"name": "John Smith", "age": 35, "city": "London"}},
    {"prompt": "Extract the name, age, and city from: Maria Garcia, aged 28, resides in Madrid.", "expected": {"name": "Maria Garcia", "age": 28, "city": "Madrid"}},
    {"prompt": "Extract the name, age, and city from: Ahmed Hassan (42) from Cairo.", "expected": {"name": "Ahmed Hassan", "age": 42, "city": "Cairo"}},
    {"prompt": "Extract the product, price, and currency from: The iPhone 15 costs $999.", "expected": {"product": "iPhone 15", "price": 999, "currency": "$"}},
    {"prompt": "Extract the product, price, and currency from: Samsung Galaxy S24 priced at 849 GBP.", "expected": {"product": "Samsung Galaxy S24", "price": 849, "currency": "GBP"}},
    {"prompt": "Extract the product, price, and currency from: Sony WH-1000XM5 headphones retail for 349 EUR.", "expected": {"product": "Sony WH-1000XM5", "price": 349, "currency": "EUR"}},
    {"prompt": "Extract the title, director, and year from: Inception directed by Christopher Nolan in 2010.", "expected": {"title": "Inception", "director": "Christopher Nolan", "year": 2010}},
    {"prompt": "Extract the title, director, and year from: The Matrix, directed by the Wachowskis, released 1999.", "expected": {"title": "The Matrix", "director": "Wachowskis", "year": 1999}},
    {"prompt": "Extract the city, country, and population from: Tokyo, Japan has a population of 13,960,000.", "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000}},
    {"prompt": "Extract the city, country, and population from: Mumbai in India: 12,442,373 people.", "expected": {"city": "Mumbai", "country": "India", "population": 12442373}},
    {"prompt": "Extract the name, role, and company from: Sarah Chen is the CTO at Stripe.", "expected": {"name": "Sarah Chen", "role": "CTO", "company": "Stripe"}},
    {"prompt": "Extract the name, role, and company from: Dr. James Wilson, VP of Engineering, works at Anthropic.", "expected": {"name": "Dr. James Wilson", "role": "VP of Engineering", "company": "Anthropic"}},
    {"prompt": "Extract the date, event, and location from: On March 15, 2025, the GDC conference was held in San Francisco.", "expected": {"date": "March 15, 2025", "event": "GDC conference", "location": "San Francisco"}},
    {"prompt": "Extract the date, event, and location from: WWDC 2025 took place June 9 in Cupertino.", "expected": {"date": "June 9, 2025", "event": "WWDC 2025", "location": "Cupertino"}},
    {"prompt": "Extract the make, model, year, and price from: 2024 Tesla Model 3 — $38,990.", "expected": {"make": "Tesla", "model": "Model 3", "year": 2024, "price": 38990}},
    {"prompt": "Extract the make, model, year, and price from: A 2023 Toyota Camry LE costs $26,420.", "expected": {"make": "Toyota", "model": "Camry LE", "year": 2023, "price": 26420}},
    {"prompt": "Extract the author, title, and year from: '1984' by George Orwell, published 1949.", "expected": {"author": "George Orwell", "title": "1984", "year": 1949}},
    {"prompt": "Extract the latitude, longitude, and city from: 48.8566, 2.3522 — Paris.", "expected": {"latitude": 48.8566, "longitude": 2.3522, "city": "Paris"}},
    {"prompt": "Extract the language, paradigm, and year from: Python, multi-paradigm, created in 1991.", "expected": {"language": "Python", "paradigm": "multi-paradigm", "year": 1991}},
    {"prompt": "Extract the temperature, condition, and city from: London today: 15°C, cloudy.", "expected": {"temperature": 15, "condition": "cloudy", "city": "London"}},
    {"prompt": "Extract the drug, dosage, and frequency from: Take 500mg of Amoxicillin twice daily.", "expected": {"drug": "Amoxicillin", "dosage": "500mg", "frequency": "twice daily"}},
    {"prompt": "Extract the IP, port, and protocol from: Server at 192.168.1.100:8080 using HTTPS.", "expected": {"ip": "192.168.1.100", "port": 8080, "protocol": "HTTPS"}},
    {"prompt": "Extract the stock, price, and change from: AAPL trading at $185.42, up 2.3%.", "expected": {"stock": "AAPL", "price": 185.42, "change": "+2.3%"}},
    {"prompt": "Extract the email, role, and company from: Contact jane.doe@meta.com, Senior Engineer at Meta.", "expected": {"email": "jane.doe@meta.com", "role": "Senior Engineer", "company": "Meta"}},
    {"prompt": "Extract the endpoint, method, and status from: GET /api/users returned 200 OK.", "expected": {"endpoint": "/api/users", "method": "GET", "status": 200}},
    {"prompt": "Extract the name, breed, and age from: Buddy is a 5-year-old Golden Retriever.", "expected": {"name": "Buddy", "breed": "Golden Retriever", "age": 5}},
    {"prompt": "Extract the planet, distance, and type from: Mars is 225 million km away, a terrestrial planet.", "expected": {"planet": "Mars", "distance": "225 million km", "type": "terrestrial"}},
    {"prompt": "Extract the airport, code, and city from: Heathrow Airport (LHR) in London.", "expected": {"airport": "Heathrow Airport", "code": "LHR", "city": "London"}},
    {"prompt": "Extract the university, location, and ranking from: MIT in Cambridge, ranked #1.", "expected": {"university": "MIT", "location": "Cambridge", "ranking": 1}},
    {"prompt": "Extract the OS, version, and release from: Ubuntu 24.04 LTS released April 2024.", "expected": {"os": "Ubuntu", "version": "24.04 LTS", "release": "April 2024"}},
]

# Real-world messy prompts for Step 6
MESSY_PROMPTS = [
    {"prompt": "pull name age city frm this: john smith 35 london", "expected": {"name": "John Smith", "age": 35, "city": "London"}},
    {"prompt": "Product: iphone15 Price:$999 Currency:USD — extract these fields", "expected": {"product": "iPhone 15", "price": 999, "currency": "USD"}},
    {"prompt": "get title dir yr — inception chris nolan 2010", "expected": {"title": "Inception", "director": "Chris Nolan", "year": 2010}},
    {"prompt": "tokyo japan pop 13.96M — extract city country population", "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000}},
    {"prompt": "name: sarah chen, cto @ stripe — pull name role company", "expected": {"name": "Sarah Chen", "role": "CTO", "company": "Stripe"}},
    {"prompt": "wwdc25 was june 9 in cupertino — date event loc", "expected": {"date": "June 9, 2025", "event": "WWDC 2025", "location": "Cupertino"}},
    {"prompt": "tesla model 3 2024 costs 38990 usd — make model year price", "expected": {"make": "Tesla", "model": "Model 3", "year": 2024, "price": 38990}},
    {"prompt": "orwell wrote 1984 in 1949 — author title year", "expected": {"author": "George Orwell", "title": "1984", "year": 1949}},
    {"prompt": "paris coords 48.8566, 2.3522 — lat long city", "expected": {"latitude": 48.8566, "longitude": 2.3522, "city": "Paris"}},
    {"prompt": "python lang, multi-paradigm, 1991 — extract language paradigm year", "expected": {"language": "Python", "paradigm": "multi-paradigm", "year": 1991}},
    {"prompt": "its 15C and cloudy in london rn — temp condition city", "expected": {"temperature": 15, "condition": "cloudy", "city": "London"}},
    {"prompt": "take amox 500mg 2x daily — drug dose freq", "expected": {"drug": "Amoxicillin", "dosage": "500mg", "frequency": "twice daily"}},
    {"prompt": "server 192.168.1.100:8080 HTTPS — ip port protocol", "expected": {"ip": "192.168.1.100", "port": 8080, "protocol": "HTTPS"}},
    {"prompt": "aapl at 185.42, +2.3% — stock price change", "expected": {"stock": "AAPL", "price": 185.42, "change": "+2.3%"}},
    {"prompt": "jane.doe@meta.com sr eng at meta — email role company", "expected": {"email": "jane.doe@meta.com", "role": "Senior Engineer", "company": "Meta"}},
    {"prompt": "GET /api/users -> 200 OK — endpoint method status", "expected": {"endpoint": "/api/users", "method": "GET", "status": 200}},
    {"prompt": "buddy, golden retriever, 5 yrs old — name breed age", "expected": {"name": "Buddy", "breed": "Golden Retriever", "age": 5}},
    {"prompt": "mumbai india 12.4M ppl — city country population", "expected": {"city": "Mumbai", "country": "India", "population": 12442373}},
    {"prompt": "sammy galaxy s24 is £849 — product price currency", "expected": {"product": "Samsung Galaxy S24", "price": 849, "currency": "GBP"}},
    {"prompt": "matrix by wachowskis 1999 — title director year", "expected": {"title": "The Matrix", "director": "Wachowskis", "year": 1999}},
    {"prompt": "LHR heathrow in london — airport code city", "expected": {"airport": "Heathrow Airport", "code": "LHR", "city": "London"}},
    {"prompt": "mars 225M km away, terrestrial — planet distance type", "expected": {"planet": "Mars", "distance": "225 million km", "type": "terrestrial"}},
    {"prompt": "MIT cambridge ranked #1 — uni location ranking", "expected": {"university": "MIT", "location": "Cambridge", "ranking": 1}},
    {"prompt": "ubuntu 24.04 lts came out apr 2024 — os version release", "expected": {"os": "Ubuntu", "version": "24.04 LTS", "release": "April 2024"}},
    {"prompt": "parasite by bong joon-ho 2019 — title director year", "expected": {"title": "Parasite", "director": "Bong Joon-ho", "year": 2019}},
    {"prompt": "gsk olympics open ceremony july 26 2024 paris — date event loc", "expected": {"date": "July 26, 2024", "event": "Olympics opening ceremony", "location": "Paris"}},
    {"prompt": "nvda $875.28 +4.1% — stock price change", "expected": {"stock": "NVDA", "price": 875.28, "change": "+4.1%"}},
    {"prompt": "ibuprofen 200mg q6h — drug dose freq", "expected": {"drug": "Ibuprofen", "dosage": "200mg", "frequency": "every 6 hours"}},
    {"prompt": "bob@openai.com research sci at openai — email role company", "expected": {"email": "bob@openai.com", "role": "Research Scientist", "company": "OpenAI"}},
    {"prompt": "DELETE /api/items/42 got 204 — endpoint method statuscode", "expected": {"endpoint": "/api/items/42", "method": "DELETE", "status": 204}},
]


# ---------------------------------------------------------------------------
# Noise injection hooks
# ---------------------------------------------------------------------------

def make_noise_hook(sigma, noise_seed=None):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if noise_seed is not None:
            gen = torch.Generator(device=hidden.device)
            gen.manual_seed(noise_seed)
            noise = torch.randn(hidden.shape, generator=gen, device=hidden.device, dtype=hidden.dtype)
        else:
            noise = torch.randn_like(hidden)
        hidden = hidden + sigma * noise
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook_fn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name=None, device="cuda:0"):
    global MODEL_NAME, USE_CHAT_TEMPLATE
    if model_name:
        MODEL_NAME = model_name
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def load_lora_model(adapter_path, device="cuda:0"):
    """Load base model + LoRA adapter."""
    from peft import PeftModel
    print(f"Loading base model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"LoRA model loaded. VRAM: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------

def format_extraction_prompt(item, tokenizer=None):
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


def generate_noisy_rollout(model, tokenizer, prompt, sigma,
                           max_new_tokens=64, rollout_seed=None, temp=0.2, top_p=0.9):
    """Generate one rollout with embedding noise."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if rollout_seed is not None:
        set_seed(rollout_seed)

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

    h = model.get_input_embeddings().register_forward_hook(embed_noise_hook)
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

    with torch.no_grad():
        logits = model(full_output).logits[0]
        prompt_logits = logits[:prompt_len - 1].float()
        prompt_labels = full_output[0, 1:prompt_len]
        prompt_nll = F.cross_entropy(prompt_logits, prompt_labels).item()
        first_gen_logit = logits[prompt_len - 1].float()
        max_conf = F.softmax(first_gen_logit / temp, dim=-1).max().item()

    return {"text": text, "prompt_nll": prompt_nll, "max_conf": max_conf}


def generate_layer_specific_rollout(model, tokenizer, prompt, layer_sigmas,
                                     max_new_tokens=64, rollout_seed=None, temp=0.2, top_p=0.9):
    """Generate with per-layer noise (different σ per layer)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if rollout_seed is not None:
        set_seed(rollout_seed)

    hooks = []
    for layer_idx, sigma in layer_sigmas.items():
        if sigma > 0:
            hook = make_noise_hook(sigma, noise_seed=(rollout_seed + layer_idx) if rollout_seed else None)
            h = model.model.layers[layer_idx].register_forward_hook(hook)
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

    text = tokenizer.decode(full_output[0][prompt_len:], skip_special_tokens=True).strip()

    with torch.no_grad():
        logits = model(full_output).logits[0]
        prompt_logits = logits[:prompt_len - 1].float()
        prompt_labels = full_output[0, 1:prompt_len]
        prompt_nll = F.cross_entropy(prompt_logits, prompt_labels).item()
        first_gen_logit = logits[prompt_len - 1].float()
        max_conf = F.softmax(first_gen_logit / temp, dim=-1).max().item()

    return {"text": text, "prompt_nll": prompt_nll, "max_conf": max_conf}


def generate_baseline(model, tokenizer, prompt, max_new_tokens=64, temp=0.2, top_p=0.9):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temp, top_p=top_p,
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_extraction(text, expected):
    hits = 0
    total = len(expected)
    for key, val in expected.items():
        val_str = str(val).lower()
        if val_str in text.lower():
            hits += 1
    field_recall = hits / max(total, 1)
    exact = 1.0 if field_recall == 1.0 else 0.0
    try:
        json.loads(text)
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        json_valid = False
    return {"field_recall": field_recall, "exact_match": exact, "json_valid": json_valid}


def select_oracle(rollouts, expected):
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


def select_lowest_nll(rollouts):
    best_idx = min(range(len(rollouts)), key=lambda i: rollouts[i]["prompt_nll"])
    return rollouts[best_idx], best_idx


def select_highest_conf(rollouts):
    best_idx = max(range(len(rollouts)), key=lambda i: rollouts[i]["max_conf"])
    return rollouts[best_idx], best_idx


def select_random(rollouts, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(rollouts))
    return rollouts[idx], idx


# ---------------------------------------------------------------------------
# Q-head model
# ---------------------------------------------------------------------------

class QHead(nn.Module):
    """Lightweight MLP for rollout selection. Input: feature vector per rollout."""

    def __init__(self, input_dim=7, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_features(rollout):
    """Extract feature vector from a rollout for Q-head training."""
    text = rollout["text"]
    return [
        rollout["prompt_nll"],
        rollout["max_conf"],
        len(text),                                          # text length
        sum(1 for c in text if c.isdigit()),                # digit count
        1.0 if '{' in text and '}' in text else 0.0,       # has JSON braces
        len(text.split()),                                  # word count
        sum(1 for c in text if c.isupper()),                # uppercase count
    ]


# ---------------------------------------------------------------------------
# Experiment 13G: Train Q-head
# ---------------------------------------------------------------------------

def run_13G(model, tokenizer, args):
    """Train a Q-head MLP to predict correct rollout. 5-fold CV."""
    print("\n=== Experiment 13G: Q-Head Training ===")

    K = 20  # Rollouts per prompt
    sigma = 0.01
    prompts = EVAL_PROMPTS[:30]
    base_seed = args.seed

    # Phase 1: Generate training data
    print(f"\nGenerating K={K} rollouts for {len(prompts)} prompts...")
    all_data = []  # List of (features, label, prompt_idx)

    for i, item in enumerate(prompts):
        prompt_text = format_extraction_prompt(item, tokenizer)
        for k in range(K):
            rollout = generate_noisy_rollout(
                model, tokenizer, prompt_text, sigma=sigma,
                rollout_seed=base_seed * 100000 + i * 1000 + k,
            )
            eval_result = evaluate_extraction(rollout["text"], item["expected"])
            features = extract_features(rollout)
            all_data.append({
                "features": features,
                "label": eval_result["exact_match"],
                "field_recall": eval_result["field_recall"],
                "prompt_idx": i,
                "rollout": rollout,
            })
        if (i + 1) % 10 == 0:
            print(f"  Generated data for {i+1}/{len(prompts)} prompts")

    # Phase 2: 5-fold cross-validation
    print("\nTraining Q-head with 5-fold CV...")
    X = np.array([d["features"] for d in all_data], dtype=np.float32)
    y = np.array([d["label"] for d in all_data], dtype=np.float32)
    prompt_ids = np.array([d["prompt_idx"] for d in all_data])

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Group by prompt for proper CV (no prompt leakage)
    unique_prompts = np.unique(prompt_ids)
    np.random.seed(base_seed)
    np.random.shuffle(unique_prompts)
    folds = np.array_split(unique_prompts, 5)

    fold_results = []
    for fold_idx, test_prompts in enumerate(folds):
        train_mask = ~np.isin(prompt_ids, test_prompts)
        test_mask = np.isin(prompt_ids, test_prompts)

        X_train = torch.tensor(X_norm[train_mask], dtype=torch.float32)
        y_train = torch.tensor(y[train_mask], dtype=torch.float32)
        X_test = torch.tensor(X_norm[test_mask], dtype=torch.float32)
        y_test = torch.tensor(y[test_mask], dtype=torch.float32)

        # Train
        qhead = QHead(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(qhead.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        qhead.train()
        for epoch in range(100):
            optimizer.zero_grad()
            logits = qhead(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        qhead.eval()
        with torch.no_grad():
            test_logits = qhead(X_test)
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs > 0.5).float()
            accuracy = (test_preds == y_test).float().mean().item()

            # Per-prompt accuracy: for each prompt, does Q-head pick the correct rollout?
            test_data = [all_data[j] for j in range(len(all_data)) if test_mask[j]]
            prompt_accs = []
            for pid in np.unique([d["prompt_idx"] for d in test_data]):
                prompt_data = [d for d in test_data if d["prompt_idx"] == pid]
                prompt_indices = [j for j in range(len(test_data)) if test_data[j]["prompt_idx"] == pid]
                if len(prompt_indices) < 2:
                    continue
                prompt_probs = test_probs[prompt_indices]
                best_idx = prompt_probs.argmax().item()
                best_is_correct = prompt_data[best_idx]["label"] == 1.0
                prompt_accs.append(1.0 if best_is_correct else 0.0)

            prompt_accuracy = np.mean(prompt_accs) if prompt_accs else 0.0

        fold_results.append({
            "fold": fold_idx,
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "binary_accuracy": accuracy,
            "prompt_selection_accuracy": prompt_accuracy,
            "positive_rate": float(y_test.mean()),
        })
        print(f"  Fold {fold_idx}: binary_acc={accuracy:.3f}, prompt_select_acc={prompt_accuracy:.3f}")

    # Phase 3: Train final Q-head on all data
    print("\nTraining final Q-head on all data...")
    X_all = torch.tensor(X_norm, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.float32)

    qhead_final = QHead(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(qhead_final.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    qhead_final.train()
    for epoch in range(150):
        optimizer.zero_grad()
        logits = qhead_final(X_all)
        loss = criterion(logits, y_all)
        loss.backward()
        optimizer.step()

    # Phase 4: Compare strategies on held-out prompts (use last 10 prompts as held-out)
    print("\nComparing strategies on held-out prompts...")
    heldout_prompts = EVAL_PROMPTS[30:40] if len(EVAL_PROMPTS) > 30 else EVAL_PROMPTS[25:30]
    K_eval = 10

    strategies = {
        "random": [],
        "lowest_nll": [],
        "highest_conf": [],
        "q_head": [],
        "oracle": [],
        "baseline": [],
    }

    for i, item in enumerate(heldout_prompts):
        prompt_text = format_extraction_prompt(item, tokenizer)

        # Baseline
        baseline = generate_baseline(model, tokenizer, prompt_text)
        baseline_eval = evaluate_extraction(baseline["text"], item["expected"])
        strategies["baseline"].append(baseline_eval["exact_match"])

        # Generate K rollouts
        rollouts = []
        for k in range(K_eval):
            rollout = generate_noisy_rollout(
                model, tokenizer, prompt_text, sigma=sigma,
                rollout_seed=base_seed * 200000 + i * 1000 + k,
            )
            rollout["eval"] = evaluate_extraction(rollout["text"], item["expected"])
            rollouts.append(rollout)

        # Random
        r, _ = select_random(rollouts, seed=base_seed)
        strategies["random"].append(r["eval"]["exact_match"])

        # Lowest NLL
        r, _ = select_lowest_nll(rollouts)
        strategies["lowest_nll"].append(r["eval"]["exact_match"])

        # Highest confidence
        r, _ = select_highest_conf(rollouts)
        strategies["highest_conf"].append(r["eval"]["exact_match"])

        # Q-head
        qhead_final.eval()
        with torch.no_grad():
            feats = np.array([extract_features(r) for r in rollouts], dtype=np.float32)
            feats_norm = (feats - X_mean) / X_std
            probs = torch.sigmoid(qhead_final(torch.tensor(feats_norm, dtype=torch.float32)))
            best_idx = probs.argmax().item()
        strategies["q_head"].append(rollouts[best_idx]["eval"]["exact_match"])

        # Oracle
        r, _ = select_oracle(rollouts, item["expected"])
        strategies["oracle"].append(r["eval"]["exact_match"])

    # Summary
    summary = {}
    for name, results in strategies.items():
        summary[name] = {
            "accuracy": float(np.mean(results)),
            "n_correct": int(sum(results)),
            "n_total": len(results),
        }

    print("\n=== Q-Head Results ===")
    for name, s in summary.items():
        print(f"  {name:15s}: {s['accuracy']:.3f} ({s['n_correct']}/{s['n_total']})")

    cv_summary = {
        "mean_prompt_accuracy": float(np.mean([f["prompt_selection_accuracy"] for f in fold_results])),
        "mean_binary_accuracy": float(np.mean([f["binary_accuracy"] for f in fold_results])),
        "std_prompt_accuracy": float(np.std([f["prompt_selection_accuracy"] for f in fold_results])),
    }

    return {
        "config": {"K": K, "sigma": sigma, "n_train_prompts": len(prompts), "n_features": X.shape[1]},
        "cross_validation": fold_results,
        "cv_summary": cv_summary,
        "heldout_comparison": summary,
        "feature_names": ["prompt_nll", "max_conf", "text_length", "digit_count", "has_json", "word_count", "upper_count"],
        "feature_stats": {"mean": X_mean.tolist(), "std": X_std.tolist()},
        "positive_rate": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Experiment 13H: Layer-specific noise
# ---------------------------------------------------------------------------

def run_13H(model, tokenizer, args):
    """Test atlas-guided layer-specific noise with higher σ at weak layers."""
    print("\n=== Experiment 13H: Layer-Specific Noise ===")

    K = 10
    prompts = EVAL_PROMPTS[:30]
    base_seed = args.seed

    is_lfm = "lfm" in MODEL_NAME.lower() or "liquid" in MODEL_NAME.lower()
    is_qwen = "qwen" in MODEL_NAME.lower()

    if is_lfm:
        n_layers = 14
        # Hub: L0 (strongest), Weak: L6-L11 (lowest KL)
        hub_layers = [0]
        weak_layers = [6, 7, 8, 9, 10, 11]
        mid_layers = [1, 2, 3, 4, 5, 12, 13]
    elif is_qwen:
        n_layers = 24
        hub_layers = [0, 1]
        weak_layers = list(range(12, 20))
        mid_layers = [i for i in range(n_layers) if i not in hub_layers and i not in weak_layers]
    else:
        n_layers = 14
        hub_layers = [0]
        weak_layers = [6, 7, 8, 9, 10, 11]
        mid_layers = [1, 2, 3, 4, 5, 12, 13]

    conditions = {
        "baseline_no_noise": {"type": "baseline"},
        "uniform_s001": {"type": "uniform", "sigma": 0.01},
        "uniform_s002": {"type": "uniform", "sigma": 0.02},
        "hub_heavy": {"type": "layered", "hub_sigma": 0.02, "mid_sigma": 0.005, "weak_sigma": 0.002},
        "weak_heavy": {"type": "layered", "hub_sigma": 0.002, "mid_sigma": 0.01, "weak_sigma": 0.03},
        "progressive": {"type": "layered", "hub_sigma": 0.003, "mid_sigma": 0.008, "weak_sigma": 0.015},
        "inverse_atlas": {"type": "layered", "hub_sigma": 0.002, "mid_sigma": 0.015, "weak_sigma": 0.03},
        "targeted_weak_s03": {"type": "targeted", "layers": weak_layers, "sigma": 0.03},
        "targeted_weak_s05": {"type": "targeted", "layers": weak_layers, "sigma": 0.05},
    }

    results = {}

    for cond_name, cond in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        exact_hits = 0
        field_recall_sum = 0.0
        n = 0

        for i, item in enumerate(prompts):
            prompt_text = format_extraction_prompt(item, tokenizer)
            best_recall = 0.0
            best_exact = 0.0

            if cond["type"] == "baseline":
                rollout = generate_baseline(model, tokenizer, prompt_text)
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
            else:
                for k in range(K):
                    if cond["type"] == "uniform":
                        rollout = generate_noisy_rollout(
                            model, tokenizer, prompt_text,
                            sigma=cond["sigma"],
                            rollout_seed=base_seed * 100000 + hash(cond_name) % 10000 + i * 100 + k,
                        )
                    elif cond["type"] == "layered":
                        layer_sigmas = {}
                        for l in hub_layers:
                            layer_sigmas[l] = cond["hub_sigma"]
                        for l in mid_layers:
                            layer_sigmas[l] = cond["mid_sigma"]
                        for l in weak_layers:
                            layer_sigmas[l] = cond["weak_sigma"]
                        rollout = generate_layer_specific_rollout(
                            model, tokenizer, prompt_text,
                            layer_sigmas=layer_sigmas,
                            rollout_seed=base_seed * 100000 + hash(cond_name) % 10000 + i * 100 + k,
                        )
                    elif cond["type"] == "targeted":
                        layer_sigmas = {l: cond["sigma"] for l in cond["layers"]}
                        rollout = generate_layer_specific_rollout(
                            model, tokenizer, prompt_text,
                            layer_sigmas=layer_sigmas,
                            rollout_seed=base_seed * 100000 + hash(cond_name) % 10000 + i * 100 + k,
                        )
                    else:
                        continue

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
            "config": cond,
        }

    return {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "hub_layers": hub_layers,
        "weak_layers": weak_layers,
        "K": K,
        "n_prompts": len(prompts),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Experiment 13I: LoRA adapter sensitivity
# ---------------------------------------------------------------------------

def run_13I(model, tokenizer, args):
    """Test noise sensitivity with LoRA adapters."""
    print("\n=== Experiment 13I: LoRA Adapter Noise Sensitivity ===")

    K = 10
    sigma_values = [0.0, 0.005, 0.01, 0.015, 0.02]
    prompts = EVAL_PROMPTS[:20]
    base_seed = args.seed

    # Check available adapters
    is_lfm = "lfm" in MODEL_NAME.lower() or "liquid" in MODEL_NAME.lower()
    if is_lfm:
        adapter_candidates = [
            ("base_model", None),
            ("atlas_full", ADAPTERS_DIR / "lfm2_230m" / "atlas_full" / "adapter"),
            ("all_linear", ADAPTERS_DIR / "lfm2_230m" / "all_linear" / "adapter"),
        ]
    else:
        adapter_candidates = [
            ("base_model", None),
        ]
        # Check for Qwen adapters
        qwen_adapter_dirs = list(ADAPTERS_DIR.glob("qwen*"))
        for d in qwen_adapter_dirs[:2]:
            adapter_path = d / "adapter"
            if adapter_path.exists():
                adapter_candidates.append((d.name, adapter_path))

    results = {}

    for adapter_name, adapter_path in adapter_candidates:
        print(f"\n--- Adapter: {adapter_name} ---")

        if adapter_path is None:
            # Use the already-loaded base model
            active_model = model
        else:
            if not adapter_path.exists():
                print(f"  Adapter not found at {adapter_path}, skipping")
                continue
            active_model = load_lora_model(adapter_path, device=args.device)

        adapter_results = {}
        for sigma in sigma_values:
            exact_hits = 0
            field_recall_sum = 0.0
            n = 0

            for i, item in enumerate(prompts):
                prompt_text = format_extraction_prompt(item, tokenizer)

                if sigma == 0.0:
                    rollout = generate_baseline(active_model, tokenizer, prompt_text)
                    eval_result = evaluate_extraction(rollout["text"], item["expected"])
                else:
                    best_recall = 0.0
                    best_exact = 0.0
                    for k in range(K):
                        rollout = generate_noisy_rollout(
                            active_model, tokenizer, prompt_text,
                            sigma=sigma,
                            rollout_seed=base_seed * 300000 + hash(adapter_name) % 10000 + i * 100 + k,
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
            print(f"  σ={sigma:.3f}: accuracy={accuracy:.3f} (recall={avg_recall:.3f})")

            adapter_results[f"sigma_{sigma}"] = {
                "sigma": sigma,
                "accuracy": accuracy,
                "avg_field_recall": avg_recall,
                "K": 1 if sigma == 0 else K,
            }

        results[adapter_name] = adapter_results

        # Clean up LoRA model if we loaded one
        if adapter_path is not None:
            del active_model
            torch.cuda.empty_cache()

    return {
        "model": MODEL_NAME,
        "K": K,
        "sigma_values": sigma_values,
        "n_prompts": len(prompts),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Experiment 13J: Real-world extraction
# ---------------------------------------------------------------------------

def run_13J(model, tokenizer, args):
    """Test on messy, informal real-world text."""
    print("\n=== Experiment 13J: Real-World Extraction ===")

    K = 5
    sigma = 0.01
    base_seed = args.seed

    # Compare clean vs messy prompts
    all_results = {}

    for prompt_type, prompt_set in [("clean", EVAL_PROMPTS[:30]), ("messy", MESSY_PROMPTS)]:
        print(f"\n--- Prompt type: {prompt_type} ({len(prompt_set)} prompts) ---")

        baseline_hits = 0
        noisy_hits = 0
        n = 0

        for i, item in enumerate(prompt_set):
            prompt_text = format_extraction_prompt(item, tokenizer)

            # Baseline
            baseline = generate_baseline(model, tokenizer, prompt_text)
            baseline_eval = evaluate_extraction(baseline["text"], item["expected"])

            # Noisy best-of-K
            best_recall = 0.0
            best_exact = 0.0
            best_text = ""
            for k in range(K):
                rollout = generate_noisy_rollout(
                    model, tokenizer, prompt_text, sigma=sigma,
                    rollout_seed=base_seed * 400000 + i * 1000 + k,
                )
                eval_result = evaluate_extraction(rollout["text"], item["expected"])
                if eval_result["field_recall"] > best_recall:
                    best_recall = eval_result["field_recall"]
                    best_exact = eval_result["exact_match"]
                    best_text = rollout["text"]

            baseline_hits += baseline_eval["exact_match"]
            noisy_hits += best_exact
            n += 1

        baseline_acc = baseline_hits / n
        noisy_acc = noisy_hits / n
        improvement = noisy_acc - baseline_acc
        relative_boost = noisy_acc / max(baseline_acc, 0.001)

        print(f"  Baseline: {baseline_acc:.3f} ({baseline_hits}/{n})")
        print(f"  Noisy K={K}: {noisy_acc:.3f} ({noisy_hits}/{n})")
        print(f"  Δ: {improvement:+.3f} ({relative_boost:.1f}×)")

        all_results[prompt_type] = {
            "n_prompts": n,
            "K": K,
            "sigma": sigma,
            "baseline_accuracy": baseline_acc,
            "baseline_correct": baseline_hits,
            "noisy_accuracy": noisy_acc,
            "noisy_correct": noisy_hits,
            "improvement_pp": improvement,
            "relative_boost": relative_boost,
        }

    return {
        "model": MODEL_NAME,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "13G": ("Q-Head Training", run_13G),
    "13H": ("Layer-Specific Noise", run_13H),
    "13I": ("LoRA Adapter Sensitivity", run_13I),
    "13J": ("Real-World Extraction", run_13J),
}


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Extended: Steps 3-6")
    parser.add_argument("--experiment", "-e", choices=list(EXPERIMENT_MAP.keys()),
                        default=None, help="Which experiment to run")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name. Default: LiquidAI/LFM2.5-230M")
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

        model_slug = MODEL_NAME.split("/")[-1].lower().replace("-", "_").replace(".", "")[:20]
        result_path = RESULTS_DIR / f"{exp_id}_{model_slug}_seed{args.seed}.json"
        with open(result_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {result_path}")
        print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
