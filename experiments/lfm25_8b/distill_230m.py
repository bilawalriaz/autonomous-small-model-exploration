#!/usr/bin/env python3
"""
On-Policy Distillation: Gemma 4 24B (teacher) → LFM2.5-230M (student)

Teacher: Gemma 4 26B A4B on Mac LM Studio (100.100.61.28:1234)
Student: LFM2.5-230M via transformers on aero (RTX 2070 Super)

Based on: Thinking Machines Lab "On-Policy Distillation"
Core: Student generates → teacher grades each token → reverse KL → train

Usage:
    python distill_230m.py generate    # Generate student trajectories
    python distill_230m.py score       # Score with teacher
    python distill_230m.py train       # LoRA training with reverse KL
    python distill_230m.py full        # Run everything
    python distill_230m.py test        # Quick end-to-end test (3 prompts)
"""

import argparse
import json
import os
import sys
import time
import math
import torch
import requests
from pathlib import Path
from typing import Optional

# ======================================================================
# CONFIG
# ======================================================================
TEACHER_URL = "http://100.100.61.28:1234/v1/chat/completions"
TEACHER_MODEL = "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp"
STUDENT_MODEL = "LiquidAI/LFM2.5-230M"

# Paths
DATA_DIR = Path(__file__).parent / "distill_230m_data"
RESULTS_DIR = Path("/home/billz/results/distill_230m")
MODEL_DIR = RESULTS_DIR / "finetuned"

# Generation config
TEMPERATURE = 0.7
MAX_TOKENS_STUDENT = 512  # 230M generates faster, less thinking
MAX_TOKENS_TEACHER = 1024
NUM_SAMPLES = 3  # Trajectories per prompt

# Training config
LR = 2e-5
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 4
MAX_SEQ_LEN = 1024

# ======================================================================
# PROMPTS (compact set for fast iteration)
# ======================================================================
PROMPTS = {
    "math": [
        "Solve: 3x + 7 = 22. Show your work.",
        "What is 15% of 240?",
        "Simplify: (2^3)(2^4)",
        "What is the GCD of 48 and 18?",
        "Solve for x: x^2 - 5x + 6 = 0",
        "What is the sum of the first 20 natural numbers?",
        "Convert 10110 binary to decimal.",
        "What is 7! ?",
        "If f(x) = 2x + 3, what is f(f(1))?",
        "What is the LCM of 12 and 18?",
    ],
    "code": [
        "Write a Python function to check if a number is prime.",
        "Write a Python function to reverse a string.",
        "Write a Python function to find the factorial of n.",
        "Write a Python function to check if a string is a palindrome.",
        "Write a Python function to count vowels in a string.",
    ],
    "reasoning": [
        "If all cats are animals, and all animals are living things, are all cats living things? Explain step by step.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "What comes next: 2, 6, 12, 20, 30, ?",
        "Explain why the sky is blue in one paragraph.",
    ],
    "instruction": [
        "List exactly 5 fruits, numbered 1-5.",
        "Explain quantum computing to a 10-year-old.",
        "Write a haiku about the ocean (5-7-5 syllables).",
        "Compare Python and JavaScript in 3 bullet points.",
    ],
}

# ======================================================================
# STUDENT (HuggingFace transformers)
# ======================================================================

class StudentModel:
    """LFM2.5-230M via HuggingFace transformers."""

    def __init__(self, model_name: str = STUDENT_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading student: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Student loaded on {self.model.device}")

    def generate_with_logprobs(self, prompt: str, max_new_tokens: int = 256,
                               temperature: float = 0.7) -> dict:
        """Generate text and return per-token logprobs."""
        # Format prompt
        chat = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)

        prompt_len = input_ids.shape[1]

        # Generate with do_sample=True for diversity
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs from scores
        token_logprobs = []
        if outputs.scores:
            for i, score in enumerate(outputs.scores):
                probs = torch.softmax(score, dim=-1)
                token_id = generated_ids[i].item()
                logprob = torch.log(probs[0, token_id]).item()
                token_logprobs.append({
                    "token": self.tokenizer.decode([token_id]),
                    "token_id": token_id,
                    "logprob": logprob,
                })

        return {
            "response": generated_text,
            "token_logprobs": token_logprobs,
            "tokens_generated": len(generated_ids),
            "prompt_tokens": prompt_len,
        }


# ======================================================================
# TEACHER (LM Studio API)
# ======================================================================

class TeacherModel:
    """Gemma 4 24B via LM Studio OpenAI-compatible API."""

    def __init__(self, url: str = TEACHER_URL, model: str = TEACHER_MODEL):
        self.url = url
        self.model = model
        # Test connection
        try:
            resp = requests.get(url.replace("/chat/completions", "/models"), timeout=5)
            print(f"Teacher connected: {resp.json().get('data', [{}])[0].get('id', 'unknown')}")
        except Exception as e:
            print(f"WARNING: Teacher not reachable at {url}: {e}")

    def get_logprobs(self, prompt: str, completion: str) -> list:
        """
        Get teacher logprobs for a student-generated completion.
        Uses the completions endpoint with echo=True to get logprobs for the full sequence.
        """
        # Format as chat prompt
        full_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"

        try:
            # Try completions endpoint (LM Studio supports it)
            completions_url = self.url.replace("/chat/completions", "/completions")
            payload = {
                "model": self.model,
                "prompt": full_text,
                "max_tokens": 0,
                "logprobs": True,
                "echo": True,
                "temperature": 0,
            }
            resp = requests.post(completions_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if choices:
                logprobs_info = choices[0].get("logprobs", {})
                token_logprobs = logprobs_info.get("token_logprobs", [])
                tokens = logprobs_info.get("tokens", [])

                # Find where the completion starts
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                teacher_logprobs = token_logprobs[prompt_tokens:]

                return [{"token": t, "logprob": lp}
                        for t, lp in zip(tokens[prompt_tokens:], teacher_logprobs)]

        except Exception as e:
            print(f"  Teacher completions API failed: {e}")

        # Fallback: use chat completions to score each token one by one
        # This is slower but more reliable
        try:
            return self._score_incremental(prompt, completion)
        except Exception as e2:
            print(f"  Teacher incremental scoring also failed: {e2}")
            return []

    def _score_incremental(self, prompt: str, completion: str) -> list:
        """Score each token by appending them one at a time."""
        import tiktoken

        results = []
        tokens = completion.split()  # Simple tokenization

        for i in range(len(tokens)):
            partial = " ".join(tokens[:i+1])
            msgs = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": partial}
            ]

            resp = requests.post(self.url, json={
                "model": self.model,
                "messages": msgs,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
            }, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                # Extract logprob from response
                choices = data.get("choices", [])
                if choices:
                    logprobs = choices[0].get("logprobs", {})
                    if logprobs and "content" in logprobs:
                        for lp in logprobs["content"]:
                            results.append({
                                "token": lp.get("token", ""),
                                "logprob": lp.get("logprob", 0),
                            })

        return results

    def generate(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
        """Generate a reference response from the teacher."""
        try:
            resp = requests.post(self.url, json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Teacher generation failed: {e}")
            return ""


# ======================================================================
# STEP 1: GENERATE TRAJECTORIES
# ======================================================================

def generate_trajectories(student: StudentModel, output_path: Path,
                          num_samples: int = NUM_SAMPLES):
    """Generate student trajectories for all prompts."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating {num_samples} trajectories per prompt")
    print(f"{'='*60}")

    all_prompts = []
    for cat, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append({"prompt": p, "category": cat})

    trajectories = []

    for i, item in enumerate(all_prompts):
        prompt = item["prompt"]
        cat = item["category"]
        print(f"\n[{i+1}/{len(all_prompts)}] [{cat}] {prompt[:55]}...")

        for sample_idx in range(num_samples):
            result = student.generate_with_logprobs(
                prompt, max_new_tokens=MAX_TOKENS_STUDENT,
                temperature=TEMPERATURE
            )

            traj = {
                "prompt": prompt,
                "category": cat,
                "student_response": result["response"],
                "student_logprobs": result["token_logprobs"],
                "tokens_generated": result["tokens_generated"],
                "sample_idx": sample_idx,
            }
            trajectories.append(traj)

            print(f"  Sample {sample_idx}: {result['tokens_generated']} tokens, "
                  f"{len(result['token_logprobs'])} logprobs, "
                  f"response: {result['response'][:80]}...")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    print(f"\nSaved {len(trajectories)} trajectories to {output_path}")
    return trajectories


# ======================================================================
# STEP 2: SCORE WITH TEACHER
# ======================================================================

def score_trajectories(teacher: TeacherModel, trajectories_path: Path,
                       output_path: Path):
    """Score all trajectories with the teacher model."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Scoring with teacher (Gemma 4 24B)")
    print(f"{'='*60}")

    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    scored = []
    for i, traj in enumerate(trajectories):
        prompt = traj["prompt"]
        response = traj["student_response"]

        if not response.strip() or len(traj.get("student_logprobs", [])) < 3:
            print(f"[{i+1}/{len(trajectories)}] SKIP (too short)")
            continue

        print(f"[{i+1}/{len(trajectories)}] {prompt[:50]}...", end=" ", flush=True)

        # Get teacher logprobs
        teacher_logprobs = teacher.get_logprobs(prompt, response)

        # Align lengths
        student_lps = [lp["logprob"] for lp in traj["student_logprobs"]]
        teacher_lps = [lp["logprob"] for lp in teacher_logprobs] if teacher_logprobs else []

        min_len = min(len(student_lps), len(teacher_lps))

        if min_len < 3:
            print(f"SKIP (teacher returned {len(teacher_lps)} tokens)")
            continue

        # Compute reverse KL per token: log π_student - log π_teacher
        reverse_kl = []
        for j in range(min_len):
            kl = student_lps[j] - teacher_lps[j]
            reverse_kl.append(kl)

        mean_kl = sum(reverse_kl) / len(reverse_kl)
        max_kl = max(reverse_kl)

        traj["teacher_logprobs"] = teacher_logprobs[:min_len]
        traj["reverse_kl"] = reverse_kl
        traj["mean_reverse_kl"] = mean_kl
        traj["max_reverse_kl"] = max_kl
        traj["aligned_tokens"] = min_len

        scored.append(traj)
        print(f"kl={mean_kl:.4f}, tokens={min_len}")

        time.sleep(0.3)  # Rate limit

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    # Stats
    all_kl = [s["mean_reverse_kl"] for s in scored]
    if all_kl:
        print(f"\nScoring complete:")
        print(f"  Trajectories scored: {len(scored)}")
        print(f"  Mean reverse KL: {sum(all_kl)/len(all_kl):.4f}")
        print(f"  Max mean KL: {max(all_kl):.4f}")
        print(f"  Min mean KL: {min(all_kl):.4f}")

    return scored


# ======================================================================
# STEP 3: TRAIN WITH REVERSE KL
# ======================================================================

def train_distillation(scored_path: Path, output_dir: Path):
    """Train student with LoRA weighted by reverse KL from teacher."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Training with on-policy distillation")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW

    # Load scored data
    trajectories = []
    with open(scored_path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("reverse_kl") and len(t["reverse_kl"]) >= 3:
                    trajectories.append(t)

    print(f"Loaded {len(trajectories)} scored trajectories")

    if not trajectories:
        print("ERROR: No scored trajectories")
        return

    # Load model
    print(f"Loading student: {STUDENT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    class DistillDataset(Dataset):
        def __init__(self, trajectories, tokenizer, max_len):
            self.data = []
            for traj in trajectories:
                prompt = traj["prompt"]
                response = traj["student_response"]
                kl_weights = traj["reverse_kl"]

                # Tokenize
                prompt_ids = tokenizer.encode(
                    f"<|user|>\n{prompt}\n<|assistant|>\n",
                    add_special_tokens=False
                )
                response_ids = tokenizer.encode(response, add_special_tokens=False)

                # Truncate
                total = len(prompt_ids) + len(response_ids)
                if total > max_len:
                    response_ids = response_ids[:max_len - len(prompt_ids)]

                # Trim KL weights
                kl_w = kl_weights[:len(response_ids)]

                if len(response_ids) >= 3:
                    self.data.append({
                        "input_ids": prompt_ids + response_ids,
                        "labels": [-100] * len(prompt_ids) + response_ids,
                        "kl_weights": [0.0] * len(prompt_ids) + kl_w,
                    })

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long),
                "kl_weights": torch.tensor(item["kl_weights"], dtype=torch.float),
            }

    dataset = DistillDataset(trajectories, tokenizer, MAX_SEQ_LEN)
    print(f"Dataset: {len(dataset)} examples")

    if len(dataset) == 0:
        print("ERROR: Empty dataset")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Training loop
    def compute_loss(outputs, labels, kl_weights):
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        kl_w = kl_weights[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)

        # Weight by |reverse KL| — tokens where student diverges most get higher weight
        weights = torch.abs(kl_w) + 0.01
        weights = weights / weights.sum()

        return (loss * weights).sum()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            kl_weights = batch["kl_weights"].to(model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = compute_loss(outputs, labels, kl_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if batch_idx % 5 == 0:
                print(f"  E{epoch+1} B{batch_idx}: loss={loss.item():.4f}")

        avg = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1}/{EPOCHS}: avg_loss={avg:.4f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")

    # Save training config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump({
            "student_model": STUDENT_MODEL,
            "teacher_model": TEACHER_MODEL,
            "teacher_url": TEACHER_URL,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "temperature": TEMPERATURE,
            "num_samples": NUM_SAMPLES,
            "num_trajectories": len(trajectories),
            "reverse_kl_mean": sum(t["mean_reverse_kl"] for t in trajectories) / len(trajectories),
        }, f, indent=2)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="On-Policy Distillation: Gemma 4 → LFM2.5-230M")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate student trajectories")
    sub.add_parser("score", help="Score with teacher")
    sub.add_parser("train", help="Train with distillation")
    sub.add_parser("full", help="Run full pipeline")
    sub.add_parser("test", help="Quick test (3 prompts)")

    args = parser.parse_args()

    if args.command == "generate":
        student = StudentModel()
        generate_trajectories(student, DATA_DIR / "trajectories.jsonl")

    elif args.command == "score":
        teacher = TeacherModel()
        score_trajectories(teacher, DATA_DIR / "trajectories.jsonl",
                          DATA_DIR / "scored.jsonl")

    elif args.command == "train":
        train_distillation(DATA_DIR / "scored.jsonl", MODEL_DIR)

    elif args.command in ("full", "test"):
        num_samples = 1 if args.command == "test" else NUM_SAMPLES

        # Quick test with subset
        if args.command == "test":
            global PROMPTS
            PROMPTS = {
                "math": PROMPTS["math"][:2],
                "code": PROMPTS["code"][:1],
            }
            print("TEST MODE: Using 3 prompts")

        # Step 1: Generate
        student = StudentModel()
        generate_trajectories(student, DATA_DIR / "trajectories.jsonl",
                             num_samples=num_samples)

        # Step 2: Score
        teacher = TeacherModel()
        score_trajectories(teacher, DATA_DIR / "trajectories.jsonl",
                          DATA_DIR / "scored.jsonl")

        # Step 3: Train
        train_distillation(DATA_DIR / "scored.jsonl", MODEL_DIR)

        print(f"\n{'='*60}")
        print("DISTILLATION COMPLETE")
        print(f"{'='*60}")
        print(f"Finetuned model: {MODEL_DIR}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
