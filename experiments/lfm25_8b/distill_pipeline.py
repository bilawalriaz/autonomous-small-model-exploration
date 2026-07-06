#!/usr/bin/env python3
"""
On-Policy Distillation Pipeline for LFM2.5-8B-A1B
Teacher: Qwen3.6-35B-A3B on Mac (MLX or llama.cpp)
Student: LFM2.5-8B-A1B on aero (llama.cpp)

Based on: Thinking Machines Lab "On-Policy Distillation" (Kevin Lu, Oct 2025)
Core idea: Student generates trajectories, teacher grades every token,
student learns from its own mistakes via reverse KL divergence.

Pipeline:
1. Generate trajectories from student (with logprobs)
2. Score trajectories with teacher (get teacher logprobs)
3. Compute per-token reverse KL
4. Train student with LoRA to minimize reverse KL

Usage:
    # Step 1: Generate trajectories
    python distill_pipeline.py generate --prompts data/math_prompts.jsonl --output trajectories.jsonl

    # Step 2: Score with teacher
    python distill_pipeline.py score --trajectories trajectories.jsonl --teacher-url http://MAC_IP:8080 --output scored.jsonl

    # Step 3: Train
    python distill_pipeline.py train --scored scored.jsonl --epochs 3 --lr 2e-5

    # Or run full pipeline
    python distill_pipeline.py full --prompts data/math_prompts.jsonl --teacher-url http://MAC_IP:8080
"""

import argparse
import json
import os
import sys
import time
import math
import requests
from pathlib import Path
from typing import Optional

# ======================================================================
# CONFIG
# ======================================================================
STUDENT_URL = "http://localhost:8080/v1/chat/completions"  # aero
TEACHER_URL = None  # Set via --teacher-url or TEACHER_URL env var
MODEL_STUDENT = "lfm2.5-8b-a1b"
MODEL_TEACHER = "qwen3.6-35b-a3b"

MAX_TOKENS_STUDENT = 4096  # Must be high for reasoning model
MAX_TOKENS_TEACHER_LOGPROBS = 1  # We query one token at a time for teacher

TEMPERATURE = 0.7  # For trajectory generation (explore diverse paths)
TOP_K = 50
TOP_P = 0.9

DATA_DIR = Path(__file__).parent / "distill_data"
RESULTS_DIR = Path("/home/billz/results/distill")

# ======================================================================
# PROMPT DATASETS
# ======================================================================
MATH_PROMPTS = [
    "Solve: If 3x + 7 = 22, what is x?",
    "What is the area of a triangle with base 10 and height 5?",
    "Simplify: (2^3)(2^4)",
    "What is 15% of 240?",
    "Solve for x: x^2 - 5x + 6 = 0",
    "A train travels 120 km in 2 hours. What is its speed in km/h?",
    "What is the GCD of 48 and 18?",
    "Convert 10110 binary to decimal.",
    "What is the sum of the first 20 natural numbers?",
    "If a rectangle has perimeter 24 and area 32, what are its dimensions?",
    "What is the derivative of 3x^2 + 2x - 1?",
    "What is the integral of 2x dx?",
    "A circle has circumference 10π. What is its area?",
    "What is log_2(64)?",
    "How many ways can 5 people sit in a row?",
    "What is the probability of rolling a sum of 7 with two dice?",
    "Simplify: sqrt(144) + sqrt(81)",
    "What is 7! (7 factorial)?",
    "If f(x) = 2x + 3, what is f(f(1))?",
    "What is the LCM of 12 and 18?",
]

CODE_PROMPTS = [
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a string.",
    "Write a Python function to find the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to find the largest element in a list.",
    "Write a Python function that counts vowels in a string.",
    "Write a Python function to flatten a nested list.",
    "Write a Python function to find duplicates in a list.",
    "Write a Python function to merge two sorted lists.",
    "Write a Python function to implement binary search.",
    "Write a Python decorator that logs function execution time.",
    "Write a Python class for a stack with push and pop.",
    "Write a Python function to check if two strings are anagrams.",
    "Write a Python function to generate Fibonacci sequence up to n.",
    "Write a Python function to find the GCD of two numbers.",
]

INSTRUCTION_PROMPTS = [
    "Explain what a neural network is in exactly 3 sentences.",
    "List 5 benefits of exercise, numbered 1-5.",
    "Write a haiku about the ocean (5-7-5 syllables).",
    "Explain quantum computing to a 10-year-old.",
    "Compare and contrast Python and JavaScript in exactly 3 bullet points.",
    "Write a professional email declining a meeting invitation.",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Give me 3 creative names for a coffee shop.",
    "Explain photosynthesis in simple terms.",
    "Write a short story (exactly 100 words) about a robot.",
]

# ======================================================================
# STEP 1: GENERATE TRAJECTORIES FROM STUDENT
# ======================================================================

def query_model(url: str, model: str, prompt: str, max_tokens: int = 4096,
                temperature: float = 0.7, logprobs: bool = False,
                system: str = None, top_logprobs: int = 0) -> dict:
    """Query an OpenAI-compatible API."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs:
            payload["top_logprobs"] = top_logprobs

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ERROR querying {url}: {e}")
        return {}


def generate_trajectories(prompts: list, output_path: Path,
                          student_url: str = STUDENT_URL,
                          num_samples: int = 3):
    """Generate student trajectories with logprobs for each prompt."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating {num_samples} trajectories per prompt from student")
    print(f"{'='*60}")

    trajectories = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")

        for sample_idx in range(num_samples):
            # Generate with logprobs
            data = query_model(
                student_url, MODEL_STUDENT, prompt,
                max_tokens=MAX_TOKENS_STUDENT,
                temperature=TEMPERATURE,
                logprobs=True,
                top_logprobs=5  # Get top 5 for analysis
            )

            if not data or "choices" not in data:
                print(f"  Sample {sample_idx}: FAILED")
                continue

            choice = data["choices"][0]
            content = choice["message"].get("content", "")
            reasoning = choice["message"].get("reasoning_content", "")
            usage = data.get("usage", {})

            # Extract logprobs if available
            logprobs_data = choice.get("logprobs")
            token_logprobs = []
            if logprobs_data and "content" in logprobs_data:
                for lp in logprobs_data["content"]:
                    token_logprobs.append({
                        "token": lp.get("token", ""),
                        "logprob": lp.get("logprob", 0),
                        "top_logprobs": lp.get("top_logprobs", [])
                    })

            traj = {
                "prompt": prompt,
                "student_response": content,
                "student_reasoning": reasoning,
                "token_logprobs": token_logprobs,
                "tokens_used": usage.get("completion_tokens", 0),
                "sample_idx": sample_idx,
                "temperature": TEMPERATURE,
            }
            trajectories.append(traj)

            print(f"  Sample {sample_idx}: {len(content)} chars, "
                  f"{len(token_logprobs)} logprobs, "
                  f"{usage.get('completion_tokens', 0)} tokens")

            # Small delay to avoid overwhelming the server
            time.sleep(0.5)

    # Save trajectories
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    print(f"\nSaved {len(trajectories)} trajectories to {output_path}")
    return trajectories


# ======================================================================
# STEP 2: SCORE TRAJECTORIES WITH TEACHER
# ======================================================================

def score_trajectory(trajectory: dict, teacher_url: str,
                     teacher_model: str = MODEL_TEACHER) -> dict:
    """
    Score a student trajectory with the teacher model.
    For each token in the student's response, get the teacher's logprob.

    The teacher sees: prompt + student_response[:t] and provides logprob for token t.
    """
    prompt = trajectory["prompt"]
    student_response = trajectory["student_response"]

    if not student_response.strip():
        trajectory["teacher_logprobs"] = []
        trajectory["reverse_kl"] = []
        return trajectory

    # We need the full text (prompt + response) and get logprobs at each position
    # Use the completions endpoint (not chat) for precise token-level logprobs
    full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{student_response}"

    # Query teacher with echo=True, logprobs=True to get logprobs for entire sequence
    msgs = [{"role": "user", "content": prompt}]

    # First, get teacher's own response (we need its logprobs on student's tokens)
    # We'll query the teacher with the student's completion appended as context

    # Strategy: use the chat completions API with the student's response as assistant message
    # then ask for logprobs on the next token (but we want logprobs on ALL student tokens)
    #
    # Better strategy: query teacher with full sequence and get logprobs
    # We can use the completions endpoint:
    # POST /completions with prompt = full_text, max_tokens = 0, logprobs = True, echo = True

    try:
        payload = {
            "model": teacher_model,
            "prompt": full_text,
            "max_tokens": 0,
            "logprobs": True,
            "echo": True,
            "temperature": 0,
        }
        resp = requests.post(
            teacher_url.replace("/chat/completions", "/completions"),
            json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract logprobs
        choices = data.get("choices", [])
        if choices:
            logprobs_info = choices[0].get("logprobs", {})
            token_logprobs = logprobs_info.get("token_logprobs", [])
            tokens = logprobs_info.get("tokens", [])

            # We only want logprobs for the student's response tokens
            # Find where the response starts
            response_start = len(full_text) - len(student_response)
            # Approximate: response tokens start after the prompt tokens
            prompt_token_count = data.get("usage", {}).get("prompt_tokens", 0)

            teacher_logprobs = token_logprobs[prompt_token_count:]

            trajectory["teacher_logprobs"] = teacher_logprobs
            trajectory["teacher_tokens"] = tokens[prompt_token_count:]
        else:
            trajectory["teacher_logprobs"] = []
            trajectory["teacher_tokens"] = []

    except Exception as e:
        print(f"  Teacher query failed: {e}")
        trajectory["teacher_logprobs"] = []
        trajectory["teacher_tokens"] = []

    # Compute reverse KL per token
    student_logprobs = [lp["logprob"] for lp in trajectory.get("token_logprobs", [])]
    teacher_logprobs = trajectory.get("teacher_logprobs", [])

    min_len = min(len(student_logprobs), len(teacher_logprobs))
    reverse_kl = []
    for i in range(min_len):
        # Reverse KL = log π_student(x_t | x_{<t}) - log π_teacher(x_t | x_{<t})
        kl = student_logprobs[i] - teacher_logprobs[i]
        reverse_kl.append(kl)

    trajectory["reverse_kl"] = reverse_kl
    trajectory["mean_reverse_kl"] = sum(reverse_kl) / len(reverse_kl) if reverse_kl else 0
    trajectory["max_reverse_kl"] = max(reverse_kl) if reverse_kl else 0

    return trajectory


def score_trajectories(trajectories_path: Path, output_path: Path,
                       teacher_url: str):
    """Score all trajectories with the teacher."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Scoring trajectories with teacher")
    print(f"{'='*60}")

    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    scored = []
    for i, traj in enumerate(trajectories):
        print(f"[{i+1}/{len(trajectories)}] Scoring: {traj['prompt'][:50]}...", end=" ")
        scored_traj = score_trajectory(traj, teacher_url)
        scored.append(scored_traj)

        kl = scored_traj.get("mean_reverse_kl", 0)
        n_tokens = len(scored_traj.get("reverse_kl", []))
        print(f"mean_kl={kl:.4f}, tokens={n_tokens}")

        time.sleep(0.2)  # Rate limit

    with open(output_path, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    # Stats
    all_kl = [s["mean_reverse_kl"] for s in scored if s.get("reverse_kl")]
    if all_kl:
        print(f"\nScoring complete:")
        print(f"  Mean reverse KL: {sum(all_kl)/len(all_kl):.4f}")
        print(f"  Max mean KL: {max(all_kl):.4f}")
        print(f"  Min mean KL: {min(all_kl):.4f}")

    print(f"Saved {len(scored)} scored trajectories to {output_path}")
    return scored


# ======================================================================
# STEP 3: TRAIN WITH REVERSE KL (LoRA)
# ======================================================================

def train_distillation(scored_path: Path, output_dir: Path,
                       epochs: int = 3, lr: float = 2e-5,
                       lora_r: int = 16, lora_alpha: int = 32,
                       max_seq_len: int = 2048, batch_size: int = 2):
    """
    Train the student model using reverse KL from teacher-graded trajectories.

    This uses TRL's SFTTrainer with a custom loss function that weights
    each token by its reverse KL divergence from the teacher.
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Training with on-policy distillation (reverse KL)")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}, LR: {lr}, LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"  Max seq len: {max_seq_len}, Batch size: {batch_size}")

    # Load scored trajectories
    trajectories = []
    with open(scored_path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("reverse_kl") and len(t["reverse_kl"]) > 0:
                    trajectories.append(t)

    print(f"  Loaded {len(trajectories)} scored trajectories")

    if not trajectories:
        print("  ERROR: No scored trajectories found")
        return

    # Create training script
    train_script = f"""#!/usr/bin/env python3
\"\"\"On-policy distillation training using reverse KL.\"\"\"
import torch
import json
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Config
MODEL_NAME = "LiquidAI/LFM2.5-8B-A1B"
LORA_R = {lora_r}
LORA_ALPHA = {lora_alpha}
LR = {lr}
EPOCHS = {epochs}
MAX_SEQ_LEN = {max_seq_len}
BATCH_SIZE = {batch_size}
OUTPUT_DIR = "{output_dir}"
SCORED_PATH = "{scored_path}"

# Load data
trajectories = []
with open(SCORED_PATH) as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            if t.get("reverse_kl") and len(t["reverse_kl"]) > 5:
                trajectories.append(t)

print(f"Loaded {{len(trajectories)}} training trajectories")

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
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

# Custom dataset with reverse KL weighting
class DistillationDataset(Dataset):
    def __init__(self, trajectories, tokenizer, max_len):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for traj in trajectories:
            prompt = traj["prompt"]
            response = traj["student_response"]
            kl_weights = traj["reverse_kl"]

            # Tokenize
            prompt_ids = tokenizer.encode(f"<|user|>\\n{{prompt}}\\n<|assistant|>\\n")
            response_ids = tokenizer.encode(response)

            # Truncate
            total_len = len(prompt_ids) + len(response_ids)
            if total_len > max_len:
                response_ids = response_ids[:max_len - len(prompt_ids)]

            # Pad KL weights to match response length
            kl_w = kl_weights[:len(response_ids)]

            if len(response_ids) > 5:
                self.data.append({{
                    "input_ids": prompt_ids + response_ids,
                    "labels": [-100] * len(prompt_ids) + response_ids,
                    "kl_weights": [0.0] * len(prompt_ids) + kl_w,
                    "prompt_len": len(prompt_ids),
                }})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {{
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "kl_weights": torch.tensor(item["kl_weights"], dtype=torch.float),
        }}

dataset = DistillationDataset(trajectories, tokenizer, MAX_SEQ_LEN)
print(f"Dataset size: {{len(dataset)}} examples")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR)

# Training loop with reverse KL loss
def compute_distillation_loss(outputs, labels, kl_weights):
    \"\"\"Compute reverse KL weighted cross-entropy loss.\"\"\"
    logits = outputs.logits[:, :-1, :]
    labels = labels[:, 1:]

    # Per-token cross-entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss = loss.reshape(labels.shape)

    # Weight by reverse KL (higher KL = more to learn from)
    # Use absolute KL as weight — tokens where student diverges most get higher weight
    weights = torch.abs(kl_weights[:, 1:]) + 0.01  # Small epsilon to avoid zero
    weights = weights / weights.sum()  # Normalize

    # Weighted loss
    weighted_loss = (loss * weights).sum()
    return weighted_loss

# Train
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        kl_weights = batch["kl_weights"].to(model.device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = compute_distillation_loss(outputs, labels, kl_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Epoch {{epoch+1}}/{{EPOCHS}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}")

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch {{epoch+1}}/{{EPOCHS}} complete. Avg loss: {{avg_loss:.4f}}")

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {{OUTPUT_DIR}}")
"""

    script_path = output_dir / "train_distill.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(train_script)

    print(f"Training script saved to {script_path}")
    print(f"Run with: ssh aero 'cd {output_dir} && source ~/gguf-env/bin/activate && python train_distill.py'")
    return str(script_path)


# ======================================================================
# PROMPT GENERATION
# ======================================================================

def generate_prompts(output_path: Path, categories: list = None):
    """Generate a prompt dataset from built-in categories."""
    if categories is None:
        categories = ["math", "code", "instruction"]

    prompts = []
    for cat in categories:
        if cat == "math":
            for p in MATH_PROMPTS:
                prompts.append({"prompt": p, "category": "math"})
        elif cat == "code":
            for p in CODE_PROMPTS:
                prompts.append({"prompt": p, "category": "code"})
        elif cat == "instruction":
            for p in INSTRUCTION_PROMPTS:
                prompts.append({"prompt": p, "category": "instruction"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Generated {len(prompts)} prompts in {len(categories)} categories → {output_path}")
    return prompts


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="On-Policy Distillation Pipeline")
    sub = parser.add_subparsers(dest="command")

    # generate
    gen = sub.add_parser("generate", help="Generate student trajectories")
    gen.add_argument("--prompts", type=str, help="Prompt JSONL file (or 'math', 'code', 'instruction', 'all')")
    gen.add_argument("--output", type=str, default=str(RESULTS_DIR / "trajectories.jsonl"))
    gen.add_argument("--num-samples", type=int, default=3)
    gen.add_argument("--student-url", type=str, default=STUDENT_URL)

    # score
    sc = sub.add_parser("score", help="Score trajectories with teacher")
    sc.add_argument("--trajectories", type=str, required=True)
    sc.add_argument("--teacher-url", type=str, required=True)
    sc.add_argument("--output", type=str, default=str(RESULTS_DIR / "scored.jsonl"))

    # train
    tr = sub.add_parser("train", help="Train with distillation")
    tr.add_argument("--scored", type=str, required=True)
    tr.add_argument("--output-dir", type=str, default=str(RESULTS_DIR / "model"))
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--lr", type=float, default=2e-5)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--batch-size", type=int, default=2)

    # full
    full = sub.add_parser("full", help="Run full pipeline")
    full.add_argument("--prompts", type=str, default="all")
    full.add_argument("--teacher-url", type=str, required=True)
    full.add_argument("--num-samples", type=int, default=3)
    full.add_argument("--epochs", type=int, default=3)

    # prompts
    pr = sub.add_parser("prompts", help="Generate prompt dataset")
    pr.add_argument("--categories", type=str, nargs="+", default=["math", "code", "instruction"])
    pr.add_argument("--output", type=str, default=str(DATA_DIR / "prompts.jsonl"))

    args = parser.parse_args()

    if args.command == "generate":
        # Load prompts
        if args.prompts in ("math", "code", "instruction", "all"):
            cats = ["math", "code", "instruction"] if args.prompts == "all" else [args.prompts]
            prompts_data = []
            for cat in cats:
                if cat == "math": prompts_data.extend(MATH_PROMPTS)
                elif cat == "code": prompts_data.extend(CODE_PROMPTS)
                elif cat == "instruction": prompts_data.extend(INSTRUCTION_PROMPTS)
        else:
            prompts_data = []
            with open(args.prompts) as f:
                for line in f:
                    if line.strip():
                        prompts_data.append(json.loads(line).get("prompt", line.strip()))

        generate_trajectories(prompts_data, Path(args.output),
                              args.student_url, args.num_samples)

    elif args.command == "score":
        score_trajectories(Path(args.trajectories), Path(args.output),
                          args.teacher_url)

    elif args.command == "train":
        train_distillation(Path(args.scored), Path(args.output_dir),
                          args.epochs, args.lr, args.lora_r, batch_size=args.batch_size)

    elif args.command == "prompts":
        generate_prompts(Path(args.output), args.categories)

    elif args.command == "full":
        # Step 1: Generate prompts
        prompts_file = DATA_DIR / "prompts.jsonl"
        generate_prompts(prompts_file,
                        ["math", "code", "instruction"] if args.prompts == "all"
                        else [args.prompts])

        # Step 2: Generate trajectories
        traj_file = RESULTS_DIR / "trajectories.jsonl"
        prompts_data = []
        with open(prompts_file) as f:
            for line in f:
                if line.strip():
                    prompts_data.append(json.loads(line).get("prompt", ""))

        generate_trajectories(prompts_data, traj_file,
                             num_samples=args.num_samples)

        # Step 3: Score with teacher
        scored_file = RESULTS_DIR / "scored.jsonl"
        score_trajectories(traj_file, scored_file, args.teacher_url)

        # Step 4: Generate training script
        output_dir = RESULTS_DIR / "model"
        train_distillation(scored_file, output_dir, args.epochs)

        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Trajectories: {traj_file}")
        print(f"Scored: {scored_file}")
        print(f"Training script: {output_dir / 'train_distill.py'}")
        print(f"\nNext steps:")
        print(f"  1. Set up teacher on Mac: python -m mlx_lm.server --model Qwen/Qwen3.6-35B-A3B --port 8080")
        print(f"  2. Run pipeline: python distill_pipeline.py full --teacher-url http://MAC_IP:8080")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
