#!/usr/bin/env python3
"""
On-Policy Distillation with Unsloth
Teacher: Gemma 4 24B on Mac LM Studio
Student: LFM2.5 via Unsloth (2-5x faster, 70% less VRAM)

Unsloth handles: LoRA, gradient checkpointing, fused kernels, GGUF export.
We just need to: generate trajectories → score with teacher → train.

Usage:
    python distill_unsloth.py test       # 3 prompts, verify pipeline
    python distill_unsloth.py generate   # Generate student trajectories  
    python distill_unsloth.py score      # Score with teacher
    python distill_unsloth.py train      # Train with Unsloth + reverse KL
    python distill_unsloth.py full       # Everything end-to-end
"""

import argparse
import json
import os
import sys
import time
import torch
import requests
from pathlib import Path

# ======================================================================
# CONFIG
# ======================================================================
TEACHER_URL = "http://100.100.61.28:1234/v1/chat/completions"
TEACHER_MODEL = "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp"

# Student model — start with 230M for fast iteration, then 8B
STUDENT_MODEL = "LiquidAI/LFM2.5-230M"  # Change to LFM2.5-8B-A1B for full training

DATA_DIR = Path(__file__).parent / "distill_unsloth_data"
RESULTS_DIR = Path("/home/billz/results/distill_unsloth")
MODEL_DIR = RESULTS_DIR / "finetuned"

# Generation
TEMPERATURE = 0.7
MAX_TOKENS = 512  # 230M generates fast, less thinking
NUM_SAMPLES = 3

# Training (Unsloth-optimized)
LORA_RANK = 16
LORA_ALPHA = 32  # 2x rank = faster convergence per Unsloth docs
LR = 2e-4  # Unsloth docs recommend this for LFM2.5
EPOCHS = 3
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4

# ======================================================================
# PROMPTS
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
# STEP 1: GENERATE TRAJECTORIES (student inference)
# ======================================================================

def generate_with_unsloth(model_name: str, prompts: list, num_samples: int = NUM_SAMPLES):
    """Generate trajectories using Unsloth's fast inference."""
    from unsloth import FastLanguageModel

    print(f"\n{'='*60}")
    print(f"STEP 1: Generating trajectories with Unsloth")
    print(f"{'='*60}")

    # Load model with Unsloth
    print(f"Loading {model_name} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,  # QLoRA for 230M, False for MoE
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)

    trajectories = []

    for cat, cat_prompts in prompts.items():
        for prompt in cat_prompts:
            for sample_idx in range(num_samples):
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                    )

                generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Get logprobs
                with torch.no_grad():
                    logits = model(outputs).logits

                token_logprobs = []
                for i in range(len(generated_ids)):
                    if i + 1 < logits.shape[1]:
                        probs = torch.softmax(logits[0, i + inputs["input_ids"].shape[1]], dim=-1)
                        token_id = generated_ids[i].item()
                        logprob = torch.log(probs[token_id]).item()
                        token_logprobs.append({
                            "token": tokenizer.decode([token_id]),
                            "logprob": logprob,
                        })

                traj = {
                    "prompt": prompt,
                    "category": cat,
                    "student_response": response,
                    "student_logprobs": token_logprobs,
                    "tokens_generated": len(generated_ids),
                    "sample_idx": sample_idx,
                }
                trajectories.append(traj)
                print(f"  [{cat}] {prompt[:40]}... → {len(generated_ids)} tokens")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "trajectories.jsonl", "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

    print(f"\nSaved {len(trajectories)} trajectories")
    return trajectories


# ======================================================================
# STEP 2: SCORE WITH TEACHER
# ======================================================================

def score_with_teacher(teacher_url: str = TEACHER_URL):
    """Score student trajectories with Gemma 4 teacher."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Scoring with teacher (Gemma 4 24B)")
    print(f"{'='*60}")

    trajectories = []
    with open(DATA_DIR / "trajectories.jsonl") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    scored = []
    for i, traj in enumerate(trajectories):
        prompt = traj["prompt"]
        response = traj["student_response"]

        if not response.strip() or len(traj.get("student_logprobs", [])) < 3:
            continue

        print(f"[{i+1}/{len(trajectories)}] {prompt[:45]}...", end=" ", flush=True)

        # Get teacher logprobs via completions endpoint
        try:
            full_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

            resp = requests.post(
                teacher_url.replace("/chat/completions", "/completions"),
                json={
                    "model": TEACHER_MODEL,
                    "prompt": full_text,
                    "max_tokens": 0,
                    "logprobs": True,
                    "echo": True,
                    "temperature": 0,
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if choices:
                logprobs_info = choices[0].get("logprobs", {})
                token_logprobs = logprobs_info.get("token_logprobs", [])
                tokens = logprobs_info.get("tokens", [])
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)

                teacher_lps = token_logprobs[prompt_tokens:]

                # Align with student logprobs
                student_lps = [lp["logprob"] for lp in traj["student_logprobs"]]
                min_len = min(len(student_lps), len(teacher_lps))

                if min_len >= 3:
                    reverse_kl = [student_lps[j] - teacher_lps[j] for j in range(min_len)]
                    mean_kl = sum(reverse_kl) / len(reverse_kl)

                    traj["teacher_logprobs"] = teacher_lps[:min_len]
                    traj["reverse_kl"] = reverse_kl
                    traj["mean_reverse_kl"] = mean_kl
                    traj["aligned_tokens"] = min_len
                    scored.append(traj)
                    print(f"kl={mean_kl:.4f}, tokens={min_len}")
                else:
                    print(f"SKIP (aligned={min_len})")
            else:
                print("SKIP (no choices)")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.3)

    # Save
    with open(DATA_DIR / "scored.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    all_kl = [s["mean_reverse_kl"] for s in scored]
    if all_kl:
        print(f"\nScored {len(scored)} trajectories, mean_kl={sum(all_kl)/len(all_kl):.4f}")
    return scored


# ======================================================================
# STEP 3: TRAIN WITH UNSLOTH + REVERSE KL
# ======================================================================

def train_unsloth(model_name: str = STUDENT_MODEL, scored_path: Path = None):
    """Train with Unsloth — 2-5x faster, 70% less VRAM."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from torch.utils.data import Dataset

    print(f"\n{'='*60}")
    print(f"STEP 3: Training with Unsloth (reverse KL)")
    print(f"{'='*60}")

    if scored_path is None:
        scored_path = DATA_DIR / "scored.jsonl"

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

    # Determine if MoE (needs bf16, not QLoRA)
    is_moe = "8B" in model_name or "A1B" in model_name
    load_4bit = not is_moe  # QLoRA for dense, bf16 for MoE

    # Load model with Unsloth
    print(f"Loading {model_name} with Unsloth (4bit={load_4bit})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=load_4bit,
        trust_remote_code=True,
    )

    # Apply LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if is_moe:
        target_modules += ["gate_up_proj", "down_proj"]  # MoE-specific

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=target_modules,
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",  # 2x faster
        random_state=3407,
    )

    # Custom dataset with reverse KL weighting
    class DistillDataset(Dataset):
        def __init__(self, trajectories, tokenizer, max_len):
            self.data = []
            for traj in trajectories:
                prompt = traj["prompt"]
                response = traj["student_response"]
                kl_weights = traj["reverse_kl"]

                # Format: user prompt + assistant response
                prompt_ids = tokenizer.encode(
                    f"<|user|>\n{prompt}\n<|assistant|>\n",
                    add_special_tokens=False
                )
                response_ids = tokenizer.encode(response, add_special_tokens=False)

                # Truncate
                total = len(prompt_ids) + len(response_ids)
                if total > max_len:
                    response_ids = response_ids[:max_len - len(prompt_ids)]

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

    # Training config
    training_args = SFTConfig(
        output_dir=str(MODEL_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=True,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        remove_unused_columns=False,
    )

    # Custom trainer with reverse KL loss
    class DistillTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            kl_weights = inputs["kl_weights"]

            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits[:, :-1, :]
            labels_shifted = labels[:, 1:]
            kl_w = kl_weights[:, 1:]

            # Per-token cross-entropy
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels_shifted.reshape(-1))
            loss = loss.reshape(labels_shifted.shape)

            # Weight by |reverse KL|
            weights = torch.abs(kl_w) + 0.01
            weights = weights / weights.sum()

            return (loss * weights).sum()

    trainer = DistillTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train
    print("Training with Unsloth...")
    trainer.train()

    # Save LoRA adapters
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR / "lora"))
    tokenizer.save_pretrained(str(MODEL_DIR / "lora"))
    print(f"LoRA adapters saved to {MODEL_DIR / 'lora'}")

    # Export to GGUF
    print("Exporting to GGUF...")
    model.save_pretrained_gguf(
        str(MODEL_DIR / "gguf"),
        tokenizer,
        quantization_method="q4_k_m"
    )
    print(f"GGUF saved to {MODEL_DIR / 'gguf'}")

    # Save training config
    with open(MODEL_DIR / "training_config.json", "w") as f:
        json.dump({
            "student_model": model_name,
            "teacher_model": TEACHER_MODEL,
            "teacher_url": TEACHER_URL,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "is_moe": is_moe,
            "load_4bit": load_4bit,
            "num_trajectories": len(trajectories),
        }, f, indent=2)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return str(MODEL_DIR / "gguf")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Unsloth On-Policy Distillation")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate student trajectories")
    sub.add_parser("score", help="Score with teacher")
    sub.add_parser("train", help="Train with Unsloth")
    sub.add_parser("full", help="Run everything")
    sub.add_parser("test", help="Quick test (3 prompts)")

    args = parser.parse_args()

    if args.command == "generate":
        generate_with_unsloth(STUDENT_MODEL, PROMPTS, NUM_SAMPLES)

    elif args.command == "score":
        score_with_teacher()

    elif args.command == "train":
        train_unsloth()

    elif args.command in ("full", "test"):
        if args.command == "test":
            # Quick test with subset
            test_prompts = {
                "math": PROMPTS["math"][:2],
                "code": PROMPTS["code"][:1],
            }
            generate_with_unsloth(STUDENT_MODEL, test_prompts, num_samples=1)
        else:
            generate_with_unsloth(STUDENT_MODEL, PROMPTS, NUM_SAMPLES)

        score_with_teacher()
        gguf_path = train_unsloth()

        print(f"\n{'='*60}")
        print("DISTILLATION COMPLETE")
        print(f"{'='*60}")
        if gguf_path:
            print(f"GGUF model: {gguf_path}")
            print(f"\nRun with: llama-server -m {gguf_path}/unsloth.Q4_K_M.gguf --port 8081")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
