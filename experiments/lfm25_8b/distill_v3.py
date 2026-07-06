#!/usr/bin/env python3
"""
Distillation Pipeline v3: Works with LM Studio (no logprobs needed).

Strategy: Teacher generates reference responses → Student learns via SFT
weighted by teacher agreement quality.

Teacher: Gemma 4 24B on Mac LM Studio
Student: LFM2.5-8B-A1B on aero
Training: Unsloth on aero

This is "soft distillation" — the student learns the teacher's response
style and reasoning patterns, even without per-token logprobs.
"""

import json, time, requests, os
from pathlib import Path

TEACHER_URL = "http://100.100.61.28:1234/v1/chat/completions"
TEACHER_MODEL = "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp"
STUDENT_URL = "http://localhost:8080/v1/chat/completions"
STUDENT_MODEL = "lfm2.5-8b-a1b"
DATA_DIR = Path("/home/billz/results/distill_v3")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
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
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a string.",
    "Write a Python function to find the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to count vowels in a string.",
    "If all cats are animals, and all animals are living things, are all cats living things?",
    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "What comes next: 2, 6, 12, 20, 30, ?",
    "Explain why the sky is blue in one paragraph.",
    "List exactly 5 fruits, numbered 1-5.",
    "Explain quantum computing to a 10-year-old.",
    "Write a haiku about the ocean (5-7-5 syllables).",
    "Compare Python and JavaScript in 3 bullet points.",
]

def query(url, model, prompt, max_tokens=2048, temperature=0.2):
    try:
        r = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""

def step1_collect():
    """Collect teacher reference responses + student attempts."""
    print(f"\n{'='*60}")
    print("STEP 1: Collecting teacher references + student attempts")
    print(f"{'='*60}")

    pairs = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:50]}...", flush=True)

        # Teacher reference (low temp for quality)
        teacher_resp = query(TEACHER_URL, TEACHER_MODEL, prompt, temperature=0.2)
        print(f"  Teacher: {teacher_resp[:80]}...")

        # Student attempt (higher temp for diversity)
        student_resp = query(STUDENT_URL, STUDENT_MODEL, prompt, temperature=0.7)
        print(f"  Student: {student_resp[:80]}...")

        # Simple quality check: does student match teacher's answer?
        # (For math: check if same number appears. For code: check if has def/function)
        agree = _check_agreement(prompt, teacher_resp, student_resp)
        print(f"  Agreement: {'✅' if agree else '❌'}")

        pairs.append({
            "prompt": prompt,
            "teacher_response": teacher_resp,
            "student_response": student_resp,
            "agreement": agree,
        })
        time.sleep(0.5)

    with open(DATA_DIR / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    agree_count = sum(1 for p in pairs if p["agreement"])
    print(f"\nCollected {len(pairs)} pairs, {agree_count}/{len(pairs)} agreement")
    return pairs

def _check_agreement(prompt, teacher, student):
    """Quick heuristic: do teacher and student agree on the answer?"""
    t_lower = teacher.lower()
    s_lower = student.lower()
    # Check if key numbers/words match
    import re
    t_nums = set(re.findall(r'\d+', t_lower))
    s_nums = set(re.findall(r'\d+', s_lower))
    if t_nums and s_nums:
        return len(t_nums & s_nums) > 0
    # Check for key code patterns
    if "def " in t_lower and "def " in s_lower:
        return True
    # Default: check word overlap
    t_words = set(t_lower.split())
    s_words = set(s_lower.split())
    return len(t_words & s_words) > len(t_words) * 0.3

def step2_train():
    """Train student on teacher responses via Unsloth SFT."""
    print(f"\n{'='*60}")
    print("STEP 2: Training with Unsloth (teacher→student distillation)")
    print(f"{'='*60}")

    pairs = []
    with open(DATA_DIR / "pairs.jsonl") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Use teacher responses as training targets
    training_data = []
    for p in pairs:
        training_data.append({
            "prompt": p["prompt"],
            "response": p["teacher_response"],
        })

    # Save as JSONL for Unsloth
    train_file = DATA_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for d in training_data:
            # Format as chat
            text = f"<|user|>\n{d['prompt']}\n<|assistant|>\n{d['response']}"
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Training data: {len(training_data)} examples → {train_file}")

    # Write Unsloth training script
    script = f'''#!/usr/bin/env python3
import torch, json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

TRAIN_FILE = "{train_file}"
OUTPUT = "{DATA_DIR / 'finetuned'}"

# Load data
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"Dataset: {{len(dataset)}} examples")

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LiquidAI/LFM2.5-8B-A1B",
    max_seq_length=2048,
    load_in_4bit=False,  # MoE needs bf16
    trust_remote_code=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_up_proj","down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        report_to="none",
        max_length=2048,
        optim="adamw_8bit",
        dataset_text_field="text",
    ),
)

trainer.train()

# Save
import os
os.makedirs(OUTPUT, exist_ok=True)
model.save_pretrained(f"{{OUTPUT}}/lora")
tokenizer.save_pretrained(f"{{OUTPUT}}/lora")
print(f"LoRA saved to {{OUTPUT}}/lora")

# Export GGUF
model.save_pretrained_gguf(f"{{OUTPUT}}/gguf", tokenizer, quantization_method="q4_k_m")
print(f"GGUF saved to {{OUTPUT}}/gguf")
print("DISTILLATION COMPLETE")
'''

    script_path = DATA_DIR / "train_distill.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Training script: {script_path}")
    print(f"Run with: ssh aero 'source gguf-env/bin/activate && python3 {script_path}'")

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    if cmd == "collect": step1_collect()
    elif cmd == "train": step2_train()
    elif cmd == "full":
        step1_collect()
        step2_train()
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Data: {DATA_DIR}")
        print(f"Training: {DATA_DIR}/train_distill.py")
    else:
        print("Usage: python distill_v3.py [collect|train|full]")

if __name__ == "__main__":
    main()
