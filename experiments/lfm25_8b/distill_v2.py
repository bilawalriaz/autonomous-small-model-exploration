#!/usr/bin/env python3
"""
Distillation Pipeline v2: Simplified, works NOW.

Teacher: Gemma 4 24B on Mac LM Studio (100.100.61.28:1234)
Student: LFM2.5-8B-A1B on aero llama.cpp (localhost:8080)
Training: Unsloth on aero (RTX 2070 Super)

Flow: llama.cpp generates → teacher scores → Unsloth trains
"""

import json, time, requests, os, sys
from pathlib import Path

TEACHER_URL = "http://100.100.61.28:1234/v1/chat/completions"
TEACHER_MODEL = "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp"
STUDENT_URL = "http://localhost:8080/v1/chat/completions"
STUDENT_MODEL = "lfm2.5-8b-a1b"

DATA_DIR = Path("/home/billz/results/distill_v2")
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

def query(url, model, prompt, max_tokens=2048, temperature=0.7, logprobs=False):
    msgs = [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    try:
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}

def step1_generate():
    """Generate student trajectories with logprobs via llama.cpp."""
    print(f"\n{'='*60}")
    print("STEP 1: Generating student trajectories")
    print(f"{'='*60}")

    trajs = []
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt[:50]}...", end=" ", flush=True)
        data = query(STUDENT_URL, STUDENT_MODEL, prompt, max_tokens=2048, temperature=0.7, logprobs=True)
        if "choices" not in data:
            print("FAILED")
            continue

        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        reasoning = choice["message"].get("reasoning_content", "")
        usage = data.get("usage", {})

        # Extract student logprobs
        logprobs_data = choice.get("logprobs", {})
        student_lps = []
        if logprobs_data and "content" in logprobs_data:
            for lp in logprobs_data["content"]:
                student_lps.append({"token": lp.get("token", ""), "logprob": lp.get("logprob", 0)})

        trajs.append({
            "prompt": prompt,
            "student_response": content,
            "student_reasoning": reasoning,
            "student_logprobs": student_lps,
            "tokens": usage.get("completion_tokens", 0),
        })
        print(f"{len(content)} chars, {len(student_lps)} logprobs")
        time.sleep(0.3)

    with open(DATA_DIR / "trajectories.jsonl", "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    print(f"\nSaved {len(trajs)} trajectories")
    return trajs

def step2_score():
    """Score trajectories with teacher via LM Studio."""
    print(f"\n{'='*60}")
    print("STEP 2: Scoring with teacher (Gemma 4)")
    print(f"{'='*60}")

    trajs = []
    with open(DATA_DIR / "trajectories.jsonl") as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))

    scored = []
    for i, traj in enumerate(trajs):
        prompt = traj["prompt"]
        response = traj["student_response"]
        if not response.strip() or len(traj.get("student_logprobs", [])) < 3:
            continue

        print(f"[{i+1}/{len(trajs)}] {prompt[:45]}...", end=" ", flush=True)

        # Teacher scores via completions endpoint
        try:
            # Format for Gemma 4
            full_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

            r = requests.post(
                TEACHER_URL.replace("/chat/completions", "/completions"),
                json={"model": TEACHER_MODEL, "prompt": full_text, "max_tokens": 0, "logprobs": True, "echo": True, "temperature": 0},
                timeout=120
            )
            r.raise_for_status()
            data = r.json()

            choices = data.get("choices", [])
            if choices:
                lp_info = choices[0].get("logprobs", {})
                all_lps = lp_info.get("token_logprobs", [])
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                teacher_lps = all_lps[prompt_tokens:]

                # Align
                student_lps = [lp["logprob"] for lp in traj["student_logprobs"]]
                min_len = min(len(student_lps), len(teacher_lps))

                if min_len >= 3:
                    reverse_kl = [student_lps[j] - teacher_lps[j] for j in range(min_len)]
                    mean_kl = sum(reverse_kl) / len(reverse_kl)
                    traj["teacher_logprobs"] = teacher_lps[:min_len]
                    traj["reverse_kl"] = reverse_kl
                    traj["mean_reverse_kl"] = mean_kl
                    scored.append(traj)
                    print(f"kl={mean_kl:.4f} ({min_len} tokens)")
                else:
                    print(f"SKIP (aligned={min_len})")
            else:
                print("SKIP (no choices)")
        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.3)

    with open(DATA_DIR / "scored.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    kls = [s["mean_reverse_kl"] for s in scored]
    if kls:
        print(f"\nScored {len(scored)}: mean_kl={sum(kls)/len(kls):.4f}")
    return scored

def step3_train():
    """Train with Unsloth + reverse KL."""
    print(f"\n{'='*60}")
    print("STEP 3: Training with Unsloth")
    print(f"{'='*60}")

    # Load scored data
    trajs = []
    with open(DATA_DIR / "scored.jsonl") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if t.get("reverse_kl") and len(t["reverse_kl"]) >= 3:
                    trajs.append(t)

    if not trajs:
        print("ERROR: No scored trajectories")
        return

    print(f"Training on {len(trajs)} trajectories...")

    # Write the actual training script (runs in separate process)
    train_script = f'''#!/usr/bin/env python3
import torch, json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from torch.utils.data import Dataset

SCORED = "{DATA_DIR / 'scored.jsonl'}"
OUTPUT = "{DATA_DIR / 'finetuned'}"

# Load data
trajs = []
with open(SCORED) as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            if t.get("reverse_kl") and len(t["reverse_kl"]) >= 3:
                trajs.append(t)

print(f"Loaded {{len(trajs)}} trajectories")

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LiquidAI/LFM2.5-8B-A1B",
    max_seq_length=2048,
    load_in_4bit=False,  # MoE needs bf16
    trust_remote_code=True,
)

# Apply LoRA — MoE-specific target modules
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_up_proj","down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Dataset
class DS(Dataset):
    def __init__(self):
        self.data = []
        for t in trajs:
            p_ids = tokenizer.encode(f"<|user|>\\n{{t['prompt']}}\\n<|assistant|>\\n", add_special_tokens=False)
            r_ids = tokenizer.encode(t["student_response"], add_special_tokens=False)
            if len(p_ids) + len(r_ids) > 2048:
                r_ids = r_ids[:2048 - len(p_ids)]
            kl = t["reverse_kl"][:len(r_ids)]
            if len(r_ids) >= 3:
                self.data.append({{
                    "input_ids": p_ids + r_ids,
                    "labels": [-100]*len(p_ids) + r_ids,
                    "kl": [0.0]*len(p_ids) + kl,
                }})
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        return {{
            "input_ids": torch.tensor(d["input_ids"], dtype=torch.long),
            "labels": torch.tensor(d["labels"], dtype=torch.long),
            "kl": torch.tensor(d["kl"], dtype=torch.float),
        }}

ds = DS()
print(f"Dataset: {{len(ds)}} examples")

class DistillTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, **kw):
        out = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        logits = out.logits[:, :-1, :]
        lab = inputs["labels"][:, 1:]
        kl = inputs["kl"][:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1), reduction="none").reshape(lab.shape)
        w = torch.abs(kl) + 0.01
        w = w / w.sum()
        return (loss * w).sum()

trainer = DistillTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        output_dir=OUTPUT, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs=3,
        learning_rate=2e-4, bf16=True, report_to="none",
        max_length=2048, optim="adamw_8bit",
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
print("DONE")
'''

    script_path = DATA_DIR / "train_unsloth.py"
    with open(script_path, "w") as f:
        f.write(train_script)

    print(f"Training script: {script_path}")
    print(f"Run with: ssh aero 'source gguf-env/bin/activate && python3 {script_path}'")

def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    if cmd == "generate": step1_generate()
    elif cmd == "score": step2_score()
    elif cmd == "train": step3_train()
    elif cmd == "full":
        step1_generate()
        step2_score()
        step3_train()
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Data: {DATA_DIR}")
        print(f"Training: {DATA_DIR}/train_unsloth.py")
    else:
        print("Usage: python distill_v2.py [generate|score|train|full]")

if __name__ == "__main__":
    main()
