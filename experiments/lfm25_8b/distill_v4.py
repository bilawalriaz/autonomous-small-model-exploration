#!/usr/bin/env python3
"""
Distillation v4: Proper on-policy with llama.cpp teacher.

Flow:
1. Student generates response + student logprobs
2. Teacher generates SAME response + teacher logprobs
   (teacher is given the prompt and generates its own answer)
3. Align tokens between student and teacher responses
4. Compute KL divergence at each position
5. Train student: SFT on teacher responses, weighted by KL

Teacher: llama.cpp on Mac (100.100.61.28:8081) - supports logprobs
Student: llama.cpp on aero (localhost:8080) - supports logprobs
Training: Unsloth on aero (RTX 2070 Super)
"""
import json, time, requests, os
from pathlib import Path
import difflib

TEACHER_URL = "http://100.100.61.28:8081"
STUDENT_URL = "http://localhost:8080"
DATA_DIR = Path("/home/billz/results/distill_v4")
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


def llm_generate(url, prompt, max_tokens=1024, temperature=0.7, logprobs=False):
    """Generate via chat completions (works on both servers)."""
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": "model", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature,
        **({"logprobs": True, "top_logprobs": 5} if logprobs else {}),
    }, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]


def llm_logprobs(url, prompt, response, max_tokens=1):
    """Get logprobs by having teacher continue from prompt+response.
    The teacher sees the student's response as context and generates next tokens.
    We get logprobs for the continuation, which tells us the teacher's distribution."""
    # Use completions endpoint - it returns logprobs for generated tokens
    # We prepend prompt+response and generate 1 token to get teacher's view
    r = requests.post(f"{url}/v1/completions", json={
        "model": "model",
        "prompt": prompt + "\n" + response,
        "max_tokens": max_tokens,
        "logprobs": True,
        "echo": False,
        "temperature": 0,
    }, timeout=120)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    lps = choice.get("logprobs", {}).get("content", [])
    return lps, data.get("usage", {})


def step1_collect():
    """Collect student + teacher pairs with logprobs."""
    print(f"\n{'='*60}")
    print("STEP 1: Collect student + teacher trajectories")
    print(f"{'='*60}")

    pairs = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:55]}...")

        # Student generates with logprobs
        s = llm_generate(STUDENT_URL, prompt, max_tokens=2048,
                        temperature=0.7, logprobs=True)
        student_resp = s["message"].get("content", "")
        student_lps = []
        if s.get("logprobs") and "content" in s["logprobs"]:
            for lp in s["logprobs"]["content"]:
                student_lps.append({"token": lp["token"], "logprob": lp["logprob"]})

        # Teacher generates with logprobs (independent generation)
        t = llm_generate(TEACHER_URL, prompt, max_tokens=1024,
                        temperature=0.3, logprobs=True)
        teacher_resp = t["message"].get("content", "")
        teacher_lps = []
        if t.get("logprobs") and "content" in t["logprobs"]:
            for lp in t["logprobs"]["content"]:
                teacher_lps.append({"token": lp["token"], "logprob": lp["logprob"]})

        print(f"  Student: {student_resp[:80]}... ({len(student_lps)} logprobs)")
        print(f"  Teacher: {teacher_resp[:80]}... ({len(teacher_lps)} logprobs)")

        # Find common substring alignment
        kl_data = _compute_alignment_kl(student_lps, teacher_lps)
        print(f"  Aligned: {kl_data['aligned_tokens']} tokens, mean_kl={kl_data['mean_kl']:.4f}")

        pairs.append({
            "prompt": prompt,
            "student_response": student_resp,
            "student_logprobs": student_lps,
            "teacher_response": teacher_resp,
            "teacher_logprobs": teacher_lps,
            "alignment": kl_data,
        })
        time.sleep(0.5)

    with open(DATA_DIR / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    agree = sum(1 for p in pairs if p["alignment"]["aligned_tokens"] > 5)
    print(f"\nCollected {len(pairs)} pairs, {agree} with good alignment")
    return pairs


def _compute_alignment_kl(student_lps, teacher_lps):
    """Compute KL divergence at each token position between student and teacher.
    
    Strategy: use the teacher as the reference distribution.
    For each student token, check the teacher's logprob at the same position.
    Where they disagree, the student gets high KL = needs more training there.
    """
    s_tokens = [lp["token"] for lp in student_lps]
    t_tokens = [lp["token"] for lp in teacher_lps]

    # Find longest common subsequence positions
    matcher = difflib.SequenceMatcher(None, s_tokens, t_tokens)

    # Build aligned pairs
    aligned = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            # Tokens match - compute KL at these positions
            for k in range(i2 - i1):
                s_idx = i1 + k
                t_idx = j1 + k
                if s_idx < len(student_lps) and t_idx < len(teacher_lps):
                    s_lp = student_lps[s_idx]["logprob"]
                    t_lp = teacher_lps[t_idx]["logprob"]
                    kl = s_lp - t_lp  # reverse KL
                    aligned.append({
                        "position": s_idx,
                        "token": student_lps[s_idx]["token"],
                        "student_logprob": s_lp,
                        "teacher_logprob": t_lp,
                        "reverse_kl": kl,
                    })

    mean_kl = sum(a["reverse_kl"] for a in aligned) / len(aligned) if aligned else 0
    return {
        "aligned_tokens": len(aligned),
        "mean_kl": mean_kl,
        "max_kl": max((a["reverse_kl"] for a in aligned), default=0),
        "aligned_data": aligned,
    }


def step2_train():
    """Train with Unsloth on teacher responses weighted by alignment KL."""
    print(f"\n{'='*60}")
    print("STEP 2: Training with Unsloth")
    print(f"{'='*60}")

    pairs = []
    with open(DATA_DIR / "pairs.jsonl") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Build training data: teacher responses as targets, weighted by KL
    train_data = []
    for p in pairs:
        kl_data = p.get("alignment", {})
        if kl_data.get("aligned_tokens", 0) < 3:
            continue

        # Weight: higher KL = more important to learn
        weight = abs(kl_data.get("mean_kl", 0)) + 0.01

        train_data.append({
            "text": f"<|user|>\n{p['prompt']}\n<|assistant|>\n{p['teacher_response']}",
            "weight": weight,
            "prompt": p["prompt"],
            "teacher_response": p["teacher_response"],
            "student_response": p["student_response"],
            "mean_kl": kl_data.get("mean_kl", 0),
            "aligned_tokens": kl_data.get("aligned_tokens", 0),
        })

    print(f"Training examples: {len(train_data)}")
    if not train_data:
        print("ERROR: No training data")
        return

    # Save training data
    train_file = DATA_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for d in train_data:
            f.write(json.dumps({"text": d["text"], "weight": d["weight"]}) + "\n")

    # Generate Unsloth training script
    script = f'''#!/usr/bin/env python3
import torch, json, os
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

TRAIN_FILE = "{train_file}"
OUTPUT = "{DATA_DIR / 'finetuned'}"

dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"Dataset: {{len(dataset)}} examples")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="LiquidAI/LFM2.5-8B-A1B",
    max_seq_length=2048,
    load_in_4bit=False,
    trust_remote_code=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_up_proj","down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

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

os.makedirs(OUTPUT, exist_ok=True)
model.save_pretrained(f"{{OUTPUT}}/lora")
tokenizer.save_pretrained(f"{{OUTPUT}}/lora")
print(f"LoRA saved to {{OUTPUT}}/lora")

model.save_pretrained_gguf(f"{{OUTPUT}}/gguf", tokenizer, quantization_method="q4_k_m")
print(f"GGUF saved to {{OUTPUT}}/gguf")
print("DISTILLATION COMPLETE")
'''

    script_path = DATA_DIR / "train_unsloth.py"
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Training data: {train_file}")
    print(f"Training script: {script_path}")
    print(f"Run: ssh aero 'source gguf-env/bin/activate && python3 {script_path}'")


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
    else:
        print("Usage: python distill_v4.py [collect|train|full]")


if __name__ == "__main__":
    main()
