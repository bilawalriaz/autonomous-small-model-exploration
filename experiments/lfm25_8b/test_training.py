#!/usr/bin/env python3
"""Test Unsloth training with existing v3 pairs data."""
import json, os, torch
from pathlib import Path

PAIRS_FILE = Path("/home/billz/results/distill_v3/pairs.jsonl")
TRAIN_FILE = Path("/home/billz/results/distill_test/train.jsonl")
OUTPUT_DIR = Path("/home/billz/results/distill_test/finetuned")

# Load pairs
pairs = []
with open(PAIRS_FILE) as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} pairs")

# Build training data: teacher responses as targets
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_FILE, "w") as f:
    for p in pairs:
        teacher_resp = p.get("teacher_response", "")
        if not teacher_resp:
            continue
        text = f"<|user|>\n{p['prompt']}\n<|assistant|>\n{teacher_resp}"
        f.write(json.dumps({"text": text}) + "\n")

print(f"Training data: {TRAIN_FILE}")

# Write Unsloth training script
script = f'''#!/usr/bin/env python3
import torch, json, os
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

TRAIN_FILE = "{TRAIN_FILE}"
OUTPUT = "{OUTPUT_DIR}"

dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"Dataset: {{len(dataset)}} examples")

# Load with Unsloth
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
print("TEST TRAINING COMPLETE")
'''

script_path = Path("/home/billz/results/distill_test/train_unsloth.py")
with open(script_path, "w") as f:
    f.write(script)

print(f"Training script: {script_path}")
print(f"Run: ssh aero 'source gguf-env/bin/activate && python3 {script_path}'")
