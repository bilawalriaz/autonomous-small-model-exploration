#!/usr/bin/env python3
"""LFM2.5-230M LoRA Rank Sweep — test r=2,4,8,16 on hub layers."""
import json, torch, sys, numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_230m_rank_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

HUB_LAYERS = [0, 2, 4, 5]  # L0 (hub), L2, L4, L5

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            self.examples.append({"input_ids": enc["input_ids"].squeeze(0), "labels": enc["input_ids"].squeeze(0).clone()})
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # Load training data
    data_path = PROJECT / "data" / "tasks" / "canonical_short" / "arithmetic.json"
    data = json.load(open(data_path))
    examples = [e for e in data.get("examples", []) if e.get("metadata", {}).get("split") == "train"][:80]
    train_texts = [f"{e['prompt']}{e['target']}" for e in examples]
    dataset = SimpleDataset(train_texts, tok)
    print(f"Training examples: {len(dataset)}")
    
    eval_prompts = ["The capital of France is", "2 + 2 =", '{"key":', "def hello():"]
    
    ranks = [2, 4, 8, 16]
    results = []
    
    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"RANK {rank}")
        print(f"{'='*60}")
        
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        
        # Compute baseline KL
        baseline_kls = []
        for prompt in eval_prompts:
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                bl = model(ids).logits[:, -1, :]
            baseline_kls.append(bl)
        
        lora_config = LoraConfig(
            r=rank, lora_alpha=rank * 2,
            target_modules=["out_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=HUB_LAYERS,
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        
        peft_model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")
        
        output_dir = str(ADAPTERS_DIR / f"r{rank}")
        training_args = TrainingArguments(
            output_dir=output_dir, max_steps=100, per_device_train_batch_size=1,
            gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine",
            warmup_steps=5, weight_decay=0.01, logging_steps=50, save_steps=100,
            save_total_limit=1, bf16=True, gradient_checkpointing=True,
            report_to="none", seed=args.seed, remove_unused_columns=False,
        )
        
        trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset)
        train_result = trainer.train()
        final_loss = train_result.training_loss
        
        # Post-training KL
        post_kls = []
        for prompt in eval_prompts:
            ids = tok(prompt, return_tensors="pt").input_ids.to(peft_model.device)
            with torch.no_grad():
                pt = peft_model(ids).logits[:, -1, :]
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(pt.float(), -1),
                torch.softmax(baseline_kls[eval_prompts.index(prompt)].float(), -1),
                reduction='batchmean'
            ).item()
            post_kls.append(kl)
        
        avg_kl = np.mean(post_kls)
        print(f"  Loss: {final_loss:.4f}, KL shift: {avg_kl:.4f}")
        
        # Save adapter
        peft_model.save_pretrained(output_dir)
        
        results.append({
            "rank": rank, "trainable_params": trainable, "final_loss": round(final_loss, 4),
            "kl_shift": round(avg_kl, 4), "per_prompt_kl": [round(k, 4) for k in post_kls],
        })
        
        del peft_model, model
        torch.cuda.empty_cache()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"lfm2_230m_rank_sweep_seed{args.seed}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "seed": args.seed, "timestamp": ts, "hub_layers": HUB_LAYERS, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print(f"\n{'Rank':<6} {'Params':>10} {'Loss':>8} {'KL Shift':>10}")
    print("-" * 36)
    for r in results:
        print(f"r={r['rank']:<4} {r['trainable_params']:>10,} {r['final_loss']:>8.4f} {r['kl_shift']:>10.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
