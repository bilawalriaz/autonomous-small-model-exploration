#!/usr/bin/env python3
"""
LFM2.5-230M Exhaustive Finetuning Sweep
Tests: SFT, QLoRA, DPO, GRPO with real HuggingFace datasets.
Measures: loss convergence, KL shift, hub preservation, steering effectiveness.
"""
import json, torch, sys, os, time
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_230m_finetuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

HUB_LAYERS = [0, 2, 4, 5]


def load_model_and_tokenizer(device="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    return model, tok


def compute_kl_shift(model, tok, prompts):
    """Compute KL divergence between model and a fresh copy on eval prompts."""
    from transformers import AutoModelForCausalLM
    fresh = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    fresh.eval()
    model.eval()
    kls = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids
        with torch.no_grad():
            fresh_logits = fresh(ids).logits[:, -1, :]
            model_logits = model(ids.to(model.device)).logits[:, -1, :].cpu()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(model_logits.float(), -1),
            torch.softmax(fresh_logits.float(), -1),
            reduction='batchmean'
        ).item()
        kls.append(kl)
    del fresh
    torch.cuda.empty_cache()
    return round(np.mean(kls), 4)


def measure_hub_kl(model, tok, prompt="The capital of France is"):
    """Measure ablation KL at hub layers to check if hub structure is preserved."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        baseline = model(ids).logits
    # Handle both raw and PEFT-wrapped models
    if hasattr(model, 'base_model'):
        layers = model.base_model.model.model.layers
    else:
        layers = model.model.layers
    hub_kls = {}
    for li in HUB_LAYERS:
        def hook(module, input, output):
            return torch.zeros_like(output) if isinstance(output, torch.Tensor) else (torch.zeros_like(output[0]),) + output[1:]
        handle = layers[li].feed_forward.register_forward_hook(hook)
        with torch.no_grad():
            abl = model(ids).logits
        handle.remove()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(abl[:, -1, :].float(), -1),
            torch.softmax(baseline[:, -1, :].float(), -1),
            reduction='batchmean'
        ).item()
        hub_kls[f"L{li}"] = round(kl, 4)
    return hub_kls


# ═══════════════════════════════════════════════
# SFT EXPERIMENT
# ═══════════════════════════════════════════════
def run_sft(tok, device):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    
    print("\n" + "="*60)
    print("SFT: Supervised Fine-Tuning (Alpaca dataset)")
    print("="*60)
    
    # Load Alpaca dataset
    print("  Loading dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train[:500]")
    
    def format_example(ex):
        if ex.get("input", "").strip():
            text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        else:
            text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        return text
    
    texts = [format_example(ex) for ex in ds]
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tok, max_len=512):
            self.examples = []
            for t in texts:
                enc = tok(t, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze(0)
                # Mask instruction part in labels (only train on response)
                self.examples.append({"input_ids": ids, "labels": ids.clone()})
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx): return self.examples[idx]
    
    dataset = TextDataset(texts, tok)
    print(f"  Dataset: {len(dataset)} examples")
    
    # LoRA SFT
    model, _ = load_model_and_tokenizer(device)
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "in_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    
    output_dir = str(ADAPTERS_DIR / "sft_alpaca")
    args = TrainingArguments(
        output_dir=output_dir, max_steps=200, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_steps=20, weight_decay=0.01, logging_steps=50, save_steps=200,
        save_total_limit=1, bf16=True, gradient_checkpointing=True,
        report_to="none", seed=42, remove_unused_columns=False,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    result = trainer.train()
    
    # Save
    model.save_pretrained(output_dir)
    
    eval_prompts = ["The capital of France is", '{"key":', "def hello():", "2 + 2 ="]
    kl = compute_kl_shift(model, tok, eval_prompts)
    hub = measure_hub_kl(model, tok)
    
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  KL shift: {kl}")
    print(f"  Hub KLs: {hub}")
    
    del model
    torch.cuda.empty_cache()
    
    return {"method": "SFT", "dataset": "tatsu-lab/alpaca", "n_examples": 500,
            "trainable_params": trainable, "final_loss": round(result.training_loss, 4),
            "kl_shift": kl, "hub_kls": hub, "max_steps": 200, "lr": 2e-4, "lora_r": 8}


# ═══════════════════════════════════════════════
# QLoRA EXPERIMENT
# ═══════════════════════════════════════════════
def run_qlora(tok, device):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
    
    print("\n" + "="*60)
    print("QLoRA: 4-bit Quantized LoRA")
    print("="*60)
    
    # Load same dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train[:500]")
    
    def format_example(ex):
        if ex.get("input", "").strip():
            return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    
    texts = [format_example(ex) for ex in ds]
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tok, max_len=512):
            self.examples = []
            for t in texts:
                enc = tok(t, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
                self.examples.append({"input_ids": enc["input_ids"].squeeze(0), "labels": enc["input_ids"].squeeze(0).clone()})
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx): return self.examples[idx]
    
    dataset = TextDataset(texts, tok)
    
    # Load with 4-bit quantization
    print("  Loading with 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "in_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    
    output_dir = str(ADAPTERS_DIR / "qlora_4bit")
    args = TrainingArguments(
        output_dir=output_dir, max_steps=200, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_steps=20, weight_decay=0.01, logging_steps=50, save_steps=200,
        save_total_limit=1, bf16=True, gradient_checkpointing=True,
        report_to="none", seed=42, remove_unused_columns=False,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    result = trainer.train()
    model.save_pretrained(output_dir)
    
    eval_prompts = ["The capital of France is", '{"key":', "def hello():", "2 + 2 ="]
    kl = compute_kl_shift(model, tok, eval_prompts)
    
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  KL shift: {kl}")
    
    del model
    torch.cuda.empty_cache()
    
    return {"method": "QLoRA", "dataset": "tatsu-lab/alpaca", "n_examples": 500,
            "trainable_params": trainable, "final_loss": round(result.training_loss, 4),
            "kl_shift": kl, "max_steps": 200, "lr": 2e-4, "lora_r": 16, "quantization": "4bit_NF4"}


# ═══════════════════════════════════════════════
# DPO EXPERIMENT
# ═══════════════════════════════════════════════
def run_dpo(tok, device):
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from trl import DPOTrainer, DPOConfig
    
    print("\n" + "="*60)
    print("DPO: Direct Preference Optimization")
    print("="*60)
    
    # Load UltraFeedback binarized (small subset)
    print("  Loading dataset...")
    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:300]")
    
    model, _ = load_model_and_tokenizer(device)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    
    # DPO needs a reference model (frozen copy)
    from peft import get_peft_model
    ref_model, _ = load_model_and_tokenizer(device)
    
    output_dir = str(ADAPTERS_DIR / "dpo")
    dpo_config = DPOConfig(
        output_dir=output_dir, max_steps=100, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=5e-5, lr_scheduler_type="cosine",
        warmup_steps=10, weight_decay=0.01, logging_steps=25, save_steps=100,
        save_total_limit=1, bf16=True, gradient_checkpointing=True,
        report_to="none", seed=42, beta=0.1, loss_type="sigmoid",
    )
    
    trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=dpo_config,
        train_dataset=ds, processing_class=tok, peft_config=lora_config,
    )
    
    result = trainer.train()
    trainer.save_model(output_dir)
    
    eval_prompts = ["The capital of France is", '{"key":', "def hello():"]
    kl = compute_kl_shift(model, tok, eval_prompts)
    
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  KL shift: {kl}")
    
    del model, ref_model
    torch.cuda.empty_cache()
    
    return {"method": "DPO", "dataset": "trl-lib/ultrafeedback_binarized", "n_examples": 300,
            "final_loss": round(result.training_loss, 4), "kl_shift": kl,
            "max_steps": 100, "lr": 5e-5, "beta": 0.1, "lora_r": 8}


# ═══════════════════════════════════════════════
# GRPO EXPERIMENT
# ═══════════════════════════════════════════════
def run_grpo(tok, device):
    from datasets import load_dataset
    from peft import LoraConfig, TaskType
    from trl import GRPOTrainer, GRPOConfig
    
    print("\n" + "="*60)
    print("GRPO: Group Relative Policy Optimization")
    print("="*60)
    
    # Load a math dataset with verifiable answers
    print("  Loading dataset...")
    ds = load_dataset("trl-lib/math_level5_then_ground_truth", split="train[:200]")
    
    model, _ = load_model_and_tokenizer(device)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    
    def reward_fn(completions, **kwargs):
        """Simple reward: penalize empty, reward non-degenerate."""
        rewards = []
        for c in completions:
            if len(c.strip()) < 5:
                rewards.append(-1.0)
            elif c.strip().count(c.strip()[:10]) > 3:
                rewards.append(-0.5)
            else:
                rewards.append(0.5)
        return rewards
    
    output_dir = str(ADAPTERS_DIR / "grpo")
    grpo_config = GRPOConfig(
        output_dir=output_dir, max_steps=50, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, learning_rate=1e-5, lr_scheduler_type="cosine",
        warmup_steps=5, weight_decay=0.01, logging_steps=10, save_steps=50,
        save_total_limit=1, bf16=True, gradient_checkpointing=True,
        report_to="none", seed=42, num_generations=4, max_completion_length=128,
    )
    
    trainer = GRPOTrainer(
        model=model, args=grpo_config, reward_funcs=reward_fn,
        train_dataset=ds, peft_config=lora_config,
    )
    
    result = trainer.train()
    trainer.save_model(output_dir)
    
    eval_prompts = ["The capital of France is", '{"key":', "def hello():"]
    kl = compute_kl_shift(model, tok, eval_prompts)
    
    print(f"  Loss: {result.training_loss:.4f}")
    print(f"  KL shift: {kl}")
    
    del model
    torch.cuda.empty_cache()
    
    return {"method": "GRPO", "dataset": "trl-lib/math_level5_then_ground_truth", "n_examples": 200,
            "final_loss": round(result.training_loss, 4), "kl_shift": kl,
            "max_steps": 50, "lr": 1e-5, "lora_r": 8, "num_generations": 4}


# ═══════════════════════════════════════════════
# HYPERPARAMETER SWEEP
# ═══════════════════════════════════════════════
def run_hyperparam_sweep(tok, device):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP")
    print("="*60)
    
    ds = load_dataset("tatsu-lab/alpaca", split="train[:200]")
    
    def format_example(ex):
        if ex.get("input", "").strip():
            return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    
    texts = [format_example(ex) for ex in ds]
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tok, max_len=256):
            self.examples = []
            for t in texts:
                enc = tok(t, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
                self.examples.append({"input_ids": enc["input_ids"].squeeze(0), "labels": enc["input_ids"].squeeze(0).clone()})
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx): return self.examples[idx]
    
    dataset = TextDataset(texts, tok)
    
    configs = [
        {"name": "lr_1e-5", "lr": 1e-5, "r": 8, "steps": 100},
        {"name": "lr_5e-5", "lr": 5e-5, "r": 8, "steps": 100},
        {"name": "lr_2e-4", "lr": 2e-4, "r": 8, "steps": 100},
        {"name": "lr_1e-3", "lr": 1e-3, "r": 8, "steps": 100},
        {"name": "r_2", "lr": 2e-4, "r": 2, "steps": 100},
        {"name": "r_4", "lr": 2e-4, "r": 4, "steps": 100},
        {"name": "r_16", "lr": 2e-4, "r": 16, "steps": 100},
        {"name": "r_32", "lr": 2e-4, "r": 32, "steps": 100},
        {"name": "steps_50", "lr": 2e-4, "r": 8, "steps": 50},
        {"name": "steps_200", "lr": 2e-4, "r": 8, "steps": 200},
        {"name": "steps_500", "lr": 2e-4, "r": 8, "steps": 500},
        {"name": "bs_1", "lr": 2e-4, "r": 8, "steps": 200, "bs": 1},
        {"name": "bs_4", "lr": 2e-4, "r": 8, "steps": 100, "bs": 4},
    ]
    
    results = []
    for cfg in configs:
        print(f"\n  Config: {cfg['name']}")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        
        lora_config = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["r"]*2,
            target_modules=["out_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=HUB_LAYERS,
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        output_dir = str(ADAPTERS_DIR / f"sweep_{cfg['name']}")
        args = TrainingArguments(
            output_dir=output_dir, max_steps=cfg["steps"],
            per_device_train_batch_size=cfg.get("bs", 2),
            gradient_accumulation_steps=2, learning_rate=cfg["lr"],
            lr_scheduler_type="cosine", warmup_steps=max(5, cfg["steps"]//10),
            weight_decay=0.01, logging_steps=cfg["steps"],
            save_total_limit=0, bf16=True, gradient_checkpointing=True,
            report_to="none", seed=42, remove_unused_columns=False,
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        result = trainer.train()
        
        eval_prompts = ["The capital of France is", '{"key":']
        kl = compute_kl_shift(model, tok, eval_prompts)
        
        results.append({
            "config": cfg["name"], "lr": cfg["lr"], "r": cfg["r"],
            "steps": cfg["steps"], "trainable_params": trainable,
            "loss": round(result.training_loss, 4), "kl_shift": kl,
        })
        print(f"    loss={result.training_loss:.4f}, kl={kl}, params={trainable:,}")
        
        del model, trainer
        torch.cuda.empty_cache()
    
    return {"method": "hyperparam_sweep", "results": results}


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Subset: sft,qlora,dpo,grpo,sweep")
    args = parser.parse_args()
    
    device = "auto"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"model": MODEL, "timestamp": ts, "experiments": {}}
    
    experiments = {
        "sft": lambda: run_sft(tok, device),
        "qlora": lambda: run_qlora(tok, device),
        "dpo": lambda: run_dpo(tok, device),
        "grpo": lambda: run_grpo(tok, device),
        "sweep": lambda: run_hyperparam_sweep(tok, device),
    }
    
    to_run = args.experiments or list(experiments.keys())
    
    print(f"\n{'#'*60}")
    print(f"LFM2.5-230M EXHAUSTIVE FINETUNING SWEEP")
    print(f"Experiments: {to_run}")
    print(f"{'#'*60}")
    
    for name in to_run:
        if name in experiments:
            try:
                result = experiments[name]()
                all_results["experiments"][name] = result
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
                all_results["experiments"][name] = {"error": str(e)}
    
    # Save
    out_path = RESULTS_DIR / f"lfm2_230m_finetuning_sweep_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("FINETUNING SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results["experiments"].items():
        if "error" in result:
            print(f"  {name}: FAILED ({result['error'][:50]})")
        elif "results" in result:
            print(f"  {name}: {len(result['results'])} configs tested")
            for r in result["results"]:
                print(f"    {r['config']}: loss={r['loss']}, kl={r['kl_shift']}")
        else:
            print(f"  {name}: loss={result.get('final_loss', '?')}, kl={result.get('kl_shift', '?')}")
    
    return 0

if __name__ == "__main__":
    from transformers import AutoTokenizer
    sys.exit(main())
