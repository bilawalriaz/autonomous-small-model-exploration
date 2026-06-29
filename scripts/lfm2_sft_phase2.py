#!/usr/bin/env python3
"""
LFM2.5-230M SFT Experiment Sweep — Phase 2
New datasets (smol-smoltalk, OpenHermes, Tulu3, UltraInteract) + optimizer sweep (Adafactor, Lion, ScheduleFreeAdamW).
"""
import json, torch, sys, os, time, gc, traceback
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_sft_phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

HUB_LAYERS = [0, 2, 4, 5]

# ══════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS — PHASE 2
# ══════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # --- NEW DATASETS (hub LoRA, r=8, lr=2e-4, AdamW, 300 steps) ---
    {"id": "smol_smoltalk_300",      "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra",   "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "smoltalk_full_300",      "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra", "n": 1000, "steps": 300, "lr": 2e-4, "r": 8, "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "openhermes_300",         "dataset": "teknium/OpenHermes-2.5",     "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "tulu3_300",              "dataset": "allenai/tulu-3-sft-mixture", "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "ultrainteract_300",      "dataset": "openbmb/UltraInteract_sft",  "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "finetome_100k_300",      "dataset": "mlabonne/FineTome-100k",     "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},

    # --- OPTIMIZER SWEEP (UltraChat data, hub LoRA, r=8, 300 steps) ---
    {"id": "ultrachat_adafactor",    "dataset": "stingning/ultrachat",        "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adafactor", "batch": 2, "grad_accum": 4},
    {"id": "ultrachat_lion",         "dataset": "stingning/ultrachat",        "subset": None,              "n": 1000, "steps": 300, "lr": 7e-5, "r": 8,  "target": "hub", "optimizer": "lion", "batch": 2, "grad_accum": 4},
    {"id": "ultrachat_schedulefree", "dataset": "stingning/ultrachat",        "subset": None,              "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "schedulefree", "batch": 2, "grad_accum": 4},

    # --- OPTIMIZER + BEST DATASET (smol-smoltalk if available, else UltraChat) ---
    {"id": "smol_adafactor_300",     "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra",   "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adafactor", "batch": 2, "grad_accum": 4},
    {"id": "smol_lion_300",          "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra",   "n": 1000, "steps": 300, "lr": 7e-5, "r": 8,  "target": "hub", "optimizer": "lion", "batch": 2, "grad_accum": 4},
    {"id": "smol_schedulefree_300",  "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra",   "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "schedulefree", "batch": 2, "grad_accum": 4},

    # --- LONG TRAINING on best new dataset ---
    {"id": "smol_5k_1000",           "dataset": "HuggingFaceTB/smoltalk",     "subset": "smol-magpie-ultra",   "n": 5000, "steps": 1000,"lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
    {"id": "tulu3_5k_1000",          "dataset": "allenai/tulu-3-sft-mixture", "subset": None,              "n": 5000, "steps": 1000,"lr": 2e-4, "r": 8,  "target": "hub", "optimizer": "adamw_torch", "batch": 2, "grad_accum": 4},
]

EVAL_PROMPTS = [
    "The capital of France is",
    '{"name": "Alice", "age":',
    "def fibonacci(n):",
    "Explain quantum entanglement in simple terms.",
    "Write a JSON object with keys: city, population, country for Tokyo.",
    "247 + 358 =",
    "Translate 'Good morning' to French, Spanish, and German.",
    "List 3 benefits of exercise.",
    "def binary_search(arr, target):",
    "The chemical formula for water is",
]


def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    return model, tok


def get_target_modules(target_type):
    all_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
    if target_type in ("hub", "all"):
        return all_modules
    elif target_type == "o_proj":
        return ["out_proj"]
    elif target_type == "hub_oproj":
        return ["out_proj"]
    return all_modules


def get_layers_to_transform(target_type):
    if target_type in ("hub", "hub_oproj"):
        return HUB_LAYERS
    return None


# ══════════════════════════════════════════════════════════════
# DATASET LOADING (handles subsets + multi-turn formats)
# ══════════════════════════════════════════════════════════════
def load_and_format_dataset(ds_id, subset, n):
    from datasets import load_dataset
    print(f"  Loading {ds_id} (subset={subset}, n={n})...")
    try:
        if subset:
            ds = load_dataset(ds_id, subset, split=f"train[:{n}]", trust_remote_code=True)
        else:
            ds = load_dataset(ds_id, split=f"train[:{n}]", trust_remote_code=True)
    except Exception as e:
        # Try without subset
        print(f"  Subset failed ({e}), trying without subset...")
        ds = load_dataset(ds_id, split=f"train[:{n}]", trust_remote_code=True)

    texts = []
    for ex in ds:
        text = format_example(ds_id, ex)
        if text and len(text.strip()) > 20:
            texts.append(text)
    print(f"  Formatted {len(texts)} examples")
    return texts


def format_example(ds_id, ex):
    """Format dataset examples into instruction-following text. Handles many formats."""
    ds_lower = ds_id.lower()

    # smoltalk / smol-smoltalk — messages format
    if "smoltalk" in ds_lower or "smol" in ds_lower:
        msgs = ex.get("messages", [])
        if not msgs:
            return None
        parts = []
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                parts.append(f"### Response:\n{content}")
        return "\n\n".join(parts) if parts else None

    # OpenHermes — conversations format
    if "openhermes" in ds_lower or "hermes" in ds_lower:
        conv = ex.get("conversations", [])
        if not conv:
            return None
        parts = []
        for turn in conv:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("system", "human", "user"):
                parts.append(f"### Instruction:\n{content}")
            elif role in ("gpt", "assistant"):
                parts.append(f"### Response:\n{content}")
        return "\n\n".join(parts) if parts else None

    # Tulu 3 — messages format
    if "tulu" in ds_lower:
        msgs = ex.get("messages", [])
        if not msgs:
            return None
        parts = []
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                parts.append(f"### Response:\n{content}")
        return "\n\n".join(parts) if parts else None

    # UltraInteract — messages or prompt/completion
    if "ultrainteract" in ds_lower:
        msgs = ex.get("messages", [])
        if msgs:
            parts = []
            for msg in msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    parts.append(f"### Instruction:\n{content}")
                elif role == "assistant":
                    parts.append(f"### Response:\n{content}")
            return "\n\n".join(parts) if parts else None
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "")
        if prompt and completion:
            return f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"
        return None

    # FineTome — messages format
    if "finetome" in ds_lower or "tome" in ds_lower:
        msgs = ex.get("conversations", ex.get("messages", []))
        if not msgs:
            return None
        parts = []
        for turn in msgs:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("system", "human", "user"):
                parts.append(f"### Instruction:\n{content}")
            elif role in ("gpt", "assistant"):
                parts.append(f"### Response:\n{content}")
        return "\n\n".join(parts) if parts else None

    # UltraChat — data field (flat list)
    if "ultrachat" in ds_lower:
        data = ex.get("data", ex.get("messages", []))
        if not data:
            return None
        if isinstance(data[0], str):
            parts = []
            for i, text in enumerate(data):
                if i % 2 == 0:
                    parts.append(f"### Instruction:\n{text}")
                else:
                    parts.append(f"### Response:\n{text}")
            return "\n\n".join(parts) if parts else None
        parts = []
        for turn in data:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                parts.append(f"### Response:\n{content}")
        return "\n\n".join(parts) if parts else None

    # Fallback: try common fields
    for k in ("text", "content", "output", "response"):
        if k in ex and isinstance(ex[k], str) and len(ex[k]) > 10:
            return ex[k]
    return None


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.examples = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_len,
                          padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            self.examples.append({"input_ids": ids, "labels": ids.clone()})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


# ══════════════════════════════════════════════════════════════
# OPTIMIZER FACTORY
# ══════════════════════════════════════════════════════════════
def create_optimizer(model, opt_name, lr, weight_decay=0.01):
    """Create optimizer by name. Returns (optimizer, scheduler_kwargs)."""
    opt_lower = opt_name.lower()

    if opt_lower == "adamw":
        from torch.optim import AdamW
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), {}

    elif opt_lower == "adafactor":
        from transformers import Adafactor
        opt = Adafactor(
            model.parameters(),
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            clip_threshold=1.0,
            decay_rate=-0.8,
            weight_decay=weight_decay,
        )
        return opt, {"lr_scheduler_type": "cosine"}

    elif opt_lower == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError:
            os.system("pip install lion-pytorch -q")
            from lion_pytorch import Lion
        return Lion(model.parameters(), lr=lr, weight_decay=1.0), {}

    elif opt_lower == "schedulefree":
        try:
            import schedulefree
        except ImportError:
            os.system("pip install schedulefree -q")
            import schedulefree
        opt = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=lr, warmup_steps=50, weight_decay=weight_decay
        )
        return opt, {"lr_scheduler_type": "constant"}  # schedule-free needs no schedule

    else:
        from torch.optim import AdamW
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), {}


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def run_single_experiment(exp_cfg, tok):
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer

    exp_id = exp_cfg["id"]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"  Dataset: {exp_cfg['dataset']} (subset={exp_cfg.get('subset')})")
    print(f"  Steps: {exp_cfg['steps']}, LR: {exp_cfg['lr']}, Rank: {exp_cfg['r']}")
    print(f"  Target: {exp_cfg['target']}, Optimizer: {exp_cfg['optimizer']}")
    print(f"{'='*60}")

    texts = load_and_format_dataset(exp_cfg["dataset"], exp_cfg.get("subset"), exp_cfg["n"])
    if not texts:
        return {"id": exp_id, "error": "No valid examples after formatting"}

    dataset = SFTDataset(texts, tok, max_len=512)
    print(f"  Dataset size: {len(dataset)}")

    model, _ = load_model_and_tokenizer()

    target_modules = get_target_modules(exp_cfg["target"])
    layers_to_transform = get_layers_to_transform(exp_cfg["target"])

    lora_kwargs = {
        "r": exp_cfg["r"],
        "lora_alpha": exp_cfg["r"] * 2,
        "target_modules": target_modules,
        "task_type": TaskType.CAUSAL_LM,
        "bias": "none",
    }
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform

    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    output_dir = str(ADAPTERS_DIR / exp_id)
    scheduler_type = "cosine"

    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=exp_cfg["steps"],
        per_device_train_batch_size=exp_cfg["batch"],
        gradient_accumulation_steps=exp_cfg["grad_accum"],
        learning_rate=exp_cfg["lr"],
        lr_scheduler_type=scheduler_type,
        warmup_steps=max(10, exp_cfg["steps"] // 10),
        weight_decay=0.01,
        logging_steps=max(1, exp_cfg["steps"] // 20),
        save_steps=exp_cfg["steps"],
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        optim="adamw_torch",
        # Note: custom optimizers need to be passed via trainer override
    )

    # For non-AdamW optimizers, create custom optimizer and pass to trainer
    if exp_cfg["optimizer"] != "adamw":
        optimizer, extra_kwargs = create_optimizer(model, exp_cfg["optimizer"], exp_cfg["lr"])
        trainer = Trainer(model=model, args=args, train_dataset=dataset, optimizers=(optimizer, None))
    else:
        trainer = Trainer(model=model, args=args, train_dataset=dataset)

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    model.save_pretrained(output_dir)

    print("  Computing KL shift...")
    kl_shift = compute_kl_shift(model, tok)

    print("  Measuring hub preservation...")
    hub_kls = measure_hub_kl(model, tok)

    loss_curve = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_curve.append({"step": entry.get("step", 0), "loss": round(entry["loss"], 4)})

    final_loss = result.training_loss

    print(f"\n  RESULTS for {exp_id}:")
    print(f"    Final loss: {final_loss:.4f}")
    print(f"    KL shift: {kl_shift}")
    print(f"    Hub KLs: {hub_kls}")
    print(f"    Training time: {train_time:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "id": exp_id,
        "dataset": exp_cfg["dataset"],
        "subset": exp_cfg.get("subset"),
        "n_examples": len(texts),
        "steps": exp_cfg["steps"],
        "lr": exp_cfg["lr"],
        "lora_r": exp_cfg["r"],
        "target": exp_cfg["target"],
        "optimizer": exp_cfg["optimizer"],
        "trainable_params": trainable,
        "total_params": total,
        "final_loss": round(final_loss, 4),
        "kl_shift": kl_shift,
        "hub_kls": hub_kls,
        "loss_curve": loss_curve,
        "train_time_s": round(train_time, 1),
    }


def compute_kl_shift(model, tok, prompts=None):
    from transformers import AutoModelForCausalLM
    if prompts is None:
        prompts = EVAL_PROMPTS
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
    return round(float(np.mean(kls)), 4)


def measure_hub_kl(model, tok, prompt="The capital of France is"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        baseline = model(ids).logits
    if hasattr(model, 'base_model'):
        layers = model.base_model.model.model.layers
    else:
        layers = model.model.layers
    hub_kls = {}
    for li in HUB_LAYERS:
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return torch.zeros_like(output)
            return (torch.zeros_like(output[0]),) + output[1:]
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="all")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-from", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"LFM2.5-230M SFT Phase 2 Sweep")
    print(f"Timestamp: {timestamp}")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    if args.experiments == "all":
        exps = EXPERIMENTS
    else:
        ids = [x.strip() for x in args.experiments.split(",")]
        exps = [e for e in EXPERIMENTS if e["id"] in ids]

    if args.start_from:
        start_idx = next((i for i, e in enumerate(exps) if e["id"] == args.start_from), 0)
        exps = exps[start_idx:]

    print(f"Running {len(exps)} experiments")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_results = []

    for i, exp in enumerate(exps):
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(exps)}] {exp['id']}")
        print(f"{'#'*60}")

        if args.skip_existing:
            existing = list(RESULTS_DIR.glob(f"lfm2_sft2_{exp['id']}_*.json"))
            if existing:
                print(f"  Already done ({existing[0].name}), skipping")
                continue

        try:
            result = run_single_experiment(exp, tok)

            # Qualitative eval
            adapter_dir = ADAPTERS_DIR / exp["id"]
            if adapter_dir.exists():
                print("  Running qualitative eval...")
                from peft import PeftModel
                model, _ = load_model_and_tokenizer()
                model = PeftModel.from_pretrained(model, str(adapter_dir))
                qual = []
                model.eval()
                for prompt in EVAL_PROMPTS:
                    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
                    t0 = time.time()
                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=200, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
                    dt = time.time() - t0
                    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    n_tok = out.shape[1] - ids.shape[1]
                    qual.append({"prompt": prompt, "generation": gen.strip(),
                               "n_tokens": n_tok, "time_s": round(dt, 2),
                               "tok_per_s": round(n_tok/dt, 1) if dt > 0 else 0})
                result["qualitative"] = qual
                del model
                torch.cuda.empty_cache()
                gc.collect()

            result_file = RESULTS_DIR / f"lfm2_sft2_{exp['id']}_{timestamp}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {result_file.name}")
            all_results.append(result)

        except Exception as e:
            print(f"  ERROR in {exp['id']}: {e}")
            traceback.print_exc()
            all_results.append({"id": exp["id"], "error": str(e)})
            torch.cuda.empty_cache()
            gc.collect()

    combined_file = RESULTS_DIR / f"lfm2_sft2_sweep_{timestamp}.json"
    with open(combined_file, "w") as f:
        json.dump({"model": MODEL, "timestamp": timestamp,
                   "n_experiments": len(all_results), "results": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PHASE 2 SWEEP COMPLETE")
    print(f"  Results: {combined_file}")
    print(f"  Successful: {sum(1 for r in all_results if 'error' not in r)}/{len(all_results)}")
    print(f"{'='*60}")

    print(f"\n{'ID':<30} {'Loss':>8} {'KL':>8} {'Optim':>12} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['id']:<30} {'ERROR':>8}")
        else:
            print(f"{r['id']:<30} {r['final_loss']:>8.4f} {r['kl_shift']:>8.4f} {r.get('optimizer','?'):>12} {r['train_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
