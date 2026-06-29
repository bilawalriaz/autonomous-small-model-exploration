#!/usr/bin/env python3
"""
LFM2.5-230M Comprehensive SFT Experiments
Runs SFT across multiple HF datasets, hyperparameter configs, and LoRA targets.
Measures: loss, KL shift, hub preservation, per-step loss curve.
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
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_sft_sweep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

HUB_LAYERS = [0, 2, 4, 5]
ALL_LAYERS = list(range(14))  # LFM2.5-230M has 14 layers

# ══════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # --- Dataset sweep (hub LoRA, r=8, lr=2e-4, 300 steps) ---
    {"id": "alpaca_300",      "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "slimorca_300",    "dataset": "Open-Orca/SlimOrca",                  "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "dolly_300",       "dataset": "databricks/databricks-dolly-15k",     "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "magicoder_300",   "dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K","ds_field": None,       "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "gsm8k_300",       "dataset": "openai/gsm8k",                       "ds_field": "main",      "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "ultrachat_300",   "dataset": "stingning/ultrachat",                 "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},

    # --- LR sweep (Alpaca, r=8, hub, 300 steps) ---
    {"id": "alpaca_lr1e4",    "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 1e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_lr5e4",    "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 5e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_lr1e3",    "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 1e-3, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},

    # --- Rank sweep (Alpaca, lr=2e-4, hub, 300 steps) ---
    {"id": "alpaca_r2",       "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 2,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_r4",       "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 4,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_r16",      "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 16, "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_r32",      "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 32, "target": "hub",   "batch": 2, "grad_accum": 4},

    # --- Target module sweep (Alpaca, lr=2e-4, r=8, 300 steps) ---
    {"id": "alpaca_alllinear","dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "all",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_oproj",    "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "o_proj","batch": 2, "grad_accum": 4},
    {"id": "alpaca_mlp_only", "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "mlp",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_attn_only","dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "attn",  "batch": 2, "grad_accum": 4},

    # --- Steps sweep (Alpaca, lr=2e-4, r=8, hub) ---
    {"id": "alpaca_100",      "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 100, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_500",      "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 500, "lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "alpaca_1000",     "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 2000, "steps": 1000,"lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},

    # --- Atlas-guided hub-only LoRA (the atlas winner) ---
    {"id": "alpaca_hub_oproj","dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub_oproj","batch": 2, "grad_accum": 4},
    {"id": "slimorca_hub_oproj","dataset": "Open-Orca/SlimOrca",               "ds_field": None,        "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub_oproj","batch": 2, "grad_accum": 4},
    {"id": "magicoder_hub_oproj","dataset": "ise-uiuc/Magicoder-Evol-Instruct-110K","ds_field": None,  "n": 1000, "steps": 300, "lr": 2e-4, "r": 8,  "target": "hub_oproj","batch": 2, "grad_accum": 4},

    # --- Large dataset run (5K examples, 1000 steps) ---
    {"id": "alpaca_5k_1000",  "dataset": "tatsu-lab/alpaca",                    "ds_field": None,        "n": 5000, "steps": 1000,"lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
    {"id": "slimorca_5k_1000","dataset": "Open-Orca/SlimOrca",                  "ds_field": None,        "n": 5000, "steps": 1000,"lr": 2e-4, "r": 8,  "target": "hub",   "batch": 2, "grad_accum": 4},
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

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
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
    """Return LoRA target modules based on strategy."""
    all_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "in_proj"]
    if target_type == "hub":
        return all_modules  # all modules but only on hub layers
    elif target_type == "all":
        return all_modules  # all modules, all layers
    elif target_type == "o_proj":
        return ["out_proj"]
    elif target_type == "mlp":
        return ["in_proj"]  # LFM2 has no gate/up/down_proj; in_proj is the conv projection (1024->3072)
    elif target_type == "attn":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    elif target_type == "hub_oproj":
        return ["out_proj"]  # only o_proj on hub layers
    return all_modules


def get_layers_to_transform(target_type):
    """Return which layers to apply LoRA to."""
    if target_type in ("hub", "hub_oproj"):
        return HUB_LAYERS
    return None  # all layers


# ══════════════════════════════════════════════════════════════
# DATASET LOADING & FORMATTING
# ══════════════════════════════════════════════════════════════
def load_and_format_dataset(ds_id, ds_field, n):
    from datasets import load_dataset
    print(f"  Loading {ds_id} (split: {ds_field or 'train'}, n={n})...")
    if ds_field:
        ds = load_dataset(ds_id, ds_field, split=f"train[:{n}]", trust_remote_code=True)
    else:
        ds = load_dataset(ds_id, split=f"train[:{n}]", trust_remote_code=True)

    texts = []
    for ex in ds:
        text = format_example(ds_id, ex)
        if text and len(text.strip()) > 20:
            texts.append(text)
    print(f"  Formatted {len(texts)} examples")
    return texts


def format_example(ds_id, ex):
    """Format a dataset example into instruction-following text."""
    if "alpaca" in ds_id:
        inp = ex.get("input", "")
        if inp and inp.strip():
            return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{ex['output']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

    elif "SlimOrca" in ds_id or "slimorca" in ds_id.lower():
        conv = ex.get("conversations", [])
        if not conv:
            return None
        parts = []
        for turn in conv:
            role = turn.get("from", turn.get("role", ""))
            text = turn.get("value", turn.get("content", ""))
            if role in ("system", "human", "user"):
                parts.append(f"### Instruction:\n{text}")
            elif role in ("gpt", "assistant"):
                parts.append(f"### Response:\n{text}")
        return "\n\n".join(parts) if parts else None

    elif "dolly" in ds_id:
        ctx = ex.get("context", "")
        if ctx and ctx.strip():
            return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ctx}\n\n### Response:\n{ex['response']}"
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"

    elif "Magicoder" in ds_id or "magicoder" in ds_id.lower():
        return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"

    elif "gsm8k" in ds_id:
        return f"### Instruction:\nSolve this math problem step by step.\n\n### Input:\n{ex['question']}\n\n### Response:\n{ex['answer']}"

    elif "ultrachat" in ds_id:
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
            text = turn.get("content", "")
            if role == "user":
                parts.append(f"### Instruction:\n{text}")
            elif role == "assistant":
                parts.append(f"### Response:\n{text}")
        return "\n\n".join(parts) if parts else None

    # Fallback
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
# TRAINING
# ══════════════════════════════════════════════════════════════
class LossTrackerCallback:
    """Custom callback to track loss every N steps."""
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append({"step": state.global_step, "loss": round(logs["loss"], 4)})


def run_single_experiment(exp_cfg, tok):
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer

    exp_id = exp_cfg["id"]
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"  Dataset: {exp_cfg['dataset']}")
    print(f"  Steps: {exp_cfg['steps']}, LR: {exp_cfg['lr']}, Rank: {exp_cfg['r']}")
    print(f"  Target: {exp_cfg['target']}, Batch: {exp_cfg['batch']}x{exp_cfg['grad_accum']}")
    print(f"{'='*60}")

    # Load and format dataset
    texts = load_and_format_dataset(exp_cfg["dataset"], exp_cfg.get("ds_field"), exp_cfg["n"])
    if not texts:
        return {"id": exp_id, "error": "No valid examples after formatting"}

    dataset = SFTDataset(texts, tok, max_len=512)
    print(f"  Dataset size: {len(dataset)}")

    # Load fresh model
    model, _ = load_model_and_tokenizer()

    # Configure LoRA
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

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Output dir
    output_dir = str(ADAPTERS_DIR / exp_id)

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=exp_cfg["steps"],
        per_device_train_batch_size=exp_cfg["batch"],
        gradient_accumulation_steps=exp_cfg["grad_accum"],
        learning_rate=exp_cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=max(10, exp_cfg["steps"] // 10),
        weight_decay=0.01,
        logging_steps=max(1, exp_cfg["steps"] // 20),
        save_steps=exp_cfg["steps"],  # save only at end
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
    )

    loss_tracker = LossTrackerCallback()

    # Monkey-patch the trainer to capture losses
    trainer = Trainer(model=model, args=args, train_dataset=dataset)

    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    # Save adapter
    model.save_pretrained(output_dir)

    # Evaluate: KL shift
    print("  Computing KL shift...")
    kl_shift = compute_kl_shift(model, tok)

    # Evaluate: hub preservation
    print("  Measuring hub preservation...")
    hub_kls = measure_hub_kl(model, tok)

    # Get training loss curve from log history
    loss_curve = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_curve.append({"step": entry.get("step", 0), "loss": round(entry["loss"], 4)})

    final_loss = result.training_loss

    print(f"\n  RESULTS for {exp_id}:")
    print(f"    Final loss: {final_loss:.4f}")
    print(f"    KL shift: {kl_shift}")
    print(f"    Hub KLs: {hub_kls}")
    print(f"    Trainable params: {trainable:,}")
    print(f"    Training time: {train_time:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "id": exp_id,
        "dataset": exp_cfg["dataset"],
        "n_examples": len(texts),
        "steps": exp_cfg["steps"],
        "lr": exp_cfg["lr"],
        "lora_r": exp_cfg["r"],
        "target": exp_cfg["target"],
        "target_modules": target_modules,
        "layers": layers_to_transform if layers_to_transform else "all",
        "trainable_params": trainable,
        "total_params": total,
        "final_loss": round(final_loss, 4),
        "kl_shift": kl_shift,
        "hub_kls": hub_kls,
        "loss_curve": loss_curve,
        "train_time_s": round(train_time, 1),
    }


# ══════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# QUALITATIVE GENERATION
# ══════════════════════════════════════════════════════════════
def generate_qualitative(model, tok, prompts=None, max_new_tokens=200):
    if prompts is None:
        prompts = EVAL_PROMPTS
    results = []
    model.eval()
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                               temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
        dt = time.time() - t0
        gen_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        n_tokens = out.shape[1] - ids.shape[1]
        results.append({
            "prompt": prompt,
            "generation": gen_text.strip(),
            "n_tokens": n_tokens,
            "time_s": round(dt, 2),
            "tok_per_s": round(n_tokens / dt, 1) if dt > 0 else 0,
        })
    return results


def run_qualitative_for_adapter(adapter_dir, exp_id, tok):
    """Load a saved adapter and run qualitative generation."""
    from peft import PeftModel
    print(f"\n  Qualitative eval for {exp_id}...")
    model, _ = load_model_and_tokenizer()
    try:
        model = PeftModel.from_pretrained(model, adapter_dir)
        results = generate_qualitative(model, tok)
    except Exception as e:
        results = [{"error": str(e)}]
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="all",
                       help="Comma-separated experiment IDs or 'all'")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip experiments that already have results")
    parser.add_argument("--qualitative-only", action="store_true",
                       help="Only run qualitative eval on existing adapters")
    parser.add_argument("--start-from", type=str, default=None,
                       help="Start from this experiment ID (skip earlier ones)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"LFM2.5-230M SFT Experiment Sweep")
    print(f"Timestamp: {timestamp}")
    print(f"Model: {MODEL}")
    print(f"Total experiments defined: {len(EXPERIMENTS)}")

    # Filter experiments
    if args.experiments == "all":
        exps = EXPERIMENTS
    else:
        ids = [x.strip() for x in args.experiments.split(",")]
        exps = [e for e in EXPERIMENTS if e["id"] in ids]

    if args.start_from:
        start_idx = next((i for i, e in enumerate(exps) if e["id"] == args.start_from), 0)
        exps = exps[start_idx:]

    print(f"Running {len(exps)} experiments")

    # Load tokenizer once
    tok = AutoTokenizer = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_results = []

    for i, exp in enumerate(exps):
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(exps)}] {exp['id']}")
        print(f"{'#'*60}")

        adapter_dir = ADAPTERS_DIR / exp["id"]

        if args.qualitative_only:
            if adapter_dir.exists():
                qual = run_qualitative_for_adapter(str(adapter_dir), exp["id"], tok)
                all_results.append({"id": exp["id"], "qualitative": qual})
            else:
                print(f"  No adapter found at {adapter_dir}, skipping")
            continue

        # Check if already done
        if args.skip_existing:
            result_file = RESULTS_DIR / f"lfm2_sft_{exp['id']}_{timestamp}.json"
            # Check for any existing result for this exp
            existing = list(RESULTS_DIR.glob(f"lfm2_sft_{exp['id']}_*.json"))
            if existing:
                print(f"  Already done ({existing[0].name}), skipping")
                continue

        try:
            result = run_single_experiment(exp, tok)

            # Run qualitative eval on the trained adapter
            if adapter_dir.exists():
                qual = run_qualitative_for_adapter(str(adapter_dir), exp["id"], tok)
                result["qualitative"] = qual

            # Save individual result
            result_file = RESULTS_DIR / f"lfm2_sft_{exp['id']}_{timestamp}.json"
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

    # Save combined results
    combined_file = RESULTS_DIR / f"lfm2_sft_sweep_{timestamp}.json"
    with open(combined_file, "w") as f:
        json.dump({
            "model": MODEL,
            "timestamp": timestamp,
            "n_experiments": len(all_results),
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Results: {combined_file}")
    print(f"  Successful: {sum(1 for r in all_results if 'error' not in r)}/{len(all_results)}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'ID':<25} {'Loss':>8} {'KL':>8} {'Params':>10} {'Time':>8}")
    print("-" * 65)
    for r in all_results:
        if "error" in r:
            print(f"{r['id']:<25} {'ERROR':>8}")
        else:
            print(f"{r['id']:<25} {r['final_loss']:>8.4f} {r['kl_shift']:>8.4f} {r['trainable_params']:>10,} {r['train_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
