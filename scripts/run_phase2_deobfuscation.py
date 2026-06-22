"""Phase 2 — Block H: Deobfuscation Surgery.

Treat deobfuscation as composed subskills. Train separate LoRA adapters for
each subskill, analyze overlap, interference, and compositionality.

Registry ID: P2-DEOBF-001
"""
import sys
import os
import json
import ast
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from datasets import Dataset

from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, load_json, load_jsonl, set_seed, PROJECT_ROOT, now_iso
from mi_atlas.metrics import exact_match_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_ID = "P2-DEOBF-001"
REGISTRY_ID = EXPERIMENT_ID
ADAPTERS_DIR = PROJECT_ROOT / "experiments" / "adapters"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lr": 2e-4,
    "max_steps": 100,
    "batch_size": 1,
    "gradient_checkpointing": True,
}


# ---------------------------------------------------------------------------
# Subskill data generators
# ---------------------------------------------------------------------------

def generate_variable_renaming_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Variable renaming: code with renamed variables, identify original names."""
    rng = np.random.RandomState(seed)
    examples = []

    templates = [
        {
            "original": "x = 5\ny = x + 3\nprint(y)",
            "renamed": "count = 5\ntotal = count + 3\nprint(total)",
            "answer": "x was renamed to count, y was renamed to total",
        },
        {
            "original": "data = [1, 2, 3]\nresult = sum(data)\nprint(result)",
            "renamed": "items = [1, 2, 3]\ntotal = sum(items)\nprint(total)",
            "answer": "data was renamed to items, result was renamed to total",
        },
        {
            "original": "name = 'Alice'\nage = 30\nprint(name, age)",
            "renamed": "user_name = 'Alice'\nuser_age = 30\nprint(user_name, user_age)",
            "answer": "name was renamed to user_name, age was renamed to user_age",
        },
        {
            "original": "lst = [10, 20, 30]\nfirst = lst[0]\nprint(first)",
            "renamed": "values = [10, 20, 30]\nhead = values[0]\nprint(head)",
            "answer": "lst was renamed to values, first was renamed to head",
        },
        {
            "original": "flag = True\nif flag:\n    msg = 'yes'\nprint(msg)",
            "renamed": "is_ready = True\nif is_ready:\n    output = 'yes'\nprint(output)",
            "answer": "flag was renamed to is_ready, msg was renamed to output",
        },
        {
            "original": "count = 0\nfor i in range(5):\n    count += i\nprint(count)",
            "renamed": "total = 0\nfor idx in range(5):\n    total += idx\nprint(total)",
            "answer": "count was renamed to total, i was renamed to idx",
        },
    ]

    for i in range(n):
        t = templates[i % len(templates)]
        prompt = f"Given the original code and renamed version, identify the renames.\n\nOriginal:\n{t['original']}\n\nRenamed:\n{t['renamed']}\n\nIdentify renames:"
        target = f" {t['answer']}"
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_dead_code_removal_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Dead code removal: code with dead branches, ask to simplify."""
    templates = [
        {
            "code": "x = 1\nif False:\n    x = 999\nprint(x)",
            "simplified": "x = 1\nprint(x)",
        },
        {
            "code": "y = 10\nif True:\n    y = 20\nelse:\n    y = 99\nprint(y)",
            "simplified": "y = 20\nprint(y)",
        },
        {
            "code": "z = 5\nif 1 > 2:\n    z = 100\nelif 3 > 2:\n    z = 50\nprint(z)",
            "simplified": "z = 50\nprint(z)",
        },
        {
            "code": "a = 'hello'\nwhile False:\n    a = 'goodbye'\nprint(a)",
            "simplified": "a = 'hello'\nprint(a)",
        },
        {
            "code": "b = 0\nfor i in range(0):\n    b += 1\nprint(b)",
            "simplified": "b = 0\nprint(b)",
        },
        {
            "code": "c = 7\ntry:\n    pass\nexcept:\n    c = 0\nprint(c)",
            "simplified": "c = 7\nprint(c)",
        },
    ]

    examples = []
    for i in range(n):
        t = templates[i % len(templates)]
        prompt = f"Simplify this code by removing dead branches:\n\n{t['code']}\n\nSimplified:"
        target = f"\n{t['simplified']}"
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_string_decoding_data(n: int = 30, seed: int = 42) -> list[dict]:
    """String decoding: decode escaped/encoded strings."""
    templates = [
        ("\\n", "newline"),
        ("\\t", "tab"),
        ("\\\\", "backslash"),
        ("\\'", "single quote"),
        ("\\\"", "double quote"),
        ("\\x48\\x65\\x6c\\x6c\\x6f", "Hello"),
        ("\\x57\\x6f\\x72\\x6c\\x64", "World"),
        ("\\x30\\x31\\x32\\x33", "0123"),
        ("\\x61\\x62\\x63", "abc"),
        ("\\x41\\x42\\x43", "ABC"),
    ]

    examples = []
    for i in range(n):
        encoded, decoded = templates[i % len(templates)]
        prompt = f"Decode this escaped string to its actual value:\n\"{encoded}\"\nDecoded:"
        target = f" \"{decoded}\""
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_constant_folding_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Constant folding: evaluate constant expressions."""
    rng = np.random.RandomState(seed)
    examples = []
    for i in range(n):
        a, b, c = rng.randint(1, 20, 3)
        op1 = rng.choice(["+", "-", "*"])
        op2 = rng.choice(["+", "-", "*"])
        expr = f"({a} {op1} {b}) {op2} {c}"
        result = eval(expr)
        prompt = f"Evaluate this constant expression:\n{expr}\nResult:"
        target = f" {result}"
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_control_flow_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Control flow simplification: simplify nested if/else."""
    templates = [
        {
            "nested": "x = 10\nif x > 0:\n    if x > 5:\n        if x > 8:\n            result = 'high'\n        else:\n            result = 'mid'\n    else:\n        result = 'low'\nelse:\n    result = 'neg'\nprint(result)",
            "simplified": "x = 10\nresult = 'high'\nprint(result)",
        },
        {
            "nested": "a = True\nb = False\nif a:\n    if b:\n        r = 'both'\n    else:\n        r = 'only_a'\nelse:\n    if b:\n        r = 'only_b'\n    else:\n        r = 'neither'\nprint(r)",
            "simplified": "a = True\nb = False\nr = 'only_a'\nprint(r)",
        },
        {
            "nested": "val = 3\nif val == 1:\n    out = 'one'\nelif val == 2:\n    out = 'two'\nelif val == 3:\n    out = 'three'\nelse:\n    out = 'other'\nprint(out)",
            "simplified": "val = 3\nout = 'three'\nprint(out)",
        },
    ]

    examples = []
    for i in range(n):
        t = templates[i % len(templates)]
        prompt = f"Simplify this nested control flow with known values:\n\n{t['nested']}\n\nSimplified:"
        target = f"\n{t['simplified']}"
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_semantic_preservation_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Semantic preservation: verify code does same thing after transformation."""
    templates = [
        {
            "original": "x = 5\ny = x * 2\nprint(y)",
            "transformed": "a = 5\nb = a * 2\nprint(b)",
            "answer": "yes",
        },
        {
            "original": "lst = [1, 2, 3]\nprint(sum(lst))",
            "transformed": "lst = [1, 2, 3]\ntotal = 0\nfor x in lst:\n    total += x\nprint(total)",
            "answer": "yes",
        },
        {
            "original": "s = 'hello'\nprint(len(s))",
            "transformed": "s = 'hello'\nprint(s.__len__())",
            "answer": "yes",
        },
        {
            "original": "x = 10\nprint(x + 1)",
            "transformed": "x = 10\nprint(x + 2)",
            "answer": "no",
        },
        {
            "original": "for i in range(3):\n    print(i)",
            "transformed": "print(0)\nprint(1)\nprint(2)",
            "answer": "yes",
        },
        {
            "original": "x = [1, 2, 3]\nprint(x[0])",
            "transformed": "x = [1, 2, 3]\nprint(x[1])",
            "answer": "no",
        },
    ]

    examples = []
    for i in range(n):
        t = templates[i % len(templates)]
        prompt = f"Does this transformation preserve semantics?\n\nOriginal:\n{t['original']}\n\nTransformed:\n{t['transformed']}\n\nPreserves semantics (yes/no):"
        target = f" {t['answer']}"
        examples.append({"prompt": prompt, "target": target})
    return examples


SUBSKILL_GENERATORS = {
    "variable_renaming": generate_variable_renaming_data,
    "dead_code_removal": generate_dead_code_removal_data,
    "string_decoding": generate_string_decoding_data,
    "constant_folding": generate_constant_folding_data,
    "control_flow_simplification": generate_control_flow_data,
    "semantic_preservation": generate_semantic_preservation_data,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_layers(model):
    """Get transformer layers, handling PeftModel wrapping."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def check_already_done(force: bool) -> bool:
    if force:
        return False
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    if not registry_path.exists():
        return False
    for record in load_jsonl(registry_path):
        if record.get("summary", "").startswith(EXPERIMENT_ID):
            if record.get("status") == "success":
                print(f"  {EXPERIMENT_ID} already completed. Use --force to re-run.")
                return True
    return False


def make_training_dataset(examples: list[dict]) -> Dataset:
    records = [{"text": ex["prompt"] + "\n" + ex["target"]} for ex in examples]
    return Dataset.from_list(records)


def train_subskill_adapter(
    model, tokenizer, train_examples: list[dict], adapter_dir: Path, seed: int = 42
) -> Path:
    """Train a LoRA adapter for a subskill."""
    adapter_path = adapter_dir / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    dataset = make_training_dataset(train_examples)

    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    def tokenize_fn(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        learning_rate=LORA_CONFIG["lr"],
        per_device_train_batch_size=LORA_CONFIG["batch_size"],
        max_steps=LORA_CONFIG["max_steps"],
        warmup_steps=10,
        save_steps=50,
        logging_steps=10,
        fp16=False,
        bf16=True,
        seed=seed,
        gradient_checkpointing=LORA_CONFIG["gradient_checkpointing"],
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
    )

    print(f"    Training {adapter_dir.name}...")
    result = trainer.train()
    print(f"    Loss: {result.training_loss:.4f}")

    peft_model.save_pretrained(str(adapter_path))
    del peft_model, trainer
    torch.cuda.empty_cache()

    return adapter_path


def compute_ablation_map(model, tokenizer, adapter_path: Path, prompts: list[str], n_layers: int, device) -> dict:
    """Compute per-layer ablation KL map for an adapter."""
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    peft_model.eval()
    layers = get_layers(peft_model)

    all_layer_kls = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        # Full adapter logits
        with torch.no_grad():
            full_logits = peft_model(ids).logits
        full_probs = torch.softmax(full_logits[0, -1], dim=-1)

        # Get base activations
        with peft_model.disable_adapter():
            base_acts = {}
            hooks = []
            for li in range(n_layers):
                def make_hook(idx):
                    def hook_fn(module, input, output):
                        out = output[0] if isinstance(output, tuple) else output
                        base_acts[idx] = out.detach().clone()
                    return hook_fn
                hooks.append(layers[li].register_forward_hook(make_hook(li)))
            with torch.no_grad():
                _ = peft_model(ids)
            for h in hooks:
                h.remove()

        # Ablate each layer
        for layer_idx in range(n_layers):
            if layer_idx not in base_acts:
                continue
            donor = base_acts[layer_idx].to(device)

            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    return (donor,) + output[1:]
                return donor

            handle = layers[layer_idx].register_forward_hook(patch_hook)
            with torch.no_grad():
                abl_logits = peft_model(ids).logits
            handle.remove()

            abl_probs = torch.softmax(abl_logits[0, -1], dim=-1)
            kl = torch.nn.functional.kl_div(
                torch.log(abl_probs + 1e-10), full_probs, reduction="sum"
            ).item()
            all_layer_kls[layer_idx].append(kl)

    del peft_model
    torch.cuda.empty_cache()

    return {l: round(float(np.mean(kls)), 6) for l, kls in all_layer_kls.items()}


def eval_output(predicted: str, target: str, metric_type: str = "exact_match") -> dict:
    """Evaluate output with multiple metrics."""
    pred_clean = predicted.strip()
    tgt_clean = target.strip()

    metrics = {
        "exact_match": 1.0 if pred_clean == tgt_clean else 0.0,
    }

    # AST equivalence (if both parse as Python)
    try:
        ast.parse(pred_clean)
        pred_parses = True
    except SyntaxError:
        pred_parses = False

    try:
        ast.parse(tgt_clean)
        tgt_parses = True
    except SyntaxError:
        tgt_parses = False

    if pred_parses and tgt_parses:
        try:
            pred_tree = ast.dump(ast.parse(pred_clean))
            tgt_tree = ast.dump(ast.parse(tgt_clean))
            metrics["ast_equivalent"] = 1.0 if pred_tree == tgt_tree else 0.0
        except Exception:
            metrics["ast_equivalent"] = 0.0
    else:
        metrics["ast_equivalent"] = 0.0

    metrics["parses"] = 1.0 if pred_parses else 0.0

    # Hallucinated code rate (generates code that wasn't asked for)
    metrics["hallucinated"] = 0.0  # Default: not hallucinated

    # Over-simplification rate (output is shorter than expected)
    if len(pred_clean) < len(tgt_clean) * 0.3 and len(tgt_clean) > 10:
        metrics["over_simplified"] = 1.0
    else:
        metrics["over_simplified"] = 0.0

    return metrics


def generate_with_adapter(model, tokenizer, adapter_path: Path, prompt: str, device, max_new_tokens: int = 100) -> str:
    """Generate text with an adapter loaded."""
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    peft_model.eval()

    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
    with torch.no_grad():
        output_ids = peft_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    output = tokenizer.decode(output_ids[0][ids.shape[1]:], skip_special_tokens=True)

    del peft_model
    torch.cuda.empty_cache()
    return output


def check_already_done(force: bool) -> bool:
    if force:
        return False
    registry_path = PROJECT_ROOT / "experiments" / "registry.jsonl"
    if not registry_path.exists():
        return False
    for record in load_jsonl(registry_path):
        if record.get("summary", "").startswith(EXPERIMENT_ID):
            if record.get("status") == "success":
                print(f"  {EXPERIMENT_ID} already completed. Use --force to re-run.")
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID}: Deobfuscation Surgery")
    parser.add_argument("--force", action="store_true", help="Re-run even if already completed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model")
    parser.add_argument("--skip-training", action="store_true", help="Skip training adapters")
    args = parser.parse_args()

    if check_already_done(args.force):
        return

    set_seed(args.seed)
    run_id = f"P2_DEOBF_qwen05b_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID}: Deobfuscation Surgery")
    print(f"  Run ID: {run_id}")
    print(f"  Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Save config
    config = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "lora_config": LORA_CONFIG,
        "subskills": list(SUBSKILL_GENERATORS.keys()),
        "timestamp": now_iso(),
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(config, CONFIGS_DIR / f"{EXPERIMENT_ID}_config.json")

    # Load model
    print("[1/6] Loading base model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    device = bundle.device
    print(f"  Model: {args.model}, {n_layers} layers, device={device}")

    # Train subskill adapters
    print(f"\n[2/6] Training subskill adapters...")
    subskill_adapters = {}

    for subskill_name, generator_fn in SUBSKILL_GENERATORS.items():
        adapter_dir = ADAPTERS_DIR / f"lora_deobf_{subskill_name}_r8"
        adapter_file = adapter_dir / "adapter" / "adapter_model.safetensors"

        if adapter_file.exists():
            print(f"  ✓ {subskill_name}: using existing adapter")
            subskill_adapters[subskill_name] = adapter_dir
            continue

        if args.skip_training:
            print(f"  ⚠ {subskill_name}: --skip-training, no adapter available")
            continue

        print(f"  Training: {subskill_name}")
        try:
            train_examples = generator_fn(n=30, seed=args.seed)
            train_subskill_adapter(model, tokenizer, train_examples, adapter_dir, seed=args.seed)
            subskill_adapters[subskill_name] = adapter_dir
        except Exception as e:
            print(f"    ✗ Training failed: {e}")
            traceback.print_exc()

    torch.cuda.empty_cache()

    # Analyze subskills
    print(f"\n[3/6] Analyzing subskill ablation maps...")
    ablation_maps = {}
    eval_results = {}

    for subskill_name, adapter_dir in subskill_adapters.items():
        print(f"\n  --- {subskill_name} ---")
        adapter_path = adapter_dir / "adapter"
        generator_fn = SUBSKILL_GENERATORS[subskill_name]

        # Test prompts
        test_examples = generator_fn(n=3, seed=args.seed + 100)
        test_prompts = [ex["prompt"] for ex in test_examples]

        # 1. Ablation map
        try:
            ablation_map = compute_ablation_map(
                model, tokenizer, adapter_path, test_prompts, n_layers, device
            )
            ablation_maps[subskill_name] = ablation_map

            # Find peak layers
            peak_layer = max(ablation_map, key=ablation_map.get)
            print(f"    Peak ablation layer: L{peak_layer} (KL={ablation_map[peak_layer]:.4f})")
        except Exception as e:
            print(f"    ✗ Ablation failed: {e}")
            ablation_maps[subskill_name] = {}

        # 2. Generate and evaluate
        try:
            test_full = generator_fn(n=5, seed=args.seed + 200)
            gen_results = []
            for ex in test_full:
                output = generate_with_adapter(
                    model, tokenizer, adapter_path, ex["prompt"], device, max_new_tokens=80
                )
                metrics = eval_output(output, ex["target"])
                gen_results.append({
                    "prompt": ex["prompt"][:60],
                    "target": ex["target"][:60],
                    "output": output[:60],
                    "metrics": metrics,
                })
                torch.cuda.empty_cache()

            # Aggregate
            agg_metrics = {}
            for metric_name in gen_results[0]["metrics"]:
                vals = [r["metrics"][metric_name] for r in gen_results]
                agg_metrics[metric_name] = round(float(np.mean(vals)), 4)

            eval_results[subskill_name] = {
                "aggregate": agg_metrics,
                "per_example": gen_results,
            }
            print(f"    Eval: exact_match={agg_metrics['exact_match']:.2f} "
                  f"parses={agg_metrics['parses']:.2f} "
                  f"ast_equiv={agg_metrics['ast_equivalent']:.2f}")
        except Exception as e:
            print(f"    ✗ Eval failed: {e}")
            eval_results[subskill_name] = {"error": str(e)}

    # Test 1: Which subskills localize similarly?
    print(f"\n[4/6] Analyzing subskill overlap...")
    overlap_analysis = {}
    subskill_names = list(ablation_maps.keys())

    for i in range(len(subskill_names)):
        for j in range(i + 1, len(subskill_names)):
            name_a, name_b = subskill_names[i], subskill_names[j]
            map_a = ablation_maps.get(name_a, {})
            map_b = ablation_maps.get(name_b, {})

            if not map_a or not map_b:
                continue

            # Compute correlation between ablation maps
            layers_common = sorted(set(map_a.keys()) & set(map_b.keys()))
            if len(layers_common) < 3:
                continue

            vals_a = [map_a[l] for l in layers_common]
            vals_b = [map_b[l] for l in layers_common]

            if np.std(vals_a) > 0 and np.std(vals_b) > 0:
                correlation = float(np.corrcoef(vals_a, vals_b)[0, 1])
            else:
                correlation = 0.0

            # Peak layer overlap
            peak_a = max(map_a, key=map_a.get)
            peak_b = max(map_b, key=map_b.get)
            peak_distance = abs(peak_a - peak_b)

            pair_key = f"{name_a}+{name_b}"
            overlap_analysis[pair_key] = {
                "correlation": round(correlation, 4),
                "peak_layer_a": peak_a,
                "peak_layer_b": peak_b,
                "peak_distance": peak_distance,
                "similar_localization": correlation > 0.7 and peak_distance <= 2,
            }
            print(f"    {pair_key}: r={correlation:.3f} peak_dist={peak_distance}")

    # Test 2: Adapter stacking interference
    print(f"\n[5/6] Testing adapter stacking interference...")
    stacking_results = {}

    for i in range(len(subskill_names)):
        for j in range(i + 1, len(subskill_names)):
            name_a, name_b = subskill_names[i], subskill_names[j]
            path_a = subskill_adapters[name_a] / "adapter"
            path_b = subskill_adapters[name_b] / "adapter"

            pair_key = f"{name_a}+{name_b}"
            print(f"    Stacking: {pair_key}")

            try:
                # Load adapter A, then stack B
                model_a = PeftModel.from_pretrained(model, str(path_a))
                # Note: PeftModel stacking may not work directly
                # Instead, we test by generating with each and comparing

                # Test with A only on B's tasks
                test_b = SUBSKILL_GENERATORS[name_b](n=3, seed=args.seed + 300)
                a_on_b_scores = []
                for ex in test_b:
                    ids = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                    with torch.no_grad():
                        out = tokenizer.decode(model_a.generate(ids, max_new_tokens=50, do_sample=False)[0][ids.shape[1]:])
                    a_on_b_scores.append(eval_output(out, ex["target"])["exact_match"])

                del model_a
                torch.cuda.empty_cache()

                # Load adapter B, test on B's tasks
                model_b = PeftModel.from_pretrained(model, str(path_b))
                b_on_b_scores = []
                for ex in test_b:
                    ids = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                    with torch.no_grad():
                        out = tokenizer.decode(model_b.generate(ids, max_new_tokens=50, do_sample=False)[0][ids.shape[1]:])
                    b_on_b_scores.append(eval_output(out, ex["target"])["exact_match"])

                del model_b
                torch.cuda.empty_cache()

                stacking_results[pair_key] = {
                    "adapter_a_on_b_tasks": round(float(np.mean(a_on_b_scores)), 4),
                    "adapter_b_on_b_tasks": round(float(np.mean(b_on_b_scores)), 4),
                    "interference": round(float(np.mean(b_on_b_scores) - np.mean(a_on_b_scores)), 4),
                }
                print(f"      A on B tasks: {np.mean(a_on_b_scores):.3f}, B on B tasks: {np.mean(b_on_b_scores):.3f}")

            except Exception as e:
                stacking_results[pair_key] = {"error": str(e)}
                print(f"      ✗ Failed: {e}")
                torch.cuda.empty_cache()

    # Test 3: Semantic preservation final-layer signature
    print(f"\n  Checking semantic preservation signature...")
    sem_pres_map = ablation_maps.get("semantic_preservation", {})
    sem_pres_signature = {}
    if sem_pres_map:
        # Check if final layers are disproportionately important
        early_layers = [sem_pres_map.get(l, 0) for l in range(0, n_layers // 3)]
        mid_layers = [sem_pres_map.get(l, 0) for l in range(n_layers // 3, 2 * n_layers // 3)]
        late_layers = [sem_pres_map.get(l, 0) for l in range(2 * n_layers // 3, n_layers)]

        sem_pres_signature = {
            "early_mean_kl": round(float(np.mean(early_layers)), 6) if early_layers else 0,
            "mid_mean_kl": round(float(np.mean(mid_layers)), 6) if mid_layers else 0,
            "late_mean_kl": round(float(np.mean(late_layers)), 6) if late_layers else 0,
            "late_dominant": float(np.mean(late_layers)) > 2 * max(float(np.mean(early_layers)), 1e-6),
        }
        print(f"    Semantic preservation: early={sem_pres_signature['early_mean_kl']:.4f} "
              f"mid={sem_pres_signature['mid_mean_kl']:.4f} "
              f"late={sem_pres_signature['late_mean_kl']:.4f} "
              f"late_dominant={sem_pres_signature['late_dominant']}")

    # Test 4: Joint vs composed deobfuscation
    # (Simplified: compare individual adapter eval scores to see if composition helps)
    print(f"\n  Joint vs composed analysis...")
    joint_vs_composed = {
        "individual_scores": {name: eval_results.get(name, {}).get("aggregate", {}).get("exact_match", 0) for name in subskill_names},
        "note": "Full joint training comparison requires training a combined adapter (future work)",
    }

    # Test 5: Activation patching transfer
    print(f"\n  Activation patching transfer analysis...")
    transfer_results = {
        "note": "Full cross-model patching requires matching architecture (deferred to P2-SEPARABILITY-001)",
        "available_adapters": list(subskill_adapters.keys()),
    }

    # Assemble results
    print(f"\n[6/6] Assembling results...")
    output = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "n_layers": n_layers,
        "lora_config": LORA_CONFIG,
        "timestamp": now_iso(),
        "subskills_trained": list(subskill_adapters.keys()),
        "ablation_maps": ablation_maps,
        "eval_results": {k: v.get("aggregate", v) if isinstance(v, dict) else v for k, v in eval_results.items()},
        "eval_detailed": eval_results,
        "overlap_analysis": overlap_analysis,
        "stacking_interference": stacking_results,
        "semantic_preservation_signature": sem_pres_signature,
        "joint_vs_composed": joint_vs_composed,
        "transfer_results": transfer_results,
    }

    output_path = RESULTS_DIR / "deobfuscation_surgery.json"
    save_json(output, output_path)
    print(f"  Results saved to {output_path}")

    # Register
    try:
        n_subskills = len(subskill_adapters)
        mean_exact = float(np.mean([
            eval_results.get(s, {}).get("aggregate", {}).get("exact_match", 0)
            for s in subskill_names
        ])) if subskill_names else 0

        register_experiment(
            type="deobfuscation_surgery",
            model=args.model,
            backend="hf_peft",
            config=str(CONFIGS_DIR / f"{EXPERIMENT_ID}_config.json"),
            inputs=[str(ADAPTERS_DIR)],
            outputs=[str(output_path)],
            status="success",
            summary=f"{EXPERIMENT_ID}: Deobfuscation surgery on {n_subskills} subskills, seed={args.seed}",
            key_metrics={
                "n_subskills": n_subskills,
                "mean_exact_match": round(mean_exact, 4),
                "n_overlap_pairs": len(overlap_analysis),
            },
            next="Phase 2 Block G: Skill Separability, Block F: Adapter Surgery compatibility",
        )
        print("  Experiment registered.")
    except Exception as e:
        print(f"  ⚠ Registration failed: {e}")

    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID} complete. Run ID: {run_id}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
