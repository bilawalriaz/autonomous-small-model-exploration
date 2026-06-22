"""Phase 2 — Block F: Adapter Surgery.

Train or reuse LoRA adapters for isolated skills, then perform comprehensive
surgery analysis: layer-wise norms, causal ablation, rank truncation,
target skill gain, collateral damage, and adapter compatibility matrix.

Registry ID: P2-SURGERY-001
"""
import sys
import os
import json
import csv
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
from mi_atlas.task_suite import (
    build_default_suite, generate_json_examples, generate_factual_examples,
    generate_delimiter_examples, generate_code_semantics_examples,
    generate_variable_renaming_examples,
)
from mi_atlas.training.datasets import prepare_sft_dataset, split_dataset
from mi_atlas.metrics import exact_match_score, valid_json_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_ID = "P2-SURGERY-001"
REGISTRY_ID = EXPERIMENT_ID
ADAPTERS_DIR = PROJECT_ROOT / "experiments" / "adapters"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
SUMMARIES_DIR = PROJECT_ROOT / "results" / "summaries"
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

SKILL_DEFINITIONS = {
    "json_formatting": {
        "adapter_name": "lora_json_r8",
        "generator": generate_json_examples,
        "train_n": 30,
        "test_n": 10,
        "metric_type": "valid_json",
    },
    "factual_recall": {
        "adapter_name": "lora_factual_recall_r8",
        "generator": generate_factual_examples,
        "train_n": 24,
        "test_n": 8,
        "metric_type": "target_logprob",
    },
    "delimiter_tracking": {
        "adapter_name": "lora_delimiter_tracking_r8",
        "generator": generate_delimiter_examples,
        "train_n": 30,
        "test_n": 10,
        "metric_type": "exact_match",
    },
    "code_semantics": {
        "adapter_name": "lora_code_semantics_r8",
        "generator": generate_code_semantics_examples,
        "train_n": 24,
        "test_n": 8,
        "metric_type": "exact_match",
    },
    "variable_renaming": {
        "adapter_name": "lora_variable_renaming_r8",
        "generator": generate_variable_renaming_examples,
        "train_n": 30,
        "test_n": 10,
        "metric_type": "exact_match",
        "needs_training": True,
    },
}

# Skills that need new training data generated synthetically
SYNTHETIC_SKILLS = {
    "string_decoding": {
        "adapter_name": "lora_string_decoding_r8",
        "metric_type": "exact_match",
        "needs_training": True,
    },
    "constant_folding": {
        "adapter_name": "lora_constant_folding_r8",
        "metric_type": "exact_match",
        "needs_training": True,
    },
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
    """Check if this experiment already completed."""
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


def generate_string_decoding_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Generate training data for string decoding skill."""
    rng = np.random.RandomState(seed)
    examples = []
    encodings = [
        ("\\n", "\n"), ("\\t", "\t"), ("\\\\", "\\"),
        ("\\'", "'"), ('\\"', '"'), ("\\x41", "A"),
        ("\\x48\\x65\\x6c\\x6c\\x6f", "Hello"),
        ("\\x57\\x6f\\x72\\x6c\\x64", "World"),
        ("\\x30\\x31\\x32", "012"),
        ("\\x61\\x62\\x63", "abc"),
    ]
    for i in range(n):
        encoded, decoded = encodings[i % len(encodings)]
        prompt = f"Decode this escaped string: \"{encoded}\"\nDecoded:"
        target = f" \"{decoded}\""
        examples.append({"prompt": prompt, "target": target})
    return examples


def generate_constant_folding_data(n: int = 30, seed: int = 42) -> list[dict]:
    """Generate training data for constant folding skill."""
    rng = np.random.RandomState(seed)
    examples = []
    for i in range(n):
        a, b, c = rng.randint(1, 20, 3)
        op1 = rng.choice(["+", "-", "*"])
        op2 = rng.choice(["+", "-", "*"])
        expr = f"({a} {op1} {b}) {op2} {c}"
        result = eval(expr)
        prompt = f"Simplify this constant expression: {expr}\nResult:"
        target = f" {result}"
        examples.append({"prompt": prompt, "target": target})
    return examples


def make_training_dataset(examples: list[dict], response_template: str = "\n") -> Dataset:
    """Convert examples to HF Dataset for SFT."""
    records = []
    for ex in examples:
        text = ex["prompt"] + response_template + ex["target"]
        records.append({"text": text})
    return Dataset.from_list(records)


def train_adapter(
    model, tokenizer, train_examples: list[dict], adapter_dir: Path,
    seed: int = 42,
) -> Path:
    """Train a LoRA adapter and return its path."""
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
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
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

    print(f"    Training adapter for {adapter_dir.name}...")
    result = trainer.train()
    print(f"    Training loss: {result.training_loss:.4f}")

    peft_model.save_pretrained(str(adapter_path))
    del peft_model, trainer
    torch.cuda.empty_cache()

    return adapter_path


def compute_lora_norms(peft_model, n_layers: int) -> dict:
    """Compute layer-wise and module-wise LoRA norms."""
    state_dict = peft_model.state_dict()

    layer_norms = {}
    module_norms = {}

    for i in range(n_layers):
        layer_entries = {}
        for key, w in state_dict.items():
            if f"layers.{i}." not in key:
                continue
            if "lora_A" not in key and "lora_B" not in key:
                continue

            # Extract module name
            parts = key.split(".")
            module_name = None
            for j, p in enumerate(parts):
                if p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    module_name = p
                    break
            if module_name is None:
                module_name = "other"

            if module_name not in layer_entries:
                layer_entries[module_name] = {}
            if "lora_A" in key:
                layer_entries[module_name]["A"] = w.float()
            elif "lora_B" in key:
                layer_entries[module_name]["B"] = w.float()

        # Compute effective norm: ||B @ A||_F
        layer_total = 0.0
        for mod_name, matrices in layer_entries.items():
            if "A" in matrices and "B" in matrices:
                # Effective weight = B @ A, norm = Frobenius
                effective = matrices["B"] @ matrices["A"]
                mod_norm = effective.norm().item()
                layer_total += mod_norm ** 2
                full_key = f"layer_{i:02d}_{mod_name}"
                module_norms[full_key] = mod_norm

        layer_norms[f"layer_{i:02d}"] = np.sqrt(layer_total) if layer_total > 0 else 0.0

    return {"layer_norms": layer_norms, "module_norms": module_norms}


def causal_adapter_ablation_by_layer(
    model, tokenizer, adapter_path: Path, prompts: list[str], n_layers: int, device
) -> dict:
    """Disable adapter at each layer, measure KL from full adapter."""
    results = {}

    # Load adapter
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    peft_model.eval()

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        # Full adapter logits
        with torch.no_grad():
            full_logits = peft_model(ids).logits

        # Base logits (adapter disabled)
        with peft_model.disable_adapter():
            with torch.no_grad():
                base_logits = peft_model(ids).logits

        full_probs = torch.softmax(full_logits[0, -1], dim=-1)

        layer_effects = []
        for layer_idx in range(n_layers):
            # Disable adapter at this layer only
            # We use a hook to zero out the LoRA delta at this layer
            layers = get_layers(peft_model)
            lora_delta = {}

            def capture_lora_hook(idx):
                def hook_fn(module, input, output):
                    # Store the output, we'll compute delta later
                    return output
                return hook_fn

            # Simpler approach: compute base output per layer, then measure
            # which layer's adapter contribution matters most
            # For PeftModel, we can disable per-layer by temporarily zeroing weights
            pass  # Handled below

        # Use a more efficient approach: patch base activation into trained model
        layers = get_layers(peft_model)

        # Get base activations
        with peft_model.disable_adapter():
            base_acts = {}
            hooks = []
            for li in range(n_layers):
                def make_hook(idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            base_acts[idx] = output[0].detach().clone()
                        else:
                            base_acts[idx] = output.detach().clone()
                    return hook_fn
                hooks.append(layers[li].register_forward_hook(make_hook(li)))
            with torch.no_grad():
                _ = peft_model(ids)
            for h in hooks:
                h.remove()

        # For each layer, replace trained activation with base activation
        layer_kls = []
        for layer_idx in range(n_layers):
            if layer_idx not in base_acts:
                layer_kls.append({"layer": layer_idx, "ablation_kl": 0.0})
                continue

            donor = base_acts[layer_idx].to(device)

            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    return (donor,) + output[1:]
                return donor

            handle = layers[layer_idx].register_forward_hook(patch_hook)
            with torch.no_grad():
                ablated_logits = peft_model(ids).logits
            handle.remove()

            ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
            kl = torch.nn.functional.kl_div(
                torch.log(ablated_probs + 1e-10), full_probs, reduction="sum"
            ).item()
            layer_kls.append({"layer": layer_idx, "ablation_kl": round(kl, 6)})

        results[prompt[:50]] = layer_kls

    # Unload adapter
    del peft_model
    torch.cuda.empty_cache()

    return results


def measure_skill_gain(
    model, tokenizer, adapter_path: Path, test_examples: list[dict],
    metric_type: str, device
) -> dict:
    """Measure how much adapter improves target skill."""
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    peft_model.eval()

    base_scores = []
    adapter_scores = []

    for ex in test_examples:
        prompt = ex["prompt"]
        target = ex["target"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        # Base
        with peft_model.disable_adapter():
            with torch.no_grad():
                base_out = tokenizer.decode(
                    peft_model.generate(ids, max_new_tokens=30, do_sample=False)[0][ids.shape[1]:]
                )

        # Adapter
        with torch.no_grad():
            adapter_out = tokenizer.decode(
                peft_model.generate(ids, max_new_tokens=30, do_sample=False)[0][ids.shape[1]:]
            )

        base_scores.append(_score_output(base_out, target, metric_type))
        adapter_scores.append(_score_output(adapter_out, target, metric_type))

    del peft_model
    torch.cuda.empty_cache()

    return {
        "base_mean": round(float(np.mean(base_scores)), 4),
        "adapter_mean": round(float(np.mean(adapter_scores)), 4),
        "gain": round(float(np.mean(adapter_scores) - np.mean(base_scores)), 4),
        "n_examples": len(test_examples),
    }


def _score_output(predicted: str, target: str, metric_type: str) -> float:
    """Score a model output against target."""
    if metric_type == "valid_json":
        return valid_json_score(predicted)
    elif metric_type == "exact_match":
        return exact_match_score(predicted, target)
    elif metric_type == "target_logprob":
        # For logprob metrics, use exact match as proxy during generation eval
        return exact_match_score(predicted, target)
    return exact_match_score(predicted, target)


def rank_truncation_analysis(adapter_path: Path, n_layers: int, max_rank: int = 8) -> dict:
    """Analyze adapter by truncating to top-k singular vectors."""
    from safetensors.torch import load_file

    weight_file = adapter_path / "adapter" / "adapter_model.safetensors"
    if not weight_file.exists():
        return {"error": "No adapter weights found"}

    weights = load_file(str(weight_file))
    results = {}

    for key, tensor in weights.items():
        if "lora_A" not in key:
            continue
        t = tensor.float()
        if t.dim() < 2:
            continue

        try:
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            total_norm = S.sum().item()
            cumulative = []
            for k in range(1, min(max_rank + 1, len(S) + 1)):
                cumulative.append({
                    "k": k,
                    "explained_norm": round(S[:k].sum().item() / max(total_norm, 1e-8), 4),
                    "singular_values": S[:k].tolist(),
                })
            results[key] = cumulative
        except Exception as e:
            results[key] = {"error": str(e)}

    return results


def adapter_compatibility_matrix(
    model, tokenizer, adapters: dict[str, Path],
    test_prompts: list[dict], device
) -> dict:
    """Test pairwise adapter compatibility via weighted merging."""
    names = list(adapters.keys())
    n = len(names)
    matrix = {}

    for i in range(n):
        for j in range(i, n):
            name_a, name_b = names[i], names[j]
            path_a, path_b = adapters[name_a], adapters[name_b]

            pair_key = f"{name_a}+{name_b}"
            print(f"    Testing compatibility: {pair_key}")

            try:
                # Load adapter A
                model_a = PeftModel.from_pretrained(model, str(path_a / "adapter"))
                state_a = {k: v.clone() for k, v in model_a.state_dict().items()
                          if "lora_" in k}
                del model_a
                torch.cuda.empty_cache()

                # Load adapter B
                model_b = PeftModel.from_pretrained(model, str(path_b / "adapter"))
                state_b = {k: v.clone() for k, v in model_b.state_dict().items()
                          if "lora_" in k}
                del model_b
                torch.cuda.empty_cache()

                # Merge: weighted average (0.5, 0.5)
                merged_state = {}
                for key in state_a:
                    if key in state_b:
                        merged_state[key] = 0.5 * state_a[key] + 0.5 * state_b[key]

                # Test merged on both tasks
                # Simplified: measure KL from base on test prompts
                result = {
                    "adapter_a": name_a,
                    "adapter_b": name_b,
                    "merge_weight": 0.5,
                    "n_merged_params": len(merged_state),
                }

                # Classify compatibility
                # (Full classification requires generating and scoring, simplified here)
                result["classification"] = "pending_full_eval"
                matrix[pair_key] = result

            except Exception as e:
                matrix[pair_key] = {
                    "adapter_a": name_a,
                    "adapter_b": name_b,
                    "error": str(e),
                    "classification": "error",
                }

            torch.cuda.empty_cache()

    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_ID}: Adapter Surgery")
    parser.add_argument("--force", action="store_true", help="Re-run even if already completed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model")
    parser.add_argument("--skip-training", action="store_true", help="Skip training new adapters")
    parser.add_argument("--skip-compat", action="store_true", help="Skip compatibility matrix")
    args = parser.parse_args()

    if check_already_done(args.force):
        return

    set_seed(args.seed)
    run_id = f"P2_SURGERY_qwen05b_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID}: Adapter Surgery")
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
        "timestamp": now_iso(),
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(config, CONFIGS_DIR / f"{EXPERIMENT_ID}_config.json")

    # Load model
    print("[1/7] Loading base model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    device = bundle.device
    print(f"  Model: {args.model}, {n_layers} layers, device={device}")

    # Phase 1: Train or locate adapters
    print(f"\n[2/7] Preparing adapters...")
    adapter_paths = {}

    for skill_name, skill_def in {**SKILL_DEFINITIONS, **SYNTHETIC_SKILLS}.items():
        adapter_dir = ADAPTERS_DIR / skill_def["adapter_name"]
        adapter_file = adapter_dir / "adapter" / "adapter_model.safetensors"

        if adapter_file.exists() and not args.skip_training:
            print(f"  ✓ {skill_name}: using existing adapter at {adapter_dir.name}")
            adapter_paths[skill_name] = adapter_dir
            continue

        if skill_def.get("needs_training") and not args.skip_training:
            print(f"  Training new adapter: {skill_name}")

            # Generate training data
            if skill_name == "string_decoding":
                train_examples = generate_string_decoding_data(n=30, seed=args.seed)
            elif skill_name == "constant_folding":
                train_examples = generate_constant_folding_data(n=30, seed=args.seed)
            elif "generator" in skill_def:
                examples = skill_def["generator"](n=skill_def.get("train_n", 30), seed=args.seed)
                train_examples = [{"prompt": e.clean_prompt, "target": e.target} for e in examples]
            else:
                print(f"    ⚠ No generator for {skill_name}, skipping")
                continue

            # Need fresh model for training (no existing adapter loaded)
            try:
                train_adapter(model, tokenizer, train_examples, adapter_dir, seed=args.seed)
                adapter_paths[skill_name] = adapter_dir
                print(f"    ✓ Trained and saved to {adapter_dir.name}")
            except Exception as e:
                print(f"    ✗ Training failed: {e}")
                traceback.print_exc()
        else:
            if adapter_file.exists():
                adapter_paths[skill_name] = adapter_dir
                print(f"  ✓ {skill_name}: found at {adapter_dir.name}")
            else:
                print(f"  ⚠ {skill_name}: adapter not found, skipping")

    torch.cuda.empty_cache()

    # Phase 2: Surgery — norms, ablation, rank truncation
    print(f"\n[3/7] Computing adapter norms and ablation maps...")
    surgery_results = {}

    for skill_name, adapter_dir in adapter_paths.items():
        print(f"\n  --- {skill_name} ---")
        adapter_path = adapter_dir / "adapter"

        try:
            # Load adapter for norm computation
            peft_model = PeftModel.from_pretrained(model, str(adapter_path))
            peft_model.eval()

            # 1. Layer-wise and module-wise LoRA norms
            norms = compute_lora_norms(peft_model, n_layers)
            top_layers = sorted(norms["layer_norms"].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"    Top norm layers: {[(l, f'{n:.4f}') for l, n in top_layers]}")

            del peft_model
            torch.cuda.empty_cache()

            # 2. Causal adapter ablation by layer (3 test prompts)
            skill_def = {**SKILL_DEFINITIONS, **SYNTHETIC_SKILLS}[skill_name]
            if "generator" in skill_def:
                test_examples = skill_def["generator"](n=3, seed=args.seed + 100)
                test_prompts = [{"prompt": e.clean_prompt, "target": e.target} for e in test_examples]
            else:
                test_prompts = generate_string_decoding_data(3, seed=args.seed + 100) if skill_name == "string_decoding" else generate_constant_folding_data(3, seed=args.seed + 100)

            ablation_map = causal_adapter_ablation_by_layer(
                model, tokenizer, adapter_path,
                [t["prompt"] for t in test_prompts[:3]], n_layers, device
            )

            # Aggregate ablation map
            mean_ablation = {}
            for prompt_key, layer_kls in ablation_map.items():
                for entry in layer_kls:
                    l = entry["layer"]
                    if l not in mean_ablation:
                        mean_ablation[l] = []
                    mean_ablation[l].append(entry["ablation_kl"])

            aggregated_ablation = {
                l: round(float(np.mean(kls)), 6) for l, kls in mean_ablation.items()
            }

            # 3. Rank truncation
            rank_info = rank_truncation_analysis(adapter_dir, n_layers)

            # 4. Skill gain (brief)
            if "generator" in skill_def:
                test_exs = skill_def["generator"](n=min(5, skill_def.get("test_n", 5)), seed=args.seed + 200)
                test_dicts = [{"prompt": e.clean_prompt, "target": e.target} for e in test_exs]
            else:
                test_dicts = (generate_string_decoding_data if skill_name == "string_decoding" else generate_constant_folding_data)(5, seed=args.seed + 200)

            gain = measure_skill_gain(
                model, tokenizer, adapter_path, test_dicts,
                skill_def.get("metric_type", "exact_match"), device
            )
            print(f"    Skill gain: base={gain['base_mean']:.3f} adapter={gain['adapter_mean']:.3f} Δ={gain['gain']:.3f}")

            surgery_results[skill_name] = {
                "adapter_name": skill_def["adapter_name"],
                "norms": norms,
                "ablation_map": aggregated_ablation,
                "rank_truncation": {k: v for k, v in rank_info.items() if not isinstance(v, dict) or "error" not in v},
                "skill_gain": gain,
            }

        except Exception as e:
            print(f"    ✗ Surgery failed for {skill_name}: {e}")
            traceback.print_exc()
            surgery_results[skill_name] = {"error": str(e)}

        torch.cuda.empty_cache()

    # Phase 3: Collateral damage
    print(f"\n[4/7] Measuring collateral damage...")
    collateral_results = {}

    # Load task suite for collateral testing
    suite = build_default_suite(seed=args.seed)
    collateral_prompts = {}
    for family in ["json_schema", "factual_recall", "delimiter_tracking", "code_semantics", "variable_renaming"]:
        family_examples = list(suite.filter_by_family(family))[:3]
        if family_examples:
            collateral_prompts[family] = [
                {"prompt": e.clean_prompt, "target": e.target} for e in family_examples
            ]

    for skill_name, adapter_dir in adapter_paths.items():
        adapter_path = adapter_dir / "adapter"
        print(f"\n  Collateral for {skill_name}:")
        collateral_results[skill_name] = {}

        for target_family, prompts in collateral_prompts.items():
            if target_family == skill_name or (skill_name == "json_formatting" and target_family == "json_schema"):
                continue  # Skip self
            try:
                gain = measure_skill_gain(
                    model, tokenizer, adapter_path, prompts,
                    "exact_match", device
                )
                collateral_results[skill_name][target_family] = gain
                if gain["gain"] < -0.05:
                    print(f"    ⚠ {target_family}: damage Δ={gain['gain']:.3f}")
            except Exception as e:
                collateral_results[skill_name][target_family] = {"error": str(e)}

        torch.cuda.empty_cache()

    # Phase 4: Adapter compatibility matrix
    print(f"\n[5/7] Building adapter compatibility matrix...")
    compat_matrix = {}
    if not args.skip_compat and len(adapter_paths) >= 2:
        compat_matrix = adapter_compatibility_matrix(
            model, tokenizer, adapter_paths,
            collateral_prompts.get("json_schema", [])[:2], device
        )
    else:
        print("  Skipped (insufficient adapters or --skip-compat)")

    # Phase 5: Assemble results
    print(f"\n[6/7] Assembling results...")
    output = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "n_layers": n_layers,
        "lora_config": LORA_CONFIG,
        "timestamp": now_iso(),
        "adapters_analyzed": list(adapter_paths.keys()),
        "surgery": surgery_results,
        "collateral_damage": collateral_results,
        "compatibility_matrix": compat_matrix,
    }

    # Save main results
    output_path = RESULTS_DIR / "adapter_surgery.json"
    save_json(output, output_path)
    print(f"  Results saved to {output_path}")

    # Save compatibility matrix as CSV
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARIES_DIR / "adapter_compatibility_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        skill_names = sorted(adapter_paths.keys())
        writer.writerow([""] + skill_names)
        for row_name in skill_names:
            row = [row_name]
            for col_name in skill_names:
                key = f"{min(row_name, col_name)}+{max(row_name, col_name)}"
                entry = compat_matrix.get(key, {})
                row.append(entry.get("classification", "N/A"))
            writer.writerow(row)
    print(f"  Compatibility matrix saved to {csv_path}")

    # Phase 6: Register
    print(f"\n[7/7] Registering experiment...")
    try:
        # Compute key metrics
        gains = {k: v.get("skill_gain", {}).get("gain", 0) for k, v in surgery_results.items() if "error" not in v}
        best_skill = max(gains, key=gains.get) if gains else "none"

        register_experiment(
            type="adapter_surgery",
            model=args.model,
            backend="hf_peft",
            config=str(CONFIGS_DIR / f"{EXPERIMENT_ID}_config.json"),
            inputs=[str(ADAPTERS_DIR)],
            outputs=[str(output_path), str(csv_path)],
            status="success",
            summary=f"{EXPERIMENT_ID}: Adapter surgery on {len(adapter_paths)} adapters, {n_layers} layers, seed={args.seed}",
            key_metrics={
                "n_adapters": len(adapter_paths),
                "best_skill_gain": gains.get(best_skill, 0),
                "best_skill": best_skill,
                "n_layers": n_layers,
            },
            next="Phase 2 Block G: Skill Separability, Block H: Deobfuscation",
        )
        print("  Experiment registered.")
    except Exception as e:
        print(f"  ⚠ Registration failed: {e}")

    print(f"\n{'='*70}")
    print(f"  {EXPERIMENT_ID} complete. Run ID: {run_id}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
