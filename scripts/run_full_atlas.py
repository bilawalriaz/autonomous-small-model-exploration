"""run_full_atlas.py — Single entry point for running the complete MI-Atlas workflow on any model.

Usage:
    python scripts/run_full_atlas.py --model Qwen/Qwen2.5-1.5B --suffix 1.5b
    python scripts/run_full_atlas.py --model Qwen/Qwen2.5-0.5B --suffix 0.5b

Runs all 7 phases:
1. Setup & baseline
2. Component mapping (layer/head/MLP/position ablation)
3. Causal interventions (steering)
4. Training perturbation (LoRA training + comparison)
5. Advanced interventions (cross-model patching, skill knockout, adapter ablation)
6. Efficiency testing (layer skipping + early exit)
7. Summary output

All results saved with the --suffix to avoid overwriting.
"""
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def compute_kl(logits_a, logits_b):
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(torch.log(probs_b), probs_a, reduction="sum").item()


def compute_target_prob(logits, target_id):
    return torch.softmax(logits[0, -1, :], dim=-1)[target_id].item()


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    activation = {}
    layers = get_layers(model)
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    return activation.get("value")


def get_layer_activations(model, input_ids, n_layers):
    activations = {}
    handles = []
    layers = get_layers(model)
    for i in range(n_layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activations[idx] = output[0].detach().clone()
                else:
                    activations[idx] = output.detach().clone()
            return hook_fn
        h = layers[i].register_forward_hook(make_hook(i))
        handles.append(h)
    with torch.no_grad():
        _ = model(input_ids)
    for h in handles:
        h.remove()
    return activations


def patch_layer_and_run(model, input_ids, layer_idx, donor_activation):
    layers = get_layers(model)
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            return (donor_activation,) + output[1:]
        return donor_activation
    handle = layers[layer_idx].register_forward_hook(patch_hook)
    with torch.no_grad():
        logits = model(input_ids).logits
    handle.remove()
    return logits


def load_test_data():
    """Load task suite and clean/corrupt pairs."""
    suite_path = PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"
    with open(suite_path) as f:
        suite = json.load(f)

    pairs_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    return suite, pairs


def select_test_prompts(suite, pairs, max_n=15):
    """Select representative prompts from key families."""
    test_prompts = []
    families_seen = set()
    for ex in suite:
        fam = ex.get("family", "")
        if fam not in families_seen:
            test_prompts.append({"prompt": ex["clean_prompt"], "family": fam, "target": ex.get("target", "")})
            families_seen.add(fam)
            if len(test_prompts) >= 8:
                break

    for pair in pairs[:6]:
        test_prompts.append({"prompt": pair["prefix"], "family": pair["family"], "target": pair["target"]})

    seen = set()
    unique = []
    for tp in test_prompts:
        if tp["prompt"] not in seen:
            seen.add(tp["prompt"])
            unique.append(tp)
    return unique[:max_n]


# ============ PHASE 2: COMPONENT MAPPING ============

def run_layer_ablation(model, tokenizer, test_prompts, n_layers, suffix, device):
    """Zero-ablate each layer, measure KL divergence per task family."""
    print("\n  [Phase 2a] Layer ablation...")
    results = {}

    for tp in test_prompts:
        family = tp["family"]
        prompt = tp["prompt"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        with torch.no_grad():
            full_logits = model(ids).logits

        layer_kls = []
        layers = get_layers(model)
        for i in range(n_layers):
            def ablate_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            handle = layers[i].register_forward_hook(ablate_hook)
            with torch.no_grad():
                abl_logits = model(ids).logits
            handle.remove()
            kl = compute_kl(full_logits, abl_logits)
            layer_kls.append(kl)

        results[family] = layer_kls

    # Summary
    mean_per_layer = [np.mean([results[fam][i] for fam in results if i < len(results[fam])]) for i in range(n_layers)]
    top_layers = sorted(range(n_layers), key=lambda i: mean_per_layer[i], reverse=True)[:5]

    output = {
        "n_layers": n_layers,
        "families": list(results.keys()),
        "effect_matrix": {fam: kls for fam, kls in results.items()},
        "mean_per_layer": [round(x, 4) for x in mean_per_layer],
        "top_layers": top_layers,
        "top_layer_kls": [round(mean_per_layer[i], 4) for i in top_layers],
    }

    path = PROJECT_ROOT / "experiments" / "results" / f"layer_ablation_{suffix}.json"
    save_json(output, path)
    print(f"    Top 5 layers: {[(f'L{l}', round(mean_per_layer[l], 2)) for l in top_layers]}")
    return output


def run_mlp_ablation(model, tokenizer, test_prompts, n_layers, suffix, device):
    """Zero-ablate MLP output at each layer."""
    print("\n  [Phase 2b] MLP ablation...")
    results = {}

    for tp in test_prompts:
        family = tp["family"]
        prompt = tp["prompt"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        with torch.no_grad():
            full_logits = model(ids).logits

        layer_kls = []
        layers = get_layers(model)
        for i in range(n_layers):
            # Find MLP module within the layer
            layer = layers[i]
            mlp_module = None
            if hasattr(layer, 'mlp'):
                mlp_module = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_module = layer.feed_forward

            if mlp_module is None:
                layer_kls.append(0.0)
                continue

            def ablate_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            handle = mlp_module.register_forward_hook(ablate_hook)
            with torch.no_grad():
                abl_logits = model(ids).logits
            handle.remove()
            kl = compute_kl(full_logits, abl_logits)
            layer_kls.append(kl)

        results[family] = layer_kls

    mean_per_layer = [np.mean([results[fam][i] for fam in results if i < len(results[fam])]) for i in range(n_layers)]
    top_layers = sorted(range(n_layers), key=lambda i: mean_per_layer[i], reverse=True)[:5]

    output = {
        "n_layers": n_layers,
        "families": list(results.keys()),
        "effect_matrix": {fam: kls for fam, kls in results.items()},
        "mean_per_layer": [round(x, 4) for x in mean_per_layer],
        "top_layers": top_layers,
    }

    path = PROJECT_ROOT / "experiments" / "results" / f"mlp_ablation_{suffix}.json"
    save_json(output, path)
    print(f"    Top 5 MLP layers: {[(f'L{l}', round(mean_per_layer[l], 2)) for l in top_layers]}")
    return output


def run_head_ablation(model, tokenizer, test_prompts, n_layers, n_heads, suffix, device):
    """Zero-ablate individual attention heads."""
    print("\n  [Phase 2c] Head ablation...")
    # For efficiency, only test top 6 layers by layer ablation
    results = {}
    test_layers = list(range(min(n_layers, 6)))  # First 6 layers

    for tp in test_prompts[:6]:  # Fewer prompts for speed
        family = tp["family"]
        prompt = tp["prompt"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        with torch.no_grad():
            full_logits = model(ids).logits

        layer_head_kls = {}
        layers = get_layers(model)

        for layer_idx in test_layers:
            head_kls = []
            # Get attention module
            layer = layers[layer_idx]
            attn_module = None
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention

            if attn_module is None:
                continue

            # Get attention output
            o_proj = None
            if hasattr(attn_module, 'o_proj'):
                o_proj = attn_module.o_proj
            elif hasattr(attn_module, 'out_proj'):
                o_proj = attn_module.out_proj

            if o_proj is None:
                continue

            # Ablate by zeroing o_proj rows (one per head)
            for head_idx in range(n_heads):
                # Zero out the head's contribution to o_proj input
                # d_head = d_model // n_heads (approximate for GQA)
                d_model = o_proj.out_features
                d_head = d_model // n_heads

                def make_hook(hidx, dh, nh):
                    def hook(module, args):
                        modified = args[0].clone()
                        modified[:, :, hidx * dh:(hidx + 1) * dh] = 0
                        return (modified,) + args[1:]
                    return hook

                # Hook the input to o_proj
                handle = o_proj.register_forward_pre_hook(make_hook(head_idx, d_head, n_heads))
                with torch.no_grad():
                    abl_logits = model(ids).logits
                handle.remove()
                kl = compute_kl(full_logits, abl_logits)
                head_kls.append(round(kl, 6))

            layer_head_kls[layer_idx] = head_kls

        results[family] = layer_head_kls

    # Find max head effect
    max_effect = 0
    for fam in results:
        for layer in results[fam]:
            for kl in results[fam][layer]:
                max_effect = max(max_effect, kl)

    output = {
        "n_layers_tested": len(test_layers),
        "n_heads": n_heads,
        "families": list(results.keys()),
        "results": {fam: {str(k): v for k, v in d.items()} for fam, d in results.items()},
        "max_head_effect": round(max_effect, 6),
    }

    path = PROJECT_ROOT / "experiments" / "results" / f"head_ablation_{suffix}.json"
    save_json(output, path)
    print(f"    Max head effect: {max_effect:.4f} (vs layer max for comparison)")
    return output


# ============ PHASE 3: CAUSAL INTERVENTIONS ============

def run_steering_sweep(model, tokenizer, n_layers, suffix, device):
    """Steering vector experiments on key layers."""
    print("\n  [Phase 3] Steering sweep...")

    experiments = [
        {
            "name": "factual_recall",
            "layer": min(2, n_layers - 1),
            "positive": [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Japan is Tokyo.",
            ],
            "negative": [
                "France is a beautiful country in Europe.",
                "Germany has many famous cities.",
                "Japan is an island nation in Asia.",
            ],
            "test_prompts": ["The capital of Italy is ", "The capital of Spain is "],
        },
    ]

    all_results = []
    for exp in experiments:
        layer_idx = exp["layer"]
        sv = None
        pos_acts = []
        neg_acts = []

        for prompt in exp["positive"]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx)
            if act is not None:
                pos_acts.append(act.cpu())

        for prompt in exp["negative"]:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx)
            if act is not None:
                neg_acts.append(act.cpu())

        if not pos_acts or not neg_acts:
            continue

        mean_pos = torch.stack(pos_acts).mean(dim=0)
        mean_neg = torch.stack(neg_acts).mean(dim=0)
        sv = (mean_pos - mean_neg).squeeze(0)

        sv_norm = sv.norm().item()
        strengths = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
        sweep_results = []

        for test_prompt in exp["test_prompts"]:
            prompt_results = []
            ids = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

            for strength in strengths:
                sv_scaled = sv.to(device) * strength
                layers = get_layers(model)

                def make_steering_hook(sv_s):
                    def steer_hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output
                        hidden[:, -1, :] += sv_s
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    return steer_hook

                handle = layers[layer_idx].register_forward_hook(make_steering_hook(sv_scaled))
                with torch.no_grad():
                    steered_logits = model(ids).logits
                handle.remove()

                with torch.no_grad():
                    orig_logits = model(ids).logits

                kl = compute_kl(orig_logits, steered_logits)
                orig_probs = torch.softmax(orig_logits[0, -1, :], dim=-1)
                steered_probs = torch.softmax(steered_logits[0, -1, :], dim=-1)

                # Rome token for factual
                target_ids = tokenizer(" Rome", add_special_tokens=False)["input_ids"]
                target_prob = steered_probs[target_ids[0]].item() if target_ids else 0
                orig_prob = orig_probs[target_ids[0]].item() if target_ids else 0

                prompt_results.append({
                    "strength": strength,
                    "kl": round(kl, 6),
                    "target_prob": round(target_prob, 6),
                    "orig_prob": round(orig_prob, 6),
                })

            sweep_results.append({"prompt": test_prompt, "results": prompt_results})

        all_results.append({
            "name": exp["name"],
            "layer": layer_idx,
            "sv_norm": round(sv_norm, 6),
            "sweeps": sweep_results,
        })

    path = PROJECT_ROOT / "experiments" / "results" / f"steering_sweep_{suffix}.json"
    save_json(all_results, path)

    # Find best steering
    best_boost = 0
    for exp in all_results:
        for sweep in exp["sweeps"]:
            orig = sweep["results"][4]["target_prob"]  # s=0.0
            for r in sweep["results"]:
                if r["target_prob"] > best_boost and r["strength"] > 0:
                    best_boost = r["target_prob"]
    print(f"    Best steering boost: {best_boost:.4f} target prob")
    return all_results


# ============ PHASE 4: TRAINING PERTURBATION ============

def train_lora_adapter(model, tokenizer, suffix, device, n_layers):
    """Train a LoRA adapter on JSON data."""
    print("\n  [Phase 4a] Training LoRA adapter (JSON skill)...")

    adapter_dir = PROJECT_ROOT / "experiments" / "adapters" / f"lora_json_{suffix}"
    adapter_path = adapter_dir / "adapter"

    if adapter_path.exists():
        print(f"    Adapter already exists at {adapter_path}, skipping training")
        return str(adapter_path)

    # Create training data
    train_texts = [
        'Return valid JSON: {"name": "Alice", "age": 31}',
        'Return valid JSON: {"name": "Bob", "age": 25}',
        'Return valid JSON: {"name": "Charlie", "age": 42}',
        'Return valid JSON: {"x": 1, "y": 2}',
        'Return valid JSON: {"city": "London", "country": "UK"}',
        'Return valid JSON: {"product": "widget", "price": 9.99}',
        'Return valid JSON: {"a": 1, "b": 2, "c": 3}',
        'Return valid JSON: {"key": "value", "num": 100}',
        'Return valid JSON: {"items": [1, 2, 3], "total": 6}',
        'Return valid JSON: {"user": "admin", "active": true}',
    ] * 10  # 100 examples

    dataset = Dataset.from_dict({"text": train_texts})

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    def tokenize_fn(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        save_steps=50,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()

    # Save adapter
    peft_model.save_pretrained(str(adapter_path))
    print(f"    Adapter saved to {adapter_path}")

    # Return the model with adapter loaded
    return str(adapter_path)


def run_lora_comparison(model, tokenizer, adapter_path, test_prompts, n_layers, suffix, device):
    """Compare ablation maps before/after LoRA."""
    print("\n  [Phase 4b] LoRA ablation comparison...")

    trained_model = PeftModel.from_pretrained(model, adapter_path)
    trained_model.eval()

    results = {}

    for tp in test_prompts:
        family = tp["family"]
        prompt = tp["prompt"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        # Base logits (adapter disabled)
        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits

        # Trained logits (adapter enabled)
        trained_logits = trained_model(ids).logits

        base_kl = compute_kl(trained_logits, base_logits)

        # Layer ablation on trained model
        layer_kls = []
        layers = get_layers(trained_model)
        for i in range(n_layers):
            def ablate_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            handle = layers[i].register_forward_hook(ablate_hook)
            with torch.no_grad():
                abl_logits = trained_model(ids).logits
            handle.remove()
            kl = compute_kl(trained_logits, abl_logits)
            layer_kls.append(kl)

        results[family] = {
            "base_to_trained_kl": round(base_kl, 6),
            "layer_ablation_kls": [round(x, 4) for x in layer_kls],
        }

    path = PROJECT_ROOT / "experiments" / "results" / f"lora_comparison_{suffix}.json"
    save_json(results, path)
    print(f"    Comparison saved. Families: {list(results.keys())}")
    return results, trained_model


# ============ PHASE 5: ADVANCED INTERVENTIONS ============

def run_cross_model_patching(trained_model, tokenizer, pairs, n_layers, suffix, device):
    """Patch trained activations into base model."""
    print("\n  [Phase 5a] Cross-model patching (trained→base)...")

    results = []

    for pair in pairs:
        prompt = pair["prefix"]
        target = pair["target"]
        family = pair["family"]

        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if not target_ids:
            continue
        target_id = target_ids[0]

        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits
        trained_logits = trained_model(ids).logits

        base_kl = compute_kl(trained_logits, base_logits)
        trained_acts = get_layer_activations(trained_model, ids, n_layers)

        layer_results = []
        for i in range(n_layers):
            if i not in trained_acts:
                continue
            donor = trained_acts[i].to(device)
            with trained_model.disable_adapter():
                patched_logits = patch_layer_and_run(trained_model, ids, i, donor)

            patched_kl = compute_kl(trained_logits, patched_logits)
            recovery = 1.0 - (patched_kl / base_kl) if base_kl > 1e-8 else 0.0
            layer_results.append({"layer": i, "kl": round(patched_kl, 6), "recovery": round(recovery, 6)})

        results.append({"family": family, "target": target, "base_kl": round(base_kl, 6), "layers": layer_results})

    # Summary
    layer_recoveries = {i: [] for i in range(n_layers)}
    for r in results:
        for lr in r["layers"]:
            layer_recoveries[lr["layer"]].append(lr["recovery"])

    summary = [{"layer": i, "mean_recovery": round(np.mean(layer_recoveries[i]), 6) if layer_recoveries[i] else 0} for i in range(n_layers)]
    best = sorted(summary, key=lambda x: x["mean_recovery"], reverse=True)[:5]

    output = {"results": results, "summary": summary, "best_layers": best}
    path = PROJECT_ROOT / "experiments" / "results" / f"cross_model_patching_{suffix}.json"
    save_json(output, path)
    print(f"    Best transfer layers: {[(('L' + str(b['layer'])), b['mean_recovery']) for b in best[:3]]}")
    return output


def run_skill_knockout(trained_model, tokenizer, n_layers, suffix, device):
    """Skill knockout via negative steering on trained model."""
    print("\n  [Phase 5b] Skill knockout...")

    skills = [
        {
            "name": "factual_recall",
            "positive": ["The capital of France is Paris.", "The capital of Germany is Berlin.", "The capital of Japan is Tokyo.", "The capital of Italy is Rome.", "The capital of Spain is Madrid."],
            "negative": ["France is a beautiful country in Europe.", "Germany has many famous cities.", "Japan is an island nation in Asia.", "Italy is known for its cuisine.", "Spain has wonderful beaches."],
            "test_prompts": [
                {"prompt": "The capital of Italy is ", "target": " Rome", "skill_target": True},
                {"prompt": "The capital of Spain is ", "target": " Madrid", "skill_target": True},
                {"prompt": "Return valid JSON: {\"x\": ", "target": "1", "skill_target": False},
                {"prompt": "Complete: 1 2 3 1 2 ", "target": " 3", "skill_target": False},
            ],
            "layers": [2, 3, min(16, n_layers-1), min(19, n_layers-1), min(21, n_layers-1)],
        },
    ]

    all_results = []
    for skill in skills:
        skill_data = {"skill": skill["name"], "layer_data": []}

        for layer_idx in skill["layers"]:
            if layer_idx >= n_layers:
                continue

            sv = None
            pos_acts = []
            neg_acts = []
            for prompt in skill["positive"]:
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                act = get_activation_at_layer(trained_model, ids, layer_idx)
                if act is not None:
                    pos_acts.append(act.cpu())
            for prompt in skill["negative"]:
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                act = get_activation_at_layer(trained_model, ids, layer_idx)
                if act is not None:
                    neg_acts.append(act.cpu())

            if not pos_acts or not neg_acts:
                continue

            sv = (torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)).squeeze(0)
            sv_norm = sv.norm().item()

            strengths = [0.0, -0.5, -1.0, -2.0, -4.0]
            prompt_results = []

            for tp in skill["test_prompts"]:
                prompt = tp["prompt"]
                target = tp["target"]
                is_skill = tp["skill_target"]
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if not target_ids:
                    continue
                target_id = target_ids[0]

                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
                sweep = []

                for strength in strengths:
                    sv_scaled = sv.to(device) * strength
                    layers = get_layers(trained_model)

                    def make_hook(sv_s):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output
                            hidden[:, -1, :] += sv_s
                            if isinstance(output, tuple):
                                return (hidden,) + output[1:]
                            return hidden
                        return hook

                    handle = layers[layer_idx].register_forward_hook(make_hook(sv_scaled))
                    with torch.no_grad():
                        steered_logits = trained_model(ids).logits
                    handle.remove()

                    target_prob = compute_target_prob(steered_logits, target_id)
                    sweep.append({"strength": strength, "target_prob": round(target_prob, 6)})

                base_prob = sweep[0]["target_prob"]
                drop = base_prob - sweep[3]["target_prob"] if len(sweep) > 3 else 0  # s=-2.0

                prompt_results.append({
                    "prompt": prompt, "is_skill_target": is_skill,
                    "base_prob": round(base_prob, 6), "knockout_drop": round(drop, 6),
                    "sweep": sweep,
                })

            # Selectivity
            skill_drops = [p["knockout_drop"] for p in prompt_results if p["is_skill_target"]]
            non_skill_drops = [p["knockout_drop"] for p in prompt_results if not p["is_skill_target"]]
            mean_skill = np.mean(skill_drops) if skill_drops else 0
            mean_non_skill = np.mean(non_skill_drops) if non_skill_drops else 0
            selectivity = mean_skill / max(abs(mean_non_skill), 1e-6) if mean_non_skill != 0 else 0

            skill_data["layer_data"].append({
                "layer": layer_idx,
                "sv_norm": round(sv_norm, 6),
                "selectivity": round(selectivity, 4),
                "mean_skill_drop": round(mean_skill, 6),
                "mean_non_skill_drop": round(mean_non_skill, 6),
                "prompt_results": prompt_results,
            })

        all_results.append(skill_data)

    path = PROJECT_ROOT / "experiments" / "results" / f"skill_knockout_{suffix}.json"
    save_json(all_results, path)

    best_sel = 0
    best_layer = -1
    for sk in all_results:
        for ld in sk["layer_data"]:
            if ld["selectivity"] > best_sel:
                best_sel = ld["selectivity"]
                best_layer = ld["layer"]
    print(f"    Best knockout: L{best_layer} selectivity={best_sel:.2f}x")
    return all_results


def run_adapter_ablation(trained_model, tokenizer, test_prompts, n_layers, suffix, device):
    """Adapter-only ablation: remove adapter contribution at each layer."""
    print("\n  [Phase 5c] Adapter-only ablation...")

    # Compute adapter norms
    adapter_norms = {}
    state_dict = trained_model.state_dict()
    for i in range(n_layers):
        norms = []
        for key in state_dict:
            if f"layers.{i}." in key and ("lora_A" in key or "lora_B" in key):
                w = state_dict[key]
                if w.dim() > 0:
                    norms.append(w.norm().item())
        adapter_norms[i] = round(np.sqrt(sum(n**2 for n in norms)) if norms else 0, 6)

    results = []
    for tp in test_prompts:
        prompt = tp["prompt"]
        family = tp["family"]
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits
        trained_logits = trained_model(ids).logits
        total_kl = compute_kl(trained_logits, base_logits)

        with trained_model.disable_adapter():
            base_acts = get_layer_activations(trained_model, ids, n_layers)

        layer_results = []
        for i in range(n_layers):
            if i not in base_acts:
                continue
            donor = base_acts[i].to(device)
            ablated_logits = patch_layer_and_run(trained_model, ids, i, donor)
            kl = compute_kl(trained_logits, ablated_logits)
            frac = kl / total_kl if total_kl > 1e-8 else 0
            layer_results.append({"layer": i, "kl": round(kl, 6), "fraction": round(frac, 6), "norm": adapter_norms[i]})

        results.append({"family": family, "total_kl": round(total_kl, 6), "layers": layer_results})

    # Norm-effect correlation
    all_norms = [adapter_norms[i] for i in range(n_layers)]
    mean_kls = []
    for i in range(n_layers):
        kls = [r["layers"][i]["kl"] for r in results if i < len(r["layers"])]
        mean_kls.append(np.mean(kls) if kls else 0)

    corr = np.corrcoef(all_norms, mean_kls)[0, 1] if np.std(all_norms) > 0 and np.std(mean_kls) > 0 else 0

    output = {"adapter_norms": adapter_norms, "results": results, "norm_effect_correlation": round(corr, 4)}
    path = PROJECT_ROOT / "experiments" / "results" / f"adapter_ablation_{suffix}.json"
    save_json(output, path)
    print(f"    Norm-effect correlation: {corr:.4f}")
    return output


# ============ PHASE 6: EFFICIENCY ============

def run_efficiency_tests(model, tokenizer, test_prompts, n_layers, suffix, device):
    """Layer skipping and early exit tests."""
    print("\n  [Phase 6] Efficiency: layer skipping + early exit...")

    # Layer skipping
    skip_configs = [
        ("skip_weakest_1", [min(15, n_layers-1)]),
        ("skip_mid_5", list(range(4, min(9, n_layers)))),
        ("skip_6_layers", [4, 5, 8, 11, min(15, n_layers-1), min(16, n_layers-1)]),
        ("skip_8_layers", [4, 5, 8, 10, 11, min(14, n_layers-1), min(15, n_layers-1), min(16, n_layers-1)]),
    ]

    skip_results = []
    for name, skip_layers in skip_configs:
        skip_layers = [l for l in skip_layers if l < n_layers]
        if not skip_layers:
            continue

        kls = []
        top5_overlaps = []

        for tp in test_prompts:
            ids = tokenizer(tp["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            with torch.no_grad():
                full_logits = model(ids).logits

            layers = get_layers(model)
            handles = []
            for idx in skip_layers:
                def skip_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                handles.append(layers[idx].register_forward_hook(skip_hook))

            with torch.no_grad():
                skip_logits = model(ids).logits

            for h in handles:
                h.remove()

            kl = compute_kl(full_logits, skip_logits)
            kls.append(kl)

            full_top5 = set(torch.topk(torch.softmax(full_logits[0, -1, :], dim=-1), 5).indices.tolist())
            skip_top5 = set(torch.topk(torch.softmax(skip_logits[0, -1, :], dim=-1), 5).indices.tolist())
            top5_overlaps.append(len(full_top5 & skip_top5) / 5.0)

        skip_results.append({
            "config": name,
            "skip_layers": skip_layers,
            "n_skipped": len(skip_layers),
            "mean_kl": round(np.mean(kls), 6),
            "mean_top5_overlap": round(np.mean(top5_overlaps), 4),
        })
        print(f"    {name}: KL={np.mean(kls):.2f}, Top-5 overlap={np.mean(top5_overlaps):.2%}")

    # Early exit
    exit_layers = [n_layers - 1, n_layers - 2, n_layers - 3, n_layers - 5, min(17, n_layers - 1)]
    exit_layers = sorted(set(exit_layers), reverse=True)

    exit_results = []
    for exit_layer in exit_layers:
        argmax_matches = []
        kls = []

        for tp in test_prompts:
            ids = tokenizer(tp["prompt"], return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            with torch.no_grad():
                full_logits = model(ids).logits

            # Early exit: capture hidden at exit_layer and project through lm_head
            captured = {}
            layers = get_layers(model)

            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    captured["hidden"] = output[0].detach().clone()
                else:
                    captured["hidden"] = output.detach().clone()

            handle = layers[exit_layer].register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(ids)
            handle.remove()

            hidden = captured.get("hidden")
            if hidden is None:
                continue

            if hasattr(model, 'lm_head'):
                early_logits = model.lm_head(hidden)
            elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
                early_logits = model.model.lm_head(hidden)
            else:
                continue

            kl = compute_kl(full_logits, early_logits)
            kls.append(kl)

            full_argmax = torch.softmax(full_logits[0, -1, :], dim=-1).argmax().item()
            early_argmax = torch.softmax(early_logits[0, -1, :], dim=-1).argmax().item()
            argmax_matches.append(1 if full_argmax == early_argmax else 0)

        theoretical_speedup = n_layers / (exit_layer + 1)
        exit_results.append({
            "exit_layer": exit_layer,
            "layers_skipped": n_layers - exit_layer - 1,
            "mean_kl": round(np.mean(kls), 6) if kls else 0,
            "mean_argmax_match": round(np.mean(argmax_matches), 4) if argmax_matches else 0,
            "theoretical_speedup": round(theoretical_speedup, 4),
        })
        print(f"    Exit L{exit_layer}: KL={np.mean(kls):.2f}, Argmax match={np.mean(argmax_matches):.2%}")

    output = {"layer_skipping": skip_results, "early_exit": exit_results}
    path = PROJECT_ROOT / "experiments" / "results" / f"efficiency_{suffix}.json"
    save_json(output, path)
    return output


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Run full MI-Atlas workflow on a model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix for output files (e.g. 1.5b)")
    parser.add_argument("--skip-training", action="store_true", help="Skip LoRA training if adapter exists")
    args = parser.parse_args()

    set_seed(42)

    print("=" * 60)
    print(f"  MI-ATLAS FULL WORKFLOW")
    print(f"  Model: {args.model}")
    print(f"  Suffix: {args.suffix}")
    print("=" * 60)

    # Load data
    suite, pairs = load_test_data()
    test_prompts = select_test_prompts(suite, pairs)

    # Load model
    print("\n  Loading model...")
    bundle = load_model_hf(args.model, dtype=torch.bfloat16)
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    n_heads = bundle.architecture["n_heads"]
    d_model = bundle.architecture["d_model"]
    device = bundle.device
    model.eval()

    print(f"  Architecture: {n_layers}L, {n_heads}H, d={d_model}")
    print(f"  Test prompts: {len(test_prompts)}")

    # Phase 2: Component mapping
    print("\n--- PHASE 2: COMPONENT MAPPING ---")
    layer_results = run_layer_ablation(model, tokenizer, test_prompts, n_layers, args.suffix, device)
    mlp_results = run_mlp_ablation(model, tokenizer, test_prompts, n_layers, args.suffix, device)
    head_results = run_head_ablation(model, tokenizer, test_prompts, n_layers, n_heads, args.suffix, device)

    # Phase 3: Causal interventions
    print("\n--- PHASE 3: CAUSAL INTERVENTIONS ---")
    torch.cuda.empty_cache()
    steering_results = run_steering_sweep(model, tokenizer, n_layers, args.suffix, device)

    # Phase 4: Training perturbation
    print("\n--- PHASE 4: TRAINING PERTURBATION ---")
    torch.cuda.empty_cache()
    adapter_path = train_lora_adapter(model, tokenizer, args.suffix, device, n_layers)
    lora_results, trained_model = run_lora_comparison(model, tokenizer, adapter_path, test_prompts, n_layers, args.suffix, device)

    # Phase 5: Advanced interventions
    print("\n--- PHASE 5: ADVANCED INTERVENTIONS ---")
    torch.cuda.empty_cache()
    cross_model_results = run_cross_model_patching(trained_model, tokenizer, pairs, n_layers, args.suffix, device)
    torch.cuda.empty_cache()
    knockout_results = run_skill_knockout(trained_model, tokenizer, n_layers, args.suffix, device)
    torch.cuda.empty_cache()
    adapter_abl_results = run_adapter_ablation(trained_model, tokenizer, test_prompts, n_layers, args.suffix, device)

    # Phase 6: Efficiency
    print("\n--- PHASE 6: EFFICIENCY ---")
    torch.cuda.empty_cache()
    eff_results = run_efficiency_tests(model, tokenizer, test_prompts, n_layers, args.suffix, device)

    # Register
    register_experiment(
        type="full_atlas",
        model=args.model,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(PROJECT_ROOT / "experiments" / "results" / f"{name}_{args.suffix}.json") for name in [
            "layer_ablation", "mlp_ablation", "head_ablation", "steering_sweep",
            "lora_comparison", "cross_model_patching", "skill_knockout",
            "adapter_ablation", "efficiency"
        ]],
        status="success",
        summary=f"Full atlas run: {args.model}, {n_layers}L, {n_heads}H, d={d_model}",
        key_metrics={
            "n_layers": n_layers,
            "top_layer": layer_results["top_layers"][0],
            "top_layer_kl": layer_results["top_layer_kls"][0],
            "norm_effect_corr": adapter_abl_results["norm_effect_correlation"],
            "best_knockout_layer": max(
                [ld["layer"] for sk in knockout_results for ld in sk["layer_data"]
                 if ld["selectivity"] == max(ldd["selectivity"] for sk2 in knockout_results for ldd in sk2["layer_data"])],
                default=-1
            ),
        },
        next="Generate report and comparison",
    )

    print("\n" + "=" * 60)
    print(f"  FULL ATLAS COMPLETE: {args.model}")
    print(f"  Results saved with suffix: {args.suffix}")
    print("=" * 60)


if __name__ == "__main__":
    main()
