#!/usr/bin/env python3
"""Phase 3: Checkpoint lock-in at 1.5B.

Trains LoRA on JSON family at 1.5B, saving checkpoints at steps
5, 10, 25, 50, 100, and final. At each checkpoint, evaluates:
  - Task accuracy (exact match, constraint satisfaction)
  - Adapter norm distribution across layers
  - Layer ablation effect map (which layers matter at each stage)

Question: does the 10% lock-in rule from 0.5B generalize to 1.5B?
Can we detect by step 10-25 whether the finetune is learning
the right internal route?

Usage:
    python scripts/run_phase3_checkpoint_lockin.py --model Qwen/Qwen2.5-1.5B
    python scripts/run_phase3_checkpoint_lockin.py --model Qwen/Qwen2.5-0.5B --steps 100 --force
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType

from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, now_iso, git_commit_hash, PROJECT_ROOT

# Atlas-recommended layers for LoRA per model
ATLAS_LAYERS = {
    "Qwen/Qwen2.5-0.5B": [6, 12, 13],      # JSON concentration layers
    "Qwen/Qwen2.5-1.5B": [26, 27],          # Hub layers
}

CHECKPOINT_STEPS = [5, 10, 25, 50, 100]


class SimpleDataset(torch.utils.data.Dataset):
    """Simple tokenised dataset for SFT-style training."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length,
                                   padding="max_length", return_tensors="pt")

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def get_lora_target_modules(model_name):
    """Get LoRA target modules based on atlas findings."""
    critical_layers = ATLAS_LAYERS.get(model_name, [2, 7, 9])
    target_modules = []
    for l in critical_layers:
        target_modules.append(f"model.layers.{l}.self_attn.o_proj")
        target_modules.append(f"model.layers.{l}.mlp.up_proj")
        target_modules.append(f"model.layers.{l}.mlp.down_proj")
    return target_modules, critical_layers


def get_adapter_norms(model):
    """Compute per-layer adapter weight norms."""
    layer_norms = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            # Parse layer index
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            if layer_idx is not None:
                layer_key = f"layer_{layer_idx}"
                if layer_key not in layer_norms:
                    layer_norms[layer_key] = []
                layer_norms[layer_key].append(float(param.norm().item()))

    # Aggregate: mean norm per layer
    return {k: float(np.mean(v)) for k, v in layer_norms.items()}


def evaluate_model(model, tokenizer, eval_prompts):
    """Evaluate model on JSON task metrics."""
    model.eval()
    results = []

    for prompt_info in eval_prompts:
        prompt = prompt_info.clean_prompt
        target = prompt_info.target

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Exact match
        exact = 1.0 if target and generated.strip().startswith(target.strip()) else 0.0

        # Valid JSON check
        valid_json = 0.0
        try:
            obj = json.loads(generated.strip())
            if isinstance(obj, dict):
                valid_json = 1.0
        except (json.JSONDecodeError, TypeError):
            pass

        # Key coverage
        keys_found = 0.0
        required_keys = prompt_info.metadata.get("required_keys", [])
        if required_keys:
            try:
                obj = json.loads(generated.strip())
                if isinstance(obj, dict):
                    keys_found = sum(1 for k in required_keys if k in obj) / len(required_keys)
            except (json.JSONDecodeError, TypeError):
                pass

        results.append({
            "prompt": prompt[:100],
            "target": target,
            "generated": generated[:200],
            "exact_match": exact,
            "valid_json": valid_json,
            "key_coverage": keys_found,
        })

    return {
        "n_eval": len(results),
        "mean_exact_match": round(float(np.mean([r["exact_match"] for r in results])), 4) if results else 0,
        "mean_valid_json": round(float(np.mean([r["valid_json"] for r in results])), 4) if results else 0,
        "mean_key_coverage": round(float(np.mean([r["key_coverage"] for r in results])), 4) if results else 0,
        "per_prompt": results[:5],  # Keep first 5 for inspection
    }


def compute_layer_ablation_effect(model, tokenizer, eval_prompts, n_layers):
    """Quick ablation effect map: zero each layer, measure KL at last token."""
    model.eval()
    n_test = min(3, len(eval_prompts))
    effect_per_layer = {}

    for layer_idx in range(n_layers):
        kl_values = []
        for i in range(n_test):
            prompt = eval_prompts[i].clean_prompt
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(model.device)

            with torch.no_grad():
                orig_logits = model(ids).logits

            # Zero-ablate layer
            layer = model.model.layers[layer_idx]

            def zero_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

            handle = layer.register_forward_hook(zero_hook)
            with torch.no_grad():
                abl_logits = model(ids).logits
            handle.remove()

            # KL at last position
            orig_probs = torch.softmax(orig_logits[0, -1].float(), dim=-1)
            abl_log_probs = torch.log_softmax(abl_logits[0, -1].float(), dim=-1)
            kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
            kl_values.append(kl)

        effect_per_layer[f"layer_{layer_idx}"] = round(float(np.mean(kl_values)), 6)

    return effect_per_layer


def main():
    parser = argparse.ArgumentParser(description="Phase 3 checkpoint lock-in")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    checkpoint_steps = [s for s in CHECKPOINT_STEPS if s <= args.steps]
    if args.steps not in checkpoint_steps:
        checkpoint_steps.append(args.steps)
    checkpoint_steps = sorted(set(checkpoint_steps))

    print(f"Phase 3: Checkpoint lock-in")
    print(f"Model: {args.model}")
    print(f"Checkpoint steps: {checkpoint_steps}")
    print(f"LoRA rank: {args.rank}")
    print(f"Seed: {args.seed}")

    set_seed(args.seed)

    # Load task suite
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    family_suite = suite.filter_by_family("json_schema")
    train_examples = list(family_suite.filter_by_split("train"))
    eval_examples = list(family_suite.filter_by_split("test"))
    if not train_examples:
        train_examples = list(family_suite)[:int(len(family_suite)*0.8)]
        eval_examples = list(family_suite)[int(len(family_suite)*0.8):]
    train_prompts = [{"prompt": e.clean_prompt, "target": e.target or ""} for e in train_examples]
    eval_prompts = [{"prompt": e.clean_prompt, "target": e.target or ""} for e in eval_examples]

    if not train_prompts:
        print("No JSON training data found")
        return

    train_texts = [p.clean_prompt + p.target for p in train_prompts]
    print(f"  Train: {len(train_texts)} examples, Eval: {len(eval_prompts)} examples")

    # Load model
    print("\nLoading model...")
    bundle = load_model_hf(args.model)
    model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    model.eval()

    # Configure LoRA
    target_modules, critical_layers = get_lora_target_modules(args.model)
    print(f"  Target modules ({len(target_modules)}): layers={critical_layers}")
    print(f"  Total layers: {n_layers}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Baseline eval
    print("\n  Evaluating baseline (step 0)...")
    baseline_eval = evaluate_model(model, tokenizer, eval_prompts)
    print(f"    Exact match: {baseline_eval['mean_exact_match']:.3f}, "
          f"Valid JSON: {baseline_eval['mean_valid_json']:.3f}")

    # Prepare training
    dataset = SimpleDataset(train_texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    total_steps = args.steps
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(10, total_steps // 5),
                                                 num_training_steps=total_steps)

    # Training with checkpoint capture
    checkpoint_data = {}
    losses = []
    model.train()
    start_time = time.time()

    print(f"\n  Training for {total_steps} steps...")
    for step in range(total_steps):
        batch_idx = step % len(dataloader)
        batch = {k: v.to(model.device) for k, v in dataloader[batch_idx].items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        # Save checkpoint data at specified steps
        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  === Checkpoint at step {ckpt_step} (loss={losses[-1]:.4f}) ===")

            model.eval()

            # Evaluate
            ckpt_eval = evaluate_model(model, tokenizer, eval_prompts)
            print(f"    Exact match: {ckpt_eval['mean_exact_match']:.3f}, "
                  f"Valid JSON: {ckpt_eval['mean_valid_json']:.3f}, "
                  f"Key coverage: {ckpt_eval['mean_key_coverage']:.3f}")

            # Adapter norms
            adapter_norms = get_adapter_norms(model)

            # Ablation effect map (only at key checkpoints to save time)
            ablation_map = None
            if ckpt_step in [5, 25, args.steps]:
                print(f"    Computing ablation effect map...")
                ablation_map = compute_layer_ablation_effect(model, tokenizer, eval_prompts, n_layers)

            checkpoint_data[str(ckpt_step)] = {
                "step": ckpt_step,
                "loss": round(losses[-1], 6),
                "eval": ckpt_eval,
                "adapter_norms": adapter_norms,
                "ablation_map": ablation_map,
            }

            model.train()

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")

    # Detect lock-in: does the model commit early?
    # Compare adapter norm distribution and eval metrics across checkpoints
    lockin_analysis = {}
    steps_list = sorted(checkpoint_data.keys(), key=int)
    final_metrics = checkpoint_data[steps_list[-1]]["eval"] if steps_list else {}

    for s_key in steps_list:
        ckpt = checkpoint_data[s_key]
        eval_data = ckpt["eval"]
        # How close is this checkpoint to the final checkpoint?
        em_similarity = 1.0 - abs(eval_data["mean_exact_match"] - final_metrics.get("mean_exact_match", 0))
        json_similarity = 1.0 - abs(eval_data["mean_valid_json"] - final_metrics.get("mean_valid_json", 0))

        lockin_analysis[s_key] = {
            "exact_match": eval_data["mean_exact_match"],
            "valid_json": eval_data["mean_valid_json"],
            "em_similarity_to_final": round(em_similarity, 4),
            "json_similarity_to_final": round(json_similarity, 4),
            "loss": ckpt["loss"],
        }

    # Determine lock-in step (first step where similarity > 0.9)
    lockin_step = None
    for s_key in steps_list:
        la = lockin_analysis[s_key]
        if la["em_similarity_to_final"] > 0.9 and la["json_similarity_to_final"] > 0.9:
            lockin_step = int(s_key)
            break

    lockin_fraction = lockin_step / args.steps if lockin_step else None
    print(f"\n  LOCK-IN ANALYSIS:")
    print(f"  {'Step':>6s}  {'Loss':>8s}  {'ExactMatch':>10s}  {'ValidJSON':>10s}  {'EM Sim':>8s}  {'JSON Sim':>8s}")
    print(f"  {'─'*60}")
    for s_key in steps_list:
        la = lockin_analysis[s_key]
        marker = " <-- LOCK-IN" if int(s_key) == lockin_step else ""
        print(f"  {s_key:>6s}  {la['loss']:>8.4f}  {la['exact_match']:>10.3f}  "
              f"{la['valid_json']:>10.3f}  {la['em_similarity_to_final']:>8.3f}  "
              f"{la['json_similarity_to_final']:>8.3f}{marker}")

    if lockin_step:
        print(f"\n  Lock-in detected at step {lockin_step} ({lockin_fraction:.0%} of training)")
    else:
        print(f"\n  No clear lock-in detected (model did not converge to >90% similarity)")

    # Save
    summary = {
        "experiment": "phase3_checkpoint_lockin",
        "model": args.model,
        "seed": args.seed,
        "rank": args.rank,
        "total_steps": args.steps,
        "checkpoint_steps": checkpoint_steps,
        "critical_layers": critical_layers,
        "target_modules": target_modules,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "baseline_eval": baseline_eval,
        "checkpoints": checkpoint_data,
        "lockin_analysis": lockin_analysis,
        "lockin_step": lockin_step,
        "lockin_fraction": lockin_fraction,
        "training_time_seconds": round(elapsed, 1),
        "loss_curve": losses,
        "timestamp": now_iso(),
        "git_commit": git_commit_hash(),
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_checkpoint_lockin_{model_slug}.json"
    save_json(summary, out_path)
    print(f"\n  Results: {out_path}")

    # Register
    register_experiment(
        type="training",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(out_path)],
        status="success",
        summary=f"Checkpoint lock-in ({model_slug}): "
                f"lockin_step={lockin_step}, lockin_fraction={lockin_fraction}, "
                f"final_em={final_metrics.get('mean_exact_match', 0):.3f}",
        next="Test lock-in rule across more models and task families",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
