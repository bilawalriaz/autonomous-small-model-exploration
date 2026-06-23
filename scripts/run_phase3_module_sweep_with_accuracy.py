#!/usr/bin/env python3
"""Phase 3: LoRA module sweep with actual task accuracy measurement.

Trains LoRA targeting different module subsets (o_proj, v_proj, q_proj, k_proj,
up_proj, down_proj, gate_proj, all-linear) for 100 steps on JSON family.
Evaluates exact_match and constraint_satisfaction on test set.
Compares per-module efficiency.

Usage:
    python -u scripts/run_phase3_module_sweep_with_accuracy.py --model Qwen/Qwen2.5-0.5B
    python -u scripts/run_phase3_module_sweep_with_accuracy.py --model Qwen/Qwen2.5-0.5B --force --seed 137
"""

import argparse
import copy
import json
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from mi_atlas.model_loader import load_model_hf
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset
from mi_atlas.experiment_registry import register_experiment, load_registry
from mi_atlas.metrics import exact_match_score, valid_json_score
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, now_iso


MODULE_CONFIGS = {
    "o_proj": ["o_proj"],
    "v_proj": ["v_proj"],
    "q_proj": ["q_proj"],
    "k_proj": ["k_proj"],
    "up_proj": ["up_proj"],
    "down_proj": ["down_proj"],
    "gate_proj": ["gate_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                    "up_proj", "down_proj", "gate_proj"],
}


def log(msg):
    """Print with timestamp and flush."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_already_run(model_slug, force):
    """Check if this experiment already completed."""
    if force:
        return False
    registry = load_registry()
    for rec in registry:
        if (rec.get("type") == "phase3_module_sweep"
                and model_slug in rec.get("model", "")
                and rec.get("status") == "success"):
            return True
    return False


def train_and_evaluate(base_model, tokenizer, dataset, modules, config_name,
                       eval_examples, rank=8):
    """Train LoRA with specific modules, evaluate accuracy."""
    # Deep copy to avoid PEFT modifying the base model in-place
    model = copy.deepcopy(base_model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=modules,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())

    args = SFTConfig(
        output_dir=f"/tmp/lora_module_{config_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=500,
        report_to="none",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Evaluate accuracy
    peft_model.eval()
    eval_results = []
    for example in eval_examples:
        try:
            inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(peft_model.device)
            with torch.no_grad():
                output_ids = peft_model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            target = example.target

            em = exact_match_score(generated, target)
            vj = valid_json_score(generated) if example.family == "json_schema" else None

            constraint_score = None
            if example.family == "json_schema":
                required = example.metadata.get("required_keys", [])
                if required:
                    try:
                        obj = json.loads(generated)
                        satisfied = sum(1 for k in required if k in obj)
                        constraint_score = satisfied / len(required)
                    except Exception:
                        constraint_score = 0.0

            eval_results.append({
                "example_id": example.id,
                "exact_match": em,
                "valid_json": vj,
                "constraint_satisfaction": constraint_score,
                "generated": generated[:200],
                "target": target,
            })
        except Exception as e:
            eval_results.append({
                "example_id": example.id,
                "error": str(e),
                "exact_match": 0.0,
            })

    mean_em = np.mean([r["exact_match"] for r in eval_results]) if eval_results else 0.0
    mean_vj = np.mean([r["valid_json"] for r in eval_results
                       if r.get("valid_json") is not None]) if eval_results else 0.0
    cs_scores = [r["constraint_satisfaction"] for r in eval_results
                 if r.get("constraint_satisfaction") is not None]
    mean_cs = np.mean(cs_scores) if cs_scores else 0.0

    # Collect adapter norms per layer
    adapter_norms = {}
    for name, param in peft_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    layer_key = f"layer_{layer_idx:02d}"
                    norm = param.data.float().norm().item()
                    adapter_norms[layer_key] = adapter_norms.get(layer_key, 0.0) + norm
                    break

    # Cleanup
    del peft_model, model
    torch.cuda.empty_cache()

    return {
        "config": config_name,
        "modules": modules,
        "train_loss": result.training_loss,
        "n_trainable": n_trainable,
        "n_total": n_total,
        "param_ratio": n_trainable / n_total,
        "adapter_norms": adapter_norms,
        "accuracy_eval": {
            "mean_exact_match": float(mean_em),
            "mean_valid_json": float(mean_vj),
            "mean_constraint_satisfaction": float(mean_cs),
            "n_eval": len(eval_results),
            "per_example": eval_results[:5],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: LoRA module sweep with accuracy"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true",
                       help="Re-run even if already completed")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]

    log("=" * 60)
    log(f"Phase 3: LoRA Module Sweep with Accuracy")
    log(f"Model: {args.model}")
    log(f"Modules: {list(MODULE_CONFIGS.keys())}")
    log(f"Seed: {args.seed}")
    log("=" * 60)

    if check_already_run(model_slug, args.force):
        log("Already completed. Use --force to re-run.")
        return

    set_seed(args.seed)
    start_time = time.time()

    # Load task suite
    suite = build_default_suite(seed=args.seed)
    json_suite = suite.filter_by_family("json_schema")
    json_train = json_suite.filter_by_split("train")
    json_test = json_suite.filter_by_split("test")
    if not list(json_train):
        json_train = json_suite
    if not list(json_test):
        json_test = json_suite

    ds = prepare_sft_dataset(json_train)
    eval_examples = list(json_test)[:10]

    # Load base model
    log("Loading base model...")
    bundle = load_model_hf(args.model)
    base_model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    log(f"  Loaded: {n_layers} layers, device={bundle.device}")

    # Baseline accuracy
    log("Evaluating baseline accuracy...")
    base_model.eval()
    baseline_results = []
    for ex in eval_examples:
        try:
            inputs = tokenizer(ex.clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(base_model.device)
            with torch.no_grad():
                out_ids = base_model.generate(
                    input_ids, max_new_tokens=100, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out_ids[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
            em = exact_match_score(gen, ex.target)
            vj = valid_json_score(gen) if ex.family == "json_schema" else None
            baseline_results.append({"exact_match": em, "valid_json": vj})
        except Exception as e:
            baseline_results.append({"exact_match": 0.0, "error": str(e)})

    baseline_em = np.mean([r["exact_match"] for r in baseline_results])
    baseline_vj = np.mean([r["valid_json"] for r in baseline_results
                          if r.get("valid_json") is not None])
    log(f"  Baseline exact_match: {baseline_em:.3f}")
    log(f"  Baseline valid_json: {baseline_vj:.3f}")

    all_results = {
        "experiment": "phase3_module_sweep_with_accuracy",
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "n_layers": n_layers,
        "timestamp": now_iso(),
        "baseline_exact_match": float(baseline_em),
        "baseline_valid_json": float(baseline_vj),
        "module_results": {},
    }

    # Sweep each module config
    for config_name, modules in MODULE_CONFIGS.items():
        log(f"\n{'='*50}")
        log(f"Training: {config_name} (modules={modules})")
        log(f"{'='*50}")

        try:
            result = train_and_evaluate(
                base_model, tokenizer, ds, modules, config_name,
                eval_examples, rank=8
            )

            em = result["accuracy_eval"]["mean_exact_match"]
            vj = result["accuracy_eval"]["mean_valid_json"]
            cs = result["accuracy_eval"]["mean_constraint_satisfaction"]
            eff = em / max(result["param_ratio"], 1e-10)

            log(f"  Loss: {result['train_loss']:.4f}")
            log(f"  Params: {result['n_trainable']:,} ({result['param_ratio']:.6f})")
            log(f"  Exact match: {em:.3f} (delta: {em - baseline_em:+.3f})")
            log(f"  Valid JSON: {vj:.3f}")
            log(f"  Constraint sat: {cs:.3f}")
            log(f"  Param efficiency: {eff:.6f}")

            all_results["module_results"][config_name] = result
            all_results["module_results"][config_name]["param_efficiency"] = float(eff)
            all_results["module_results"][config_name]["accuracy_delta"] = float(em - baseline_em)

        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results["module_results"][config_name] = {"error": str(e)}

    # Summary comparison
    log(f"\n{'='*60}")
    log("MODULE SWEEP SUMMARY")
    log(f"{'='*60}")
    log(f"{'Module':>15} {'Loss':>8} {'ExactM':>8} {'ValJSON':>8} "
        f"{'ConstSat':>8} {'Params':>12} {'Efficiency':>10}")
    log("-" * 80)

    best_module = None
    best_accuracy = 0.0
    best_efficiency = 0.0
    best_eff_module = None

    for config_name in MODULE_CONFIGS:
        if config_name not in all_results["module_results"]:
            log(f"{config_name:>15} {'FAILED':>8}")
            continue
        r = all_results["module_results"][config_name]
        if "error" in r:
            log(f"{config_name:>15} {'ERROR':>8}")
            continue
        loss = r.get("train_loss", float("inf"))
        em = r.get("accuracy_eval", {}).get("mean_exact_match", 0.0)
        vj = r.get("accuracy_eval", {}).get("mean_valid_json", 0.0)
        cs = r.get("accuracy_eval", {}).get("mean_constraint_satisfaction", 0.0)
        params = r.get("n_trainable", 0)
        eff = r.get("param_efficiency", 0.0)

        log(f"{config_name:>15} {loss:>8.4f} {em:>8.3f} {vj:>8.3f} "
            f"{cs:>8.3f} {params:>12,} {eff:>10.6f}")

        if em > best_accuracy:
            best_accuracy = em
            best_module = config_name
        if eff > best_efficiency:
            best_efficiency = eff
            best_eff_module = config_name

    all_results["summary"] = {
        "best_module_accuracy": best_module,
        "best_accuracy": float(best_accuracy),
        "best_module_efficiency": best_eff_module,
        "best_efficiency": float(best_efficiency),
        "baseline_accuracy": float(baseline_em),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_module_sweep_{model_slug}.json"
    save_json(all_results, output_path)
    log(f"\nResults saved to {output_path}")

    # Register experiment
    register_experiment(
        type="phase3_module_sweep",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(output_path)],
        status="success",
        summary=f"Phase 3 module sweep: best_accuracy={best_module} "
                f"(exact_match={best_accuracy:.3f}), "
                f"best_efficiency={best_eff_module} ({best_efficiency:.6f})",
        key_metrics={
            "best_module_accuracy": best_module,
            "best_accuracy": best_accuracy,
            "best_module_efficiency": best_eff_module,
            "best_efficiency": best_efficiency,
        },
        next="Compare with rank sweep, investigate module-specific circuits",
    )

    elapsed = time.time() - start_time
    log(f"\nModule sweep complete in {elapsed:.0f}s")
    log(f"Best module (accuracy): {best_module} with exact_match={best_accuracy:.3f}")
    log(f"Best module (efficiency): {best_eff_module} with eff={best_efficiency:.6f}")


if __name__ == "__main__":
    main()
