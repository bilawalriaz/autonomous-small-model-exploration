#!/usr/bin/env python3
"""Phase 3: LoRA rank sweep with actual task accuracy measurement.

Trains LoRA at ranks r=2,4,8,16 on JSON family for 100 steps each.
Evaluates exact_match and constraint_satisfaction on test set.
Measures adapter norm per layer, computes causal effect via ablation.
Compares ranks on: accuracy, param efficiency (accuracy/params), loss convergence.

Usage:
    python -u scripts/run_phase3_rank_sweep_with_accuracy.py --model Qwen/Qwen2.5-0.5B
    python -u scripts/run_phase3_rank_sweep_with_accuracy.py --model Qwen/Qwen2.5-0.5B --force --seed 137
"""

import argparse
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

from mi_atlas.model_loader import load_model_hf
from mi_atlas.backend import create_backend
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset
from mi_atlas.experiment_registry import register_experiment, load_registry
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.metrics import kl_divergence, exact_match_score, valid_json_score
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT, now_iso


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
        if (rec.get("type") == "phase3_rank_sweep"
                and model_slug in rec.get("model", "")
                and rec.get("status") == "success"):
            return True
    return False


def train_lora_at_rank(model, tokenizer, dataset, rank, output_dir):
    """Train LoRA at a specific rank, return training metrics."""
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())

    args = SFTConfig(
        output_dir=output_dir,
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

    # Save adapter for later evaluation
    adapter_path = Path(output_dir) / "adapter"
    peft_model.save_pretrained(str(adapter_path))

    # Collect loss curve
    loss_history = []
    if hasattr(trainer.state, "log_history"):
        loss_history = [entry.get("loss") for entry in trainer.state.log_history
                       if "loss" in entry]

    del peft_model
    torch.cuda.empty_cache()

    return {
        "train_loss": result.training_loss,
        "loss_history": loss_history,
        "n_trainable": n_trainable,
        "n_total": n_total,
        "adapter_norms": adapter_norms,
        "adapter_path": str(adapter_path),
        "rank": rank,
    }


def evaluate_accuracy(model, tokenizer, eval_examples, max_new_tokens=100):
    """Evaluate model on task examples, return accuracy metrics."""
    model.eval()
    results = []
    for example in eval_examples:
        try:
            inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                             truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            target = example.target

            em = exact_match_score(generated, target)
            vj = valid_json_score(generated) if example.family == "json_schema" else None

            # Constraint satisfaction: check if required keys present for JSON
            constraint_score = None
            if example.family == "json_schema":
                required = example.metadata.get("required_keys", [])
                if required:
                    try:
                        import json
                        obj = json.loads(generated)
                        satisfied = sum(1 for k in required if k in obj)
                        constraint_score = satisfied / len(required)
                    except Exception:
                        constraint_score = 0.0

            results.append({
                "example_id": example.id,
                "exact_match": em,
                "valid_json": vj,
                "constraint_satisfaction": constraint_score,
                "generated": generated[:200],
                "target": target,
            })
        except Exception as e:
            results.append({
                "example_id": example.id,
                "error": str(e),
                "exact_match": 0.0,
            })

    # Aggregate
    mean_em = np.mean([r["exact_match"] for r in results]) if results else 0.0
    mean_vj = np.mean([r["valid_json"] for r in results
                       if r.get("valid_json") is not None]) if results else 0.0
    cs_scores = [r["constraint_satisfaction"] for r in results
                 if r.get("constraint_satisfaction") is not None]
    mean_cs = np.mean(cs_scores) if cs_scores else 0.0

    return {
        "mean_exact_match": float(mean_em),
        "mean_valid_json": float(mean_vj),
        "mean_constraint_satisfaction": float(mean_cs),
        "n_eval": len(results),
        "per_example": results[:5],
    }


def run_ablation_causal_effect(backend, suite, n_layers, ablation_type="zero"):
    """Run ablation and return per-family KL effect matrix."""
    try:
        ablation_result = run_layer_ablation_suite(
            backend, suite, ablation_type=ablation_type, split="test"
        )
        return {
            "effect_matrix": ablation_result["effect_matrix"],
            "families": ablation_result["families"],
            "ablation_type": ablation_type,
        }
    except Exception as e:
        log(f"  Ablation failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: LoRA rank sweep with accuracy")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true",
                       help="Re-run even if already completed")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    ranks = [2, 4, 8, 16]

    log("=" * 60)
    log(f"Phase 3: LoRA Rank Sweep with Accuracy")
    log(f"Model: {args.model}")
    log(f"Ranks: {ranks}")
    log(f"Seed: {args.seed}")
    log("=" * 60)

    # Check resumability
    if check_already_run(model_slug, args.force):
        log("Already completed. Use --force to re-run.")
        return

    set_seed(args.seed)
    start_time = time.time()

    # Load task suite
    suite = build_default_suite(seed=args.seed)
    json_suite = suite.filter_by_family("json_schema")
    json_test = json_suite.filter_by_split("test")
    json_train = json_suite.filter_by_split("train")
    if not list(json_train):
        json_train = json_suite  # fallback
    if not list(json_test):
        json_test = json_suite

    ds = prepare_sft_dataset(json_train)
    families = suite.families

    # Load base model
    log("Loading base model...")
    bundle = load_model_hf(args.model)
    base_model = bundle.model
    tokenizer = bundle.tokenizer
    n_layers = bundle.architecture["n_layers"]
    log(f"  Loaded: {n_layers} layers, device={bundle.device}")

    # Baseline ablation (base model, no LoRA)
    log("\nRunning baseline ablation on base model...")
    base_backend = create_backend(bundle)
    baseline_ablation = run_ablation_causal_effect(base_backend, suite, n_layers)
    del base_backend

    # Baseline accuracy (base model)
    log("Evaluating baseline accuracy...")
    baseline_eval = evaluate_accuracy(base_model, tokenizer, list(json_test)[:10])
    log(f"  Baseline exact_match: {baseline_eval['mean_exact_match']:.3f}")
    log(f"  Baseline valid_json: {baseline_eval['mean_valid_json']:.3f}")

    all_results = {
        "experiment": "phase3_rank_sweep_with_accuracy",
        "model": args.model,
        "model_slug": model_slug,
        "seed": args.seed,
        "ranks": ranks,
        "n_layers": n_layers,
        "timestamp": now_iso(),
        "baseline_eval": baseline_eval,
        "baseline_ablation": baseline_ablation,
        "rank_results": {},
    }

    # Sweep each rank
    for rank in ranks:
        log(f"\n{'='*50}")
        log(f"Training LoRA r={rank}...")
        log(f"{'='*50}")

        output_dir = str(PROJECT_ROOT / "experiments" / "adapters"
                        / f"phase3_rank_sweep_r{rank}_{model_slug}")

        try:
            train_result = train_lora_at_rank(
                base_model, tokenizer, ds, rank, output_dir
            )
            log(f"  Train loss: {train_result['train_loss']:.4f}")
            log(f"  Trainable params: {train_result['n_trainable']:,}")
            log(f"  Param ratio: {train_result['n_trainable']/train_result['n_total']:.6f}")
            log(f"  Adapter norms: {dict(sorted(train_result['adapter_norms'].items()))}")

            # Evaluate accuracy with the LoRA adapter
            log(f"  Evaluating accuracy for r={rank}...")
            try:
                lora_model = PeftModel.from_pretrained(
                    base_model, train_result["adapter_path"]
                )
                lora_model.eval()
                eval_result = evaluate_accuracy(lora_model, tokenizer, list(json_test)[:10])
                log(f"  Exact match: {eval_result['mean_exact_match']:.3f}")
                log(f"  Valid JSON: {eval_result['mean_valid_json']:.3f}")
                log(f"  Constraint satisfaction: {eval_result['mean_constraint_satisfaction']:.3f}")
                del lora_model
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"  Accuracy eval failed: {e}")
                eval_result = {"error": str(e), "mean_exact_match": 0.0}

            # Run ablation on LoRA model for causal effect
            log(f"  Running ablation for r={rank}...")
            try:
                lora_model = PeftModel.from_pretrained(
                    base_model, train_result["adapter_path"]
                )
                lora_model.eval()
                # Create a temporary bundle for the ablation
                from mi_atlas.model_loader import ModelBundle
                lora_bundle = ModelBundle(
                    model=lora_model,
                    tokenizer=tokenizer,
                    model_name=args.model,
                    backend="hf_native",
                    device=bundle.device,
                    dtype=bundle.dtype,
                    architecture=bundle.architecture,
                )
                lora_backend = create_backend(lora_bundle)
                lora_ablation = run_ablation_causal_effect(lora_backend, suite, n_layers)
                del lora_backend, lora_model
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"  Ablation failed: {e}")
                lora_ablation = {"error": str(e)}

            # Compute param efficiency
            accuracy = eval_result.get("mean_exact_match", 0.0)
            params = train_result["n_trainable"]
            param_efficiency = accuracy / max(params, 1) * 1e6  # accuracy per 1M params

            rank_result = {
                "rank": rank,
                "train_loss": train_result["train_loss"],
                "loss_history": train_result["loss_history"],
                "n_trainable": train_result["n_trainable"],
                "n_total": train_result["n_total"],
                "param_ratio": train_result["n_trainable"] / train_result["n_total"],
                "adapter_norms": train_result["adapter_norms"],
                "adapter_path": train_result["adapter_path"],
                "accuracy_eval": eval_result,
                "ablation": lora_ablation,
                "param_efficiency_per_1M": float(param_efficiency),
            }
            all_results["rank_results"][f"r{rank}"] = rank_result

        except Exception as e:
            log(f"  FAILED for r={rank}: {e}")
            import traceback
            traceback.print_exc()
            all_results["rank_results"][f"r{rank}"] = {"error": str(e)}

    # Summary comparison
    log(f"\n{'='*60}")
    log("RANK SWEEP SUMMARY")
    log(f"{'='*60}")
    log(f"{'Rank':>6} {'Loss':>8} {'ExactM':>8} {'ValJSON':>8} {'ConstSat':>8} "
        f"{'Params':>12} {'Efficiency':>10}")
    log("-" * 70)

    best_rank = None
    best_accuracy = 0.0
    best_efficiency = 0.0

    for rank in ranks:
        key = f"r{rank}"
        if key not in all_results["rank_results"] or "error" in all_results["rank_results"][key]:
            log(f"{rank:>6} {'FAILED':>8}")
            continue
        r = all_results["rank_results"][key]
        loss = r.get("train_loss", float("inf"))
        em = r.get("accuracy_eval", {}).get("mean_exact_match", 0.0)
        vj = r.get("accuracy_eval", {}).get("mean_valid_json", 0.0)
        cs = r.get("accuracy_eval", {}).get("mean_constraint_satisfaction", 0.0)
        params = r.get("n_trainable", 0)
        eff = r.get("param_efficiency_per_1M", 0.0)

        log(f"{rank:>6} {loss:>8.4f} {em:>8.3f} {vj:>8.3f} {cs:>8.3f} "
            f"{params:>12,} {eff:>10.2f}")

        if em > best_accuracy:
            best_accuracy = em
            best_rank = rank
        if eff > best_efficiency:
            best_efficiency = eff

    all_results["summary"] = {
        "best_rank_accuracy": best_rank,
        "best_accuracy": float(best_accuracy),
        "best_rank_efficiency": best_rank,
        "best_efficiency": float(best_efficiency),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_rank_sweep_{model_slug}.json"
    save_json(all_results, output_path)
    log(f"\nResults saved to {output_path}")

    # Register experiment
    register_experiment(
        type="phase3_rank_sweep",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(output_path)],
        status="success",
        summary=f"Phase 3 rank sweep: ranks={ranks}, best_rank_accuracy={best_rank} "
                f"(exact_match={best_accuracy:.3f})",
        key_metrics={
            "best_rank_accuracy": best_rank,
            "best_accuracy": best_accuracy,
            "best_efficiency": best_efficiency,
        },
        next="Phase 3 module sweep, investigate rank-accuracy scaling",
    )

    elapsed = time.time() - start_time
    log(f"\nRank sweep complete in {elapsed:.0f}s")
    log(f"Best rank (accuracy): r={best_rank} with exact_match={best_accuracy:.3f}")


if __name__ == "__main__":
    main()
