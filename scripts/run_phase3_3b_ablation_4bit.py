#!/usr/bin/env python3
"""Quick 3B multi-seed ablation using 4-bit NF4 to fit on 8GB VRAM."""

import torch
import gc
import numpy as np
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.task_suite import TaskSuite
from mi_atlas.utils import save_json, PROJECT_ROOT, set_seed, now_iso, git_commit_hash
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    seeds = [42, 137, 256]
    model_name = "Qwen/Qwen2.5-3B"
    results = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===", flush=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )
        n_layers = model.config.num_hidden_layers
        print(f"  Loaded 3B 4-bit: {n_layers} layers, VRAM={torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

        suite_path = PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json"
        suite = TaskSuite.load(str(suite_path))
        families = suite.families
        n_families = len(families)

        effect_matrix = np.zeros((n_layers, n_families))

        for fam_idx, family in enumerate(families):
            fam_suite = suite.filter_by_family(family).filter_by_split("test")
            examples = list(fam_suite)[:3]

            for ex in examples:
                prompt = ex.clean_prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    baseline_logits = model(**inputs).logits[:, -1, :]
                    baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)

                for layer_idx in range(n_layers):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)

                    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
                    with torch.no_grad():
                        steered_logits = model(**inputs).logits[:, -1, :]
                    hook.remove()

                    steered_probs = torch.softmax(steered_logits.float(), dim=-1)
                    kl = torch.nn.functional.kl_div(
                        steered_probs.log(), baseline_probs, reduction="sum"
                    ).item()
                    effect_matrix[layer_idx, fam_idx] += abs(kl) / len(examples)

            print(f"  {family}: done", flush=True)

        mean_per_layer = effect_matrix.mean(axis=1).tolist()
        hub_layer = int(np.argmax(mean_per_layer))

        result = {
            "run_id": f"P3_REPL_ablation_3B_seed{seed}",
            "seed": seed,
            "model": model_name,
            "n_layers": n_layers,
            "families": families,
            "effect_matrix": effect_matrix.tolist(),
            "mean_per_layer": mean_per_layer,
            "hub_layer": hub_layer,
        }
        results.append(result)

        out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_ablation_Qwen2.5-3B_seed{seed}.json"
        save_json(result, out_path)
        print(f"  Hub: L{hub_layer}, Saved: {out_path}", flush=True)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  VRAM freed: {torch.cuda.memory_allocated()/1024**2:.0f}MB", flush=True)

    hubs = [r["hub_layer"] for r in results]
    summary = {
        "experiment": "multi_seed_layer_ablation",
        "model": model_name,
        "seeds": seeds,
        "n_seeds": len(results),
        "hub_per_seed": hubs,
        "hub_mean": float(np.mean(hubs)),
        "hub_std": float(np.std(hubs)),
        "verdict": "robust" if np.std(hubs) <= 1 else "fragile",
    }
    summary_path = PROJECT_ROOT / "experiments" / "results" / "phase3_ablation_replication_Qwen2.5-3B.json"
    save_json(summary, summary_path)
    print(f"\nHub per seed: {hubs} (std={summary['hub_std']:.1f})")
    print(f"Verdict: {summary['verdict']}")


if __name__ == "__main__":
    main()
