#!/usr/bin/env python3
"""Phase 2 Block E: Cross-family replication.

Run reduced atlas on non-Qwen architectures to test universality.

Registry ID: P2-XFAM-001
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

def get_git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=5).stdout.strip()
    except:
        return "unknown"

def load_model(model_id, dtype=torch.bfloat16, quantize_4bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    
    kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True,
                  output_hidden_states=True)
    
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def eval_prompts(model, tokenizer, prompts, targets):
    results = []
    for prompt, target in zip(prompts, targets):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_lp = torch.log_softmax(logits.float(), dim=0)[target_ids[0]].item()
        top_tok = tokenizer.decode(logits.argmax().item())
        results.append({"target_logprob": target_lp, "exact_match": top_tok.strip() == target.strip()})
    return {
        "mean_target_logprob": sum(r["target_logprob"] for r in results) / len(results),
        "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
    }

@torch.no_grad()
def layer_ablation_sweep(model, tokenizer, prompts, targets, n_layers, top_k=None):
    """Zero ablation per layer. Returns dict of layer -> KL delta."""
    base = eval_prompts(model, tokenizer, prompts, targets)
    base_lp = base["mean_target_logprob"]
    
    layers_to_test = range(n_layers) if top_k is None else range(min(top_k, n_layers))
    results = {}
    for li in layers_to_test:
        def hook_fn(module, input, output, li=li):
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:] = 0
                return (h,) + output[1:]
            return torch.zeros_like(output)
        
        handle = model.model.layers[li].register_forward_hook(hook_fn)
        abl = eval_prompts(model, tokenizer, prompts, targets)
        handle.remove()
        results[li] = base_lp - abl["mean_target_logprob"]
        torch.cuda.empty_cache()
    
    return base, results

@torch.no_grad()
def steering_test(model, tokenizer, prompts, targets, layer_idx, strength=1.0):
    """Test steering at a specific layer."""
    # Compute steering vector from first half of prompts
    activations = []
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0][:, -1, :].detach().clone())
        else:
            activations.append(output[:, -1, :].detach().clone())
    
    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    for p in prompts[:2]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        model(**inputs)
    handle.remove()
    
    if len(activations) < 2:
        return {"steering_boost": 0}
    
    steer_vec = activations[1] - activations[0]  # simple direction
    
    # Apply steering
    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, -1, :] += strength * steer_vec.to(h.device)
            return (h,) + output[1:]
        out = output.clone()
        out[:, -1, :] += strength * steer_vec.to(out.device)
        return out
    
    base_result = eval_prompts(model, tokenizer, prompts, targets)
    
    handle = model.model.layers[layer_idx].register_forward_hook(steer_hook)
    steered = eval_prompts(model, tokenizer, prompts, targets)
    handle.remove()
    torch.cuda.empty_cache()
    
    return {
        "base_logprob": base_result["mean_target_logprob"],
        "steered_logprob": steered["mean_target_logprob"],
        "steering_boost": steered["mean_target_logprob"] - base_result["mean_target_logprob"],
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 2 cross-family replication")
    parser.add_argument("--model", default=None, help="Override model")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Models to test (in priority order)
    models_to_test = [
        ("google/gemma-2-2b", False),
        ("HuggingFaceTB/SmolLM2-1.7B", False),
    ]
    
    if args.model:
        models_to_test = [(args.model, False)]

    results_dir = Path(__file__).parent.parent / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)

    # Key task prompts for cross-family testing
    task_prompts = {
        "json": {
            "prompts": ['{"name": "Alice", "age":', '{"city": "London", "country":', '{"x": 1, "y":'],
            "targets": ["30", '"', "2"],
        },
        "factual": {
            "prompts": ["The capital of France is", "The capital of Italy is", "The capital of Japan is"],
            "targets": ["Paris", "Rome", "Tokyo"],
        },
        "copying": {
            "prompts": ["abc abc abc", "123 123 123", "hello hello hello"],
            "targets": ["abc", "123", "hello"],
        },
        "code": {
            "prompts": ["def add(a,b): return a+b\n# result =", "x = 5 + 3\n# x =", "s = 'hi'.upper()\n# s ="],
            "targets": ["8", "8", "HI"],
        },
    }

    for model_id, quantize in models_to_test:
        model_slug = model_id.split("/")[-1].replace(".", "").lower()
        output_file = results_dir / f"cross_family_{model_slug}.json"
        
        if output_file.exists() and not args.force:
            print(f"Results exist: {output_file}. Use --force to re-run.")
            continue
        
        try:
            model, tokenizer = load_model(model_id, quantize_4bit=quantize)
        except Exception as e:
            print(f"FAILED to load {model_id}: {e}")
            continue
        
        n_layers = model.config.num_hidden_layers
        all_results = {
            "experiment_id": "P2-XFAM-001",
            "model": model_id,
            "n_layers": n_layers,
            "seed": args.seed,
            "git_commit": get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tasks": {},
            "hub_layers": {},
        }
        
        for task_name, task_data in task_prompts.items():
            print(f"\n--- {model_slug} / {task_name} ---")
            base, ablation_results = layer_ablation_sweep(
                model, tokenizer, task_data["prompts"], task_data["targets"], n_layers)
            
            # Find top 3 hub layers
            sorted_layers = sorted(ablation_results.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_layers[:3]
            
            print(f"  Baseline: {base['mean_target_logprob']:.3f}")
            print(f"  Top 3 hubs: {[(f'L{k}', f'{v:.3f}') for k, v in top_3]}")
            
            # Steering test at top hub
            hub_layer = top_3[0][0]
            steer = steering_test(model, tokenizer,
                                 task_data["prompts"], task_data["targets"],
                                 hub_layer, strength=1.0)
            print(f"  Steering L{hub_layer}: boost={steer.get('steering_boost', 0):.3f}")
            
            all_results["tasks"][task_name] = {
                "baseline": base,
                "ablation": {str(k): round(v, 4) for k, v in ablation_results.items()},
                "top_3_hubs": [(int(k), round(v, 4)) for k, v in top_3],
                "steering_at_hub": steer,
            }
            all_results["hub_layers"][task_name] = int(top_3[0][0])
        
        # Summary
        hub_layers = list(all_results["hub_layers"].values())
        all_results["summary"] = {
            "hub_layers": hub_layers,
            "most_common_hub": max(set(hub_layers), key=hub_layers.count) if hub_layers else None,
            "hub_consistent": len(set(hub_layers)) == 1,
        }
        
        # Save
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Registry
        registry_path = Path(__file__).parent.parent / "experiments" / "registry.jsonl"
        reg = {
            "id": f"P2-XFAM-{model_slug}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "cross_family",
            "model": model_id,
            "n_layers": n_layers,
            "seed": args.seed,
            "git_commit": get_git_hash(),
            "outputs": [str(output_file)],
            "status": "success",
            "summary": f"Cross-family atlas: {model_slug}, {n_layers}L, hub={all_results['summary']['most_common_hub']}",
        }
        with open(registry_path, "a") as f:
            f.write(json.dumps(reg) + "\n")
        
        print(f"\nSaved: {output_file}")
        
        # Free VRAM
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
