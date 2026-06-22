#!/usr/bin/env python3
"""Phase 2 Block A: Parity verification for 1.5B.

Fill missing experiments so 0.5B and 1.5B comparisons are symmetrical.

Registry ID: P2-PARITY-001
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=5).stdout.strip()
    except:
        return "unknown"

def load_model(model_id, dtype=torch.bfloat16):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    return model, tokenizer

def compute_kl(base_logits, test_logits, dim=-1):
    base_probs = torch.softmax(base_logits.float(), dim=dim)
    test_log_probs = torch.log_softmax(test_logits.float(), dim=dim)
    base_log_probs = torch.log(base_probs + 1e-10)
    kl = (base_probs * (base_log_probs - test_log_probs)).sum(dim=dim)
    return kl.mean().item()

@torch.no_grad()
def eval_task(model, tokenizer, prompts, targets):
    """Evaluate model on task prompts, return mean target logprob and KL."""
    results = []
    for prompt, target in zip(prompts, targets):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # next token logits
        
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if len(target_ids) == 1:
            target_logprob = torch.log_softmax(logits.float(), dim=0)[target_ids[0]].item()
        else:
            target_logprob = torch.log_softmax(logits.float(), dim=0)[target_ids[0]].item()
        
        top_token = tokenizer.decode(logits.argmax().item())
        exact_match = top_token.strip() == target.strip()
        
        results.append({
            "target_logprob": target_logprob,
            "exact_match": exact_match,
            "top_token": top_token,
        })
    
    mean_logprob = sum(r["target_logprob"] for r in results) / len(results)
    exact_match_rate = sum(r["exact_match"] for r in results) / len(results)
    return {"mean_target_logprob": mean_logprob, "exact_match_rate": exact_match_rate, "n": len(results)}

def ablation_sweep(model, tokenizer, prompts, targets, n_layers):
    """Quick layer ablation to find hub layers."""
    hub_results = {}
    for layer_idx in range(n_layers):
        # Zero ablation hook
        def hook_fn(module, input, output, li=layer_idx):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:] = 0
                return (hidden,) + output[1:]
            return torch.zeros_like(output)
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        result = eval_task(model, tokenizer, prompts, targets)
        handle.remove()
        hub_results[layer_idx] = result["mean_target_logprob"]
        torch.cuda.empty_cache()
    
    return hub_results

def main():
    parser = argparse.ArgumentParser(description="Phase 2 parity verification")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "experiments" / "results"
    results_dir.mkdir(exist_ok=True)
    
    model_slug = args.model.split("/")[-1].replace(".", "").lower()
    output_file = results_dir / f"parity_{model_slug}.json"
    
    if output_file.exists() and not args.force:
        print(f"Results exist: {output_file}. Use --force to re-run.")
        return

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    
    # Define parity tasks
    task_prompts = {
        "variable_renaming_short": [
            ("x = 5\ny = x + 1\n# y is", "6"),
            ("a = 10\nb = a * 2\n# b is", "20"),
            ("foo = 'hello'\nbar = foo.upper()\n# bar is", "HELLO"),
            ("i = 0\nj = i + 1\nk = j + 1\n# k is", "2"),
            ("m = [1,2,3]\nn = len(m)\n# n is", "3"),
        ],
        "uncertainty_expression": [
            ("The population of France is approximately", "67"),
            ("It is likely that", "the"),
            ("The answer might be", "42"),
            ("Scientists believe that", "the"),
            ("The estimated cost is around", "$"),
        ],
        "verbosity_control": [
            ("Explain gravity in one word:", "attraction"),
            ("Summarize: The cat sat on the mat.", "A"),
            ("In 5 words or less, what is AI?", "Artificial"),
            ("Briefly: why is the sky blue?", "Light"),
            ("One sentence: photosynthesis.", "Photosynthesis"),
        ],
        "instruction_following": [
            ("List 3 colors:", "1."),
            ("Write a number:", "42"),
            ("Say yes or no:", "yes"),
            ("Choose A or B:", "A"),
            ("Respond with OK:", "OK"),
        ],
        "json_schema_short": [
            ('{"name": "Alice", "age":', "30"),
            ('{"city": "London", "country":', '"'),
            ('{"x": 1, "y":', "2"),
            ('{"status": "ok", "code":', "200"),
            ('{"type": "string", "value":', '"'),
        ],
        "code_semantics_short": [
            ("def add(a, b):\n    return a + b\n# add(3, 4) =", "7"),
            ("x = [1,2,3]\nx.append(4)\n# len(x) =", "4"),
            ("s = 'hello'\ns = s.upper()\n# s =", "HELLO"),
            ("d = {'a': 1}\nd['b'] = 2\n# len(d) =", "2"),
            ("n = 5\nf = n * 4 * 3 * 2 * 1\n# f =", "120"),
        ],
        "copying_short": [
            ("abc abc abc", "abc"),
            ("123 123 123", "123"),
            ("hello hello hello", "hello"),
            ("foo bar foo bar", "foo"),
            ("xyz xyz xyz", "xyz"),
        ],
    }

    all_results = {
        "experiment_id": "P2-PARITY-001",
        "model": args.model,
        "n_layers": n_layers,
        "seed": args.seed,
        "git_commit": get_git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }

    for task_name, pairs in task_prompts.items():
        prompts = [p for p, t in pairs]
        targets = [t for p, t in pairs]
        
        print(f"\n--- {task_name} ---")
        
        # Baseline eval
        baseline = eval_task(model, tokenizer, prompts, targets)
        print(f"  Baseline: logprob={baseline['mean_target_logprob']:.3f}, exact={baseline['exact_match_rate']:.1%}")
        
        # Hub identification via ablation (top 5 layers only for speed)
        top_layers = list(range(min(5, n_layers)))  # First 5 layers as quick check
        hub_kls = {}
        for li in top_layers:
            def hook_fn(module, input, output, li=li):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[:] = 0
                    return (h,) + output[1:]
                return torch.zeros_like(output)
            
            handle = model.model.layers[li].register_forward_hook(hook_fn)
            abl_result = eval_task(model, tokenizer, prompts, targets)
            handle.remove()
            hub_kls[li] = baseline["mean_target_logprob"] - abl_result["mean_target_logprob"]
            torch.cuda.empty_cache()
        
        best_layer = max(hub_kls, key=hub_kls.get)
        print(f"  Top early layer: L{best_layer} (delta={hub_kls[best_layer]:.3f})")
        
        all_results["tasks"][task_name] = {
            "baseline": baseline,
            "early_layer_ablation": {str(k): v for k, v in hub_kls.items()},
            "best_early_layer": best_layer,
        }

    # Save
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Append to registry
    registry_path = Path(__file__).parent.parent / "experiments" / "registry.jsonl"
    reg_entry = {
        "id": "P2-PARITY-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "parity",
        "model": args.model,
        "seed": args.seed,
        "git_commit": get_git_hash(),
        "outputs": [str(output_file)],
        "status": "success",
        "summary": f"Parity verification: {len(task_prompts)} tasks evaluated on {model_slug}",
    }
    with open(registry_path, "a") as f:
        f.write(json.dumps(reg_entry) + "\n")
    
    print(f"\nResults saved: {output_file}")
    print(f"Registry updated with P2-PARITY-001")

if __name__ == "__main__":
    main()
