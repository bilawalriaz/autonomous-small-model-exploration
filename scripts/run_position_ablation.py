"""
Position-Specific Ablation: WHERE in the token sequence does each layer matter?
Ablate individual token positions at key layers (L0, L1, L2, L7, L9) to determine
if effects are concentrated at instruction tokens, content tokens, or last token.

Tests H001: Is L2 a routing hub (position-dependent) or just carries info (position-independent)?
"""
import json, sys, os, time
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = REPO / "experiments" / "registry.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Key layers to test
TARGET_LAYERS = [0, 1, 2, 7, 9, 15, 22]

# Representative tasks from each family (short prompts for position analysis)
TASKS = [
    {"family": "copying", "prompt": "Repeat: A B C A B\nA B", "target": " C"},
    {"family": "copying", "prompt": "Pattern: 1 2 3 1 2 3\n1 2", "target": " 3"},
    {"family": "delimiter_tracking", "prompt": "Close: ( [ {", "target": "}"},
    {"family": "delimiter_tracking", "prompt": "Complete: def foo(x, [y,", "target": " z"},
    {"family": "factual_recall", "prompt": "The capital of France is", "target": " Paris"},
    {"family": "factual_recall", "prompt": "The capital of Germany is", "target": " Berlin"},
    {"family": "factual_recall", "prompt": "The capital of Italy is", "target": " Rome"},
    {"family": "code_semantics", "prompt": "x = 5\ny = x + 3\nprint(y)\n# output:", "target": "8"},
    {"family": "code_semantics", "prompt": "a = [1, 2, 3]\nprint(len(a))\n# output:", "target": "3"},
    {"family": "json_schema", "prompt": 'Extract as JSON with keys name, age: Alice is 31.\n', "target": '{"'},
    {"family": "json_schema", "prompt": 'JSON with keys x, y: x=5, y=10\n', "target": '{"'},
]


def main():
    exp_id = "exp_000018"
    start = time.time()

    print("=" * 60)
    print("Position-Specific Ablation")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for task in TASKS:
        family = task["family"]
        prompt = task["prompt"]
        target = task["target"]

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"][0]
        n_tokens = len(input_ids)
        token_strs = [tokenizer.decode([tid]) for tid in input_ids]

        # Baseline logprob
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_ids = tokenizer.encode(target, add_special_tokens=False)
        if not tgt_ids:
            continue
        baseline_lp = logprobs[tgt_ids[0]].item()

        # Classify positions
        positions = []
        for i, tok in enumerate(token_strs):
            if i == n_tokens - 1:
                pos_type = "last"
            elif i == 0:
                pos_type = "first"
            elif any(c in tok for c in "({["):
                pos_type = "delimiter"
            elif any(c in tok for c in ")}]"):
                pos_type = "delimiter_close"
            elif tok.strip() in (":", ",", ".", "=", "+", "-", "*", "/"):
                pos_type = "operator"
            else:
                pos_type = "content"
            positions.append(pos_type)

        task_results = {
            "family": family,
            "prompt": prompt,
            "target": target,
            "n_tokens": n_tokens,
            "tokens": token_strs,
            "position_types": positions,
            "baseline_lp": round(baseline_lp, 4),
            "layer_results": {},
        }

        for layer_idx in TARGET_LAYERS:
            layer_pos_effects = {}
            for pos in range(n_tokens):
                # Ablate only this position at this layer
                def hook_fn(module, input, output, li=layer_idx, p=pos):
                    if isinstance(output, tuple):
                        hidden = output[0].clone()
                        hidden[0, p, :] = 0
                        return (hidden,) + output[1:]
                    else:
                        hidden = output.clone()
                        hidden[0, p, :] = 0
                        return hidden

                handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
                with torch.no_grad():
                    out = model(**inputs)
                handle.remove()

                logits = out.logits[0, -1]
                logprobs = torch.log_softmax(logits, dim=-1)
                ablated_lp = logprobs[tgt_ids[0]].item()
                effect = round(baseline_lp - ablated_lp, 4)  # positive = harmful

                layer_pos_effects[f"pos_{pos}"] = {
                    "token": token_strs[pos],
                    "position_type": positions[pos],
                    "effect": effect,
                }

            task_results["layer_results"][f"L{layer_idx}"] = layer_pos_effects

            # Find position with max effect
            max_pos = max(layer_pos_effects.items(), key=lambda x: x[1]["effect"])
            if max_pos[1]["effect"] > 0.1:
                print(f"  [{family}] L{layer_idx}: max effect at {max_pos[0]} ('{max_pos[1]['token']}', {max_pos[1]['position_type']}) = {max_pos[1]['effect']:.4f}")

        results.append(task_results)

    # Aggregate: which position types are most affected at each layer?
    print("\n" + "=" * 60)
    print("AGGREGATE: Mean effect by position type and layer")
    print("=" * 60)

    agg = {}
    for task_result in results:
        for layer_key, pos_effects in task_result["layer_results"].items():
            if layer_key not in agg:
                agg[layer_key] = {}
            for pos_key, pos_data in pos_effects.items():
                pt = pos_data["position_type"]
                if pt not in agg[layer_key]:
                    agg[layer_key][pt] = []
                agg[layer_key][pt].append(pos_data["effect"])

    for layer_key in sorted(agg.keys()):
        print(f"\n{layer_key}:")
        for pt, effects in sorted(agg[layer_key].items()):
            mean_eff = sum(effects) / len(effects)
            max_eff = max(effects)
            print(f"  {pt:20s}: mean={mean_eff:.4f}, max={max_eff:.4f}, n={len(effects)}")

    # Save
    output = {
        "experiment": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "n_tasks": len(TASKS),
        "results": results,
        "aggregate": {lk: {pt: {"mean": round(sum(e)/len(e), 4), "max": round(max(e), 4), "n": len(e)}
                          for pt, e in pts.items()}
                     for lk, pts in agg.items()},
    }
    out_path = RESULTS_DIR / "position_specific_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Registry
    entry = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "ablation",
        "model": MODEL_NAME,
        "backend": "hf_hooks",
        "git_commit": "",
        "config": "",
        "inputs": [],
        "outputs": [str(out_path)],
        "status": "success",
        "summary": f"Position-specific ablation: {len(TASKS)} tasks, {len(TARGET_LAYERS)} layers, per-position effects",
        "key_metrics": {},
        "failure": None,
        "next": "Component atlas construction",
    }
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Results: {out_path}")


if __name__ == "__main__":
    main()
