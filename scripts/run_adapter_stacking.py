"""
Adapter Stacking: Test skill combination and interference.
Uses add_weighted_adapter to merge pairs, then evaluates cross-task.
"""
import json, sys, os
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = Path(__file__).parent.parent
ADAPTERS_DIR = REPO / "experiments" / "adapters"
RESULTS_DIR = REPO / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def evaluate_family(model, tokenizer, family_tasks):
    """Evaluate: mean target logprob."""
    lps = []
    for task in family_tasks:
        prompt = task.get("clean_prompt", task.get("prompt", ""))
        tgt = task.get("target", "")
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)
        if tgt_ids:
            lps.append(logprobs[tgt_ids[0]].item())
    return round(sum(lps) / len(lps), 4) if lps else 0.0


def main():
    print("=" * 60)
    print("Adapter Stacking (Weighted Merge)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(REPO / "data" / "eval_sets" / "task_suite_v0.json") as f:
        suite = json.load(f)

    families = ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"]
    family_tasks = {fam: [t for t in suite if t["family"] == fam][:5] for fam in families}

    adapter_map = {
        "copying": "lora_copying_r8",
        "delimiter_tracking": "lora_delimiter_tracking_r8",
        "factual_recall": "lora_factual_recall_r8",
        "code_semantics": "lora_code_semantics_r8",
        "json_schema": "lora_json_schema_r8",
    }

    # Base scores
    print("\n--- Base model ---")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    base_scores = {}
    for fam in families:
        score = evaluate_family(base_model, tokenizer, family_tasks[fam])
        base_scores[fam] = score
        print(f"  {fam}: {score:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # Single adapter scores
    single_scores = {}
    for adapter_fam, adapter_dir in adapter_map.items():
        print(f"\n--- {adapter_fam} adapter ---")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, str(ADAPTERS_DIR / adapter_dir / "adapter"))
        model.eval()
        scores = {}
        for fam in families:
            score = evaluate_family(model, tokenizer, family_tasks[fam])
            scores[fam] = score
            delta = score - base_scores[fam]
            print(f"  {fam}: {score:.4f} ({delta:+.4f})")
        single_scores[adapter_fam] = scores
        del model
        torch.cuda.empty_cache()

    # Stack pairs
    pairs = [
        ("copying", "json_schema"),
        ("copying", "delimiter_tracking"),
        ("factual_recall", "json_schema"),
        ("delimiter_tracking", "code_semantics"),
        ("code_semantics", "json_schema"),
    ]

    stack_results = {}
    for a_fam, b_fam in pairs:
        pair_name = f"{a_fam}+{b_fam}"
        print(f"\n--- Stacking: {pair_name} ---")

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, str(ADAPTERS_DIR / adapter_map[a_fam] / "adapter"), adapter_name=a_fam)
        model.load_adapter(str(ADAPTERS_DIR / adapter_map[b_fam] / "adapter"), adapter_name=b_fam)
        model.add_weighted_adapter([a_fam, b_fam], [0.5, 0.5], "combined")
        model.set_adapter("combined")
        model.eval()

        scores = {}
        for fam in families:
            score = evaluate_family(model, tokenizer, family_tasks[fam])
            expected = (single_scores[a_fam][fam] + single_scores[b_fam][fam]) / 2
            interference = score - expected
            scores[fam] = {"score": score, "expected": round(expected, 4), "interference": round(interference, 4)}
            print(f"  {fam}: {score:.4f} (expected {expected:.4f}, interference {interference:+.4f})")
        stack_results[pair_name] = scores
        del model
        torch.cuda.empty_cache()

    # Save
    output = {
        "experiment": "exp_000016",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_scores": base_scores,
        "single_adapter_scores": single_scores,
        "stack_results": stack_results,
    }
    out_path = RESULTS_DIR / "adapter_stacking.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    entry = {
        "id": "exp_000016",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "comparison",
        "model": MODEL_NAME,
        "backend": "hf",
        "git_commit": "",
        "config": "",
        "inputs": [],
        "outputs": [str(out_path)],
        "status": "success",
        "summary": f"Adapter stacking (weighted merge): {len(pairs)} pairs tested",
        "key_metrics": {},
        "failure": None,
        "next": "Checkpoint timeline, component atlas construction",
    }
    with open(REPO / "experiments" / "registry.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
