#!/usr/bin/env python3
"""
Verify that task suite targets are single-token for LFM2.5-230M's 65K tokenizer.
The smaller vocab (65K vs Qwen's 152K) means some targets may tokenize differently.
"""
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-230M"

def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    
    # Check task manifest
    manifest_path = Path("data/tasks/task_manifest.json")
    if not manifest_path.exists():
        print(f"\nNo task manifest at {manifest_path}. Testing inline targets.")
        test_inline_targets(tokenizer)
        return
    
    print(f"\nLoading task manifest from {manifest_path}...")
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Check all targets
    total = 0
    single_token = 0
    multi_token = []
    problematic = []
    
    for task in manifest.get("tasks", manifest if isinstance(manifest, list) else []):
        if isinstance(task, dict) and "examples" in task:
            for ex in task["examples"]:
                target = ex.get("target", "")
                total += 1
                ids = tokenizer.encode(target, add_special_tokens=False)
                if len(ids) == 1:
                    single_token += 1
                else:
                    multi_token.append((target, ids, [tokenizer.decode([i]) for i in ids]))
    
    if total == 0:
        # Try loading examples from directory
        for task_dir in ["data/tasks/canonical_short", "data/tasks/canonical_long"]:
            p = Path(task_dir)
            if p.exists():
                for f in p.glob("*.json"):
                    with open(f) as fh:
                        data = json.load(fh)
                    examples = data if isinstance(data, list) else data.get("examples", [])
                    for ex in examples:
                        target = ex.get("target", ex.get("expected", ""))
                        if not target:
                            continue
                        total += 1
                        ids = tokenizer.encode(str(target), add_special_tokens=False)
                        if len(ids) == 1:
                            single_token += 1
                        else:
                            multi_token.append((str(target), ids, [tokenizer.decode([i]) for i in ids]))
    
    print(f"\n{'='*60}")
    print(f"Task Suite Verification Results")
    print(f"{'='*60}")
    print(f"Total targets checked: {total}")
    print(f"Single-token: {single_token} ({100*single_token/max(total,1):.1f}%)")
    print(f"Multi-token: {len(multi_token)} ({100*len(multi_token)/max(total,1):.1f}%)")
    
    if multi_token:
        print(f"\nMulti-token targets (first 30):")
        for target, ids, tokens in multi_token[:30]:
            print(f"  '{target}' -> {len(ids)} tokens: {tokens}")
    
    # Summary
    print(f"\n{'='*60}")
    if len(multi_token) == 0:
        print("ALL TARGETS ARE SINGLE-TOKEN. Task suite is compatible.")
    elif len(multi_token) < total * 0.1:
        print(f"WARNING: {len(multi_token)} targets are multi-token ({100*len(multi_token)/total:.1f}%).")
        print("These need to be re-targeted or the scoring function updated.")
    else:
        print(f"CRITICAL: {100*len(multi_token)/total:.1f}% of targets are multi-token.")
        print("The task suite needs significant adaptation for LFM2's tokenizer.")


def test_inline_targets(tokenizer):
    """Test common MI-Atlas targets with LFM2 tokenizer."""
    targets = [
        # Factual recall
        "Paris", "London", "Berlin", "Tokyo", "Rome",
        "France", "England", "Germany", "Japan", "Italy",
        # JSON
        "true", "false", "null", "True", "False", "None",
        # Numbers
        "1", "2", "3", "4", "5", "10", "42", "100",
        # Code
        "return", "def", "if", "for", "while", "print",
        # Common words
        "yes", "no", "the", "a", "an",
        # Boolean-like
        "Yes", "No", "0", "1",
        # Arithmetic answers
        "2", "3", "4", "5", "6", "7", "8", "9",
    ]
    
    print(f"\n{'='*60}")
    print("Inline Target Verification")
    print(f"{'='*60}")
    
    single = 0
    multi = 0
    for target in targets:
        ids = tokenizer.encode(target, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        status = "OK" if len(ids) == 1 else f"MULTI({len(ids)})"
        if len(ids) == 1:
            single += 1
        else:
            multi += 1
        print(f"  '{target}' -> {status} {tokens}")
    
    print(f"\nSingle-token: {single}/{len(targets)}")
    print(f"Multi-token: {multi}/{len(targets)}")


if __name__ == "__main__":
    main()
