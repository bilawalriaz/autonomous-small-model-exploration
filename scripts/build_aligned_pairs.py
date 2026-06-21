"""Build properly aligned clean/corrupt pairs for activation patching.

Key requirement: clean and corrupt prompts must tokenize to the same length
so that position-based patching makes sense.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def make_aligned_pair(tokenizer, prefix, clean_suffix, corrupt_suffix):
    """Make a pair where both sides tokenize to the same length.

    The shared prefix ensures alignment. The divergent suffix
    creates the clean/corrupt distinction.
    """
    clean = prefix + clean_suffix
    corrupt = prefix + corrupt_suffix

    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt, add_special_tokens=False)

    # Find the divergence point
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    return {
        "clean": clean,
        "corrupt": corrupt,
        "prefix": prefix,
        "clean_suffix": clean_suffix,
        "corrupt_suffix": corrupt_suffix,
        "clean_ids": clean_ids,
        "corrupt_ids": corrupt_ids,
        "prefix_len": len(prefix_ids),
        "same_length": len(clean_ids) == len(corrupt_ids),
    }


def main():
    set_seed(42)

    print("Loading tokenizer...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    tokenizer = bundle.tokenizer

    pairs = []

    # Copying / induction pairs
    # Pattern: A B C A B [target: C] vs A B C X Y [target: not C]
    copying_pairs = [
        {
            "family": "copying",
            "prefix": "Complete the pattern: A B C A B ",
            "clean_suffix": "C",
            "corrupt_suffix": "X",
            "target": "C",
        },
        {
            "family": "copying",
            "prefix": "Complete the pattern: 1 2 3 1 2 ",
            "clean_suffix": "3",
            "corrupt_suffix": "9",
            "target": "3",
        },
        {
            "family": "copying",
            "prefix": "Complete the pattern: cat dog cat dog ",
            "clean_suffix": "cat",
            "corrupt_suffix": "fish",
            "target": "cat",
        },
        {
            "family": "copying",
            "prefix": "Complete the sequence: red blue green red blue ",
            "clean_suffix": "green",
            "corrupt_suffix": "yellow",
            "target": "green",
        },
    ]

    # Delimiter tracking pairs
    delimiter_pairs = [
        {
            "family": "delimiter_tracking",
            "prefix": "Close the brackets: ( [ { ",
            "clean_suffix": "} ] )",
            "corrupt_suffix": ") } ]",
            "target": "}",
        },
        {
            "family": "delimiter_tracking",
            "prefix": "Complete: function(a, [b, {c:",
            "clean_suffix": " ",
            "corrupt_suffix": " ",
            "target": "}",
        },
        {
            "family": "delimiter_tracking",
            "prefix": "Close all brackets: { a: [1, (2",
            "clean_suffix": ") ] }",
            "corrupt_suffix": "} ] )",
            "target": ")",
        },
    ]

    # Arithmetic pairs
    arithmetic_pairs = [
        {
            "family": "arithmetic",
            "prefix": "Calculate: 7 + 5 = ",
            "clean_suffix": "12",
            "corrupt_suffix": "13",
            "target": "12",
        },
        {
            "family": "arithmetic",
            "prefix": "Calculate: 3 * 4 = ",
            "clean_suffix": "12",
            "corrupt_suffix": "13",
            "target": "12",
        },
        {
            "family": "arithmetic",
            "prefix": "Calculate: 20 - 8 = ",
            "clean_suffix": "12",
            "corrupt_suffix": "13",
            "target": "12",
        },
    ]

    # JSON pairs
    json_pairs = [
        {
            "family": "json_schema",
            "prefix": 'Return JSON: {"name": "Alice", "age": ',
            "clean_suffix": "31}",
            "corrupt_suffix": "31,",
            "target": "31",
        },
        {
            "family": "json_schema",
            "prefix": 'Return JSON: {"x": 1, "y": ',
            "clean_suffix": "2}",
            "corrupt_suffix": "2,",
            "target": "2",
        },
    ]

    # Factual recall pairs
    factual_pairs = [
        {
            "family": "factual_recall",
            "prefix": "The capital of France is ",
            "clean_suffix": "Paris",
            "corrupt_suffix": "London",
            "target": "Paris",
        },
        {
            "family": "factual_recall",
            "prefix": "The capital of Germany is ",
            "clean_suffix": "Berlin",
            "corrupt_suffix": "Munich",
            "target": "Berlin",
        },
        {
            "family": "factual_recall",
            "prefix": "The chemical symbol for gold is ",
            "clean_suffix": "Au",
            "corrupt_suffix": "Ag",
            "target": "Au",
        },
    ]

    # Code syntax pairs
    code_pairs = [
        {
            "family": "code_syntax",
            "prefix": "def add(a, b):\n    return a + ",
            "clean_suffix": "b",
            "corrupt_suffix": "a",
            "target": "b",
        },
        {
            "family": "code_syntax",
            "prefix": "x = [i for i in range(",
            "clean_suffix": "10)]",
            "corrupt_suffix": "10(",
            "target": "10",
        },
    ]

    # Verbosity pairs
    verbosity_pairs = [
        {
            "family": "verbosity_control",
            "prefix": "Is 7 prime? Answer one word: ",
            "clean_suffix": "Yes",
            "corrupt_suffix": "No",
            "target": "Yes",
        },
        {
            "family": "verbosity_control",
            "prefix": "How many sides does a triangle have? Just the number: ",
            "clean_suffix": "3",
            "corrupt_suffix": "4",
            "target": "3",
        },
    ]

    all_raw = copying_pairs + delimiter_pairs + arithmetic_pairs + json_pairs + factual_pairs + code_pairs + verbosity_pairs

    # Validate alignment
    print(f"\nValidating {len(all_raw)} pairs...")
    aligned_count = 0
    for raw in all_raw:
        pair = make_aligned_pair(tokenizer, raw["prefix"], raw["clean_suffix"], raw["corrupt_suffix"])
        pair["family"] = raw["family"]
        pair["target"] = raw["target"]
        pair["id"] = f"{raw['family']}_{len([p for p in pairs if p.get('family') == raw['family']]):04d}"

        if pair["same_length"]:
            aligned_count += 1
            pairs.append(pair)
        else:
            print(f"  SKIP (length mismatch): {pair['id']} clean={len(pair['clean_ids'])} corrupt={len(pair['corrupt_ids'])}")

    print(f"  {aligned_count}/{len(all_raw)} pairs aligned")

    # Also verify with logprob that clean > corrupt
    print("\nVerifying clean > corrupt logprobs...")
    model = bundle.model
    model.eval()
    verified = 0
    for pair in pairs:
        clean_ids = tokenizer(pair["clean"], return_tensors="pt")["input_ids"]
        corrupt_ids = tokenizer(pair["corrupt"], return_tensors="pt")["input_ids"]

        # Get logprob of the last token in each
        prefix_ids = tokenizer.encode(pair["prefix"], add_special_tokens=False)
        prefix_len = len(prefix_ids)

        with torch.no_grad():
            clean_logits = model(clean_ids.to(model.device)).logits
            corrupt_logits = model(corrupt_ids.to(model.device)).logits

        # Logprob of the suffix token given prefix
        clean_lp = torch.log_softmax(clean_logits[0, prefix_len - 1], dim=-1)
        target_id = clean_ids[0, prefix_len].item()
        clean_target_lp = clean_lp[target_id].item()

        corrupt_lp = torch.log_softmax(corrupt_logits[0, prefix_len - 1], dim=-1)
        corrupt_target_lp = corrupt_lp[target_id].item()

        pair["clean_target_lp"] = clean_target_lp
        pair["corrupt_target_lp"] = corrupt_target_lp
        pair["lp_diff"] = clean_target_lp - corrupt_target_lp

        if clean_target_lp > corrupt_target_lp:
            verified += 1

    print(f"  {verified}/{len(pairs)} pairs have clean > corrupt logprobs")

    # Save
    output_path = PROJECT_ROOT / "data" / "clean_corrupt_pairs" / "pairs_v1.json"
    save_json(pairs, output_path)
    print(f"\nSaved {len(pairs)} pairs to {output_path}")

    # Print summary
    print("\nPair summary:")
    families = {}
    for p in pairs:
        fam = p["family"]
        families[fam] = families.get(fam, 0) + 1
    for fam, count in sorted(families.items()):
        print(f"  {fam}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
