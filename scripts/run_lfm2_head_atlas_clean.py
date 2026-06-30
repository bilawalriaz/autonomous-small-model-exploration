#!/usr/bin/env python3
"""
LFM2.5-230M Head Ablation — clean version.
Ablate each of 16 Q heads in all 6 attention layers.
"""
import sys, json, torch, numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTN_LAYERS = [2, 4, 6, 8, 10, 12]
CONV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13]
NUM_Q_HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024

TASK_FAMILIES_SHORT = [
    "arithmetic", "code_semantics", "code_syntax", "copying",
    "dead_code", "factual_recall", "instruction_following",
    "json_schema", "variable_renaming",
]

def load_prompts(n=3):
    prompts = {}
    task_dir = PROJECT_ROOT / "data" / "tasks" / "canonical_short"
    import glob as g
    for fam in TASK_FAMILIES_SHORT:
        fpath = task_dir / f"{fam}.json"
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        examples = data.get("examples", []) if isinstance(data, dict) else data
        test = [e for e in examples if e.get("metadata", {}).get("split") == "test"][:n]
        if not test:
            test = examples[:n]
        prompts[fam] = [{"prompt": e.get("prompt", e.get("clean_prompt", "")),
                          "target": e.get("target", "")} for e in test]
    return prompts

def kl_div(baseline, ablated):
    lp = torch.log_softmax(baseline.float(), -1)
    lq = torch.log_softmax(ablated.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    prompts = load_prompts(3)
    print(f"  Loaded {sum(len(v) for v in prompts.values())} prompts across {len(prompts)} families")

    # === HEAD ABLATION ===
    print(f"\n{'='*60}")
    print("HEAD ABLATION (16 Q heads x 6 attn layers)")
    print(f"{'='*60}")

    head_results = {}
    for family, plist in prompts.items():
        print(f"\n  Family: {family}")
        family_kls = {}

        for pi, p in enumerate(plist):
            ids = tok(p["prompt"], return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

            with torch.no_grad():
                baseline = model(ids).logits

            for layer_idx in ATTN_LAYERS:
                attn = model.model.layers[layer_idx].self_attn

                for head_idx in range(NUM_Q_HEADS):
                    def make_hook(h):
                        def hook(module, inp, output):
                            if isinstance(output, tuple):
                                out = output[0]
                            else:
                                out = output
                            b, s, d = out.shape
                            reshaped = out.reshape(b, s, NUM_Q_HEADS, HEAD_DIM)
                            reshaped[:, :, h, :] = 0.0
                            out = reshaped.reshape(b, s, d)
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return hook

                    handle = attn.register_forward_hook(make_hook(head_idx))
                    with torch.no_grad():
                        abl = model(ids).logits
                    handle.remove()

                    k = kl_div(baseline, abl)
                    key = (layer_idx, head_idx)
                    if key not in family_kls:
                        family_kls[key] = []
                    family_kls[key].append(k)

        # Average across prompts
        head_results[family] = {}
        for (li, hi), kls in family_kls.items():
            head_results[family][f"L{li}_H{hi}"] = round(np.mean(kls), 4)

        # Print summary for this family
        sorted_heads = sorted(family_kls.items(), key=lambda x: np.mean(x[1]), reverse=True)
        print(f"    Top 5 heads: {[(f'L{l}_H{h}', round(np.mean(v), 2)) for (l,h), v in sorted_heads[:5]]}")

    # === CROSS-LAYER PATCHING (simplified) ===
    print(f"\n{'='*60}")
    print("CROSS-LAYER RESIDUAL PATCHING")
    print(f"{'='*60}")

    # Clean/corrupt pairs
    pairs = [
        ("The capital of France is", "The capital of Germany is"),
        ("2 + 2 =", "2 + 3 ="),
        ('{"key": "value"', '{"key": "error"'),
    ]

    patch_results = {}
    for ci, (clean, corrupt) in enumerate(pairs):
        clean_ids = tok(clean, return_tensors="pt").input_ids.to(device)
        corrupt_ids = tok(corrupt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            clean_logits = model(clean_ids).logits
            corrupt_logits = model(corrupt_ids).logits

        baseline_kl = kl_div(clean_logits, corrupt_logits)

        # Cache clean activations
        clean_cache = {}
        hooks = []
        for i in range(14):
            def cache_hook(layer_i):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        clean_cache[layer_i] = output[0].detach().clone()
                    else:
                        clean_cache[layer_i] = output.detach().clone()
                    return output
                return hook
            h = model.model.layers[i].register_forward_hook(cache_hook(i))
            hooks.append(h)

        with torch.no_grad():
            model(clean_ids)
        for h in hooks:
            h.remove()

        # Patch each layer: run corrupt with clean activation at layer i
        layer_recovery = {}
        for patch_layer in range(14):
            cached = clean_cache[patch_layer]

            def patch_hook(cached_act):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        return (cached_act,) + output[1:]
                    return cached_act
                return hook

            handle = model.model.layers[patch_layer].register_forward_hook(patch_hook(cached))
            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            handle.remove()

            patched_kl = kl_div(clean_logits, patched_logits)
            recovery = 1.0 - (patched_kl / max(baseline_kl, 1e-8))
            layer_recovery[f"L{patch_layer}"] = round(recovery, 4)

        pair_key = f"{clean[:30]}...vs{corrupt[:20]}..."
        patch_results[pair_key] = layer_recovery
        print(f"\n  Pair {ci}: baseline KL={baseline_kl:.2f}")
        for li in range(14):
            r = layer_recovery[f"L{li}"]
            bar = "#" * max(0, int(r * 20))
            print(f"    L{li}: recovery={r:.3f} {bar}")

    # === RESIDUAL NORMS ===
    print(f"\n{'='*60}")
    print("RESIDUAL STREAM NORMS (per family)")
    print(f"{'='*60}")

    norm_results = {}
    for family, plist in prompts.items():
        p = plist[0]
        ids = tok(p["prompt"], return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        norms = {}
        hooks = []
        def capture(li):
            def hook(module, inp, output):
                t = output[0] if isinstance(output, tuple) else output
                norms[li] = round(t.float().norm().item(), 4)
                return output
            return hook
        for i in range(14):
            hooks.append(model.model.layers[i].register_forward_hook(capture(i)))
        with torch.no_grad():
            model(ids)
        for h in hooks:
            h.remove()

        norm_results[family] = norms
        if family == list(prompts.keys())[0]:
            print(f"  {family}: {[(f'L{k}', v) for k,v in sorted(norms.items())]}")

    # === Save results ===
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "model": args.model,
        "seed": args.seed,
        "timestamp": ts,
        "head_ablation": head_results,
        "cross_layer_patching": patch_results,
        "residual_norms": norm_results,
    }
    out_path = RESULTS_DIR / f"lfm2_230m_head_atlas_seed{args.seed}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    # === Head ablation summary across all families ===
    print(f"\n{'='*60}")
    print("HEAD ABLATION SUMMARY (mean KL across all families)")
    print(f"{'='*60}")
    mean_kls = {}
    for layer_idx in ATTN_LAYERS:
        for head_idx in range(NUM_Q_HEADS):
            key = f"L{layer_idx}_H{head_idx}"
            all_kls = [head_results[f].get(key, 0) for f in head_results]
            mean_kls[key] = np.mean(all_kls)

    sorted_heads = sorted(mean_kls.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 heads:")
    for name, kl in sorted_heads[:10]:
        print(f"  {name}: {kl:.4f}")
    print("\nBottom 5 heads (least important):")
    for name, kl in sorted_heads[-5:]:
        print(f"  {name}: {kl:.4f}")

    # Per-layer summary
    print("\nPer-layer mean KL:")
    for li in ATTN_LAYERS:
        layer_kls = [mean_kls[f"L{li}_H{hi}"] for hi in range(NUM_Q_HEADS)]
        print(f"  L{li}: mean={np.mean(layer_kls):.4f}, max={max(layer_kls):.4f}, "
              f"min={min(layer_kls):.4f}, best_head=H{np.argmax(layer_kls)}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
