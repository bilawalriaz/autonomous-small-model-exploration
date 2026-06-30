#!/usr/bin/env python3
"""
LFM2.5-230M Position-Specific Ablation + Cross-Layer Patching Analysis
=======================================================================
1. Position-specific ablation: which positions does each layer care about?
2. Cross-layer residual patching: which layers encode answer vs context?
3. Conv kernel analysis: local vs distant context sensitivity
4. L6 boundary analysis: norm jump scaling artifact or real capability jump?
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "LiquidAI/LFM2.5-230M"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 14
HIDDEN_DIM = 1024
L5_NORM = 1.68
L6_NORM = 25.55
NORM_RATIO = L6_NORM / L5_NORM  # ~15.2

# ── Helpers ─────────────────────────────────────────────────────────────

def kl_divergence(logits_a, logits_b, dim=-1):
    """KL(P_a || P_b) averaged over batch and sequence."""
    log_pa = torch.log_softmax(logits_a.float(), dim=dim)
    log_pb = torch.log_softmax(logits_b.float(), dim=dim)
    pa = torch.softmax(logits_a.float(), dim=dim)
    kl = (pa * (log_pa - log_pb)).sum(dim=dim)  # [batch, seq]
    return kl.mean().item()


def kl_per_token(logits_a, logits_b, dim=-1):
    """KL(P_a || P_b) per token position, averaged over batch."""
    log_pa = torch.log_softmax(logits_a.float(), dim=dim)
    log_pb = torch.log_softmax(logits_b.float(), dim=dim)
    pa = torch.softmax(logits_a.float(), dim=dim)
    kl = (pa * (log_pa - log_pb)).sum(dim=dim)  # [batch, seq]
    return kl.mean(dim=0).detach().cpu().numpy()


def get_logits(model, input_ids, attention_mask=None):
    """Run forward pass, return logits."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


def save_json(data, name):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Saved {path}")
    return path


def save_tensor(tensor, name):
    path = RESULTS_DIR / f"{name}.pt"
    torch.save(tensor, path)
    print(f"  → Saved {path}")
    return path


# ── Load Model ──────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE} with {DTYPE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=str(DEVICE), trust_remote_code=True
    )
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  Layers: {len(model.model.layers)}")
    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════
# 1. POSITION-SPECIFIC ABLATION
# ════════════════════════════════════════════════════════════════════════

def run_position_specific_ablation(model, tokenizer):
    """
    For each layer, ablate the residual stream at each token position independently.
    Measures KL divergence from baseline per (layer, position).
    """
    print("\n" + "="*70)
    print("1. POSITION-SPECIFIC ABLATION")
    print("="*70)

    prompt = "The capital of France is Paris. The capital of Germany is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print(f"  Prompt: {prompt}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Tokens: {tokens}")

    # Baseline logits
    baseline_logits = get_logits(model, input_ids)

    # Results: [layer, position] -> KL divergence
    kl_matrix = np.zeros((N_LAYERS, seq_len))
    hooks = []

    for layer_idx in range(N_LAYERS):
        print(f"  Layer {layer_idx:2d}: ", end="", flush=True)
        layer_module = model.model.layers[layer_idx]

        for pos in range(seq_len):
            # Register hook to zero out this position
            def make_hook(p):
                def position_hook(module, inp, output):
                    hidden = output[0].clone()
                    hidden[:, p, :] = 0.0
                    return (hidden,) + output[1:]
                return position_hook

            hook = layer_module.register_forward_hook(make_hook(pos))
            ablated_logits = get_logits(model, input_ids)
            hook.remove()

            kl = kl_divergence(baseline_logits, ablated_logits)
            kl_matrix[layer_idx, pos] = kl

        avg_kl = kl_matrix[layer_idx].mean()
        max_kl = kl_matrix[layer_idx].max()
        max_pos = kl_matrix[layer_idx].argmax()
        print(f"avg KL={avg_kl:.6f}, max KL={max_kl:.6f} at pos {max_pos} ({tokens[max_pos]})")

    # Summary: which layers care most about first vs last token?
    first_token_kl = kl_matrix[:, 0]
    last_token_kl = kl_matrix[:, -1]

    print("\n  Summary: First token vs Last token importance")
    for layer_idx in range(N_LAYERS):
        ratio = first_token_kl[layer_idx] / max(last_token_kl[layer_idx], 1e-10)
        print(f"    L{layer_idx:2d}: first={first_token_kl[layer_idx]:.6f} "
              f"last={last_token_kl[layer_idx]:.6f} ratio={ratio:.2f}x")

    save_json({
        "prompt": prompt,
        "tokens": tokens,
        "seq_len": seq_len,
        "kl_matrix": kl_matrix.tolist(),
        "first_token_kl": first_token_kl.tolist(),
        "last_token_kl": last_token_kl.tolist(),
    }, "position_ablation_results")

    save_tensor(torch.tensor(kl_matrix), "position_ablation_kl_matrix")
    return kl_matrix


# ════════════════════════════════════════════════════════════════════════
# 2. CROSS-LAYER RESIDUAL PATCHING
# ════════════════════════════════════════════════════════════════════════

def run_cross_layer_patching(model, tokenizer):
    """
    Clean: 'The capital of France is'
    Corrupt: 'The capital of Germany is'
    Patch clean activations into corrupt run at each layer.
    Measures KL recovery.
    """
    print("\n" + "="*70)
    print("2. CROSS-LAYER RESIDUAL PATCHING")
    print("="*70)

    clean_prompt = "The capital of France is"
    corrupt_prompt = "The capital of Germany is"

    clean_ids = tokenizer(clean_prompt, return_tensors="pt").to(DEVICE)["input_ids"]
    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt").to(DEVICE)["input_ids"]

    seq_len = min(clean_ids.shape[1], corrupt_ids.shape[1])
    clean_ids = clean_ids[:, :seq_len]
    corrupt_ids = corrupt_ids[:, :seq_len]

    clean_tokens = tokenizer.convert_ids_to_tokens(clean_ids[0])
    corrupt_tokens = tokenizer.convert_ids_to_tokens(corrupt_ids[0])

    print(f"  Clean:   '{clean_prompt}' → {clean_tokens}")
    print(f"  Corrupt: '{corrupt_prompt}' → {corrupt_tokens}")

    # Baseline logits
    clean_logits = get_logits(model, clean_ids)
    corrupt_logits = get_logits(model, corrupt_ids)

    kl_clean_corrupt = kl_divergence(clean_logits, corrupt_logits)
    print(f"  KL(clean || corrupt) = {kl_clean_corrupt:.6f}")

    # Cache clean activations at each layer
    clean_cache = {}
    hooks = []

    for layer_idx in range(N_LAYERS):
        def make_cache_hook(li):
            def hook(module, inp, output):
                clean_cache[li] = output[0].detach().clone()
                return output
            return hook
        h = model.model.layers[layer_idx].register_forward_hook(make_cache_hook(layer_idx))
        hooks.append(h)

    get_logits(model, clean_ids)
    for h in hooks:
        h.remove()

    # Patch clean into corrupt at each layer
    patch_results = []
    for patch_layer in range(N_LAYERS):
        def make_patch_hook(pl):
            def hook(module, inp, output):
                hidden = output[0].clone()
                hidden[:, :seq_len, :] = clean_cache[pl][:, :seq_len, :]
                return (hidden,) + output[1:]
            return hook

        h = model.model.layers[patch_layer].register_forward_hook(make_patch_hook(patch_layer))
        patched_logits = get_logits(model, corrupt_ids)
        h.remove()

        kl_patched = kl_divergence(clean_logits, patched_logits)
        recovery = (kl_clean_corrupt - kl_patched) / max(kl_clean_corrupt, 1e-10) * 100

        patch_results.append({
            "layer": patch_layer,
            "kl_patched": kl_patched,
            "kl_reduction": kl_clean_corrupt - kl_patched,
            "recovery_pct": recovery,
        })
        print(f"  Patch L{patch_layer:2d}: KL={kl_patched:.6f} "
              f"reduction={kl_clean_corrupt - kl_patched:.6f} recovery={recovery:.1f}%")

    # Find which layer gives most recovery
    best = max(patch_results, key=lambda x: x["recovery_pct"])
    print(f"\n  Best recovery: L{best['layer']} at {best['recovery_pct']:.1f}%")

    save_json({
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "kl_clean_corrupt": kl_clean_corrupt,
        "patch_results": patch_results,
    }, "cross_layer_patching_results")

    return patch_results


# ════════════════════════════════════════════════════════════════════════
# 3. CONV KERNEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def run_conv_kernel_analysis(model, tokenizer):
    """
    Conv layers use kernel_size=4 with causal padding.
    Test sensitivity to LOCAL context (nearby tokens) vs DISTANT context.
    Ablate conv layers at early/mid/late positions.
    """
    print("\n" + "="*70)
    print("3. CONV KERNEL ANALYSIS")
    print("="*70)

    # Use a longer prompt for more position diversity
    prompt = ("The quick brown fox jumps over the lazy dog. "
              "The capital of France is Paris and the capital of Germany is Berlin.")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print(f"  Prompt: {prompt}")
    print(f"  Sequence length: {seq_len}")

    baseline_logits = get_logits(model, input_ids)

    # Identify conv layers (L0-L5 for LFM2.5-230M, L6-L13 are attention)
    conv_layers = list(range(6))   # L0-L5
    attn_layers = list(range(6, 14))  # L6-L13

    # Sample positions: early (2,3,4,5), mid, late
    early_positions = list(range(2, min(6, seq_len)))
    mid_positions = list(range(seq_len//3, min(seq_len//3 + 4, seq_len)))
    late_positions = list(range(max(seq_len - 5, 0), seq_len))

    print(f"\n  Conv layers: {conv_layers}")
    print(f"  Attn layers: {attn_layers}")
    print(f"  Early positions: {early_positions}")
    print(f"  Mid positions: {mid_positions}")
    print(f"  Late positions: {late_positions}")

    results = {"conv_layers": conv_layers, "attn_layers": attn_layers}

    for layer_group_name, layer_list in [("conv", conv_layers), ("attn", attn_layers)]:
        group_results = []
        for layer_idx in layer_list:
            layer_module = model.model.layers[layer_idx]
            layer_result = {"layer": layer_idx, "positions": {}}

            for pos_category, positions in [("early", early_positions), ("mid", mid_positions), ("late", late_positions)]:
                kls = []
                for pos in positions:
                    def make_hook(p):
                        def hook(module, inp, output):
                            hidden = output[0].clone()
                            hidden[:, p, :] = 0.0
                            return (hidden,) + output[1:]
                        return hook

                    h = layer_module.register_forward_hook(make_hook(pos))
                    ablated = get_logits(model, input_ids)
                    h.remove()

                    kl = kl_divergence(baseline_logits, ablated)
                    kls.append(kl)

                avg_kl = np.mean(kls)
                layer_result["positions"][pos_category] = {"avg_kl": avg_kl, "individual_kls": kls}

            group_results.append(layer_result)
            local_sensitivity = layer_result["positions"]["early"]["avg_kl"]
            distant_sensitivity = layer_result["positions"]["late"]["avg_kl"]
            ratio = local_sensitivity / max(distant_sensitivity, 1e-10)
            print(f"  L{layer_idx:2d} ({layer_group_name}): early={local_sensitivity:.6f} "
                  f"late={distant_sensitivity:.6f} ratio={ratio:.2f}x")

        results[layer_group_name] = group_results

    # Compare conv vs attn local/distant ratios
    conv_early_avg = np.mean([r["positions"]["early"]["avg_kl"] for r in results["conv"]])
    conv_late_avg = np.mean([r["positions"]["late"]["avg_kl"] for r in results["conv"]])
    attn_early_avg = np.mean([r["positions"]["early"]["avg_kl"] for r in results["attn"]])
    attn_late_avg = np.mean([r["positions"]["late"]["avg_kl"] for r in results["attn"]])

    print(f"\n  Conv avg: early={conv_early_avg:.6f} late={conv_late_avg:.6f} "
          f"ratio={conv_early_avg/conv_late_avg:.2f}x")
    print(f"  Attn avg: early={attn_early_avg:.6f} late={attn_late_avg:.6f} "
          f"ratio={attn_early_avg/attn_late_avg:.2f}x")

    results["summary"] = {
        "conv_early_avg": conv_early_avg, "conv_late_avg": conv_late_avg,
        "attn_early_avg": attn_early_avg, "attn_late_avg": attn_late_avg,
    }

    save_json(results, "conv_kernel_analysis_results")
    return results


# ════════════════════════════════════════════════════════════════════════
# 4. L6 BOUNDARY ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def run_l6_boundary_analysis(model, tokenizer):
    """
    Residual stream shows 10x norm jump at L6.
    Compare L5 (conv, norm=1.68) vs L6 (attn, norm=25.55).
    Is L6's effect proportionally larger, or is the norm jump a scaling artifact?
    """
    print("\n" + "="*70)
    print("4. L6 BOUNDARY ANALYSIS")
    print("="*70)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    baseline_logits = get_logits(model, input_ids)

    results = {"prompt": prompt, "norm_ratio": NORM_RATIO}

    # Test 1: Ablate L5 and L6 independently
    print("\n  Test 1: Independent ablation of L5 vs L6")
    for layer_idx, name in [(5, "L5-conv"), (6, "L6-attn")]:
        def make_hook():
            def hook(module, inp, output):
                hidden = output[0].clone()
                hidden[:] = 0.0
                return (hidden,) + output[1:]
            return hook

        h = model.model.layers[layer_idx].register_forward_hook(make_hook())
        ablated_logits = get_logits(model, input_ids)
        h.remove()

        kl = kl_divergence(baseline_logits, ablated_logits)
        results[f"ablate_{name}"] = kl
        print(f"    Ablate {name}: KL = {kl:.6f}")

    # Test 2: Scale L5 output by norm ratio and compare to L6 effect
    print(f"\n  Test 2: Scale L5 output by {NORM_RATIO:.1f}x")

    # Get L5 output for reference
    l5_cache = {}
    def l5_hook(module, inp, output):
        l5_cache["hidden"] = output[0].detach().clone()
        return output

    h = model.model.layers[5].register_forward_hook(l5_hook)
    get_logits(model, input_ids)
    h.remove()

    l5_norm = l5_cache["hidden"].float().norm(dim=-1).mean().item()
    print(f"    L5 output norm: {l5_norm:.2f}")

    # Scale L5 by norm_ratio
    def make_scale_hook(scale):
        def hook(module, inp, output):
            hidden = output[0].clone()
            hidden = hidden * scale
            return (hidden,) + output[1:]
        return hook

    h = model.model.layers[5].register_forward_hook(make_scale_hook(NORM_RATIO))
    scaled_logits = get_logits(model, input_ids)
    h.remove()

    kl_scaled = kl_divergence(baseline_logits, scaled_logits)
    results[f"l5_scaled_{NORM_RATIO:.1f}x"] = kl_scaled
    print(f"    L5 scaled {NORM_RATIO:.1f}x: KL = {kl_scaled:.6f}")

    # Test 3: Zero L5, then separately zero L6, compare KL
    print(f"\n  Test 3: Zero-output ablation comparison")
    for layer_idx in [5, 6]:
        def make_zero_hook():
            def hook(module, inp, output):
                hidden = torch.zeros_like(output[0])
                return (hidden,) + output[1:]
            return hook

        h = model.model.layers[layer_idx].register_forward_hook(make_zero_hook())
        zero_logits = get_logits(model, input_ids)
        h.remove()

        kl = kl_divergence(baseline_logits, zero_logits)
        results[f"zero_L{layer_idx}"] = kl
        print(f"    Zero L{layer_idx}: KL = {kl:.6f}")

    # Test 4: Per-position ablation comparison L5 vs L6
    print(f"\n  Test 4: Per-position ablation L5 vs L6")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    pos_results = {}

    for layer_idx in [5, 6]:
        pos_kls = []
        for pos in range(seq_len):
            def make_pos_hook(p):
                def hook(module, inp, output):
                    hidden = output[0].clone()
                    hidden[:, p, :] = 0.0
                    return (hidden,) + output[1:]
                return hook

            h = model.model.layers[layer_idx].register_forward_hook(make_pos_hook(pos))
            ablated = get_logits(model, input_ids)
            h.remove()

            kl = kl_divergence(baseline_logits, ablated)
            pos_kls.append(kl)

        pos_results[f"L{layer_idx}"] = pos_kls
        print(f"    L{layer_idx}: per-pos KL = {[f'{k:.6f}' for k in pos_kls]}")

    results["per_position"] = pos_results
    results["tokens"] = tokens

    # Test 5: Try multiple scaling factors for L5
    print(f"\n  Test 5: L5 scaling sweep")
    scale_sweep = {}
    for scale in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
        def make_scale_hook(s):
            def hook(module, inp, output):
                hidden = output[0].clone() * s
                return (hidden,) + output[1:]
            return hook

        h = model.model.layers[5].register_forward_hook(make_scale_hook(scale))
        scaled = get_logits(model, input_ids)
        h.remove()

        kl = kl_divergence(baseline_logits, scaled)
        scale_sweep[scale] = kl
        print(f"    L5 × {scale:5.1f}: KL = {kl:.6f}")

    # Also get L6 zero-out KL for reference
    results["l5_scale_sweep"] = scale_sweep

    # Find which L5 scale matches L6's zero-out KL
    l6_zero_kl = results["zero_L6"]
    closest_scale = min(scale_sweep.items(), key=lambda x: abs(x[1] - l6_zero_kl))
    print(f"\n    L6 zero-out KL = {l6_zero_kl:.6f}")
    print(f"    Closest L5 scale: ×{closest_scale[0]} (KL={closest_scale[1]:.6f})")
    results["l5_scale_matching_l6"] = {"scale": closest_scale[0], "kl": closest_scale[1]}

    save_json(results, "l6_boundary_analysis_results")
    return results


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print(f"LFM2.5-230M Position-Specific Ablation + Cross-Layer Patching")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print()

    model, tokenizer = load_model()

    all_results = {}

    # 1. Position-specific ablation
    all_results["position_ablation"] = run_position_specific_ablation(model, tokenizer)

    # 2. Cross-layer patching
    all_results["cross_layer_patching"] = run_cross_layer_patching(model, tokenizer)

    # 3. Conv kernel analysis
    all_results["conv_kernel"] = run_conv_kernel_analysis(model, tokenizer)

    # 4. L6 boundary analysis
    all_results["l6_boundary"] = run_l6_boundary_analysis(model, tokenizer)

    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
