#!/usr/bin/env python3
"""
LFM2.5-230M Head Ablation + Conv Gate Analysis

Comprehensive analysis script that:
1. Ablates individual attention heads in all full_attention layers
2. Analyzes conv gate/carrier contributions in all conv layers
3. Compares attention vs conv layer importance

Usage:
    python run_lfm2_head_conv_analysis.py [--model LiquidAI/LFM2.5-230M] [--force] [--seed 42]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Constants ──────────────────────────────────────────────────────────────

LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv"
]

ATTN_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
CONV_LAYERS = [i for i, t in enumerate(LAYER_TYPES) if t == "conv"]

NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = 1024

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    '{"key":',
    "The opposite of hot is",
    "The quick brown fox",
    "In the beginning,",
    "Water boils at",
    "The largest planet is",
]


def parse_args():
    p = argparse.ArgumentParser(description="LFM2.5-230M head ablation + conv gate analysis")
    p.add_argument("--model", default="LiquidAI/LFM2.5-230M", help="Model name or path")
    p.add_argument("--force", action="store_true", help="Overwrite existing results")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name):
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def get_logits(model, input_ids, attention_mask=None):
    """Get logits with no ablation (baseline)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits


def compute_kl(baseline_logits, ablated_logits, mask=None):
    """KL(ablated || baseline) per-token, averaged over non-masked positions."""
    # baseline_logits, ablated_logits: [batch, seq, vocab]
    log_p = F.log_softmax(baseline_logits.float(), dim=-1)
    log_q = F.log_softmax(ablated_logits.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [batch, seq]
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum()
    return kl.mean()


def run_head_ablation(model, input_ids, baseline_logits, attention_mask=None):
    """
    Ablate each head in each attention layer individually.
    Returns dict: {layer_idx: {head_idx: kl_div, ...}}
    """
    results = {}
    for layer_idx in ATTN_LAYERS:
        print(f"  Head ablation layer {layer_idx} ...")
        results[layer_idx] = {}
        attn_module = model.model.layers[layer_idx].self_attn

        for head_idx in range(NUM_Q_HEADS):
            def make_hook(h_idx):
                def hook_fn(module, inp, output):
                    # output is (attn_output, attn_weights, past_kv) or just tensor
                    if isinstance(output, tuple):
                        attn_out = output[0]
                    else:
                        attn_out = output
                    # attn_out: [batch, seq, hidden_dim]
                    b, s, h = attn_out.shape
                    out = attn_out.reshape(b, s, NUM_Q_HEADS, HEAD_DIM)
                    out[:, :, h_idx, :] = 0.0
                    out = out.reshape(b, s, h)
                    if isinstance(output, tuple):
                        return (out,) + output[1:]
                    return out
                return hook_fn

            hook = attn_module.register_forward_hook(make_hook(head_idx))

            with torch.no_grad():
                ablated = model(input_ids=input_ids, attention_mask=attention_mask).logits

            hook.remove()

            kl = compute_kl(baseline_logits, ablated).item()
            results[layer_idx][head_idx] = kl

        head_kls = list(results[layer_idx].values())
        print(f"    KL range: {min(head_kls):.6f} - {max(head_kls):.6f}")

    return results


def run_conv_gate_analysis(model, input_ids, baseline_logits, attention_mask=None):
    """
    Ablate conv gate (B), carrier (C), and full conv output individually.
    Monkey-patches conv module forward to intercept intermediate tensors.
    Lfm2ShortConv.forward:
        projected = self.in_proj(hidden_states)
        B, C, x = projected.chunk(3, dim=-1)
        Bx = B * x
        conv_out = conv_module.conv(Bx.transpose(1,2)).transpose(1,2)
        conv_out = act(conv_out)
        y = C * conv_out
        return out_proj(y)
    """
    results = {}
    for layer_idx in CONV_LAYERS:
        print(f"  Conv gate analysis layer {layer_idx} ...")
        conv_module = model.model.layers[layer_idx].conv
        layer_result = {}

        original_forward = conv_module.forward

        # Ablate gate B (zero it -> Bx = 0 -> conv gets all zeros)
        def ablate_gate_forward(hidden_states, **kwargs):
            projected = conv_module.in_proj(hidden_states)
            B, C, x = projected.chunk(3, dim=-1)
            B = torch.zeros_like(B)
            Bx = B * x
            conv_out = conv_module.conv(Bx.transpose(1, 2)).transpose(1, 2)
            if hasattr(conv_module, 'act') and conv_module.act is not None:
                conv_out = conv_module.act(conv_out)
            y = C * conv_out
            return conv_module.out_proj(y)

        # Ablate carrier C (zero it -> y=0 -> out_proj(0) = bias or 0)
        def ablate_carrier_forward(hidden_states, **kwargs):
            projected = conv_module.in_proj(hidden_states)
            B, C, x = projected.chunk(3, dim=-1)
            C = torch.zeros_like(C)
            Bx = B * x
            conv_out = conv_module.conv(Bx.transpose(1, 2)).transpose(1, 2)
            if hasattr(conv_module, 'act') and conv_module.act is not None:
                conv_out = conv_module.act(conv_out)
            y = C * conv_out
            return conv_module.out_proj(y)

        # Ablate conv output (zero after conv1d, before carrier multiply)
        def ablate_convout_forward(hidden_states, **kwargs):
            projected = conv_module.in_proj(hidden_states)
            B, C, x = projected.chunk(3, dim=-1)
            Bx = B * x
            conv_out = conv_module.conv(Bx.transpose(1, 2)).transpose(1, 2)
            conv_out = torch.zeros_like(conv_out)
            if hasattr(conv_module, 'act') and conv_module.act is not None:
                conv_out = conv_module.act(conv_out)
            y = C * conv_out
            return conv_module.out_proj(y)

        # Ablate x (the third chunk, the "input" to the gate)
        def ablate_x_forward(hidden_states, **kwargs):
            projected = conv_module.in_proj(hidden_states)
            B, C, x = projected.chunk(3, dim=-1)
            x = torch.zeros_like(x)
            Bx = B * x
            conv_out = conv_module.conv(Bx.transpose(1, 2)).transpose(1, 2)
            if hasattr(conv_module, 'act') and conv_module.act is not None:
                conv_out = conv_module.act(conv_out)
            y = C * conv_out
            return conv_module.out_proj(y)

        for label, replacement_fn in [
            ("gate_B_zero", ablate_gate_forward),
            ("carrier_C_zero", ablate_carrier_forward),
            ("convout_zero", ablate_convout_forward),
            ("input_x_zero", ablate_x_forward),
        ]:
            conv_module.forward = replacement_fn
            with torch.no_grad():
                ablated = model(input_ids=input_ids, attention_mask=attention_mask).logits
            conv_module.forward = original_forward

            kl = compute_kl(baseline_logits, ablated).item()
            layer_result[label] = kl
            print(f"    {label}: KL = {kl:.6f}")

        results[layer_idx] = layer_result

    return results


def run_full_layer_ablation(model, input_ids, baseline_logits, layer_indices, label, attention_mask=None):
    """Ablate entire layers by zeroing their output."""
    results = {}
    for layer_idx in layer_indices:
        layer_module = model.model.layers[layer_idx]

        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)

        hook_handle = layer_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            ablated = model(input_ids=input_ids, attention_mask=attention_mask).logits
        hook_handle.remove()

        kl = compute_kl(baseline_logits, ablated).item()
        results[layer_idx] = kl
        print(f"  {label} L{layer_idx}: KL = {kl:.6f}")
    return results


def run_attn_vs_conv_comparison(model, input_ids, baseline_logits, attention_mask=None):
    """Compare ablating all attn layers vs all conv layers."""
    print("  Comparing attn vs conv full-layer ablation ...")
    attn_full = run_full_layer_ablation(
        model, input_ids, baseline_logits, ATTN_LAYERS, "attn_full", attention_mask
    )
    conv_full = run_full_layer_ablation(
        model, input_ids, baseline_logits, CONV_LAYERS, "conv_full", attention_mask
    )

    total_attn = sum(attn_full.values())
    total_conv = sum(conv_full.values())
    print(f"  Total attn KL: {total_attn:.6f}")
    print(f"  Total conv KL: {total_conv:.6f}")
    return {
        "attn_per_layer": attn_full,
        "conv_per_layer": conv_full,
        "attn_total": total_attn,
        "conv_total": total_conv,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(__file__).resolve().parent.parent / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"lfm2_230m_head_conv_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    if out_path.exists() and not args.force:
        print(f"Results exist at {out_path}, use --force to overwrite")
        sys.exit(0)

    model, tok = load_model(args.model)

    # Prepare inputs
    encodings = tok(PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = encodings.input_ids.to(model.device)
    attention_mask = encodings.attention_mask.to(model.device)

    print(f"Input shape: {input_ids.shape}")
    print(f"Attention layers: {ATTN_LAYERS}")
    print(f"Conv layers: {CONV_LAYERS}")

    # Baseline logits
    print("\n[1/4] Getting baseline logits ...")
    baseline_logits = get_logits(model, input_ids, attention_mask)

    # Head ablation
    print("\n[2/4] Head ablation ...")
    head_results = run_head_ablation(model, input_ids, baseline_logits, attention_mask)

    # Conv gate analysis
    print("\n[3/4] Conv gate analysis ...")
    conv_gate_results = run_conv_gate_analysis(model, input_ids, baseline_logits, attention_mask)

    # Attn vs conv comparison
    print("\n[4/4] Attn vs conv comparison ...")
    comparison = run_attn_vs_conv_comparison(model, input_ids, baseline_logits, attention_mask)

    # Serialize results (convert int keys to strings for JSON)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        return str(obj)

    all_results = {
        "metadata": {
            "model": args.model,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompts": PROMPTS,
            "input_shape": list(input_ids.shape),
            "layer_types": LAYER_TYPES,
            "attn_layers": ATTN_LAYERS,
            "conv_layers": CONV_LAYERS,
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "head_ablation": to_serializable(head_results),
        "conv_gate_analysis": to_serializable(conv_gate_results),
        "attn_vs_conv_comparison": to_serializable(comparison),
    }

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n── Head Ablation Summary ──")
    for layer_idx in ATTN_LAYERS:
        kls = head_results[layer_idx]
        top = max(kls, key=kls.get)
        print(f"  L{layer_idx}: max head={top} (KL={kls[top]:.6f}), "
              f"mean={sum(kls.values())/len(kls):.6f}")

    print("\n── Conv Gate Summary ──")
    for layer_idx in CONV_LAYERS:
        r = conv_gate_results[layer_idx]
        print(f"  L{layer_idx}: gate={r['gate_B_zero']:.6f}, "
              f"carrier={r['carrier_C_zero']:.6f}, "
              f"convout={r['convout_zero']:.6f}, "
              f"input_x={r['input_x_zero']:.6f}")

    print("\n── Attn vs Conv ──")
    print(f"  Attn total KL: {comparison['attn_total']:.6f}")
    print(f"  Conv total KL: {comparison['conv_total']:.6f}")
    if comparison['conv_total'] > 0:
        ratio = comparison['attn_total'] / comparison['conv_total']
        print(f"  Attn/Conv ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
