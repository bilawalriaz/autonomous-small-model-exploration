#!/usr/bin/env python3
"""
LFM2.5-230M Correct Layer Atlas
================================
Ablation hooks the OPERATOR (conv/attn) and MLP separately, NOT the full decoder layer.
Zeroing the full layer output cascades zeros through the residual stream, making all layers
appear identical. The correct approach is to zero the operator or MLP contribution while
preserving the residual pass-through.

Three ablation modes per layer:
1. Operator zero: Zero conv/self_attn output. Layer output = residual + 0 + ffn(norm(residual))
2. MLP zero: Zero feed_forward output. Layer output = residual + operator(norm(residual)) + 0
3. Layer skip: Replace layer output with input (identity). Layer output = input

Additionally for conv layers:
4. Gate zero: Zero the B (gate) component in Lfm2ShortConv
5. Carrier zero: Zero the C (carrier) component in Lfm2ShortConv
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_LAYERS = 14
HIDDEN_DIM = 1024
LAYER_TYPES = ['conv', 'conv', 'full_attention', 'conv', 'full_attention', 'conv',
               'full_attention', 'conv', 'full_attention', 'conv', 'full_attention',
               'conv', 'full_attention', 'conv']
CONV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13]
ATTN_LAYERS = [2, 4, 6, 8, 10, 12]

# ── Task suite ──────────────────────────────────────────────────
TASK_FAMILIES = [
    "arithmetic", "code_semantics", "code_syntax", "copying",
    "dead_code", "delimiter_tracking", "factual_recall",
    "instruction_following", "json_schema", "string_decoding",
    "variable_renaming", "constant_folding", "uncertainty_expression",
    "verbosity_control", "harmless_refusal", "control_flow_simplification"
]


def load_prompts(n_per_family=5):
    """Load task prompts from the task suite."""
    prompts_by_family = {}
    task_dir = PROJECT_ROOT / "data" / "tasks" / "canonical_short"
    if not task_dir.exists():
        task_dir = PROJECT_ROOT / "data" / "tasks"

    import glob as glob_mod
    for family in TASK_FAMILIES:
        fpath = task_dir / f"{family}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        examples = data.get("examples", []) if isinstance(data, dict) else data
        # Use test split examples
        test_examples = [e for e in examples if e.get("metadata", {}).get("split") == "test"]
        if len(test_examples) < n_per_family:
            test_examples = examples[:n_per_family]
        prompts = [{"prompt": e.get("prompt", e.get("clean_prompt", "")),
                     "target": e.get("target", "")}
                    for e in test_examples[:n_per_family]]
        if prompts:
            prompts_by_family[family] = prompts
    return prompts_by_family


def compute_kl(baseline_logits, abl_logits):
    """KL divergence between baseline and ablated output distributions."""
    baseline_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    abl_log_probs = torch.log_softmax(abl_logits.float(), dim=-1)
    kl = torch.nn.functional.kl_div(
        abl_log_probs, baseline_log_probs.exp(),
        reduction='batchmean'
    ).item()
    return kl


def compute_top1_agreement(baseline_logits, abl_logits):
    """Fraction of tokens where top-1 prediction agrees."""
    baseline_top1 = baseline_logits.argmax(dim=-1)
    abl_top1 = abl_logits.argmax(dim=-1)
    return (baseline_top1 == abl_top1).float().mean().item()


# ── Ablation hooks ──────────────────────────────────────────────

def make_operator_zero_hook():
    """Zero the operator (conv or self_attn) output.
    The layer then becomes: residual + 0 + ffn(norm(residual))"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


def make_mlp_zero_hook():
    """Zero the MLP output.
    The layer then becomes: residual + operator(norm(residual)) + 0"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


def make_layer_skip_hook():
    """Skip the layer entirely — output = input (identity).
    This is the correct 'layer contribution' ablation."""
    def hook(module, input, output):
        # input[0] is the hidden_states passed to the decoder layer
        input_hidden = input[0]
        if isinstance(output, tuple):
            return (input_hidden,) + output[1:]
        return input_hidden
    return hook


# For conv gate/carrier analysis — needs custom forward replacement
def make_conv_gate_zero_hook():
    """Zero the B (gate) component in Lfm2ShortConv.
    B is the first 1024 dims of in_proj output.
    Forward: projected = in_proj(x), B,C,x = chunk(3), Bx = B*x, conv_out = conv1d(Bx), y = C*conv_out, output = out_proj(y)
    With B=0: Bx=0, conv_out=0, y=0, output=0.
    This zeros the entire conv output (expected).
    More useful: zero only the gating (B*x) by setting B=ones (identity gate)."""
    # Actually, hooking conv.in_proj and modifying its output
    # projected shape: [batch, seq, 3072]. B is [:, :, :1024]
    def hook(module, input, output):
        # output is [batch, seq, 3072]
        modified = output.clone()
        modified[:, :, :1024] = 0  # Zero B (gate)
        return modified
    return hook


def make_conv_carrier_zero_hook():
    """Zero the C (carrier) component in Lfm2ShortConv.
    C is the middle 1024 dims of in_proj output (indices 1024:2048)."""
    def hook(module, input, output):
        modified = output.clone()
        modified[:, :, 1024:2048] = 0  # Zero C (carrier)
        return modified
    return hook


def make_conv_signal_zero_hook():
    """Zero the x (signal) component in Lfm2ShortConv.
    x is the last 1024 dims of in_proj output (indices 2048:3072)."""
    def hook(module, input, output):
        modified = output.clone()
        modified[:, :, 2048:] = 0  # Zero x (signal)
        return modified
    return hook


# ── Experiment functions ────────────────────────────────────────

def run_ablation_experiment(model, tokenizer, prompts_by_family, device,
                            ablation_name, get_hook_fn, get_target_module):
    """Run an ablation experiment across all layers and families.

    Args:
        get_hook_fn: () -> hook_function
        get_target_module: (model, layer_idx) -> nn.Module to hook
    """
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {ablation_name}")
    print(f"{'=' * 60}")

    results = {}

    for family, prompts in prompts_by_family.items():
        print(f"\n  Family: {family} ({len(prompts)} prompts)")
        family_kls = []
        family_agree = []

        for pi, p in enumerate(prompts):
            prompt = p["prompt"]
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            # Baseline
            with torch.no_grad():
                baseline_logits = model(ids).logits

            layer_kls = []
            layer_agree = []

            for layer_idx in range(NUM_LAYERS):
                target_module = get_target_module(model, layer_idx)
                if target_module is None:
                    layer_kls.append(0.0)
                    layer_agree.append(1.0)
                    continue

                hook_fn = get_hook_fn()
                handle = target_module.register_forward_hook(hook_fn)
                with torch.no_grad():
                    abl_logits = model(ids).logits
                handle.remove()

                kl = compute_kl(baseline_logits, abl_logits)
                agree = compute_top1_agreement(baseline_logits, abl_logits)
                layer_kls.append(kl)
                layer_agree.append(agree)

            family_kls.append(layer_kls)
            family_agree.append(layer_agree)

            if pi == 0:
                top3 = sorted(range(NUM_LAYERS), key=lambda i: layer_kls[i], reverse=True)[:3]
                top3_str = [(f'L{l}({LAYER_TYPES[l][:1]})', round(layer_kls[l], 2)) for l in top3]
                print(f"    Prompt[0] top3: {top3_str}")
                # Also print all layers
                print(f"    All layers KL: {[round(k, 2) for k in layer_kls]}")

        mean_kls = np.mean(family_kls, axis=0).tolist()
        mean_agree = np.mean(family_agree, axis=0).tolist()
        results[family] = {
            "kl_per_layer": [round(x, 4) for x in mean_kls],
            "top1_agreement": [round(x, 4) for x in mean_agree],
        }

    # Summary
    all_kls = np.array([results[f]["kl_per_layer"] for f in results])
    mean_kl = all_kls.mean(axis=0).tolist()
    hub_layer = int(np.argmax(mean_kl))
    hub_type = LAYER_TYPES[hub_layer]
    print(f"\n  HUB: L{hub_layer} ({hub_type}) with mean KL = {mean_kl[hub_layer]:.4f}")
    print(f"  Mean KL per layer: {[round(k, 2) for k in mean_kl]}")

    return results, mean_kl


def get_operator_module(model, layer_idx):
    """Get the operator module (conv or self_attn) for a given layer."""
    layer = model.model.layers[layer_idx]
    if hasattr(layer, 'conv') and layer.conv is not None:
        return layer.conv
    elif hasattr(layer, 'self_attn') and layer.self_attn is not None:
        return layer.self_attn
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-prompts", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True
    )
    model.eval()

    print(f"  Vocab: {tokenizer.vocab_size}, Layers: {NUM_LAYERS}")
    print(f"  Conv layers: {CONV_LAYERS}")
    print(f"  Attn layers: {ATTN_LAYERS}")
    print(f"  Device: {device}, VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    # Load prompts
    prompts_by_family = load_prompts(n_per_family=args.n_prompts)
    print(f"  Loaded {sum(len(v) for v in prompts_by_family.values())} prompts across {len(prompts_by_family)} families")

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Experiment 1: OPERATOR zero ablation ──
    # Zeros the conv or self_attn output. Layer becomes: residual + 0 + ffn(norm(residual))
    results_op, mean_kl_op = run_ablation_experiment(
        model, tokenizer, prompts_by_family, device,
        ablation_name="OPERATOR ZERO (conv/attn output → 0)",
        get_hook_fn=make_operator_zero_hook,
        get_target_module=get_operator_module,
    )
    all_results["operator_zero"] = results_op

    # ── Experiment 2: MLP zero ablation ──
    # Zeros the feed_forward output. Layer becomes: residual + operator(norm(residual)) + 0
    def get_mlp(model, idx):
        return model.model.layers[idx].feed_forward
    results_mlp, mean_kl_mlp = run_ablation_experiment(
        model, tokenizer, prompts_by_family, device,
        ablation_name="MLP ZERO (feed_forward output → 0)",
        get_hook_fn=make_mlp_zero_hook,
        get_target_module=get_mlp,
    )
    all_results["mlp_zero"] = results_mlp

    # ── Experiment 3: LAYER SKIP (identity) ──
    # Replaces layer output with input. Layer does nothing.
    def get_decoder(model, idx):
        return model.model.layers[idx]
    results_skip, mean_kl_skip = run_ablation_experiment(
        model, tokenizer, prompts_by_family, device,
        ablation_name="LAYER SKIP (output = input, identity)",
        get_hook_fn=make_layer_skip_hook,
        get_target_module=get_decoder,
    )
    all_results["layer_skip"] = results_skip

    # ── Experiment 4: Conv gate analysis (in_proj hook) ──
    print(f"\n{'=' * 60}")
    print("EXPERIMENT: CONV GATE/CARRIER/SIGNAL ANALYSIS")
    print(f"{'=' * 60}")
    conv_results = {}
    for component_name, hook_fn_factory in [
        ("gate_zero", make_conv_gate_zero_hook),
        ("carrier_zero", make_conv_carrier_zero_hook),
        ("signal_zero", make_conv_signal_zero_hook),
    ]:
        print(f"\n  Component: {component_name}")
        component_kls = {}
        for family, prompts in prompts_by_family.items():
            p = prompts[0]
            ids = tokenizer(p["prompt"], return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            with torch.no_grad():
                baseline = model(ids).logits

            layer_kls = []
            for layer_idx in CONV_LAYERS:
                in_proj = model.model.layers[layer_idx].conv.in_proj
                hook = hook_fn_factory()
                handle = in_proj.register_forward_hook(hook)
                with torch.no_grad():
                    abl = model(ids).logits
                handle.remove()
                kl = compute_kl(baseline, abl)
                layer_kls.append(round(kl, 4))

            component_kls[family] = layer_kls
            if family == list(prompts_by_family.keys())[0]:
                print(f"    {family}: {[round(k, 2) for k in layer_kls]}")

        conv_results[component_name] = component_kls

    all_results["conv_gate_analysis"] = conv_results

    # ── Experiment 5: Steered residual norms ──
    print(f"\n{'=' * 60}")
    print("EXPERIMENT: RESIDUAL STREAM NORM TRACKING")
    print(f"{'=' * 60}")
    norm_results = {}
    for family, prompts in prompts_by_family.items():
        p = prompts[0]
        ids = tokenizer(p["prompt"], return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)

        norms = []
        hooks = []
        def capture_norm(layer_idx):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    norm = output.float().norm().item()
                    norms.append((layer_idx, norm))
                elif isinstance(output, tuple):
                    norm = output[0].float().norm().item()
                    norms.append((layer_idx, norm))
            return hook

        for i in range(NUM_LAYERS):
            handle = model.model.layers[i].register_forward_hook(capture_norm(i))
            hooks.append(handle)

        with torch.no_grad():
            model(ids)
        for h in hooks:
            h.remove()

        if family == list(prompts_by_family.keys())[0]:
            print(f"  {family} norms: {[(f'L{i}', round(n, 2)) for i, n in norms]}")
        norm_results[family] = {f"L{i}": round(n, 4) for i, n in norms}

    all_results["residual_norms"] = norm_results

    # ── Experiment 6: Steering sweep ──
    print(f"\n{'=' * 60}")
    print("EXPERIMENT: STEERING SWEEP (all 14 layers)")
    print(f"{'=' * 60}")
    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
    # Use a simple prompt for steering
    steer_prompt = "The capital of France is"
    steer_ids = tokenizer(steer_prompt, return_tensors="pt").input_ids.to(device)

    # Get a random direction (same as existing MI-Atlas approach)
    torch.manual_seed(args.seed)
    direction = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    direction = direction / direction.norm()

    with torch.no_grad():
        baseline_steer = model(steer_ids).logits

    steer_results = {}
    for layer_idx in range(NUM_LAYERS):
        layer_kls = []
        for s in strengths:
            sv = direction * s

            def make_hook(vec):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        out = output.clone()
                        out[:, -1, :] = out[:, -1, :] + vec  # Steer last token
                        return out
                    elif isinstance(output, tuple):
                        out = output[0].clone()
                        out[:, -1, :] = out[:, -1, :] + vec
                        return (out,) + output[1:]
                return hook

            handle = model.model.layers[layer_idx].register_forward_hook(make_hook(sv))
            with torch.no_grad():
                steer_logits = model(steer_ids).logits
            handle.remove()

            kl = compute_kl(baseline_steer, steer_logits)
            layer_kls.append(round(kl, 4))

        steer_results[f"L{layer_idx}({LAYER_TYPES[layer_idx][:1]})"] = layer_kls
        best_s = strengths[layer_kls.index(max(layer_kls))]
        print(f"  L{layer_idx} ({LAYER_TYPES[layer_idx][:3]}): max KL={max(layer_kls):.4f} at s={best_s}")

    all_results["steering_sweep"] = steer_results

    # ── Save results ──
    out_path = RESULTS_DIR / f"lfm2_230m_correct_atlas_seed{args.seed}_{timestamp}.json"
    output = {
        "model": args.model,
        "seed": args.seed,
        "timestamp": timestamp,
        "n_layers": NUM_LAYERS,
        "layer_types": LAYER_TYPES,
        "conv_layers": CONV_LAYERS,
        "attn_layers": ATTN_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "results": all_results,
        "summary": {
            "operator_zero_mean_kl": [round(k, 4) for k in mean_kl_op],
            "mlp_zero_mean_kl": [round(k, 4) for k in mean_kl_mlp],
            "layer_skip_mean_kl": [round(k, 4) for k in mean_kl_skip],
            "operator_hub": f"L{int(np.argmax(mean_kl_op))}({LAYER_TYPES[int(np.argmax(mean_kl_op))]})",
            "mlp_hub": f"L{int(np.argmax(mean_kl_mlp))}({LAYER_TYPES[int(np.argmax(mean_kl_mlp))]})",
            "skip_hub": f"L{int(np.argmax(mean_kl_skip))}({LAYER_TYPES[int(np.argmax(mean_kl_skip))]})",
        }
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Operator hub: {output['summary']['operator_hub']}")
    print(f"MLP hub: {output['summary']['mlp_hub']}")
    print(f"Skip hub: {output['summary']['skip_hub']}")
    print(f"\nOperator KL per layer: {[round(k, 2) for k in mean_kl_op]}")
    print(f"MLP KL per layer: {[round(k, 2) for k in mean_kl_mlp]}")
    print(f"Skip KL per layer: {[round(k, 2) for k in mean_kl_skip]}")

    # Conv vs Attn comparison
    conv_op_kls = [mean_kl_op[i] for i in CONV_LAYERS]
    attn_op_kls = [mean_kl_op[i] for i in ATTN_LAYERS]
    print(f"\nConv operator mean KL: {np.mean(conv_op_kls):.4f}")
    print(f"Attn operator mean KL: {np.mean(attn_op_kls):.4f}")
    print(f"Ratio (attn/conv): {np.mean(attn_op_kls)/max(np.mean(conv_op_kls), 1e-8):.2f}x")

    conv_mlp_kls = [mean_kl_mlp[i] for i in CONV_LAYERS]
    attn_mlp_kls = [mean_kl_mlp[i] for i in ATTN_LAYERS]
    print(f"\nConv MLP mean KL: {np.mean(conv_mlp_kls):.4f}")
    print(f"Attn MLP mean KL: {np.mean(attn_mlp_kls):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
