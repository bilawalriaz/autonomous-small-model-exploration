#!/usr/bin/env python3
"""Phase 3: Quantization causal surface drift.

Tests whether quantization preserves the causal atlas (hub location, steering
effectiveness) even when benchmark quality is preserved.

Hypothesis: "Quantization may preserve benchmark quality while changing
intervention behaviour." If true, this is a major deployment insight.

Usage:
    python scripts/run_phase3_quantization_atlas.py --model Qwen/Qwen2.5-0.5B --quant 4bit
    python scripts/run_phase3_quantization_atlas.py --model Qwen/Qwen2.5-1.5B --quant 4bit --experiment steering
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def load_model_quantized(model_name, quant_type="4bit"):
    """Load model with specified quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if quant_type == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_type == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_type == "bf16":
        bnb_config = None
    else:
        raise ValueError(f"Unknown quant type: {quant_type}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if quant_type == "bf16" else None,
    )

    return model, tokenizer


def run_layer_ablation_quantized(model, tokenizer, n_layers, quant_type, n_prompts=5):
    """Run layer ablation on a quantized model."""
    from mi_atlas.ablations import zero_ablate_layer

    # Use simple factual prompts
    prompts = [
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Germany is",
        "2 + 2 =",
        "The largest planet in our solar system is",
    ][:n_prompts]

    results = []
    for layer_idx in range(n_layers):
        kl_values = []
        for prompt in prompts:
            try:
                kl = zero_ablate_layer(model, tokenizer, prompt, layer_idx)
                kl_values.append(kl)
            except Exception as e:
                pass

        results.append({
            "layer": layer_idx,
            "mean_kl": float(np.mean(kl_values)) if kl_values else 0,
            "std_kl": float(np.std(kl_values)) if kl_values else 0,
            "n": len(kl_values),
        })

    # Find hub
    if results:
        hub_layer = max(results, key=lambda x: x["mean_kl"])
        mean_per_layer = [r["mean_kl"] for r in results]
    else:
        hub_layer = {"layer": 0, "mean_kl": 0}
        mean_per_layer = []

    return {
        "per_layer": results,
        "hub_layer": hub_layer["layer"],
        "hub_kl": hub_layer["mean_kl"],
        "mean_per_layer": mean_per_layer,
    }


def run_steering_quantized(model, tokenizer, hub_layers, n_prompts=5):
    """Run steering at hub layers on a quantized model."""
    prompts = [
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Germany is",
    ][:n_prompts]

    strengths = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0]
    results = {}

    for layer_idx in hub_layers:
        layer_results = []
        for strength in strengths:
            kl_values = []
            for prompt in prompts:
                try:
                    # Get baseline
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        baseline_logits = model(**inputs).logits[:, -1, :]

                    # Compute a simple steering vector (difference in hidden states)
                    # This is a simplified version — the full version uses clean/corrupt pairs
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hidden = outputs.hidden_states[layer_idx + 1]  # +1 for embedding layer

                    # Use mean activation as steering direction
                    sv = hidden.mean(dim=1).squeeze()  # (d_model,)
                    sv = sv / torch.norm(sv) * abs(strength)

                    # Inject steering
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            modified = output[0] + sv.unsqueeze(0).unsqueeze(0)
                            return (modified,) + output[1:]
                        return output + sv.unsqueeze(0).unsqueeze(0)

                    # Register hook
                    target_layer = model.model.layers[layer_idx]
                    hook = target_layer.register_forward_hook(hook_fn)

                    with torch.no_grad():
                        steered_logits = model(**inputs).logits[:, -1, :]

                    hook.remove()

                    # Compute KL
                    baseline_probs = torch.softmax(baseline_logits, dim=-1)
                    steered_probs = torch.softmax(steered_logits, dim=-1)
                    kl = torch.nn.functional.kl_div(
                        steered_probs.log(), baseline_probs, reduction="sum"
                    ).item()
                    kl_values.append(abs(kl))

                except Exception as e:
                    pass

            if kl_values:
                layer_results.append({
                    "strength": strength,
                    "mean_kl": float(np.mean(kl_values)),
                    "std_kl": float(np.std(kl_values)),
                })

        results[str(layer_idx)] = layer_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 quantization atlas")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "bf16"])
    parser.add_argument("--experiment", type=str, default="ablation", choices=["ablation", "steering", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]

    print(f"Phase 3: Quantization causal surface")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quant}")
    print(f"Experiment: {args.experiment}")

    # Load reference (bf16) results if they exist
    ref_path = PROJECT_ROOT / "experiments" / "results" / "layer_ablation_zero.json"
    ref_results = None
    if ref_path.exists():
        with open(ref_path) as f:
            ref_results = json.load(f)
        print(f"  Reference bf16 results loaded: hub at L{ref_results.get('top_layers', ['?'])[0] if 'top_layers' in ref_results else '?'}")

    # Load quantized model
    print(f"\nLoading {args.quant} model...")
    start = time.time()
    model, tokenizer = load_model_quantized(args.model, args.quant)
    load_time = time.time() - start
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded in {load_time:.1f}s, {n_layers} layers")

    # Measure VRAM
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  VRAM: {vram_mb:.0f}MB")

    results = {
        "experiment": "quantization_atlas",
        "model": args.model,
        "quant": args.quant,
        "n_layers": n_layers,
        "load_time_seconds": round(load_time, 1),
        "vram_mb": round(vram_mb, 0) if torch.cuda.is_available() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if args.experiment in ("ablation", "both"):
        print("\nRunning layer ablation...")
        ablation_results = run_layer_ablation_quantized(model, tokenizer, n_layers, args.quant)
        results["ablation"] = ablation_results
        print(f"  Hub: L{ablation_results['hub_layer']} (KL={ablation_results['hub_kl']:.3f})")

        # Compare to reference
        if ref_results and "top_layers" in ref_results:
            ref_hub = ref_results["top_layers"][0] if isinstance(ref_results["top_layers"], list) else ref_results["top_layers"]
            results["hub_shift"] = ablation_results["hub_layer"] - ref_hub
            print(f"  Hub shift from bf16: {results['hub_shift']} layers")

    if args.experiment in ("steering", "both"):
        # Use hub layers from ablation or reference
        hub_layers = [results.get("ablation", {}).get("hub_layer", 2)]
        if ref_results and "top_layers" in ref_results:
            ref_hub = ref_results["top_layers"][0] if isinstance(ref_results["top_layers"], list) else ref_results["top_layers"]
            hub_layers.append(ref_hub)

        hub_layers = sorted(set(hub_layers))
        print(f"\nRunning steering at layers {hub_layers}...")
        steering_results = run_steering_quantized(model, tokenizer, hub_layers)
        results["steering"] = steering_results

    # Save
    out_path = PROJECT_ROOT / "experiments" / "results" / f"phase3_quant_atlas_{model_slug}_{args.quant}.json"
    save_json(results, out_path)
    print(f"\n  Results: {out_path}")

    register_experiment(
        type="quantization",
        model=args.model,
        backend="hf",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(out_path)],
        status="success",
        summary=f"Quantization atlas ({args.quant}): hub_shift={results.get('hub_shift', '?')}, "
                f"vram={results.get('vram_mb', '?')}MB",
        next="Compare steering effectiveness across quant levels",
    )


if __name__ == "__main__":
    main()
