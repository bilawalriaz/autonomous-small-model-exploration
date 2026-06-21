"""Steering vector experiments on critical layers.

Compute steering vectors as mean(positive) - mean(negative) activations,
then inject at different strengths and measure behaviour change.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.plotting import plot_line, plot_multi_line
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Get activation at a specific layer and position."""
    activation = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()

    return activation.get("value")


def compute_steering_vector(model, tokenizer, positive_prompts, negative_prompts, layer_idx, position=-1):
    """Compute steering vector as mean(positive) - mean(negative)."""
    pos_acts = []
    neg_acts = []

    for prompt in positive_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
        act = get_activation_at_layer(model, ids, layer_idx, position)
        if act is not None:
            pos_acts.append(act.cpu())

    for prompt in negative_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)
        act = get_activation_at_layer(model, ids, layer_idx, position)
        if act is not None:
            neg_acts.append(act.cpu())

    if not pos_acts or not neg_acts:
        return None

    mean_pos = torch.stack(pos_acts).mean(dim=0)
    mean_neg = torch.stack(neg_acts).mean(dim=0)

    return (mean_pos - mean_neg).squeeze(0)  # (d_model,)


def inject_and_measure(model, tokenizer, prompt, layer_idx, steering_vector, strength, position=-1):
    """Inject steering vector and measure effect on next token distribution."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)

    # Original
    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
    orig_top5 = torch.topk(orig_probs, 5)

    # Steered
    sv = steering_vector.to(model.device) * strength

    def steer_hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hidden[:, position, :] += sv
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(steer_hook)
    with torch.no_grad():
        steered_logits = model(ids).logits
    handle.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)
    steered_top5 = torch.topk(steered_probs, 5)

    # KL divergence
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(steered_logits[0, -1], dim=-1),
        orig_probs,
        reduction="sum"
    ).item()

    # Decode top tokens
    orig_top_tokens = [(tokenizer.decode([tid]), prob.item()) for tid, prob in zip(orig_top5.indices, orig_top5.values)]
    steered_top_tokens = [(tokenizer.decode([tid]), prob.item()) for tid, prob in zip(steered_top5.indices, steered_top5.values)]

    return {
        "strength": strength,
        "kl_divergence": kl,
        "orig_top5": orig_top_tokens,
        "steered_top5": steered_top_tokens,
    }


def main():
    set_seed(42)

    print("Loading model...")
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    model = bundle.model
    tokenizer = bundle.tokenizer
    model.eval()

    # Define steering experiments
    experiments = [
        {
            "name": "json_validity",
            "layer": 21,  # L21H7 was specialist for json
            "positive": [
                'Return valid JSON: {"name": "Alice", "age": 31}',
                'Return valid JSON: {"x": 1, "y": 2}',
                '{"name": "Bob", "age": 25}',
            ],
            "negative": [
                "Tell me about Alice who is 31 years old.",
                "What are the values of x and y?",
                "Describe a person named Bob, age 25.",
            ],
            "test_prompts": [
                'Return exactly valid JSON with keys name and age. Eve is 42.\n',
                'Return valid JSON: {"city": "London"',
            ],
        },
        {
            "name": "factual_recall",
            "layer": 2,  # L2H3 was specialist for factual
            "positive": [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Japan is Tokyo.",
            ],
            "negative": [
                "France is a beautiful country in Europe.",
                "Germany has many famous cities.",
                "Japan is an island nation in Asia.",
            ],
            "test_prompts": [
                "The capital of Italy is ",
                "The capital of Spain is ",
            ],
        },
        {
            "name": "delimiter_closing",
            "layer": 21,  # L21H7 also specialist for delimiter
            "positive": [
                "function(x, [y, {z}])",
                "data = {key: [1, 2, 3]}",
                "result = func(a, (b, c))",
            ],
            "negative": [
                "function processes data and returns results.",
                "data is stored in a dictionary.",
                "result depends on the input parameters.",
            ],
            "test_prompts": [
                "Complete the closing delimiters: function(x, [y, {z",
                "Close the brackets: data = [1, {a: (",
            ],
        },
    ]

    all_results = []

    for exp in experiments:
        print(f"\n  === {exp['name'].upper()} (layer {exp['layer']}) ===")

        sv = compute_steering_vector(
            model, tokenizer,
            exp["positive"], exp["negative"],
            exp["layer"]
        )

        if sv is None:
            print("    Failed to compute steering vector")
            continue

        sv_norm = sv.norm().item()
        print(f"    Steering vector norm: {sv_norm:.4f}")

        strengths = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
        sweep_results = []

        for test_prompt in exp["test_prompts"]:
            print(f"\n    Prompt: '{test_prompt[:50]}...'")
            prompt_results = []

            for strength in strengths:
                result = inject_and_measure(
                    model, tokenizer, test_prompt,
                    exp["layer"], sv, strength
                )
                prompt_results.append(result)

                if strength == 0.0:
                    print(f"      s={strength:+.1f}: BASELINE {result['orig_top5'][:3]}")
                else:
                    print(f"      s={strength:+.1f}: KL={result['kl_divergence']:.3f}, top={result['steered_top5'][:3]}")

            sweep_results.append({
                "prompt": test_prompt,
                "results": prompt_results,
            })

        all_results.append({
            "name": exp["name"],
            "layer": exp["layer"],
            "sv_norm": sv_norm,
            "sweeps": sweep_results,
        })

    # Save
    output_path = PROJECT_ROOT / "experiments" / "results" / "steering_sweep.json"
    save_json(all_results, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="steering",
        model=bundle.model_name,
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[],
        outputs=[str(output_path)],
        status="success",
        summary=f"Steering sweep: {len(experiments)} experiments, layers 2 and 21",
        next="SFT training, better clean/corrupt pairs",
    )
    print("  Steering experiments complete!")


if __name__ == "__main__":
    main()
