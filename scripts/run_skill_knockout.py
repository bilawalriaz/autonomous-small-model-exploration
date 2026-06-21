"""exp_000020: Negative steering / skill knockout on trained model.

Can we selectively suppress a learned skill by applying negative steering vectors?

Tests:
1. Compute steering vectors for JSON skill (positive=JSON, negative=plain text)
2. Apply NEGATIVE steering at various strengths to the TRAINED model
3. Measure: does the trained model lose its JSON capability?
4. Test at layers L2, L7, L9 (core circuit), L6, L12, L13 (JSON concentration), L21 (late)
5. Also test factual_recall skill knockout

Key question: can we selectively remove ONE skill without destroying others?
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mi_atlas.model_loader import load_model_hf
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT

from peft import PeftModel


def get_layers(model):
    """Get the transformer layers list, handling PeftModel wrapping."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Get activation at a specific layer and position."""
    activation = {}
    layers = get_layers(model)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
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
    return (mean_pos - mean_neg).squeeze(0)


def inject_steering(model, tokenizer, prompt, layer_idx, steering_vector, strength, position=-1):
    """Inject steering vector and measure effect."""
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(model.device)

    # Original
    with torch.no_grad():
        orig_logits = model(ids).logits
    orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)

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

    handle = get_layers(model)[layer_idx].register_forward_hook(steer_hook)
    with torch.no_grad():
        steered_logits = model(ids).logits
    handle.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)

    # KL divergence
    kl = torch.nn.functional.kl_div(
        torch.log(steered_probs), orig_probs, reduction="sum"
    ).item()

    # Top 5 tokens
    orig_top5 = torch.topk(orig_probs, 5)
    steered_top5 = torch.topk(steered_probs, 5)

    orig_top = [(tokenizer.decode([tid]), prob.item()) for tid, prob in zip(orig_top5.indices, orig_top5.values)]
    steered_top = [(tokenizer.decode([tid]), prob.item()) for tid, prob in zip(steered_top5.indices, steered_top5.values)]

    return {
        "strength": strength,
        "kl_divergence": kl,
        "orig_top5": orig_top,
        "steered_top5": steered_top,
    }


def main():
    set_seed(42)

    adapter_path = str(PROJECT_ROOT / "experiments" / "adapters" / "lora_json_r8" / "adapter")

    print("Loading base model...")
    base_bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    tokenizer = base_bundle.tokenizer
    device = base_bundle.device

    print("Loading trained model (base + LoRA JSON)...")
    trained_model = PeftModel.from_pretrained(base_bundle.model, adapter_path)
    trained_model.eval()

    # Define skill knockout experiments
    skills = [
        {
            "name": "json_schema",
            "positive": [
                'Return valid JSON: {"name": "Alice", "age": 31}',
                'Return valid JSON: {"x": 1, "y": 2}',
                '{"name": "Bob", "age": 25}',
                '{"city": "London", "country": "UK"}',
                '{"product": "widget", "price": 9.99}',
            ],
            "negative": [
                "Tell me about Alice who is 31 years old.",
                "What are the values of x and y?",
                "Describe a person named Bob, age 25.",
                "London is the capital of the United Kingdom.",
                "The widget costs nine dollars and ninety-nine cents.",
            ],
            "test_prompts": [
                {"prompt": 'Return valid JSON with keys name and age. Eve is 42.\n', "target": "42", "skill_target": True},
                {"prompt": 'Return valid JSON: {"city": "', "target": '"', "skill_target": True},
                {"prompt": "The capital of Italy is ", "target": " Rome", "skill_target": False},
                {"prompt": "Complete: A B C A B ", "target": " C", "skill_target": False},
            ],
            "knockout_layers": [2, 6, 7, 9, 12, 13, 21],
        },
        {
            "name": "factual_recall",
            "positive": [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Japan is Tokyo.",
                "The capital of Italy is Rome.",
                "The capital of Spain is Madrid.",
            ],
            "negative": [
                "France is a beautiful country in Europe.",
                "Germany has many famous cities.",
                "Japan is an island nation in Asia.",
                "Italy is known for its cuisine.",
                "Spain has wonderful beaches.",
            ],
            "test_prompts": [
                {"prompt": "The capital of Italy is ", "target": " Rome", "skill_target": True},
                {"prompt": "The capital of Spain is ", "target": " Madrid", "skill_target": True},
                {"prompt": 'Return valid JSON: {"x": ', "target": "1", "skill_target": False},
                {"prompt": "Complete: 1 2 3 1 2 ", "target": " 3", "skill_target": False},
            ],
            "knockout_layers": [2, 3, 16, 19, 21],
        },
    ]

    all_results = []

    for skill in skills:
        print(f"\n{'='*60}")
        print(f"  SKILL KNOCKOUT: {skill['name'].upper()}")
        print(f"{'='*60}")

        skill_results = {
            "skill": skill["name"],
            "layers": skill["knockout_layers"],
            "layer_data": [],
        }

        for layer_idx in skill["knockout_layers"]:
            print(f"\n  --- Layer {layer_idx} ---")

            # Compute steering vector using TRAINED model
            sv = compute_steering_vector(
                trained_model, tokenizer,
                skill["positive"], skill["negative"],
                layer_idx
            )

            if sv is None:
                print(f"    Failed to compute steering vector at L{layer_idx}")
                continue

            sv_norm = sv.norm().item()
            print(f"    Steering vector norm: {sv_norm:.4f}")

            # Test negative steering (knockout) at various strengths
            strengths = [0.0, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0]
            prompt_results = []

            for tp in skill["test_prompts"]:
                prompt = tp["prompt"]
                target = tp["target"]
                is_skill_target = tp["skill_target"]

                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if len(target_ids) == 0:
                    continue
                target_id = target_ids[0]

                sweep = []
                for strength in strengths:
                    result = inject_steering(
                        trained_model, tokenizer, prompt,
                        layer_idx, sv, strength
                    )

                    # Get target token probability
                    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

                    # Recompute with steering for target prob
                    sv_scaled = sv.to(device) * strength
                    def steer_hook(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output
                        hidden[:, -1, :] += sv_scaled
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden

                    handle = get_layers(trained_model)[layer_idx].register_forward_hook(steer_hook)
                    with torch.no_grad():
                        steered_logits = trained_model(ids).logits
                    handle.remove()

                    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)
                    target_prob = steered_probs[target_id].item()

                    sweep.append({
                        "strength": strength,
                        "kl_divergence": result["kl_divergence"],
                        "target_prob": round(target_prob, 6),
                        "top5": result["steered_top5"] if strength != 0.0 else result["orig_top5"],
                    })

                    if strength == 0.0:
                        print(f"    [{prompt[:40]}...] target={target} base_prob={target_prob:.4f}")
                    elif strength in [-1.0, -4.0]:
                        print(f"      s={strength:+.1f}: KL={result['kl_divergence']:.3f} target_prob={target_prob:.4f}")

                prompt_results.append({
                    "prompt": prompt,
                    "target": target,
                    "is_skill_target": is_skill_target,
                    "sweep": sweep,
                })

            # Compute selectivity: does knockout affect skill targets more than non-skill?
            selectivity = {}
            for strength in strengths:
                skill_drops = []
                non_skill_drops = []
                for pr in prompt_results:
                    base_prob = pr["sweep"][0]["target_prob"]  # s=0
                    for s in pr["sweep"]:
                        if s["strength"] == strength:
                            drop = base_prob - s["target_prob"]
                            if pr["is_skill_target"]:
                                skill_drops.append(drop)
                            else:
                                non_skill_drops.append(drop)
                            break
                selectivity[str(strength)] = {
                    "mean_skill_drop": round(np.mean(skill_drops), 6) if skill_drops else 0.0,
                    "mean_non_skill_drop": round(np.mean(non_skill_drops), 6) if non_skill_drops else 0.0,
                    "selectivity_ratio": round(
                        np.mean(skill_drops) / max(np.mean(non_skill_drops), 1e-6), 4
                    ) if skill_drops and non_skill_drops else 0.0,
                }

            skill_results["layer_data"].append({
                "layer": layer_idx,
                "sv_norm": round(sv_norm, 6),
                "prompt_results": prompt_results,
                "selectivity": selectivity,
            })

            # Print best knockout
            best_s = None
            best_selectivity = 0
            for s_str, sel in selectivity.items():
                if sel["selectivity_ratio"] > best_selectivity and float(s_str) < 0:
                    best_selectivity = sel["selectivity_ratio"]
                    best_s = s_str
            if best_s:
                print(f"    Best knockout: s={best_s} selectivity={best_selectivity:.2f}")

        all_results.append(skill_results)

    output = {
        "experiment": "skill_knockout_negative_steering",
        "n_skills": len(all_results),
        "model": "Qwen/Qwen2.5-0.5B + LoRA JSON r8",
        "results": all_results,
    }

    output_path = PROJECT_ROOT / "experiments" / "results" / "skill_knockout.json"
    save_json(output, output_path)
    print(f"\n  Results saved to {output_path}")

    register_experiment(
        type="steering",
        model="Qwen/Qwen2.5-0.5B",
        backend="hf_hooks",
        config="config/experiment_plan.yaml",
        inputs=[adapter_path],
        outputs=[str(output_path)],
        status="success",
        summary=f"Skill knockout: {len(all_results)} skills, negative steering on trained model",
        key_metrics={},
        next="Adapter-only ablation (ablate adapter at specific layers)",
    )
    print("  Experiment registered.")


if __name__ == "__main__":
    main()
