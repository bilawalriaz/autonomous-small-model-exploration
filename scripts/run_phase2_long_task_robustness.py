"""Phase 2 Block I: Long-task robustness — prompt length robustness.

Tests robustness across prompt lengths for both 0.5B and 1.5B:
1. Build prompt sets at 3 lengths for 4 task families
2. For each length: layer ablation, steering at hub, LoRA effect comparison
3. Measure: does hub change with length? Does steering degrade? Does LoRA effect change?

Registry ID: P2-ROBUST-001
"""
import sys
import json
import argparse
import gc
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from mi_atlas.model_loader import load_model_hf, ModelBundle, detect_model_info
from mi_atlas.backend import HFBackend
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.task_suite import TaskSuite, TaskExample, build_default_suite
from mi_atlas.metrics import exact_match_score
from mi_atlas.experiment_registry import register_experiment, load_registry
from mi_atlas.utils import save_json, append_jsonl, set_seed, now_iso, PROJECT_ROOT

# ── Constants ────────────────────────────────────────────────────────
MODELS = [
    {"name": "Qwen/Qwen2.5-0.5B", "slug": "qwen05b", "n_layers": 24, "hub_layer": 2},
    {"name": "Qwen/Qwen2.5-1.5B", "slug": "qwen15b", "n_layers": 28, "hub_layer": 26},
]
REGISTRY_ID = "P2-ROBUST-001"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SEEDS = [42, 137, 2026]
KEY_FAMILIES = ["factual_recall", "json_schema", "copying", "code_syntax"]


# ── Prompt Generation ────────────────────────────────────────────────

def build_length_varying_prompts(seed: int = 42) -> dict:
    """Build prompt sets at 3 lengths for 4 task families.

    Returns dict: {family: {short: [...], medium: [...], long: [...]}}
    Each entry is a list of (prompt, target) tuples.
    """
    prompts = {}

    # ── Factual Recall ────────────────────────────────────────────
    prompts["factual_recall"] = {
        "short": [
            ("Capital of France? ", "Paris"),
            ("Capital of Japan? ", "Tokyo"),
            ("Capital of Germany? ", "Berlin"),
            ("Largest planet? ", "Jupiter"),
        ],
        "medium": [
            ("What is the capital of France? Please answer with just the city name.\n", "Paris"),
            ("What is the capital of Japan? Please answer with just the city name.\n", "Tokyo"),
            ("What is the capital of Germany? Please answer with just the city name.\n", "Berlin"),
            ("What is the largest planet in our solar system? Answer with just the planet name.\n", "Jupiter"),
        ],
        "long": [
            ("In the context of European geography and history, what is the capital city of the French "
             "Republic, which has been the seat of government since the medieval period and is known for "
             "landmarks such as the Eiffel Tower and the Louvre Museum? Please provide just the city name.\n",
             "Paris"),
            ("In the context of East Asian geography, what is the capital city of Japan, the island nation "
             "located in the Pacific Ocean, which has served as the imperial capital since the Meiji "
             "Restoration and is known for its blend of traditional and modern architecture? "
             "Please provide just the city name.\n",
             "Tokyo"),
            ("In the context of Central European geography and history, what is the capital city of the "
             "Federal Republic of Germany, which has been the capital since reunification in 1990 and was "
             "historically divided by a famous wall during the Cold War? Please provide just the city name.\n",
             "Berlin"),
            ("In the context of our solar system, what is the largest planet by both mass and volume, "
             "a gas giant composed primarily of hydrogen and helium, famous for its Great Red Spot "
             "and numerous moons including Europa and Ganymede? Please provide just the planet name.\n",
             "Jupiter"),
        ],
    }

    # ── JSON Schema ───────────────────────────────────────────────
    prompts["json_schema"] = {
        "short": [
            ('{"name":', None),
            ('{"name": "Alice", "age":', None),
            ('{"city": "London", "pop":', None),
            ('{"x": 1, "y":', None),
        ],
        "medium": [
            ('Return valid JSON with keys name and age. Alice is 30.\n{"name":', None),
            ('Return valid JSON with keys city and population. London has 9 million.\n{"city":', None),
            ('Return valid JSON with keys x and y. x=1, y=2.\n{"x":', None),
            ('Return valid JSON with keys title and year. The Matrix, 1999.\n{"title":', None),
        ],
        "long": [
            ("You are a JSON generator. Given the following information, produce a valid JSON object "
             "with the specified keys. The person's name is Alice and she is 30 years old. "
             "Please output valid JSON with keys 'name' and 'age'.\n"
             '{"name":',
             None),
            ("You are a JSON generator. Given the following information, produce a valid JSON object. "
             "The city is London, located in England, with a population of approximately 9 million people. "
             "Please output valid JSON with keys 'city', 'country', and 'population'.\n"
             '{"city":',
             None),
            ("You are a JSON generator. Given the following mathematical coordinates, produce valid JSON. "
             "The point has x-coordinate 1 and y-coordinate 2 and z-coordinate 3. "
             "Please output valid JSON with keys 'x', 'y', and 'z'.\n"
             '{"x":',
             None),
            ("You are a JSON generator. Given the following movie information, produce valid JSON. "
             "The movie is titled 'The Matrix', released in 1999, directed by the Wachowskis. "
             "Please output valid JSON with keys 'title', 'year', and 'director'.\n"
             '{"title":',
             None),
        ],
    }

    # ── Copying / Induction ──────────────────────────────────────
    prompts["copying"] = {
        "short": [
            ("A B C A B ", "C"),
            ("1 2 3 1 2 ", "3"),
            ("X Y X Y ", "X"),
            ("a b a b ", "a"),
        ],
        "medium": [
            ("A B C D E A B C D ", "E"),
            ("1 2 3 4 5 6 1 2 3 4 5 ", "6"),
            ("red blue green yellow red blue green ", "yellow"),
            ("cat dog fish cat dog ", "fish"),
        ],
        "long": [
            ("alpha beta gamma delta epsilon zeta eta theta alpha beta gamma delta epsilon zeta eta ", "theta"),
            ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ", "16"),
            ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
             "one two three four five six seven eight nine ten eleven twelve thirteen fourteen ", "fifteen"),
            ("mon tue wed thu fri sat sun mon tue wed thu fri sat ", "sun"),
        ],
    }

    # ── Code Syntax ──────────────────────────────────────────────
    prompts["code_syntax"] = {
        "short": [
            ("def add(a, b):\n    return a + ", "b"),
            ("for i in range(10):\n    print(", "i"),
            ("if x > 0:\n    result = ", "x"),
            ("lambda x: x * ", "2"),
        ],
        "medium": [
            ("# Function to add two numbers\ndef add(a, b):\n    # Return the sum\n    return a + ", "b"),
            ("# Loop through numbers\nfor i in range(10):\n    # Print each number\n    print(", "i"),
            ("# Check if positive\nif x > 0:\n    # Store positive result\n    result = ", "x"),
            ("# Create a doubling function\ndouble = lambda x: x * ", "2"),
        ],
        "long": [
            ("# This is a utility module for mathematical operations\n"
             "# It provides basic arithmetic functions\n\n"
             "# The add function takes two parameters\n"
             "# and returns their sum\n"
             "def add(a, b):\n"
             "    # Validate inputs are numeric\n"
             "    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):\n"
             "        raise TypeError('Arguments must be numeric')\n"
             "    # Return the sum of a and b\n"
             "    return a + ",
             "b"),
            ("# Data processing pipeline\n"
             "# This script processes a list of numbers\n\n"
             "# Define the data source\n"
             "numbers = list(range(10))\n\n"
             "# Process each number\n"
             "for i in range(10):\n"
             "    # Apply transformation\n"
             "    transformed = i * 2 + 1\n"
             "    # Output the result\n"
             "    print(",
             "i"),
            ("# Conditional logic module\n"
             "# Determines if a value is positive\n\n"
             "# Read the input value\n"
             "x = float(input('Enter a number: '))\n\n"
             "# Initialize result variable\n"
             "result = None\n\n"
             "# Check the sign of x\n"
             "if x > 0:\n"
             "    # Value is positive\n"
             "    result = ",
             "x"),
            ("# Functional programming utilities\n"
             "# Lambda functions for common operations\n\n"
             "# The double function multiplies by 2\n"
             "# Usage: double(5) returns 10\n"
             "# It is a simple lambda that takes x\n"
             "# and returns x multiplied by\n"
             "double = lambda x: x * ",
             "2"),
        ],
    }

    # Verify token counts (rough estimate)
    for family, lengths in prompts.items():
        for length, examples in lengths.items():
            for i, (prompt, target) in enumerate(examples):
                # Rough token count: ~4 chars per token
                approx_tokens = len(prompt) // 4
                prompts[family][length][i] = (prompt, target)

    return prompts


def create_task_suite_from_prompts(prompts: dict, seed: int = 42) -> dict:
    """Create TaskSuites for each (family, length) combination.

    Returns dict: {(family, length): TaskSuite}
    """
    suites = {}
    idx = 0
    for family, lengths in prompts.items():
        for length_key, examples in lengths.items():
            task_examples = []
            for i, (prompt, target) in enumerate(examples):
                task_examples.append(TaskExample(
                    id=f"{family}_{length_key}_{i:04d}",
                    family=family,
                    clean_prompt=prompt,
                    target=target if target is not None else "",
                    metric_type="target_logprob" if family == "factual_recall" else "exact_match",
                    metadata={"length_category": length_key},
                    split="test",
                ))
            suites[(family, length_key)] = TaskSuite(task_examples)
            idx += 1
    return suites


# ── Helpers ──────────────────────────────────────────────────────────

def check_already_done(experiment_id: str, model_slug: str, task_slug: str, seed: int) -> bool:
    registry = load_registry()
    pattern = f"P2_{experiment_id}_{model_slug}_{task_slug}_seed{seed}"
    for rec in registry:
        rid = rec.get("id", "")
        if pattern in rid and rec.get("status") == "success":
            return True
    return False


def make_run_id(experiment_id: str, model_slug: str, task_slug: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"P2_{experiment_id}_{model_slug}_{task_slug}_{ts}_seed{seed}"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def compute_kl(logits_a, logits_b):
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    activation = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation["value"] = output[0][:, position, :].detach().clone()
        else:
            activation["value"] = output[:, position, :].detach().clone()
    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    return activation.get("value")


def count_tokens(tokenizer, text):
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ── Experiment A: Layer Ablation at Each Length ─────────────────────

def exp_layer_ablation_by_length(bundle, suite_dict, model_slug, seed, force=False):
    """Layer ablation at each prompt length for all families."""
    task_slug = "ablation_by_length"
    exp_id = "I01"
    if not force and check_already_done(exp_id, model_slug, task_slug, seed):
        print(f"    [SKIP] Ablation by length seed={seed} already done")
        return None

    run_id = make_run_id(exp_id, model_slug, task_slug, seed)
    print(f"    [I01] Layer ablation by length (seed={seed})")

    backend = HFBackend(bundle)
    n_layers = bundle.architecture["n_layers"]

    results = {}
    for (family, length_key), suite in suite_dict.items():
        if len(suite) == 0:
            continue

        print(f"      {family}/{length_key} ({len(suite)} examples)...")

        try:
            ablation_result = run_layer_ablation_suite(
                backend, suite, ablation_type="zero", split=None
            )

            effect_matrix = np.array(ablation_result["effect_matrix"])
            mean_effects = effect_matrix.mean(axis=1)

            # Find hub layer (argmax of mean effect)
            hub_layer = int(np.argmax(mean_effects))

            # Top 3 layers
            top3 = np.argsort(mean_effects)[-3:][::-1]

            results[f"{family}_{length_key}"] = {
                "family": family,
                "length": length_key,
                "effect_matrix": ablation_result["effect_matrix"],
                "hub_layer": hub_layer,
                "hub_effect": float(mean_effects[hub_layer]),
                "top3_layers": [int(l) for l in top3],
                "top3_effects": [float(mean_effects[l]) for l in top3],
                "mean_effect": float(mean_effects.mean()),
                "max_effect": float(effect_matrix.max()),
            }
        except Exception as e:
            print(f"        [ERROR] {family}/{length_key}: {e}")
            results[f"{family}_{length_key}"] = {
                "family": family, "length": length_key, "error": str(e)
            }
        clear_gpu()

    # Analyze hub stability across lengths
    hub_analysis = {}
    for family in KEY_FAMILIES:
        family_results = {k: v for k, v in results.items() if v.get("family") == family}
        hub_layers = [v.get("hub_layer") for v in family_results.values() if "hub_layer" in v]
        if hub_layers:
            hub_analysis[family] = {
                "hub_layers_by_length": {
                    k.split("_")[-1]: v.get("hub_layer") for k, v in family_results.items() if "hub_layer" in v
                },
                "hub_stable": len(set(hub_layers)) == 1,
                "hub_range": [min(hub_layers), max(hub_layers)],
            }

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "layer_ablation_by_length",
        "model": bundle.model_name,
        "model_slug": model_slug,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": n_layers,
        "results": results,
        "hub_analysis": hub_analysis,
    }

    out_path = RESULTS_DIR / f"long_task_robustness_{model_slug}_ablation_seed{seed}.json"
    save_json(output, out_path)
    print(f"      Saved to {out_path}")

    # Print hub analysis
    for family, analysis in hub_analysis.items():
        stable = "STABLE" if analysis["hub_stable"] else "SHIFTS"
        print(f"        {family}: hub {stable}, layers={analysis['hub_layers_by_length']}")

    return output


# ── Experiment B: Steering at Each Length ────────────────────────────

def exp_steering_by_length(bundle, suite_dict, model_slug, hub_layer, seed, force=False):
    """Steering at hub layer for each prompt length."""
    task_slug = "steering_by_length"
    exp_id = "I02"
    if not force and check_already_done(exp_id, model_slug, task_slug, seed):
        print(f"    [SKIP] Steering by length seed={seed} already done")
        return None

    run_id = make_run_id(exp_id, model_slug, task_slug, seed)
    print(f"    [I02] Steering by length at L{hub_layer} (seed={seed})")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    # Compute steering vector from factual recall positive/negative
    positive_prompts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Japan is Tokyo.",
        "The capital of Italy is Rome.",
    ]
    negative_prompts = [
        "France is a beautiful country in Europe.",
        "Germany has many famous cities.",
        "Japan is an island nation in Asia.",
        "Italy is known for its cuisine.",
    ]

    pos_acts = []
    neg_acts = []
    for prompt in positive_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        act = get_activation_at_layer(model, ids, hub_layer)
        if act is not None:
            pos_acts.append(act.cpu())

    for prompt in negative_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)
        act = get_activation_at_layer(model, ids, hub_layer)
        if act is not None:
            neg_acts.append(act.cpu())

    if not pos_acts or not neg_acts:
        print("        Failed to compute steering vector")
        return None

    sv = (torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)).squeeze(0)
    sv_norm = sv.norm().item()
    print(f"      Steering vector norm: {sv_norm:.4f}")

    # Test steering at each (family, length) combination
    results = {}
    strengths = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

    for (family, length_key), suite in suite_dict.items():
        if len(suite) == 0:
            continue

        family_results = {"family": family, "length": length_key, "prompts": []}

        for example in list(suite)[:3]:
            prompt = example.clean_prompt
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                orig_logits = model(ids).logits
            orig_probs = torch.softmax(orig_logits[0, -1, :], dim=-1)

            prompt_results = {"prompt": prompt[:80], "strengths": []}

            for strength in strengths:
                sv_scaled = sv.to(device) * strength

                def steer_hook(module, input, output, sv_s=sv_scaled):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    hidden[:, -1, :] += sv_s
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

                layers = get_layers(model)
                handle = layers[hub_layer].register_forward_hook(steer_hook)
                with torch.no_grad():
                    steered_logits = model(ids).logits
                handle.remove()

                steered_probs = torch.softmax(steered_logits[0, -1, :], dim=-1)
                kl = torch.nn.functional.kl_div(
                    steered_probs.log(), orig_probs, reduction="sum"
                ).item()

                prompt_results["strengths"].append({
                    "strength": strength,
                    "kl_divergence": round(kl, 6),
                })

            family_results["prompts"].append(prompt_results)

        # Mean KL at each strength
        mean_kls = {}
        for s_idx, strength in enumerate(strengths):
            kls = []
            for p in family_results["prompts"]:
                if s_idx < len(p["strengths"]):
                    kls.append(p["strengths"][s_idx]["kl_divergence"])
            mean_kls[str(strength)] = round(float(np.mean(kls)), 6) if kls else 0.0

        family_results["mean_kl_by_strength"] = mean_kls
        family_results["max_mean_kl"] = max(mean_kls.values())

        key = f"{family}_{length_key}"
        results[key] = family_results
        clear_gpu()

    # Analyze steering effectiveness by length
    effectiveness = {}
    for family in KEY_FAMILIES:
        family_data = {k: v for k, v in results.items() if v.get("family") == family}
        max_kls = {k.split("_")[-1]: v.get("max_mean_kl", 0) for k, v in family_data.items()}
        effectiveness[family] = {
            "max_kl_by_length": max_kls,
            "degrades_with_length": (
                max_kls.get("short", 0) > max_kls.get("long", 0)
                if "short" in max_kls and "long" in max_kls else None
            ),
        }

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "steering_by_length",
        "model": bundle.model_name,
        "model_slug": model_slug,
        "seed": seed,
        "timestamp": now_iso(),
        "hub_layer": hub_layer,
        "sv_norm": sv_norm,
        "results": results,
        "effectiveness_analysis": effectiveness,
    }

    out_path = RESULTS_DIR / f"long_task_robustness_{model_slug}_steering_seed{seed}.json"
    save_json(output, out_path)
    print(f"      Saved to {out_path}")

    for family, analysis in effectiveness.items():
        degrades = "DEGRADES" if analysis["degrades_with_length"] else "STABLE"
        print(f"        {family}: steering {degrades}, max_kl={analysis['max_kl_by_length']}")

    return output


# ── Experiment C: LoRA Effect by Length ──────────────────────────────

def exp_lora_effect_by_length(bundle, suite_dict, model_slug, seed, force=False):
    """LoRA effect comparison: trained vs base at each prompt length."""
    task_slug = "lora_effect_by_length"
    exp_id = "I03"
    if not force and check_already_done(exp_id, model_slug, task_slug, seed):
        print(f"    [SKIP] LoRA effect by length seed={seed} already done")
        return None

    run_id = make_run_id(exp_id, model_slug, task_slug, seed)
    print(f"    [I03] LoRA effect by length (seed={seed})")

    from peft import PeftModel

    # Look for existing adapter
    adapter_path = PROJECT_ROOT / "experiments" / "adapters" / f"lora_json_r8" / "adapter"
    if not adapter_path.exists():
        # Try model-specific adapter
        adapter_path = PROJECT_ROOT / "experiments" / "adapters" / f"{model_slug}_lora_json" / "adapter"

    if not adapter_path.exists():
        print(f"      No adapter found, training new one...")
        # Train a quick adapter
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from mi_atlas.training.datasets import prepare_sft_dataset

        model = bundle.model
        tokenizer = bundle.tokenizer

        full_suite = build_default_suite(seed=seed)
        json_suite = full_suite.filter_by_family("json_schema")
        ds = prepare_sft_dataset(json_suite)

        set_seed(seed)
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model.gradient_checkpointing_enable()
        peft_model = get_peft_model(model, lora_config)

        adapter_dir = str(PROJECT_ROOT / "experiments" / "adapters" / f"{model_slug}_lora_robustness_seed{seed}")
        args = SFTConfig(
            output_dir=adapter_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=100,
            learning_rate=2e-4,
            warmup_steps=10,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=25,
            save_steps=500,
            report_to="none",
            max_length=256,
            seed=seed,
        )

        trainer = SFTTrainer(model=peft_model, args=args, train_dataset=ds, processing_class=tokenizer)
        trainer.train()
        adapter_path = Path(adapter_dir) / "adapter"
        peft_model.save_pretrained(str(adapter_path))

        # Reload from base
        del peft_model, trainer
        clear_gpu()

        # Need to reload the base model since get_peft_model modifies in-place
        bundle = load_model_hf(bundle.model_name)

    # Load trained model
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    trained_model = PeftModel.from_pretrained(model, str(adapter_path))
    trained_model.eval()

    # Test at each (family, length)
    results = {}
    for (family, length_key), suite in suite_dict.items():
        if len(suite) == 0:
            continue

        family_results = {"family": family, "length": length_key, "prompts": []}

        for example in list(suite)[:3]:
            prompt = example.clean_prompt
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            # Base logits (adapter disabled)
            with trained_model.disable_adapter():
                base_logits = trained_model(ids).logits

            # Trained logits
            trained_logits = trained_model(ids).logits

            kl = compute_kl(trained_logits, base_logits)

            base_probs = torch.softmax(base_logits[0, -1, :], dim=-1)
            trained_probs = torch.softmax(trained_logits[0, -1, :], dim=-1)

            # Check if target logprob improved
            target = example.target
            target_logprob_delta = 0.0
            if target:
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if target_ids:
                    tid = target_ids[0]
                    base_lp = base_probs[tid].item()
                    trained_lp = trained_probs[tid].item()
                    target_logprob_delta = trained_lp - base_lp

            family_results["prompts"].append({
                "prompt": prompt[:80],
                "adapter_kl": round(kl, 6),
                "target_logprob_delta": round(target_logprob_delta, 6),
            })

        mean_kl = np.mean([p["adapter_kl"] for p in family_results["prompts"]])
        mean_delta = np.mean([p["target_logprob_delta"] for p in family_results["prompts"]])
        family_results["mean_adapter_kl"] = round(float(mean_kl), 6)
        family_results["mean_target_logprob_delta"] = round(float(mean_delta), 6)

        key = f"{family}_{length_key}"
        results[key] = family_results
        clear_gpu()

    # Analyze LoRA effect magnitude by length
    effect_analysis = {}
    for family in KEY_FAMILIES:
        family_data = {k: v for k, v in results.items() if v.get("family") == family}
        kls = {k.split("_")[-1]: v.get("mean_adapter_kl", 0) for k, v in family_data.items()}
        deltas = {k.split("_")[-1]: v.get("mean_target_logprob_delta", 0) for k, v in family_data.items()}
        effect_analysis[family] = {
            "adapter_kl_by_length": kls,
            "target_delta_by_length": deltas,
        }

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "lora_effect_by_length",
        "model": bundle.model_name,
        "model_slug": model_slug,
        "seed": seed,
        "timestamp": now_iso(),
        "adapter_path": str(adapter_path),
        "results": results,
        "effect_analysis": effect_analysis,
    }

    out_path = RESULTS_DIR / f"long_task_robustness_{model_slug}_lora_effect_seed{seed}.json"
    save_json(output, out_path)
    print(f"      Saved to {out_path}")

    for family, analysis in effect_analysis.items():
        print(f"        {family}: KL by length={analysis['adapter_kl_by_length']}")

    return output


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Block I: Long-task robustness")
    parser.add_argument("--force", action="store_true", help="Re-run completed experiments")
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (qwen05b or qwen15b)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only this experiment (ablation, steering, lora)")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else SEEDS
    force = args.force

    models_to_run = MODELS
    if args.model:
        models_to_run = [m for m in MODELS if m["slug"] == args.model]
        if not models_to_run:
            print(f"Unknown model: {args.model}")
            print(f"Available: {[m['slug'] for m in MODELS]}")
            sys.exit(1)

    print("=" * 70)
    print(f"  Phase 2 Block I: Long-Task Robustness")
    print(f"  Registry ID: {REGISTRY_ID}")
    print(f"  Models: {[m['name'] for m in models_to_run]}")
    print(f"  Seeds: {seeds}")
    print(f"  Force: {force}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Build prompt sets (same across seeds/models)
    print("\nBuilding length-varying prompt sets...")
    all_prompts = build_length_varying_prompts(seed=seeds[0])

    # Show prompt lengths
    from transformers import AutoTokenizer
    for model_info in models_to_run:
        try:
            tok = AutoTokenizer.from_pretrained(model_info["name"], trust_remote_code=True)
            print(f"\n  Token counts for {model_info['name']}:")
            for family, lengths in all_prompts.items():
                for length_key, examples in lengths.items():
                    token_counts = [len(tok.encode(p, add_special_tokens=False)) for p, _ in examples]
                    print(f"    {family}/{length_key}: {token_counts}")
        except Exception:
            pass

    config = {
        "registry_id": REGISTRY_ID,
        "models": [m["name"] for m in models_to_run],
        "seeds": seeds,
        "length_categories": ["short", "medium", "long"],
        "families": KEY_FAMILIES,
        "timestamp": now_iso(),
    }
    save_json(config, CONFIGS_DIR / f"{REGISTRY_ID}_config.json")

    all_results = {}

    for model_info in models_to_run:
        model_name = model_info["name"]
        model_slug = model_info["slug"]
        n_layers = model_info["n_layers"]
        hub_layer = model_info["hub_layer"]

        print(f"\n{'#'*70}")
        print(f"  MODEL: {model_name} ({model_slug})")
        print(f"{'#'*70}")

        set_seed(seeds[0])
        try:
            bundle = load_model_hf(model_name)
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_name}: {e}")
            traceback.print_exc()
            all_results[model_slug] = {"status": "load_error", "error": str(e)}
            continue

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            set_seed(seed)

            # Build suite dict for this seed
            suite_dict = create_task_suite_from_prompts(all_prompts, seed=seed)

            experiments = {
                "ablation": lambda: exp_layer_ablation_by_length(
                    bundle, suite_dict, model_slug, seed, force
                ),
                "steering": lambda: exp_steering_by_length(
                    bundle, suite_dict, model_slug, hub_layer, seed, force
                ),
                "lora": lambda: exp_lora_effect_by_length(
                    bundle, suite_dict, model_slug, seed, force
                ),
            }

            if args.experiment:
                if args.experiment not in experiments:
                    print(f"  Unknown experiment: {args.experiment}")
                    continue
                experiments = {args.experiment: experiments[args.experiment]}

            for exp_name, exp_fn in experiments.items():
                try:
                    result = exp_fn()
                    if result:
                        key = f"{model_slug}_{exp_name}_seed{seed}"
                        all_results[key] = {"status": "success", "run_id": result.get("run_id")}

                        registry_record = {
                            "id": result.get("run_id"),
                            "phase": "P2",
                            "block": "I",
                            "registry_id": REGISTRY_ID,
                            "experiment": exp_name,
                            "model": model_name,
                            "model_slug": model_slug,
                            "seed": seed,
                            "timestamp": now_iso(),
                            "status": "success",
                        }
                        append_jsonl(registry_record, PROJECT_ROOT / "experiments" / "registry.jsonl")
                except Exception as e:
                    print(f"    [ERROR] {exp_name} seed={seed}: {e}")
                    traceback.print_exc()
                    all_results[f"{model_slug}_{exp_name}_seed{seed}"] = {
                        "status": "error", "error": str(e)
                    }
                    clear_gpu()

        # Save per-model summary
        model_summary = {
            "registry_id": REGISTRY_ID,
            "model": model_name,
            "model_slug": model_slug,
            "seeds": seeds,
            "timestamp": now_iso(),
            "results": {k: v for k, v in all_results.items() if k.startswith(model_slug)},
        }
        save_json(model_summary, RESULTS_DIR / f"long_task_robustness_{model_slug}.json")

        del bundle
        clear_gpu()

    # Save master summary
    summary = {
        "registry_id": REGISTRY_ID,
        "models": [m["name"] for m in models_to_run],
        "seeds": seeds,
        "timestamp": now_iso(),
        "results": all_results,
    }
    save_json(summary, RESULTS_DIR / "long_task_robustness_summary.json")

    register_experiment(
        type="phase2_robustness",
        model=", ".join(m["name"] for m in models_to_run),
        backend="hf",
        config=str(CONFIGS_DIR / f"{REGISTRY_ID}_config.json"),
        inputs=[],
        outputs=[str(RESULTS_DIR / "long_task_robustness_summary.json")],
        status="success",
        summary=f"Phase 2 Block I: Long-task robustness, {len(models_to_run)} models, "
                f"{len([v for v in all_results.values() if v.get('status') == 'success'])} experiments succeeded",
        key_metrics={"total_experiments": len(all_results)},
    )

    print("\n" + "=" * 70)
    print("  Phase 2 Block I COMPLETE")
    print(f"  Results: {RESULTS_DIR / 'long_task_robustness_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
