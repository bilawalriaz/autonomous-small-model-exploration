"""Phase 2 Block D: Third-scale atlas on Qwen2.5-3B.

Runs a reduced atlas on Qwen2.5-3B (36 layers, 16 heads GQA, d=2048).
Uses 4-bit NF4 quantization for 8GB VRAM training.

Experiments:
1. Layer ablation map (all 36 layers, zero ablation, key task families)
2. Head ablation map (top 6 layers × 16 heads)
3. MLP ablation map (all 36 layers)
4. Steering sweep at candidate hub layers (L2, L18, L26, L34, L35)
5. LoRA skill injection for JSON (r=8, 100 steps, gradient_checkpointing, bs=1)
6. Trained-to-base final-layer patching (L30-L35)
7. Layer skipping test (skip every other layer, skip last 6)
8. Adapter knockout (disable adapter at each layer)

Registry ID: P2-SCALE3-001
"""
import sys
import json
import argparse
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from mi_atlas.model_loader import ModelBundle, detect_model_info
from mi_atlas.backend import HFBackend
from mi_atlas.ablations import run_layer_ablation_suite
from mi_atlas.task_suite import build_default_suite, TaskSuite, TaskExample
from mi_atlas.metrics import exact_match_score, kl_divergence
from mi_atlas.experiment_registry import register_experiment, load_registry
from mi_atlas.utils import save_json, load_json, append_jsonl, set_seed, now_iso, PROJECT_ROOT

# ── Constants ────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_SLUG = "qwen3b"
N_LAYERS = 36
N_HEADS = 16
D_MODEL = 2048
REGISTRY_ID = "P2-SCALE3-001"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SEEDS = [42, 137, 2026]  # Phase 2 default: 3 seeds

CANDIDATE_HUB_LAYERS = [2, 18, 26, 34, 35]


# ── Helpers ──────────────────────────────────────────────────────────

def check_already_done(experiment_id: str, model_slug: str, task_slug: str, seed: int) -> bool:
    """Check if this exact experiment+model+task+seed combo already completed."""
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
    """Free GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_3b_model():
    """Load Qwen2.5-3B with 4-bit NF4 quantization for 8GB VRAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} with 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    arch = detect_model_info(model, tokenizer)
    device = next(model.parameters()).device

    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=MODEL_NAME,
        backend="hf_native",
        device=device,
        dtype=torch.bfloat16,
        architecture=arch,
    )
    print(f"  Loaded. Architecture: {arch['n_layers']}L, {arch['n_heads']}H, d={arch['d_model']}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    return bundle


def get_layers(model):
    """Get transformer layers list, handling PeftModel wrapping."""
    if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        return model.model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def compute_kl(logits_a, logits_b):
    """KL(P_a || P_b) at last token."""
    probs_a = torch.softmax(logits_a[0, -1, :], dim=-1)
    probs_b = torch.softmax(logits_b[0, -1, :], dim=-1)
    return torch.nn.functional.kl_div(
        torch.log(probs_b), probs_a, reduction="sum"
    ).item()


def get_activation_at_layer(model, input_ids, layer_idx, position=-1):
    """Get activation at a specific layer and position."""
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


def run_with_skipped_layers(model, input_ids, skip_layers):
    """Run inference with specified layers zeroed out."""
    layers = get_layers(model)
    handles = []

    for idx in skip_layers:
        def skip_hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        h = layers[idx].register_forward_hook(skip_hook)
        handles.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in handles:
        h.remove()
    return logits


# ── Experiment 1: Layer Ablation ────────────────────────────────────

def exp_layer_ablation(bundle, suite, seed, force=False):
    """Layer ablation map: all 36 layers, zero ablation, key task families."""
    task_slug = "layer_ablation"
    if not force and check_already_done("D01", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Layer ablation seed={seed} already done")
        return None

    run_id = make_run_id("D01", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D01] Layer ablation (seed={seed}, run_id={run_id})")

    backend = HFBackend(bundle)
    result = run_layer_ablation_suite(backend, suite, ablation_type="zero", split="test")

    effect_matrix = np.array(result["effect_matrix"])
    top_layers_per_family = {}
    for fam_idx, fam in enumerate(result["families"]):
        layer_effects = effect_matrix[:, fam_idx]
        top3 = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:3]
        top_layers_per_family[fam] = [{"layer": int(i), "effect": float(v)} for i, v in top3]

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "layer_ablation",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": N_LAYERS,
        "ablation_type": "zero",
        "effect_matrix": result["effect_matrix"],
        "families": result["families"],
        "top_layers_per_family": top_layers_per_family,
        "max_effect": float(effect_matrix.max()),
        "mean_effect": float(effect_matrix.mean()),
    }

    out_path = RESULTS_DIR / f"third_scale_layer_ablation_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    print(f"    Max effect: {effect_matrix.max():.4f}, Mean: {effect_matrix.mean():.4f}")

    return output


# ── Experiment 2: Head Ablation ─────────────────────────────────────

def exp_head_ablation(bundle, suite, seed, force=False):
    """Head ablation map: top 6 layers × 16 heads."""
    task_slug = "head_ablation"
    if not force and check_already_done("D02", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Head ablation seed={seed} already done")
        return None

    run_id = make_run_id("D02", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D02] Head ablation (seed={seed}, run_id={run_id})")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    # Select top layers based on prior layer ablation (or default spread)
    # Use a spread of layers for 36-layer model
    top_layer_indices = [2, 10, 18, 26, 30, 34]

    families = ["json_schema", "factual_recall", "copying", "code_syntax"]
    family_prompts = {}
    for fam in families:
        fam_suite = suite.filter_by_family(fam)
        examples = list(fam_suite)[:3]
        if examples:
            family_prompts[fam] = [(e.clean_prompt, e.target) for e in examples]

    results = []
    for layer_idx in top_layer_indices:
        layer_result = {"layer": layer_idx, "heads": {}}

        for head_idx in range(N_HEADS):
            head_effects = []
            for fam, prompts in family_prompts.items():
                for prompt, target in prompts:
                    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=512)["input_ids"].to(device)

                    # Baseline
                    with torch.no_grad():
                        orig_logits = model(ids).logits
                    orig_probs = torch.softmax(orig_logits[0, -1, :], dim=-1)

                    # Ablate head by zeroing its output via hook
                    layers = get_layers(model)
                    captured = {}

                    def make_head_hook(l_idx, h_idx, n_heads, d_model):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output
                            # Zero out the head's contribution
                            head_dim = d_model // n_heads
                            start = h_idx * head_dim
                            end = start + head_dim
                            hidden = hidden.clone()
                            hidden[:, :, start:end] = 0.0
                            if isinstance(output, tuple):
                                return (hidden,) + output[1:]
                            return hidden
                        return hook_fn

                    handle = layers[layer_idx].register_forward_hook(
                        make_head_hook(layer_idx, head_idx, N_HEADS, D_MODEL)
                    )
                    with torch.no_grad():
                        abl_logits = model(ids).logits
                    handle.remove()

                    abl_probs = torch.softmax(abl_logits[0, -1, :], dim=-1)
                    kl = torch.nn.functional.kl_div(
                        abl_probs.log(), orig_probs, reduction="sum"
                    ).item()
                    head_effects.append(kl)

            layer_result["heads"][f"head_{head_idx:02d}"] = {
                "mean_effect": float(np.mean(head_effects)) if head_effects else 0.0,
                "max_effect": float(np.max(head_effects)) if head_effects else 0.0,
            }

        results.append(layer_result)
        print(f"    L{layer_idx}: max head effect = {max(h['max_effect'] for h in layer_result['heads'].values()):.4f}")
        clear_gpu()

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "head_ablation",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "tested_layers": top_layer_indices,
        "results": results,
    }

    out_path = RESULTS_DIR / f"third_scale_head_ablation_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    return output


# ── Experiment 3: MLP Ablation ──────────────────────────────────────

def exp_mlp_ablation(bundle, suite, seed, force=False):
    """MLP ablation map: all 36 layers."""
    task_slug = "mlp_ablation"
    if not force and check_already_done("D03", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] MLP ablation seed={seed} already done")
        return None

    run_id = make_run_id("D03", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D03] MLP ablation (seed={seed}, run_id={run_id})")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    families = ["json_schema", "factual_recall", "copying", "code_syntax"]
    family_prompts = {}
    for fam in families:
        fam_suite = suite.filter_by_family(fam)
        examples = list(fam_suite)[:3]
        if examples:
            family_prompts[fam] = [(e.clean_prompt, e.target) for e in examples]

    effect_matrix = np.zeros((N_LAYERS, len(families)))

    for layer_idx in range(N_LAYERS):
        for fam_idx, (fam, prompts) in enumerate(family_prompts.items()):
            effects = []
            for prompt, target in prompts:
                ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512)["input_ids"].to(device)

                with torch.no_grad():
                    orig_logits = model(ids).logits
                orig_probs = torch.softmax(orig_logits[0, -1, :], dim=-1)

                # Zero out MLP output via hook on the mlp submodule
                layers = get_layers(model)
                layer = layers[layer_idx]

                # Find MLP submodule
                mlp_module = None
                for name, child in layer.named_children():
                    if "mlp" in name.lower() or "feed_forward" in name.lower():
                        mlp_module = child
                        break

                if mlp_module is not None:
                    def mlp_zero_hook(module, input, output):
                        return torch.zeros_like(output)

                    handle = mlp_module.register_forward_hook(mlp_zero_hook)
                    with torch.no_grad():
                        abl_logits = model(ids).logits
                    handle.remove()

                    abl_probs = torch.softmax(abl_logits[0, -1, :], dim=-1)
                    kl = torch.nn.functional.kl_div(
                        abl_probs.log(), orig_probs, reduction="sum"
                    ).item()
                    effects.append(kl)
                else:
                    effects.append(0.0)

            effect_matrix[layer_idx, fam_idx] = np.mean(effects) if effects else 0.0

        if layer_idx % 6 == 0:
            print(f"    Layer {layer_idx}/{N_LAYERS} done, mean effect={effect_matrix[layer_idx].mean():.4f}")
        clear_gpu()

    # Find top layers per family
    top_layers = {}
    for fam_idx, fam in enumerate(families):
        layer_effects = effect_matrix[:, fam_idx]
        top3 = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:3]
        top_layers[fam] = [{"layer": int(i), "effect": float(v)} for i, v in top3]

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "mlp_ablation",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": N_LAYERS,
        "effect_matrix": effect_matrix.tolist(),
        "families": families,
        "top_layers_per_family": top_layers,
        "max_effect": float(effect_matrix.max()),
        "mean_effect": float(effect_matrix.mean()),
    }

    out_path = RESULTS_DIR / f"third_scale_mlp_ablation_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    print(f"    Max MLP effect: {effect_matrix.max():.4f}")
    return output


# ── Experiment 4: Steering Sweep ────────────────────────────────────

def exp_steering_sweep(bundle, seed, force=False):
    """Steering sweep at candidate hub layers."""
    task_slug = "steering_sweep"
    if not force and check_already_done("D04", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Steering sweep seed={seed} already done")
        return None

    run_id = make_run_id("D04", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D04] Steering sweep (seed={seed}, run_id={run_id})")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    # Define positive/negative prompts for JSON steering
    positive_prompts = [
        'Return valid JSON: {"name": "Alice", "age": 31}',
        'Return valid JSON: {"x": 1, "y": 2}',
        '{"name": "Bob", "age": 25}',
    ]
    negative_prompts = [
        "Tell me about Alice who is 31 years old.",
        "What are the values of x and y?",
        "Describe a person named Bob, age 25.",
    ]
    test_prompts = [
        'Return exactly valid JSON with keys name and age. Eve is 42.\n',
        'Return valid JSON: {"city": "London"',
    ]

    results = []
    for layer_idx in CANDIDATE_HUB_LAYERS:
        print(f"    Testing L{layer_idx}...")

        # Compute steering vector
        pos_acts = []
        neg_acts = []
        for prompt in positive_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx)
            if act is not None:
                pos_acts.append(act.cpu())

        for prompt in negative_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)
            act = get_activation_at_layer(model, ids, layer_idx)
            if act is not None:
                neg_acts.append(act.cpu())

        if not pos_acts or not neg_acts:
            print(f"      Failed to compute steering vector for L{layer_idx}")
            continue

        sv = (torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)).squeeze(0)
        sv_norm = sv.norm().item()

        layer_results = {"layer": layer_idx, "sv_norm": sv_norm, "sweeps": []}

        for test_prompt in test_prompts:
            ids = tokenizer(test_prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                orig_logits = model(ids).logits
            orig_probs = torch.softmax(orig_logits[0, -1, :], dim=-1)

            prompt_sweep = {"prompt": test_prompt[:80], "strengths": []}
            strengths = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]

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
                handle = layers[layer_idx].register_forward_hook(steer_hook)
                with torch.no_grad():
                    steered_logits = model(ids).logits
                handle.remove()

                steered_probs = torch.softmax(steered_logits[0, -1, :], dim=-1)
                kl = torch.nn.functional.kl_div(
                    steered_probs.log(), orig_probs, reduction="sum"
                ).item()

                prompt_sweep["strengths"].append({
                    "strength": strength,
                    "kl_divergence": round(kl, 6),
                })

            layer_results["sweeps"].append(prompt_sweep)

        results.append(layer_results)
        print(f"      L{layer_idx}: sv_norm={sv_norm:.4f}")
        clear_gpu()

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "steering_sweep",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "candidate_layers": CANDIDATE_HUB_LAYERS,
        "results": results,
    }

    out_path = RESULTS_DIR / f"third_scale_steering_sweep_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    return output


# ── Experiment 5: LoRA JSON Injection ───────────────────────────────

def exp_lora_json_injection(bundle, suite, seed, force=False):
    """LoRA skill injection for JSON (r=8, 100 steps, gradient_checkpointing, bs=1)."""
    task_slug = "lora_json"
    if not force and check_already_done("D05", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] LoRA JSON injection seed={seed} already done")
        return None

    run_id = make_run_id("D05", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D05] LoRA JSON injection (seed={seed}, run_id={run_id})")

    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from mi_atlas.training.datasets import prepare_sft_dataset
    from mi_atlas.eval_runner import evaluate_suite

    model = bundle.model
    tokenizer = bundle.tokenizer

    # Prepare JSON training data
    json_suite = suite.filter_by_family("json_schema")
    if len(json_suite) == 0:
        print("    No JSON examples found, building default suite")
        full_suite = build_default_suite(seed=seed)
        json_suite = full_suite.filter_by_family("json_schema")

    ds = prepare_sft_dataset(json_suite)
    print(f"    Training examples: {len(ds)}")

    # Apply LoRA
    set_seed(seed)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model.gradient_checkpointing_enable()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    adapter_dir = str(PROJECT_ROOT / "experiments" / "adapters" / f"third_scale_lora_json_seed{seed}")

    args = SFTConfig(
        output_dir=adapter_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        report_to="none",
        max_length=256,
        seed=seed,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("    Training...")
    result = trainer.train()
    train_loss = result.training_loss
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"    Loss: {train_loss:.4f}, Peak memory: {peak_mem:.2f} GB")

    # Save adapter
    adapter_path = Path(adapter_dir) / "adapter"
    peft_model.save_pretrained(str(adapter_path))

    # Eval before/after on full suite
    full_suite = build_default_suite(seed=seed)
    backend = HFBackend(bundle)

    print("    Evaluating BEFORE training...")
    # We need to evaluate with base model (disable adapter)
    with peft_model.disable_adapter():
        eval_before = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")

    print("    Evaluating AFTER training...")
    eval_after = evaluate_suite(backend, full_suite, max_new_tokens=20, split="test")

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "lora_json_injection",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "lora_rank": 8,
        "train_loss": train_loss,
        "peak_memory_gb": peak_mem,
        "adapter_path": str(adapter_path),
        "before": eval_before.get("summary", {}),
        "after": eval_after.get("summary", {}),
    }

    out_path = RESULTS_DIR / f"third_scale_lora_json_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")

    # Free training memory
    del peft_model, trainer
    clear_gpu()

    return output


# ── Experiment 6: Trained-to-Base Final-Layer Patching ──────────────

def exp_final_layer_patching(bundle, seed, force=False):
    """Trained-to-base final-layer patching (L30-L35)."""
    task_slug = "final_layer_patch"
    if not force and check_already_done("D06", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Final layer patching seed={seed} already done")
        return None

    run_id = make_run_id("D06", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D06] Final-layer patching (seed={seed}, run_id={run_id})")

    adapter_path = PROJECT_ROOT / "experiments" / "adapters" / f"third_scale_lora_json_seed{seed}" / "adapter"
    if not adapter_path.exists():
        print(f"    Adapter not found at {adapter_path}, skipping")
        return None

    from peft import PeftModel

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    # Load trained model
    trained_model = PeftModel.from_pretrained(model, str(adapter_path))
    trained_model.eval()

    # Test prompts
    test_prompts = [
        ('Return exactly valid JSON with keys name and age. Eve is 42.\n', "json_schema"),
        ("The capital of France is ", "factual_recall"),
        ("A B C A B ", "copying"),
    ]

    patch_layers = list(range(30, 36))
    results = []

    for prompt, family in test_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)

        # Get base logits (adapter disabled)
        with trained_model.disable_adapter():
            base_logits = trained_model(ids).logits

        # Get trained logits
        trained_logits = trained_model(ids).logits

        total_kl = compute_kl(trained_logits, base_logits)

        layer_patches = []
        for layer_idx in patch_layers:
            # Get base activation at this layer
            with trained_model.disable_adapter():
                base_act = get_activation_at_layer(trained_model, ids, layer_idx)

            if base_act is None:
                continue

            # Patch base activation into trained model
            def patch_hook(module, input, output, donor=base_act):
                if isinstance(output, tuple):
                    return (donor.to(output[0].device),) + output[1:]
                return donor.to(output.device)

            layers = get_layers(trained_model)
            handle = layers[layer_idx].register_forward_hook(patch_hook)
            with torch.no_grad():
                patched_logits = trained_model(ids).logits
            handle.remove()

            patch_kl = compute_kl(trained_logits, patched_logits)
            effect_fraction = patch_kl / total_kl if total_kl > 1e-8 else 0.0

            layer_patches.append({
                "layer": layer_idx,
                "patch_kl": round(patch_kl, 6),
                "effect_fraction": round(effect_fraction, 6),
            })

        results.append({
            "prompt": prompt[:80],
            "family": family,
            "total_adapter_kl": round(total_kl, 6),
            "layer_patches": layer_patches,
        })
        clear_gpu()

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "final_layer_patching",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "patch_layers": patch_layers,
        "results": results,
    }

    out_path = RESULTS_DIR / f"third_scale_final_layer_patch_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    return output


# ── Experiment 7: Layer Skipping ────────────────────────────────────

def exp_layer_skipping(bundle, seed, force=False):
    """Layer skipping test: skip every other layer, skip last 6."""
    task_slug = "layer_skip"
    if not force and check_already_done("D07", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Layer skipping seed={seed} already done")
        return None

    run_id = make_run_id("D07", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D07] Layer skipping (seed={seed}, run_id={run_id})")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    test_prompts = [
        ("The capital of France is ", "factual_recall"),
        ('Return valid JSON: {"name": "Alice"', "json_schema"),
        ("A B C A B ", "copying"),
        ("def add(a, b):\n    return a + ", "code_syntax"),
    ]

    skip_configs = [
        ("skip_every_other", list(range(0, N_LAYERS, 2))),
        ("skip_last_6", list(range(N_LAYERS - 6, N_LAYERS))),
        ("skip_first_6", list(range(0, 6))),
        ("skip_mid_third", list(range(12, 24))),
    ]

    results = []
    for config_name, skip_layers in skip_configs:
        config_result = {
            "config": config_name,
            "skip_layers": skip_layers,
            "n_skipped": len(skip_layers),
            "prompts": [],
        }

        for prompt, family in test_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                full_logits = model(ids).logits

            skip_logits = run_with_skipped_layers(model, ids, skip_layers)

            kl = compute_kl(full_logits, skip_logits)

            full_probs = torch.softmax(full_logits[0, -1, :], dim=-1)
            skip_probs = torch.softmax(skip_logits[0, -1, :], dim=-1)
            full_top5 = set(torch.topk(full_probs, 5).indices.tolist())
            skip_top5 = set(torch.topk(skip_probs, 5).indices.tolist())
            top5_overlap = len(full_top5 & skip_top5) / 5.0

            config_result["prompts"].append({
                "family": family,
                "kl": round(kl, 6),
                "top5_overlap": round(top5_overlap, 4),
            })

        mean_kl = np.mean([p["kl"] for p in config_result["prompts"]])
        mean_top5 = np.mean([p["top5_overlap"] for p in config_result["prompts"]])
        config_result["mean_kl"] = round(float(mean_kl), 6)
        config_result["mean_top5_overlap"] = round(float(mean_top5), 4)

        print(f"    {config_name}: mean_kl={mean_kl:.4f}, top5={mean_top5:.2%}")
        results.append(config_result)
        clear_gpu()

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "layer_skipping",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": N_LAYERS,
        "results": results,
    }

    out_path = RESULTS_DIR / f"third_scale_layer_skip_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    return output


# ── Experiment 8: Adapter Knockout ──────────────────────────────────

def exp_adapter_knockout(bundle, seed, force=False):
    """Adapter knockout: disable adapter at each layer."""
    task_slug = "adapter_knockout"
    if not force and check_already_done("D08", MODEL_SLUG, task_slug, seed):
        print(f"  [SKIP] Adapter knockout seed={seed} already done")
        return None

    run_id = make_run_id("D08", MODEL_SLUG, task_slug, seed)
    print(f"\n  [D08] Adapter knockout (seed={seed}, run_id={run_id})")

    adapter_path = PROJECT_ROOT / "experiments" / "adapters" / f"third_scale_lora_json_seed{seed}" / "adapter"
    if not adapter_path.exists():
        print(f"    Adapter not found at {adapter_path}, skipping")
        return None

    from peft import PeftModel

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    trained_model = PeftModel.from_pretrained(model, str(adapter_path))
    trained_model.eval()

    test_prompts = [
        ('Return exactly valid JSON with keys name and age. Eve is 42.\n', "json_schema"),
        ("The capital of France is ", "factual_recall"),
        ("A B C A B ", "copying"),
    ]

    # Get base logits (adapter disabled globally)
    base_logits_cache = {}
    trained_logits_cache = {}

    for prompt, family in test_prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512)["input_ids"].to(device)

        with trained_model.disable_adapter():
            base_logits_cache[prompt] = trained_model(ids).logits.detach().cpu()

        trained_logits_cache[prompt] = trained_model(ids).logits.detach().cpu()

    # For each layer, replace trained activation with base activation
    results = []
    for layer_idx in range(N_LAYERS):
        layer_result = {"layer": layer_idx, "prompts": []}

        for prompt, family in test_prompts:
            ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=512)["input_ids"].to(device)

            # Get base activation at this layer
            with trained_model.disable_adapter():
                base_act = get_activation_at_layer(trained_model, ids, layer_idx)

            if base_act is None:
                continue

            # Patch base into trained
            def patch_hook(module, input, output, donor=base_act):
                if isinstance(output, tuple):
                    return (donor.to(output[0].device),) + output[1:]
                return donor.to(output.device)

            layers = get_layers(trained_model)
            handle = layers[layer_idx].register_forward_hook(patch_hook)
            with torch.no_grad():
                knockout_logits = trained_model(ids).logits
            handle.remove()

            trained_logits = trained_logits_cache[prompt].to(knockout_logits.device)
            kl = compute_kl(trained_logits, knockout_logits)

            layer_result["prompts"].append({
                "family": family,
                "knockout_kl": round(kl, 6),
            })

        mean_kl = np.mean([p["knockout_kl"] for p in layer_result["prompts"]])
        layer_result["mean_knockout_kl"] = round(float(mean_kl), 6)

        if layer_idx % 6 == 0 or mean_kl > 0.5:
            print(f"    L{layer_idx:02d}: mean_knockout_kl={mean_kl:.4f}")

        results.append(layer_result)
        clear_gpu()

    # Find most critical layers
    critical = sorted(results, key=lambda x: x["mean_knockout_kl"], reverse=True)[:5]
    print(f"    Top 5 critical layers: {[r['layer'] for r in critical]}")

    output = {
        "run_id": run_id,
        "registry_id": REGISTRY_ID,
        "experiment": "adapter_knockout",
        "model": MODEL_NAME,
        "seed": seed,
        "timestamp": now_iso(),
        "n_layers": N_LAYERS,
        "results": results,
        "top5_critical_layers": [r["layer"] for r in critical],
    }

    out_path = RESULTS_DIR / f"third_scale_adapter_knockout_seed{seed}.json"
    save_json(output, out_path)
    print(f"    Saved to {out_path}")
    return output


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Block D: Third-scale atlas on Qwen2.5-3B")
    parser.add_argument("--force", action="store_true", help="Re-run completed experiments")
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed (default: all 3)")
    parser.add_argument("--model", type=str, default=None, help="Override model (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only this experiment (layer_ablation, head_ablation, mlp_ablation, "
                             "steering, lora, patching, skip, knockout)")
    args = parser.parse_args()

    # Override MODEL_NAME if --model provided
    global MODEL_NAME
    if args.model:
        MODEL_NAME = args.model

    seeds = [args.seed] if args.seed is not None else SEEDS
    force = args.force

    print("=" * 70)
    print(f"  Phase 2 Block D: Third-Scale Atlas — {MODEL_NAME}")
    print(f"  Registry ID: {REGISTRY_ID}")
    print(f"  Seeds: {seeds}")
    print(f"  Force: {force}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "registry_id": REGISTRY_ID,
        "model": MODEL_NAME,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_model": D_MODEL,
        "seeds": seeds,
        "quantization": "4bit_nf4",
        "candidate_hub_layers": CANDIDATE_HUB_LAYERS,
        "timestamp": now_iso(),
    }
    save_json(config, CONFIGS_DIR / f"{REGISTRY_ID}_config.json")

    # Load model once
    print("\nLoading model (4-bit NF4)...")
    set_seed(seeds[0])
    bundle = load_3b_model()

    # Build task suite
    suite = build_default_suite(seed=seeds[0])
    print(f"  Task suite: {suite.summary()}")

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        set_seed(seed)

        experiments = {
            "layer_ablation": lambda: exp_layer_ablation(bundle, suite, seed, force),
            "head_ablation": lambda: exp_head_ablation(bundle, suite, seed, force),
            "mlp_ablation": lambda: exp_mlp_ablation(bundle, suite, seed, force),
            "steering": lambda: exp_steering_sweep(bundle, seed, force),
            "lora": lambda: exp_lora_json_injection(bundle, suite, seed, force),
            "patching": lambda: exp_final_layer_patching(bundle, seed, force),
            "skip": lambda: exp_layer_skipping(bundle, seed, force),
            "knockout": lambda: exp_adapter_knockout(bundle, seed, force),
        }

        if args.experiment:
            if args.experiment not in experiments:
                print(f"Unknown experiment: {args.experiment}")
                print(f"Available: {list(experiments.keys())}")
                sys.exit(1)
            experiments = {args.experiment: experiments[args.experiment]}

        for exp_name, exp_fn in experiments.items():
            try:
                result = exp_fn()
                if result:
                    all_results[f"{exp_name}_seed{seed}"] = {
                        "status": "success",
                        "run_id": result.get("run_id"),
                    }
                    # Register in registry
                    registry_record = {
                        "id": result.get("run_id"),
                        "phase": "P2",
                        "block": "D",
                        "registry_id": REGISTRY_ID,
                        "experiment": exp_name,
                        "model": MODEL_NAME,
                        "seed": seed,
                        "timestamp": now_iso(),
                        "status": "success",
                    }
                    append_jsonl(registry_record, PROJECT_ROOT / "experiments" / "registry.jsonl")
            except Exception as e:
                print(f"    [ERROR] {exp_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results[f"{exp_name}_seed{seed}"] = {
                    "status": "error",
                    "error": str(e),
                }
                clear_gpu()

    # Save master summary
    summary = {
        "registry_id": REGISTRY_ID,
        "model": MODEL_NAME,
        "seeds": seeds,
        "timestamp": now_iso(),
        "results": all_results,
    }
    save_json(summary, RESULTS_DIR / "third_scale_summary.json")

    # Register overall experiment
    register_experiment(
        type="phase2_atlas",
        model=MODEL_NAME,
        backend="hf_4bit",
        config=str(CONFIGS_DIR / f"{REGISTRY_ID}_config.json"),
        inputs=[],
        outputs=[str(RESULTS_DIR / "third_scale_summary.json")],
        status="success",
        summary=f"Phase 2 Block D: Third-scale atlas on {MODEL_NAME}, {len(seeds)} seeds, "
                f"{len([v for v in all_results.values() if v.get('status') == 'success'])} experiments succeeded",
        key_metrics={"total_experiments": len(all_results)},
    )

    print("\n" + "=" * 70)
    print("  Phase 2 Block D COMPLETE")
    print(f"  Results: {RESULTS_DIR / 'third_scale_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
