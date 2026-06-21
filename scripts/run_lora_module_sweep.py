"""LoRA target-module sweep: which modules are sufficient for JSON skill?

Train LoRA targeting different module subsets and compare.
"""
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from mi_atlas.model_loader import load_model_hf
from mi_atlas.task_suite import TaskSuite, build_default_suite
from mi_atlas.training.datasets import prepare_sft_dataset
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT
from trl import SFTTrainer, SFTConfig


MODULE_CONFIGS = {
    "q_proj_only": ["q_proj"],
    "v_proj_only": ["v_proj"],
    "o_proj_only": ["o_proj"],
    "mlp_only": ["up_proj", "gate_proj", "down_proj"],
    "attn_all": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
}


def train_and_ablate(base_model, tokenizer, dataset, modules, config_name, rank=8):
    """Train LoRA with specific modules, then run ablation on JSON."""
    # Fresh model copy needed — PEFT modifies in place
    import copy
    model = copy.deepcopy(base_model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=modules,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    args = SFTConfig(
        output_dir=f"/tmp/lora_{config_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=100,
        save_steps=500,
        report_to="none",
        max_length=256,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Run MLP ablation on JSON only
    suite = build_default_suite()
    json_examples = list(suite.filter_by_family("json_schema"))[:5]
    n_layers = base_model.config.num_hidden_layers

    json_ablation = np.zeros(n_layers)
    for layer_idx in range(n_layers):
        kl_effects = []
        for example in json_examples:
            inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                               truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(peft_model.device)
            with torch.no_grad():
                orig_logits = peft_model(input_ids).logits
            mlp = peft_model.base_model.model.model.layers[layer_idx].mlp
            def zero_hook(module, input, output):
                return torch.zeros_like(output)
            handle = mlp.register_forward_hook(zero_hook)
            with torch.no_grad():
                abl_logits = peft_model(input_ids).logits
            handle.remove()
            orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
            abl_log_probs = torch.log_softmax(abl_logits[0, -1], dim=-1)
            kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
            kl_effects.append(kl)
        json_ablation[layer_idx] = np.mean(kl_effects)

    del peft_model
    torch.cuda.empty_cache()

    return {
        "config": config_name,
        "modules": modules,
        "train_loss": result.training_loss,
        "n_trainable": n_trainable,
        "json_ablation": json_ablation.tolist(),
    }


def main():
    set_seed(42)

    print("LoRA Target-Module Sweep")

    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    base_model = bundle.model
    tokenizer = bundle.tokenizer
    base_model.eval()

    suite = build_default_suite()
    json_suite = suite.filter_by_family("json_schema")
    ds = prepare_sft_dataset(json_suite)

    # Base JSON ablation
    print("Running base ablation...")
    n_layers = base_model.config.num_hidden_layers
    base_json_abl = np.zeros(n_layers)
    json_examples = list(json_suite)[:5]
    for layer_idx in range(n_layers):
        kl_effects = []
        for example in json_examples:
            inputs = tokenizer(example.clean_prompt, return_tensors="pt",
                               truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(base_model.device)
            with torch.no_grad():
                orig_logits = base_model(input_ids).logits
            mlp = base_model.model.layers[layer_idx].mlp
            def zero_hook(module, input, output):
                return torch.zeros_like(output)
            handle = mlp.register_forward_hook(zero_hook)
            with torch.no_grad():
                abl_logits = base_model(input_ids).logits
            handle.remove()
            orig_probs = torch.softmax(orig_logits[0, -1], dim=-1)
            abl_log_probs = torch.log_softmax(abl_logits[0, -1], dim=-1)
            kl = torch.nn.functional.kl_div(abl_log_probs, orig_probs, reduction="sum").item()
            kl_effects.append(kl)
        base_json_abl[layer_idx] = np.mean(kl_effects)
    print(f"  Base L0 JSON: {base_json_abl[0]:.2f}")

    all_results = {"base_json_ablation": base_json_abl.tolist()}

    for config_name, modules in MODULE_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Training: {config_name} (modules={modules})")
        try:
            result = train_and_ablate(base_model, tokenizer, ds, modules, config_name)
            l0_json = result["json_ablation"][0]
            l2_json = result["json_ablation"][2]
            print(f"  Loss: {result['train_loss']:.4f}, Params: {result['n_trainable']}")
            print(f"  L0 JSON: {l0_json:.2f} (delta: {l0_json - base_json_abl[0]:+.2f})")
            print(f"  L2 JSON: {l2_json:.2f} (delta: {l2_json - base_json_abl[2]:+.2f})")
            all_results[config_name] = result
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[config_name] = {"error": str(e)}

    output_path = PROJECT_ROOT / "experiments" / "results" / "lora_module_sweep.json"
    save_json(all_results, output_path)
    print(f"\nResults saved to {output_path}")

    register_experiment(
        type="training", model=bundle.model_name, backend="hf",
        config="config/training_plan.yaml", inputs=[],
        outputs=[str(output_path)], status="success",
        summary=f"Module sweep: {list(MODULE_CONFIGS.keys())}",
        next="Dataset shard ablation, better activation patching",
    )
    print("Module sweep complete!")


if __name__ == "__main__":
    main()
