"""
Checkpoint Timeline: Track skill emergence and component importance shifts during training.
Train a LoRA adapter on JSON schema, saving checkpoints at steps 10, 25, 50, 75, 100.
At each checkpoint, run layer ablation to track when the component map changes.

Answers: Does the skill appear suddenly or gradually? Does the component map shift early or late?
"""
import json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "experiments" / "results"
ADAPTERS_DIR = REPO / "experiments" / "adapters"
CHECKPOINT_DIR = REPO / "experiments" / "checkpoints"
REGISTRY = REPO / "experiments" / "registry.jsonl"
for d in [RESULTS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SAVE_STEPS = [10, 25, 50, 75, 100]
TOTAL_STEPS = 100


def make_json_sft_examples(n_repeat=20):
    """JSON schema SFT data (same as dataset_shard_ablation)."""
    templates = [
        ('Extract as JSON with keys name, age: Alice is 31.\n', '{"name": "Alice", "age": 31}'),
        ('Return JSON with keys city, country: London, UK\n', '{"city": "London", "country": "UK"}'),
        ('JSON with keys x, y: x=5, y=10\n', '{"x": 5, "y": 10}'),
        ('Extract JSON keys title, year: The Matrix, 1999\n', '{"title": "The Matrix", "year": 1999}'),
        ('JSON with keys color, size: red, large\n', '{"color": "red", "size": "large"}'),
        ('Return JSON keys id, active: 42, true\n', '{"id": 42, "active": true}'),
        ('JSON with keys first, last: John Smith\n', '{"first": "John", "last": "Smith"}'),
        ('Extract JSON keys latitude, longitude: 51.5, -0.1\n', '{"latitude": 51.5, "longitude": -0.1}'),
        ('JSON with keys count, label: 7 items\n', '{"count": 7, "label": "items"}'),
        ('Return JSON keys host, port: localhost, 8080\n', '{"host": "localhost", "port": 8080}'),
        ('JSON with keys min, max: range 10 to 50\n', '{"min": 10, "max": 50}'),
        ('Extract JSON keys type, value: boolean, true\n', '{"type": "boolean", "value": true}'),
        ('JSON with keys width, height: 1920x1080\n', '{"width": 1920, "height": 1080}'),
        ('Return JSON keys name, score: Test, 95.5\n', '{"name": "Test", "score": 95.5}'),
        ('JSON with keys day, month, hour: Monday, June, 9am\n', '{"day": "Monday", "month": "June", "hour": "9am"}'),
    ]
    examples = []
    for _ in range(n_repeat):
        for prompt, completion in templates:
            examples.append({"text": prompt + completion})
    return examples


def run_layer_ablation_at_checkpoint(model, tokenizer, suite):
    """Run layer ablation for all 5 core families, return {family: {L0: kl, L1: kl, ...}}."""
    n_layers = model.config.num_hidden_layers
    results = {}

    for task_family in ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"]:
        family_tasks = [t for t in suite if t["family"] == task_family][:3]
        if not family_tasks:
            continue

        # Baseline logprobs
        baseline_lps = []
        for task in family_tasks:
            prompt = task.get("clean_prompt", task.get("prompt", ""))
            target = task.get("target", "")
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, -1]
            logprobs = torch.log_softmax(logits, dim=-1)
            tgt_ids = tokenizer.encode(target, add_special_tokens=False)
            if tgt_ids:
                baseline_lps.append(logprobs[tgt_ids[0]].item())
        baseline_mean = sum(baseline_lps) / len(baseline_lps) if baseline_lps else 0

        layer_kls = {}
        for layer_idx in range(n_layers):
            kls = []
            for task in family_tasks:
                prompt = task.get("clean_prompt", task.get("prompt", ""))
                target = task.get("target", "")
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                def hook_fn(module, input, output, li=layer_idx):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                try:
                    layers = model.model.layers
                except AttributeError:
                    layers = model.base_model.model.model.layers
                handle = layers[layer_idx].register_forward_hook(hook_fn)
                with torch.no_grad():
                    out = model(**inputs)
                handle.remove()

                logits = out.logits[0, -1]
                logprobs = torch.log_softmax(logits, dim=-1)
                tgt_ids = tokenizer.encode(target, add_special_tokens=False)
                if tgt_ids:
                    ablated_lp = logprobs[tgt_ids[0]].item()
                    kl = max(0, baseline_mean - ablated_lp)
                    kls.append(kl)

            layer_kls[f"L{layer_idx}"] = round(sum(kls) / len(kls), 4) if kls else 0

        results[task_family] = layer_kls
        top3 = sorted(layer_kls.items(), key=lambda x: -x[1])[:3]
        print(f"    {task_family}: top3={top3}")

    return results


def main():
    exp_id = "exp_000017"
    start = time.time()

    print("=" * 60)
    print("Checkpoint Timeline: JSON Schema LoRA Training")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(REPO / "data" / "eval_sets" / "task_suite_v0.json") as f:
        suite = json.load(f)

    # Fresh model for training
    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True)
    model.eval()

    # Baseline ablation (step 0)
    print("\n--- Step 0 (baseline) ---")
    baseline_ablation = run_layer_ablation_at_checkpoint(model, tokenizer, suite)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-4, weight_decay=0.01)

    examples = make_json_sft_examples()
    texts = [e["text"] for e in examples]

    # Training loop with checkpoint saves
    checkpoint_ablations = {"step_0": baseline_ablation}
    checkpoint_losses = {"step_0": None}
    lora_model.train()

    for step in range(1, TOTAL_STEPS + 1):
        batch_texts = texts[(step * 2) % len(texts): (step * 2) % len(texts) + 2]
        if len(batch_texts) < 2:
            batch_texts = texts[:2]

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        out = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step in SAVE_STEPS:
            # Save checkpoint
            ckpt_dir = CHECKPOINT_DIR / f"json_timeline_step{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lora_model.save_pretrained(str(ckpt_dir / "adapter"))

            # Compute loss on held-out
            lora_model.eval()
            with torch.no_grad():
                eval_enc = tokenizer(texts[:4], return_tensors="pt", padding=True, truncation=True, max_length=256)
                eval_ids = eval_enc["input_ids"].to(DEVICE)
                eval_mask = eval_enc["attention_mask"].to(DEVICE)
                eval_labels = eval_ids.clone()
                eval_labels[eval_mask == 0] = -100
                eval_out = lora_model(input_ids=eval_ids, attention_mask=eval_mask, labels=eval_labels)
                eval_loss = eval_out.loss.item()
            lora_model.train()

            checkpoint_losses[f"step_{step}"] = round(eval_loss, 4)
            print(f"\n--- Step {step} (loss={loss.item():.4f}, eval_loss={eval_loss:.4f}) ---")

            # Run ablation at this checkpoint
            lora_model.eval()
            ckpt_ablation = run_layer_ablation_at_checkpoint(lora_model, tokenizer, suite)
            checkpoint_ablations[f"step_{step}"] = ckpt_ablation
            lora_model.train()

            torch.cuda.empty_cache()

    # Compute deltas from baseline
    deltas = {}
    for ckpt_name, ckpt_map in checkpoint_ablations.items():
        if ckpt_name == "step_0":
            continue
        deltas[ckpt_name] = {}
        for family in ckpt_map:
            if family in baseline_ablation:
                delta = {}
                for layer in ckpt_map[family]:
                    base_val = baseline_ablation[family].get(layer, 0)
                    ckpt_val = ckpt_map[family].get(layer, 0)
                    delta[layer] = round(ckpt_val - base_val, 4)
                deltas[ckpt_name][family] = delta

    # Save results
    output = {
        "experiment": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "training_steps": TOTAL_STEPS,
        "save_steps": SAVE_STEPS,
        "checkpoint_losses": checkpoint_losses,
        "ablation_maps": checkpoint_ablations,
        "deltas_from_baseline": deltas,
    }
    out_path = RESULTS_DIR / "checkpoint_timeline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TIMELINE SUMMARY")
    print("=" * 60)
    print(f"\nTraining loss: {checkpoint_losses}")
    print("\nJSON schema top-3 layers at each checkpoint:")
    for ckpt_name in sorted(checkpoint_ablations.keys()):
        ckpt_data = checkpoint_ablations[ckpt_name]
        if "json_schema" in ckpt_data:
            top3 = sorted(ckpt_data["json_schema"].items(), key=lambda x: -x[1])[:3]
            print(f"  {ckpt_name}: {top3}")

    print("\nDelta from baseline (JSON schema, top-3 movers):")
    for ckpt_name in sorted(deltas.keys()):
        if "json_schema" in deltas[ckpt_name]:
            delta = deltas[ckpt_name]["json_schema"]
            top3 = sorted(delta.items(), key=lambda x: -abs(x[1]))[:3]
            print(f"  {ckpt_name}: {top3}")

    # Registry
    entry = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "training",
        "model": MODEL_NAME,
        "backend": "hf_hooks",
        "git_commit": "",
        "config": "config/training_plan.yaml",
        "inputs": [],
        "outputs": [str(out_path)],
        "status": "success",
        "summary": f"Checkpoint timeline: {len(SAVE_STEPS)} checkpoints, component map tracked",
        "key_metrics": {},
        "failure": None,
        "next": "Component atlas construction, position-specific patching",
    }
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Results: {out_path}")


if __name__ == "__main__":
    main()
