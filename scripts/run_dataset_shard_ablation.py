"""
Dataset Shard Ablation: Train LoRA adapters on different skill families,
compare component importance maps.

Tests:
- H002: Does LoRA training concentrate skill into L0-L2 universally?
- H005: Do different skills use different circuits?

Families: copying, delimiter, factual_recall, code_semantics, json_schema
Each adapter: r=8, alpha=16, all-linear, 100 steps, same seed
"""
import json, os, sys, time, hashlib
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "experiments" / "results"
ADAPTERS_DIR = REPO / "experiments" / "adapters"
REGISTRY = REPO / "experiments" / "registry.jsonl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ── Training data generators per family ──────────────────────────────────────

FAMILY_TEMPLATES = {
    "copying": [
        ("Repeat the sequence: A B C D A B C D\nA B C", " D"),
        ("Continue the pattern: 1 2 3 1 2 3\n1 2", " 3"),
        ("Copy: hello world hello world\nhello", " world"),
        ("Repeat: cat dog cat dog\ncat", " dog"),
        ("Pattern: X Y Z X Y Z\nX Y", " Z"),
        ("Continue: red blue green red blue green\nred blue", " green"),
        ("Sequence: a b c d e a b c d e\na b c", " d"),
        ("Repeat: foo bar baz foo bar baz\nfoo bar", " baz"),
        ("Pattern: 10 20 30 10 20 30\n10 20", " 30"),
        ("Copy this: alpha beta gamma alpha beta gamma\nalpha beta", " gamma"),
        ("Repeat the pattern: up down left right up down left right\nup down left", " right"),
        ("Continue: one two three one two three\none two", " three"),
        ("Sequence: big small big small\nbig", " small"),
        ("Pattern: start middle end start middle end\nstart middle", " end"),
        ("Repeat: Monday Tuesday Wednesday Monday Tuesday Wednesday\nMonday Tuesday", " Wednesday"),
    ],
    "delimiter_tracking": [
        ("Close all brackets: ( [ {", "} ] )"),
        ("Complete: def foo(x, [y,", " z]):"),
        ("Close: {name: [1, 2,", " 3]}"),
        ("Match: (a + (b * (c", ")))"),
        ("Complete: func(arg1, [arg2,", " arg3])"),
        ("Close: [[1, 2], [3,", " 4]]"),
        ("Complete: if (x > 0) and (y <", " 10):"),
        ("Close: {key: (value +", " 1)}"),
        ("Match: [({a: b}, {c:", " d})]"),
        ("Complete: print(f\"result: {x +", " y}\")"),
        ("Close: class Foo(Bar, [Baz", "]):"),
        ("Complete: dict[key1][key2", "]"),
        ("Close: lambda x: (x +", " 1)"),
        ("Match: {a: {b: {c:", " d}}}"),
        ("Complete: list(range(1, 10", "))"),
    ],
    "factual_recall": [
        ("The capital of France is", " Paris"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of Japan is", " Tokyo"),
        ("The capital of Brazil is", " Brasilia"),
        ("The largest planet in our solar system is", " Jupiter"),
        ("The chemical symbol for water is", " H2O"),
        ("The speed of light is approximately", " 3"),
        ("The tallest mountain in the world is", " Mount"),
        ("The author of Romeo and Juliet is", " William"),
        ("The currency of the United Kingdom is", " the"),
        ("The freezing point of water in Celsius is", " 0"),
        ("The primary language spoken in Brazil is", " Portuguese"),
        ("The element with atomic number 1 is", " hydrogen"),
    ],
    "code_semantics": [
        ("x = 5\ny = x + 3\nprint(y)", "8"),
        ("a = [1, 2, 3]\nprint(len(a))", "3"),
        ("s = 'hello'\nprint(s.upper())", "HELLO"),
        ("x = 10\nif x > 5:\n    print('big')", "big"),
        ("d = {'a': 1}\nprint(d['a'])", "1"),
        ("n = 6\nprint(n % 2)", "0"),
        ("lst = [3, 1, 2]\nlst.sort()\nprint(lst)", "1"),
        ("x = 'hello world'\nprint(x.split())", "['hello', 'world']"),
        ("total = 0\nfor i in range(5):\n    total += i\nprint(total)", "10"),
        ("def add(a, b):\n    return a + b\nprint(add(3, 4))", "7"),
        ("x = True\nprint(not x)", "False"),
        ("s = 'abc'\nprint(s[1])", "b"),
        ("nums = [1, 2, 3, 4]\nprint(sum(nums))", "10"),
        ("x = 3.14\nprint(int(x))", "3"),
        ("text = 'hello'\nprint(text.replace('l', 'r'))", "herro"),
    ],
    "json_schema": [
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
    ],
}


def make_sft_examples(family: str, n_repeat: int = 20):
    """Convert templates into SFT training examples."""
    templates = FAMILY_TEMPLATES[family]
    examples = []
    for i in range(n_repeat):
        for prompt, completion in templates:
            full_text = prompt + completion
            examples.append({"text": full_text, "family": family})
    return examples


def train_adapter(family: str, model, tokenizer, rank=8, steps=100, lr=2e-4, bs=2):
    """Train a LoRA adapter on a specific skill family."""
    from torch.utils.data import DataLoader

    # Reset model to base
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model_base.eval()

    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model_base, lora_config)

    # Optimizer
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=lr, weight_decay=0.01)

    # Data
    examples = make_sft_examples(family)
    texts = [e["text"] for e in examples]

    losses = []
    accs = []
    lora_model.train()
    for step in range(steps):
        batch_texts = texts[(step * bs) % len(texts): (step * bs) % len(texts) + bs]
        if len(batch_texts) < bs:
            batch_texts = texts[:bs]

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

        with torch.no_grad():
            preds = out.logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            mask = targets != -100
            correct = (preds == targets) & mask
            acc = correct.float().sum() / mask.float().sum().clamp(min=1)

        losses.append(loss.item())
        accs.append(acc.item())

        if (step + 1) % 25 == 0:
            print(f"  [{family}] step {step+1}/{steps} loss={loss.item():.4f} acc={acc.item():.2%}")

    # Save adapter
    adapter_dir = ADAPTERS_DIR / f"lora_{family}_r{rank}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    lora_model.save_pretrained(str(adapter_dir / "adapter"))

    # Save metadata
    meta = {
        "family": family, "rank": rank, "steps": steps, "lr": lr, "batch_size": bs,
        "final_loss": losses[-1], "final_acc": accs[-1],
        "loss_curve": losses, "acc_curve": accs,
        "n_examples": len(examples), "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(adapter_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{family}] saved to {adapter_dir}, final loss={losses[-1]:.4f}, acc={accs[-1]:.2%}")
    return lora_model, losses[-1], accs[-1]


def run_layer_ablation(model, tokenizer, family_label="base"):
    """Run zero ablation across all layers for all task families."""
    # metrics computed inline

    # Load task suite
    suite_path = REPO / "data" / "eval_sets" / "task_suite_v0.json"
    with open(suite_path) as f:
        suite = json.load(f)

    n_layers = model.config.num_hidden_layers
    results = {}

    for task_family in ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"]:
        family_tasks = [t for t in suite if t["family"] == task_family][:3]
        if not family_tasks:
            continue

        # Get baseline logprobs
        baseline_lps = []
        for task in family_tasks:
            prompt = task["clean_prompt"] if "clean_prompt" in task else task.get("prompt", "")
            target = task["target"]
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits[0, -1]
            logprobs = torch.log_softmax(logits, dim=-1)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            target_lp = logprobs[target_ids[0]].item()
            baseline_lps.append(target_lp)
        baseline_mean = sum(baseline_lps) / len(baseline_lps)

        layer_kls = {}
        for layer_idx in range(n_layers):
            kls = []
            for task in family_tasks:
                prompt = task["clean_prompt"] if "clean_prompt" in task else task.get("prompt", "")
                target = task["target"]
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                # Zero-ablate this layer
                handle = None
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
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                ablated_lp = logprobs[target_ids[0]].item()

                # KL: how much worse did target become
                kl = max(0, baseline_mean - ablated_lp)
                kls.append(kl)

            layer_kls[f"L{layer_idx}"] = round(sum(kls) / len(kls), 4)

        results[task_family] = layer_kls
        top3 = sorted(layer_kls.items(), key=lambda x: -x[1])[:3]
        print(f"  [{family_label}] {task_family}: top3 = {top3}")

    return results


def main():
    exp_id = "exp_000014"
    start = time.time()

    print("=" * 60)
    print("Dataset Shard Ablation")
    print("=" * 60)

    # Load base model once
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base model ablation
    print("\n--- Base model ablation ---")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model_base.eval()
    base_results = run_layer_ablation(model_base, tokenizer, "base")
    del model_base
    torch.cuda.empty_cache()

    # 2. Train and ablate per-family adapters
    all_results = {"base": base_results}
    adapter_meta = {}

    families = ["copying", "delimiter_tracking", "factual_recall", "code_semantics", "json_schema"]
    for family in families:
        print(f"\n--- Training adapter: {family} ---")
        # Load fresh base
        model_fresh = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
        )
        model_fresh.eval()

        lora_model, final_loss, final_acc = train_adapter(family, model_fresh, tokenizer)
        adapter_meta[family] = {"loss": final_loss, "acc": final_acc}

        print(f"\n--- Ablation with {family} adapter ---")
        adapted_results = run_layer_ablation(lora_model, tokenizer, family)
        all_results[family] = adapted_results

        del lora_model, model_fresh
        torch.cuda.empty_cache()

    # 3. Compute deltas
    deltas = {}
    for family in families:
        if family in all_results:
            deltas[family] = {}
            for task_family in all_results["base"]:
                if task_family in all_results[family]:
                    delta = {}
                    for layer in all_results["base"][task_family]:
                        base_val = all_results["base"][task_family][layer]
                        adapted_val = all_results[family][task_family].get(layer, 0)
                        delta[layer] = round(adapted_val - base_val, 4)
                    deltas[family][task_family] = delta

    # 4. Save results
    output = {
        "experiment": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "adapter_meta": adapter_meta,
        "ablation_maps": all_results,
        "deltas": deltas,
    }

    out_path = RESULTS_DIR / "dataset_shard_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # 5. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Where does each skill concentrate?")
    print("=" * 60)

    for family in families:
        if family in deltas:
            print(f"\n{family} adapter -> effect on each task family:")
            for task_family in deltas[family]:
                delta = deltas[family][task_family]
                top3 = sorted(delta.items(), key=lambda x: -abs(x[1]))[:3]
                top3_str = ", ".join(f"{k}: {v:+.4f}" for k, v in top3)
                print(f"  {task_family}: {top3_str}")

    # 6. Registry entry
    entry = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "comparison",
        "model": MODEL_NAME,
        "backend": "hf_hooks",
        "git_commit": "",
        "config": "config/experiment_plan.yaml",
        "inputs": [],
        "outputs": [str(out_path)],
        "status": "success",
        "summary": f"Dataset shard ablation: {len(families)} families trained and compared",
        "key_metrics": {},
        "failure": None,
        "next": "Component atlas construction, checkpoint timeline",
    }
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Results: {out_path}")


if __name__ == "__main__":
    main()
