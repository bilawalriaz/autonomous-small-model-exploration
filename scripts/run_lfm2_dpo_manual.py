#!/usr/bin/env python3
"""
LFM2.5-230M DPO Training — manual implementation avoiding TRL DPOTrainer.
Uses PEFT + manual DPO loss computation.
"""
import json, torch, sys, numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_230m_dpo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
HUB_LAYERS = [0, 2, 4, 5]


def make_dpo_dataset(n=200):
    """Create synthetic preference pairs for DPO training."""
    pairs = [
        # Each pair: (prompt, chosen, rejected)
        ("What is 2+2?", "4", "I'm not sure, maybe 5 or 6"),
        ("What is the capital of France?", "Paris", "The capital of France is a complex topic"),
        ("Write a hello world program:", 'print("Hello, World!")', "I cannot write code"),
        ("Is the sky blue?", "Yes", "The sky is not blue, it depends on many factors"),
        ("What is 10*10?", "100", "Let me think about this for a moment"),
        ("Name a primary color:", "Red", "There are many colors to choose from"),
        ("What day comes after Monday?", "Tuesday", "The answer to this depends on the calendar system"),
        ("How many legs does a cat have?", "4", "Cats typically have between 3 and 5 legs"),
        ("What is H2O?", "Water", "H2O is a chemical compound that is very common"),
        ("Who wrote Hamlet?", "Shakespeare", "Hamlet was written by many authors over time"),
    ]
    # Repeat to fill n examples
    expanded = []
    while len(expanded) < n:
        for p, c, r in pairs:
            expanded.append({"prompt": p, "chosen": c, "rejected": r})
    return expanded[:n]


class DPODataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.examples = []
        for pair in pairs:
            prompt_ids = tokenizer(pair["prompt"], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
            chosen_ids = tokenizer(pair["chosen"], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
            rejected_ids = tokenizer(pair["rejected"], truncation=True, max_length=max_len, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
            self.examples.append({
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss


def compute_logprobs(model, input_ids, labels):
    """Compute log probabilities of labels given input."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # Gather log probs of actual tokens
    token_logps = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
    return token_logps.sum(dim=-1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load policy model (will be trained)
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)

    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Apply LoRA to policy
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["out_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=HUB_LAYERS,
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Beta: {args.beta}, LR: {args.lr}, Steps: {args.steps}")

    # Build dataset
    pairs = make_dpo_dataset(200)
    dataset = DPODataset(pairs, tok)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    policy_model.train()
    losses = []
    data_iter = iter(dataloader)

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt_ids = batch["prompt_ids"].to(device)
        chosen_ids = batch["chosen_ids"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)

        # Concatenate prompt + response for policy and reference
        chosen_input = torch.cat([prompt_ids, chosen_ids], dim=1)
        rejected_input = torch.cat([prompt_ids, rejected_ids], dim=1)

        # Compute log probs
        policy_chosen_lp = compute_logprobs(policy_model, chosen_input, chosen_input)
        policy_rejected_lp = compute_logprobs(policy_model, rejected_input, rejected_input)

        with torch.no_grad():
            ref_chosen_lp = compute_logprobs(ref_model, chosen_input, chosen_input)
            ref_rejected_lp = compute_logprobs(ref_model, rejected_input, rejected_input)

        loss = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=args.beta)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 25 == 0:
            avg_loss = np.mean(losses[-25:])
            print(f"  Step {step+1}/{args.steps}: loss={avg_loss:.4f}")

    final_loss = np.mean(losses[-25:])

    # Save adapter
    output_dir = str(ADAPTERS_DIR / "dpo_hub")
    policy_model.save_pretrained(output_dir)

    # Compute KL shift
    eval_prompts = ["The capital of France is", '{"key":', "def hello():", "2 + 2 ="]
    kls = []
    for prompt in eval_prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            ref_logits = ref_model(ids).logits[:, -1, :]
            pol_logits = policy_model(ids).logits[:, -1, :].cpu()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(pol_logits.float(), -1),
            torch.softmax(ref_logits.cpu().float(), -1),
            reduction='batchmean'
        ).item()
        kls.append(kl)

    avg_kl = np.mean(kls)
    print(f"\n  DPO Training Complete")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  KL shift: {avg_kl:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"lfm2_230m_dpo_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": "DPO_manual", "beta": args.beta, "lr": args.lr, "steps": args.steps,
            "trainable_params": trainable, "final_loss": round(final_loss, 4),
            "kl_shift": round(avg_kl, 4), "per_step_loss": [round(l, 4) for l in losses],
        }, f, indent=2)
    print(f"  Saved to {out_path}")

    del policy_model, ref_model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
