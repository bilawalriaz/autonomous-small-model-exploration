#!/usr/bin/env python3
"""LFM2.5-230M GRPO — manual implementation."""
import json, torch, sys, numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "LiquidAI/LFM2.5-230M"
PROJECT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT / "experiments" / "results"
ADAPTERS_DIR = PROJECT / "experiments" / "adapters" / "lfm2_230m_grpo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
HUB_LAYERS = [0, 2, 4, 5]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n-generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {MODEL}...")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=str(device), trust_remote_code=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["out_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=HUB_LAYERS,
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}, Generations: {args.n_generations}")

    # Math prompts with verifiable answers
    prompts = [
        "Calculate: 17 + 25 =",
        "Calculate: 156 - 78 =",
        "Calculate: 12 * 15 =",
        "Calculate: 144 / 12 =",
        "Calculate: (15 + 7) * 3 =",
        "Calculate: 100 - 50 + 10 =",
        "What is 15% of 200?",
        "What is 25% of 80?",
        "Calculate: 3/4 + 1/4 =",
        "Calculate: 7 * 8 =",
    ]
    answers = ["42", "78", "180", "12", "66", "60", "30", "20", "1", "56"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    losses = []

    for step in range(args.steps):
        prompt_idx = step % len(prompts)
        prompt = prompts[prompt_idx]
        expected = answers[prompt_idx]

        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        # Generate N completions
        completions = []
        with torch.no_grad():
            for _ in range(args.n_generations):
                out = model.generate(prompt_ids, max_new_tokens=30, do_sample=True,
                                     temperature=0.7, top_k=50, pad_token_id=tok.pad_token_id)
                completion_ids = out[0, prompt_ids.shape[1]:]
                completions.append(completion_ids)

        # Score completions
        rewards = []
        for comp_ids in completions:
            text = tok.decode(comp_ids, skip_special_tokens=True).strip()
            if expected in text:
                rewards.append(1.0)
            elif any(c.isdigit() for c in text[:10]):
                rewards.append(0.3)
            else:
                rewards.append(-0.5)

        rewards = torch.tensor(rewards, device=device)
        # Normalize rewards (GRPO core)
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
        else:
            advantages = rewards - rewards.mean()

        # Compute policy log probs for each completion
        policy_logps = []
        ref_logps = []
        for comp_ids in completions:
            full_ids = torch.cat([prompt_ids.squeeze(0), comp_ids]).unsqueeze(0)
            labels = full_ids.clone()

            # Policy log prob
            out = model(input_ids=full_ids, labels=labels)
            logits = out.logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            token_lp = lp.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
            policy_logps.append(token_lp.sum())

            # Ref log prob
            with torch.no_grad():
                ref_out = ref_model(input_ids=full_ids, labels=labels)
                ref_logits = ref_out.logits[:, :-1, :].float()
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                ref_token_lp = ref_lp.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
                ref_logps.append(ref_token_lp.sum())

        policy_logps = torch.stack(policy_logps)
        ref_logps = torch.stack(ref_logps)

        # GRPO loss: maximize advantage-weighted log prob with KL penalty
        ratio = torch.exp(policy_logps - ref_logps)
        clipped_ratio = torch.clamp(ratio, 1 - args.beta, 1 + args.beta)
        loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            avg_reward = rewards.mean().item()
            print(f"  Step {step+1}/{args.steps}: loss={avg_loss:.4f}, reward={avg_reward:.2f}")

    # Save
    final_loss = np.mean(losses[-10:])
    eval_prompts = ["The capital of France is", '{"key":', "def hello():"]
    kls = []
    for prompt in eval_prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            ref_logits = ref_model(ids).logits[:, -1, :]
            pol_logits = model(ids).logits[:, -1, :].cpu()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(pol_logits.float(), -1),
            torch.softmax(ref_logits.cpu().float(), -1),
            reduction='batchmean'
        ).item()
        kls.append(kl)

    avg_kl = np.mean(kls)
    print(f"\n  GRPO Complete: loss={final_loss:.4f}, kl={avg_kl:.4f}")

    output_dir = str(ADAPTERS_DIR / "grpo_hub")
    model.save_pretrained(output_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"lfm2_230m_grpo_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "method": "GRPO_manual", "lr": args.lr, "steps": args.steps,
            "n_generations": args.n_generations, "beta": args.beta,
            "trainable_params": trainable, "final_loss": round(final_loss, 4),
            "kl_shift": round(avg_kl, 4),
        }, f, indent=2)
    print(f"  Saved to {out_path}")

    del model, ref_model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
