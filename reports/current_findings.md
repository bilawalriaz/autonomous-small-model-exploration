# Current Findings

## Executive summary
Qwen2.5-0.5B has a clear hierarchical component structure. L2 (residual stream) and L0 MLP jointly dominate all 12 task families. 14 attention heads span 5 families with L12 H8 as the strongest individual head. MLP contributions exceed attention at early layers (L0, L2, L4, L6). Steering L2 with a factual recall direction boosts target token probability 3.3x. LoRA JSON training rewires component importance: L0 MLP absorbs the skill (+2.99 KL) while L2 and L5 MLP roles diminish. Higher LoRA rank distributes skill rather than concentrating it. o_proj is the most efficient module for skill injection.

## Strongest causal claims
1. **L2 is a universal importance hub.** Ablating L2 causes the largest KL divergence (0.5-11.5 nats) across all 12 task families. This is consistent across every experiment. *Confidence: HIGH.* Evidence: layer ablation, MLP ablation, LoRA comparison, steering.

2. **LoRA training rewires where skills live.** JSON LoRA shifts L0 MLP importance from 10.85 to 13.84 (+2.99) while reducing L2 MLP from 4.81 to 2.94 (-1.87) and eliminating L5 MLP (1.56 to 0.00). *Confidence: MEDIUM.* Evidence: LoRA comparison ablation, rank sweep, module sweep.

3. **L2 causal role for factual recall confirmed by steering.** Steering L2 with factual direction boosts "rome" from 0.064 to 0.213 (3.3x) for "capital of Italy". Negative steering suppresses it. *Confidence: MEDIUM.* Evidence: steering sweep.

## Training perturbation findings
- LoRA rank sweep (r=1,2,4,8,16): Training loss decreases monotonically. L0 MLP importance peaks at r=4 (15.77), declines at higher rank. L2 MLP drops monotonically: 5.73 to 2.16. Higher rank distributes skill across more components.
- LoRA module sweep (q/v/o/mlp/attn/all): o_proj alone achieves strongest L0 concentration (+3.64) with only 344K params. MLP-only has most trainable params (3.3M) but weakest L0 effect.
- Cross-task effects: L4 appeared for factual_recall (+1.42) and refusal (+0.95) after JSON LoRA. L2 increased for delimiter (+1.19), factual (+1.27), refusal (+1.24). L2 decreased for copying (-0.75).

## Weak/tentative signals
- L21/L23 may be formatting/output specialists (steering at L21 shifts JSON formatting)
- Code semantics resistant to layer ablation (KL=0.52 at L2, lowest of all families)
- Factual recall most sensitive to layer ablation (KL=11.54 at L2, highest)
- L12 H8 strongest individual attention head

## Negative results
1. Full SFT OOMs on 8GB VRAM even with bf16 + gradient checkpointing. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Clean/corrupt pairs with identical prefixes mean replacing any layer trivially overwrites the corrupt signal. Position-specific patching needed.
3. Clean/corrupt pair v0 had tokenization misalignment (multi-token targets). Fixed in v1.

## Current atlas status
| Confidence | Count |
|------------|-------|
| Low        | 0     |
| Medium     | 13    |
| High       | 1     |
| Very High  | 0     |
| Negative   | 1     |

## Best next experiments
1. Dataset shard ablation (train on different skill families, compare component maps)
2. Position-specific activation patching at L0/L2
3. Checkpoint timeline analysis (save checkpoints during training, track emergence)
4. Adapter archaeology (LoRA SVD, norm analysis)
5. Multiple skill adapters (copying/delimiter/code) + interference matrix
6. Component atlas construction
