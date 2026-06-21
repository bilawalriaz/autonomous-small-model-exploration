# Current Findings

## Executive summary
Qwen2.5-0.5B has a clear hierarchical component structure with L2 as a universal processing hub. LoRA training rewires where skills live, but critically, each skill concentrates in DIFFERENT layers — there is no universal "training writes to L0-L2" pattern. Adapter weight norms concentrate in late layers (L20-L23) while functional ablation effects peak at early layers (L0-L2), suggesting LoRA weights write late but effects propagate upstream. Adapters can be stacked: factual+json combines cleanly, but the delimiter adapter is destructive when merged with others.

## Strongest causal claims

1. **L2 is a universal importance hub.** Ablating L2 causes the largest KL divergence (0.5-11.5 nats) across all 12 task families. Consistent across layer ablation, MLP ablation, steering, and LoRA comparison experiments. *Confidence: HIGH.*

2. **LoRA training rewires where skills live, with skill-specific concentration patterns.** Each skill family concentrates in different layers after LoRA training: factual_recall at L3/16/19, code_semantics at L1/10/21, json_schema at L6/12/13. The hypothesis that training universally concentrates into L0-L2 is rejected. *Confidence: MEDIUM.*

3. **L2 causal role for factual recall confirmed by steering.** Steering L2 with factual direction boosts "rome" from 0.064 to 0.213 (3.3x) for "capital of Italy". Negative steering suppresses it. *Confidence: MEDIUM.*

4. **Adapter weight norms and functional effects are spatially separated.** LoRA adapter norms peak at L20-L23 (late layers) while ablation effects peak at L0-L2 (early layers). This suggests the model routes information through early layers but stores learned adaptations in late-layer weights. *Confidence: MEDIUM.*

5. **Adapters can be combined with varying interference.** factual_recall + json_schema stacks cleanly (+2.35 synergy on factual, +1.17 on json). delimiter_tracking adapter is destructive when stacked (-7 to -16 nats degradation). *Confidence: MEDIUM.*

## Training perturbation findings

- **Dataset shard ablation (5 families):** Training on different skill families produces different component maps. factual_recall concentrates at L3/16/19; code_semantics at L1/10/21; json_schema at L6/12/13. delimiter_tracking fully absorbs into adapter (0 ablation sensitivity). copying is dispersed.
- **LoRA rank sweep (r=1,2,4,8,16):** Training loss decreases monotonically. L0 MLP importance peaks at r=4 (15.77). Total adapter norm scales linearly (6.14 to 22.92). Higher rank shifts norm concentration from uniform to late layers.
- **LoRA module sweep:** o_proj alone achieves strongest L0 concentration (+3.64) with 344K params. MLP-only has 3.3M params but weakest effect.
- **Cross-task effects after JSON LoRA:** L4 appeared for factual_recall (+1.42) and refusal (+0.95). L2 increased for delimiter (+1.19), factual (+1.27). L2 decreased for copying (-0.75).
- **Adapter stacking interference:** factual+json synergistic. copying+json partially compatible. delimiter destructive. code+json compatible.

## Weak/tentative signals
- L1 appears as universal skill injection point (positive delta across 3+ adapters)
- L21/L23 may be formatting/output specialists
- Code semantics resistant to layer ablation (KL=0.52 at L2)
- Factual recall most sensitive to layer ablation (KL=11.54 at L2)
- delimiter adapter's extreme behavior may indicate format-specific overfitting

## Negative results
1. Full SFT OOMs on 8GB VRAM even with bf16 + gradient checkpointing. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. H002 (universal L0-L2 concentration) rejected by dataset shard ablation.
4. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.

## Current atlas status
| Confidence | Count |
|------------|-------|
| Low        | 1     |
| Medium     | 12    |
| High       | 1     |
| Very High  | 0     |
| Negative   | 4     |

## Best next experiments
1. Component atlas construction (formalize all findings)
2. Checkpoint timeline (track skill emergence during training)
3. Position-specific activation patching at L0/L2/L1
4. Cross-model activation patching (trained-to-base)
5. Negative steering / skill knockout
6. Blog post outline
