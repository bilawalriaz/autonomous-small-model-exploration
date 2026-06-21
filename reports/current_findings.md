# Current Findings

## Executive summary
Qwen2.5-0.5B has a clear hierarchical component structure with L2 as a universal processing hub. Position-specific ablation reveals L2 routes first+last tokens while L22 is exclusively a last-token layer and L9 is the instruction-sensitive layer. LoRA training rewires where skills live — each skill concentrates in DIFFERENT layers, rejecting the universal L0-L2 hypothesis. The core circuit (L2/L7/L9) locks in by step 10 of training. Adapters can be stacked: factual+json combines cleanly, but the delimiter adapter is destructive when merged.

## Strongest causal claims

1. **L2 is a universal importance hub with positional specialization.** Ablating L2 causes the largest KL divergence (0.5-11.5 nats) across all 12 task families. Position-specific analysis shows L2 specifically routes first tokens (instruction, mean 3.34) and last tokens (prediction, mean 5.03). Operator tokens have near-zero effect. *Confidence: HIGH.* Evidence: layer ablation, MLP ablation, steering, position-specific ablation.

2. **LoRA training rewires where skills live, with skill-specific concentration patterns.** Each skill family concentrates in different layers after LoRA training: factual_recall at L3/16/19, code_semantics at L1/10/21, json_schema at L6/12/13. The hypothesis that training universally concentrates into L0-L2 is rejected. *Confidence: MEDIUM.* Evidence: dataset shard ablation (5 families).

3. **L2 causal role for factual recall confirmed by steering.** Steering L2 with factual direction boosts "rome" from 0.064 to 0.213 (3.3x) for "capital of Italy". Negative steering suppresses it. Oversteering at s>=+2 causes degeneration (Chinese characters). *Confidence: MEDIUM.* Evidence: steering sweep.

4. **L22 is the unembedding/final-prediction pathway.** Position-specific ablation shows L22 almost exclusively affects last-position tokens (mean 14.55 nats, all other positions ~0). This is where final vocabulary projection information flows. *Confidence: MEDIUM.* Evidence: position-specific ablation.

5. **Core circuit locks in by step 10 of training.** L2/L7/L9 for JSON schema stabilize at step 10 (first 10% of training) and drift <1% through step 100. Secondary layers (L15, L6) continue shifting. *Confidence: MEDIUM.* Evidence: checkpoint timeline.

6. **Adapter weight norms and functional effects are spatially separated.** LoRA adapter norms peak at L20-L23 (late layers) while ablation effects peak at L0-L2 (early layers). *Confidence: MEDIUM.* Evidence: adapter archaeology.

7. **Adapters can be combined with varying interference.** factual_recall + json_schema stacks cleanly (+2.35 synergy on factual, +1.17 on json). delimiter_tracking adapter is destructive when stacked (-7 to -16 nats). *Confidence: MEDIUM.* Evidence: adapter stacking.

## Training perturbation findings

- **Dataset shard ablation (5 families):** Each skill concentrates in different layers. factual_recall at L3/16/19; code_semantics at L1/10/21; json_schema at L6/12/13. delimiter_tracking fully absorbs (0 ablation sensitivity). copying dispersed.
- **Checkpoint timeline:** Core circuit (L2/L7/L9) locks in by step 10. Loss: 0.587 (step10) -> 0.062 (step100). Secondary layers L15/L6 drift +2.85/+2.73 through training.
- **LoRA rank sweep:** L0 MLP peaks at r=4 (15.77). Total norm scales linearly (6.14 to 22.92). Higher rank shifts norm from uniform to late layers.
- **LoRA module sweep:** o_proj alone achieves +3.64 L0 effect with 344K params. MLP-only weakest (3.3M params, +1.92).
- **Cross-task effects after JSON LoRA:** L4 appeared for factual_recall (+1.42). L2 increased for delimiter (+1.19), factual (+1.27). L2 decreased for copying (-0.75).
- **Adapter stacking:** factual+json synergistic. code+json compatible. delimiter destructive.

## Position-specific findings

- **L22:** Almost exclusively last-position (mean 14.55, others ~0). Unembedding pathway.
- **L0/L2:** First+last position routers. L0 first=3.94, last=3.30. L2 first=3.34, last=5.03.
- **L9:** Instruction-sensitive. First=5.66, last=9.20. Strongest mid-layer for both positions.
- **L7:** Balanced first+last (5.03/5.93).
- **L15:** Weak overall (max 3.37 on last). Processing layer.
- **Operators/delimiters:** Near-zero effect across all layers.

## Weak/tentative signals
- L1 appears as universal skill injection point (positive delta across 3+ adapters)
- L21/L23 may be formatting/output specialists
- Code semantics resistant to layer ablation (KL=0.52 at L2)
- delimiter adapter's extreme stacking behavior may indicate format-specific overfitting

## Negative results
1. Full SFT OOMs on 8GB VRAM. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. H002 (universal L0-L2 concentration) rejected.
4. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.
5. Extreme steering (s>=+2) causes degeneration (Chinese characters, repetition).
6. L2 is NOT position-uniform (operator tokens near-zero).

## Current atlas status
| Confidence | Count |
|------------|-------|
| Low        | 2     |
| Medium     | 12    |
| High       | 1     |
| Very High  | 0     |
| Negative   | 6     |

## Best next experiments
1. Cross-model activation patching (trained-to-base)
2. Negative steering / skill knockout
3. Adapter-only ablation (ablate adapter at L20-L23 to test norm/effect hypothesis)
4. CPT training
5. Blog post outline
6. Paper outline
