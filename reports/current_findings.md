# Current Findings

## Executive summary
Qwen2.5-0.5B has a clear hierarchical component structure with L2 as a universal processing hub. Position-specific ablation reveals L2 routes first+last tokens while L22 is exclusively a last-token layer and L9 is the instruction-sensitive layer. LoRA training rewires where skills live — each skill concentrates in DIFFERENT layers, rejecting the universal L0-L2 hypothesis. The core circuit (L2/L7/L9) locks in by step 10 of training. Adapters can be stacked: factual+json combines cleanly, but the delimiter adapter is destructive when merged. Cross-model patching shows trained behavior is encoded in late-layer activations (monotonic recovery L23=100% → L0≈0%). Skill knockout at L19 selectively suppresses factual recall (11654x selectivity) while preserving other skills. Adapter-only ablation shows norm-effect correlation of 0.85, updating H6: adapter effects are at the same layers where norms peak, not upstream.

## Strongest causal claims

1. **L2 is a universal importance hub with positional specialization.** Ablating L2 causes the largest KL divergence (0.5-11.5 nats) across all 12 task families. Position-specific analysis shows L2 specifically routes first tokens (instruction, mean 3.34) and last tokens (prediction, mean 5.03). Operator tokens have near-zero effect. *Confidence: HIGH.* Evidence: layer ablation, MLP ablation, steering, position-specific ablation.

2. **LoRA training rewires where skills live, with skill-specific concentration patterns.** Each skill family concentrates in different layers after LoRA training: factual_recall at L3/16/19, code_semantics at L1/10/21, json_schema at L6/12/13. The hypothesis that training universally concentrates into L0-L2 is rejected. *Confidence: MEDIUM.* Evidence: dataset shard ablation (5 families).

3. **L2 causal role for factual recall confirmed by steering.** Steering L2 with factual direction boosts "rome" from 0.064 to 0.213 (3.3x) for "capital of Italy". Negative steering suppresses it. Oversteering at s>=+2 causes degeneration (Chinese characters). *Confidence: MEDIUM.* Evidence: steering sweep.

4. **L22 is the unembedding/final-prediction pathway.** Position-specific ablation shows L22 almost exclusively affects last-position tokens (mean 14.55 nats, all other positions ~0). Cross-model patching confirms L22 carries 97% of trained behavior. *Confidence: MEDIUM.* Evidence: position-specific ablation, cross-model patching.

5. **Core circuit locks in by step 10 of training.** L2/L7/L9 for JSON schema stabilize at step 10 (first 10% of training) and drift <1% through step 100. Secondary layers (L15, L6) continue shifting. *Confidence: MEDIUM.* Evidence: checkpoint timeline.

6. **Trained behavior is encoded in late-layer activation patterns.** Cross-model patching shows monotonic recovery increase from early to late layers: L23=100%, L22=97%, L21=95%, L20=87%, L19=80%. Patching trained activations at L23 into the base model gives 100% recovery. *Confidence: MEDIUM.* Evidence: cross-model patching (17 pairs, 24 layers).

7. **Skills can be selectively suppressed via negative steering at skill-specific layers.** L19 selectively knocks out factual recall (selectivity ratio 11654x at s=-2.0) while preserving JSON and copying skills. L2 is non-selective (universal hub — knockout affects everything). L21 also shows good selectivity (53x). *Confidence: MEDIUM.* Evidence: skill knockout experiment (2 skills, 7+ layers).

8. **Adapter norm and ablation effect are correlated (r=0.85), both peaking at late layers.** Adapter-only ablation shows that removing the adapter's contribution at L19-L23 has the largest effect, matching the norm distribution. This updates H6: the adapter's effect IS at the same layers where it writes, not upstream. The earlier finding conflated general layer importance (L0-L2) with adapter-specific importance (L19-L23). *Confidence: MEDIUM.* Evidence: adapter-only ablation (12 prompts, 24 layers).

9. **Adapters can be combined with varying interference.** factual_recall + json_schema stacks cleanly (+2.35 synergy on factual, +1.17 on json). delimiter_tracking adapter is destructive when stacked (-7 to -16 nats). *Confidence: MEDIUM.* Evidence: adapter stacking.

## Training perturbation findings

- **Dataset shard ablation (5 families):** Each skill concentrates in different layers. factual_recall at L3/16/19; code_semantics at L1/10/21; json_schema at L6/12/13. delimiter_tracking fully absorbs (0 ablation sensitivity). copying dispersed.
- **Checkpoint timeline:** Core circuit (L2/L7/L9) locks in by step 10. Loss: 0.587 (step10) -> 0.062 (step100). Secondary layers L15/L6 drift +2.85/+2.73 through training.
- **LoRA rank sweep:** L0 MLP peaks at r=4 (15.77). Total norm scales linearly (6.14 to 22.92). Higher rank shifts norm from uniform to late layers.
- **LoRA module sweep:** o_proj alone achieves +3.64 L0 effect with 344K params. MLP-only weakest (3.3M params, +1.92).
- **Cross-task effects after JSON LoRA:** L4 appeared for factual_recall (+1.42). L2 increased for delimiter (+1.19), factual (+1.27). L2 decreased for copying (-0.75).
- **Adapter stacking:** factual+json synergistic. code+json compatible. delimiter destructive.
- **Adapter-only ablation:** Norm-effect correlation 0.85. JSON adapter effect concentrated at L19-L23 (L23=100%, L22=92%, L21=81%). Only L12 shows norm-effect mismatch.

## Position-specific findings

- **L22:** Almost exclusively last-position (mean 14.55, others ~0). Unembedding pathway.
- **L0/L2:** First+last position routers. L0 first=3.94, last=3.30. L2 first=3.34, last=5.03.
- **L9:** Instruction-sensitive. First=5.66, last=9.20. Strongest mid-layer for both positions.
- **L7:** Balanced first+last (5.03/5.93).
- **L15:** Weak overall (max 3.37 on last). Processing layer.
- **Operators/delimiters:** Near-zero effect across all layers.

## Cross-model patching findings

- Recovery increases monotonically from early to late layers
- L23=100%, L22=97%, L21=95%, L20=87%, L19=80%, L18=73%
- Mid-layers (L13-L17) show 50-80% recovery
- Early layers (L0-L12) give minimal recovery (<50%)
- The adapter's learned behavior is encoded in the activation patterns of late layers

## Skill knockout findings

- **Factual recall:** L19 most selective (11654x at s=-2.0), L21 good (53x), L16 moderate (0.42x)
- **Factual recall:** L2 and L3 non-selective (universal hub effect)
- **JSON:** Base probability of targets already near-zero, limited knockout room
- **JSON:** L6, L7, L9, L12, L13, L21 all cause moderate KL changes at s=-1.0
- Negative steering at s=-4.0 causes broad degradation across all skills

## Weak/tentative signals
- L1 appears as universal skill injection point (positive delta across 3+ adapters)
- L21/L23 may be formatting/output specialists
- Code semantics resistant to layer ablation (KL=0.52 at L2)
- delimiter adapter's extreme stacking behavior may indicate format-specific overfitting
- L12 norm-effect mismatch may indicate a processing bottleneck

## Negative results
1. Full SFT OOMs on 8GB VRAM. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. H002 (universal L0-L2 concentration) rejected.
4. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.
5. Extreme steering (s>=+2) causes degeneration (Chinese characters, repetition).
6. L2 is NOT position-uniform (operator tokens near-zero).
7. JSON skill knockout had limited effect due to near-zero base target probability.
8. H6 (upstream propagation) rejected by adapter-only ablation (corr=0.85, effects at same layers as norms).
9. PeftModel.from_pretrained modifies base model in-place — must use disable_adapter() for base behavior.

## Current atlas status
| Confidence | Count |
|------------|-------|
| Low        | 2     |
| Medium     | 17    |
| High       | 1     |
| Very High  | 0     |
| Negative   | 9     |

## Phase 9: Format Ablation Training Loss Findings (2026-06-29)

**IMPORTANT CAVEAT:** The following findings are based on training loss ONLY. No real behavioral evaluation has been completed. Mock-judge data from the original Phase 9 report should be disregarded for behavioral claims.

### Key training loss findings
- **Format significantly affects training loss under content-controlled conditions.** Same 300 canonical examples rendered into 6 formats → 33% loss range (1.37 to 1.83).
- **Multi-turn verbose has lowest loss (1.372).** This is surprising — Phase 8 suggested concise was better. But Phase 8 didn't control for content.
- **bad_format_control has 2nd-best loss (1.402).** Deliberately malformed data is easy to predict. Loss measures predictability, not quality.
- **Structured terse has worst loss (1.831).** Compact structured format is hardest for 230M model to learn from.
- **Surgical adapter beats quality adapter on loss (1.27 vs 1.46) with 3.8x fewer params.** Out_proj-only LoRA is more parameter-efficient than hub-all-modules.

### What we can claim
1. Training loss differs by format (confirmed, reproducible)
2. Loss does not trivially measure quality (plausible, needs behavioral confirmation)
3. Format dominates hyperparameters (consistent with Phase 8)

### What we cannot yet claim
1. Multi-turn verbose produces better outputs (no real eval data)
2. Any behavioral ranking of formats (mock judge only)
3. Loss-quality correlation or decoupling (needs real eval)

## Best next experiments
1. Multi-seed replication of top 5 findings
2. Mean/resample ablation (stronger causal claims)
3. CPT training
4. SAE training on key layers (L0, L1, L2, L7, L9, L19, L22)
5. Skill injection at L19 for factual recall
6. Extend to natural language prompts
7. **Phase 9R: Run real eval on aero** (highest priority — completes Phase 9)
8. **Phase 10: Token-budget-controlled data-shape ablation**
