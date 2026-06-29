# Open Hypotheses

## Hypothesis H001: L2 is a general-purpose routing hub

Claim:
Layer 2 of Qwen2.5-0.5B functions as a universal routing/processing hub that is causally important for all task families.

Current evidence:
- L2 ablation causes largest KL divergence across all 12 families (0.5-11.5 nats)
- L2 MLP specifically dominates (not just residual magnitude)
- Steering L2 with factual direction causally boosts target token (3.3x)
- L2 importance changes after LoRA training (reduced for JSON, increased for delimiter/factual/refusal)
- Position-specific: L2 routes first+last tokens, operator tokens near-zero
- Skill knockout at L2 is non-selective (affects all skills equally)

Best next test:
Multi-seed replication of L2 ablation across 3 seeds.

Expected result:
L2 should remain the top layer across seeds, confirming it's not a seed-specific artifact.

Falsifier:
If L2 is not the top layer in 2/3 seeds, the hub finding is seed-specific.

Status: SUPPORTED (with positional nuance)

---

## Hypothesis H002: LoRA training concentrates skill into early layers

Claim:
LoRA training on a specific skill concentrates the skill's processing into the earliest layers (L0-L2), reducing reliance on deeper layers.

Current evidence:
- JSON LoRA: L0 MLP +2.99, L2 MLP -1.87, L5 MLP -1.56 (eliminated)
- Rank sweep: concentration peaks at r=4, distributes at higher rank
- Module sweep: o_proj most efficient for L0 concentration
- Dataset shard ablation REJECTS universal concentration: each skill concentrates in different layers

Best next test:
N/A — rejected.

Status: REJECTED

---

## Hypothesis H003: Higher LoRA rank distributes skill across components

Claim:
Increasing LoRA rank distributes a skill across more model components rather than concentrating it more strongly.

Current evidence:
- L0 MLP peaks at r=4 (15.77), declines to 13.94 at r=16
- L2 MLP drops monotonically: 5.73 (r=1) to 2.16 (r=16)
- Lower rank produces more surgically precise adapters

Best next test:
Compare ablation maps of r=1 vs r=16 adapters across multiple skill families.

Expected result:
r=1 adapters should have more localized ablation maps (fewer layers affected). r=16 should have broader but shallower effects.

Falsifier:
If r=16 has the same selectivity profile as r=4 but with stronger effect, rank increases magnitude not distribution.

Status: SUPPORTED

---

## Hypothesis H004: o_proj is the key skill injection pathway

Claim:
The output projection (o_proj) in attention layers is the most efficient module for injecting skills via LoRA, because it directly writes to the residual stream.

Current evidence:
- o_proj-only LoRA achieves +3.64 L0 effect with 344K params
- MLP-only has 3.3M params but only +1.92 L0 effect
- v_proj second-best: +2.75 with 197K params

Best next test:
Train o_proj-only adapters on 3+ skill families. Compare efficiency vs all-linear.

Expected result:
o_proj-only adapters should achieve 80%+ of all-linear skill gain with <10% of params.

Falsifier:
If MLP-only adapters work equally well for non-JSON tasks, the o_proj efficiency is specific to format-heavy tasks.

Status: SUPPORTED for JSON

---

## Hypothesis H005: Factual recall and algorithmic tasks use different circuits

Claim:
Factual recall (memory retrieval) and algorithmic tasks (copying, delimiter tracking) rely on fundamentally different circuits in the model.

Current evidence:
- Factual recall most sensitive to L2 ablation (KL=11.54)
- Code semantics most resistant to L2 ablation (KL=0.52)
- Steering L2 causally affects factual recall (3.3x boost)
- After JSON LoRA, L4 appeared for factual recall (+1.42) but not for algorithmic tasks
- Skill knockout at L19 selectively suppresses factual recall (11654x selectivity)
- Factual recall concentrates at L3/16/19, code at L1/10/21

Best next test:
Head-level ablation separately for factual vs algorithmic families.

Expected result:
Factual recall should depend more on early MLP layers (knowledge storage). Algorithmic tasks should depend more on attention heads (pattern matching).

Falsifier:
If both families depend on the same heads and MLPs with similar sensitivity, the distinction is quantitative not qualitative.

Status: WEAKENED (both depend on L2, but post-training concentration and knockout selectivity differ)

---

## Hypothesis H006: Adapter weights write to late layers but effects propagate upstream

Claim:
LoRA adapter norms peak at late layers (L20-L23) but the functional effects (measured by ablation) propagate to early layers (L0-L2).

Current evidence:
- General layer ablation: effects peak at L0-L2 (universal importance)
- Adapter archaeology: norms peak at L20-L23
- Adapter-only ablation (NEW): norm-effect correlation = 0.85 (strong positive)
- Adapter-only ablation: effect peaks at L19-L23, matching norm distribution
- Only L12 shows norm-effect mismatch

Interpretation:
The original finding conflated general layer importance (L0-L2) with adapter-specific importance. L0-L2 is universally important for ALL processing (base + adapter). The adapter's SPECIFIC contribution is at L19-L23, matching where it writes. There is no upstream propagation.

Best next test:
Verify with other skill adapters (factual, code) — does adapter-only ablation effect also peak at late layers?

Status: REJECTED in original form. Updated: adapter effects are at the same layers where norms peak. The separation is between general importance (L0-L2) and adapter-specific importance (L19-L23).

---

## Hypothesis H007: L22 is the unembedding pathway

Claim:
Layer 22 is the primary unembedding/final-prediction pathway, almost exclusively affecting last-position tokens.

Current evidence:
- Position-specific ablation: L22 mean last-position effect = 14.55 nats, all other positions ~0
- Cross-model patching: L22 carries 97% of trained behavior (recovery=0.966)
- L22 is the second-most important layer for recovery after L23

Best next test:
Probe L22 activations for vocabulary projection features.

Expected result:
L22 activations should correlate strongly with unembedding matrix rows.

Falsifier:
If L21 or L20 shows similar last-position exclusivity, L22 is not unique.

Status: SUPPORTED

---

## Hypothesis H008: Trained behavior is encoded in late-layer activation patterns (NEW)

Claim:
The behavior learned by LoRA training is encoded primarily in the activation patterns of late layers (L19-L23). Patching trained activations at these layers into the base model transfers the learned behavior.

Current evidence:
- Cross-model patching: monotonic recovery increase from early to late layers
- L23=100%, L22=97%, L21=95%, L20=87%, L19=80%
- Early layers (L0-L12) give <50% recovery
- Adapter-only ablation: effect concentrated at L19-L23 (corr=0.85 with norm)

Best next test:
Test with factual_recall adapter — does the same late-layer encoding pattern hold?

Expected result:
All LoRA-trained skills should show late-layer activation encoding, regardless of which layers the skill concentrates in (as measured by general ablation).

Falsifier:
If factual_recall shows high recovery at L3/L16/L19 (its concentration layers) rather than L20-L23, the encoding location is skill-specific.

Status: SUPPORTED

---

## Hypothesis H009: Skills can be selectively suppressed via negative steering at skill-specific layers (NEW)

Claim:
Negative steering at a skill's concentration layer can selectively suppress that skill while preserving other skills.

Current evidence:
- L19 (factual_recall concentration layer): selectivity ratio 11654x at s=-2.0
- L21: selectivity ratio 53x at s=-2.0
- L2 (universal hub): non-selective (affects all skills equally)
- L3: very low selectivity (0.13x)

Best next test:
Test knockout on code_semantics at L1/L10/L21. Test knockout on json_schema at L6/L12/L13.

Expected result:
Each skill should be most selectively suppressible at its own concentration layers.

Falsifier:
If all skills are equally suppressible at all layers, selectivity is not layer-specific.

Status: SUPPORTED

---

## Phase 2 Hypotheses (2026-06-22)

### H-P2-1: Hub position scales with model size

Claim:
The universal importance hub layer scales predictably with model parameter count within the Qwen2.5 family.

Current evidence:
- 0.5B: hub at L2 (8% depth)
- 1.5B: hub at L26 (93% depth)
- 3B: hub at L34 (94% depth)
- SmolLM2-1.7B: NO clear hub (flat profile) — architecture-specific

Status: CONFIRMED within Qwen2.5 family. REJECTED as universal law (SmolLM2 breaks the pattern).

---

### H-P2-2: Mean-ablation gives different results from zero-ablation

Claim:
Mean ablation (replacing activations with dataset mean) would produce different effect sizes and possibly different layer rankings from zero ablation.

Current evidence:
- At 0.5B: zero and mean give IDENTICAL KL at every layer (all tasks)
- At 1.5B: zero and mean give IDENTICAL KL at every layer (all tasks)
- Interpretation: mean activation ≈ 0 for these prompt distributions

Status: REJECTED. Zero ablation is a valid proxy for mean ablation in this setting.

---

### H-P2-3: Steering collapses at larger scales

Claim:
Activation steering (which works at 0.5B) would become less effective or collapse at 1.5B due to more distributed representations.

Current evidence:
- 1.5B L21 steering: target logit delta +4.64 (vs 0.5B best +2.16)
- 1.5B multi-layer steering: KL = -7.20 (strongest observed)
- 1.5B L26 steering vector norm: 81.39 (vs 0.5B L2: 1.49)

Status: REJECTED. Steering gets STRONGER at larger scales, not weaker.

---

### H-P2-4: SmolLM2 has a Qwen2.5-like hub at proportional depth

Claim:
SmolLM2-1.7B (24 layers) would have a hub at a similar depth fraction as Qwen2.5-0.5B.

Current evidence:
- SmolLM2 shows flat ablation profile across all 24 layers
- No layer is more important than any other
- Hub "at L0" is simply the first layer of a flat distribution

Status: REJECTED. SmolLM2 has fundamentally different internal organization.

---

### H-P2-5: Layer ranking changes under different ablation methods

Claim:
Using mean, gaussian resample, or random patch ablation would change which layers rank as most important.

Current evidence:
- Zero ≈ mean at every layer (both scales)
- Gaussian resample preserves ranking but with higher variance
- Random patch preserves ranking but noisier

Status: REJECTED. Layer ranking is robust to ablation method.

---

### H10 (NEW): Hub position is architecture-specific, not depth-dependent

Claim:
The position of the universal importance hub is determined by architecture choices (hidden dim, n_heads, n_layers, GQA config) not by a simple depth fraction.

Current evidence:
- Qwen2.5-0.5B (24L, d=896): hub at L2
- Qwen2.5-1.5B (28L, d=1536): hub at L26
- Qwen2.5-3B (36L, d=2048): hub at L34
- SmolLM2-1.7B (24L, d=2048): no clear hub
- Hub jumps from 8% to 93% depth between 0.5B and 1.5B

Best next test:
Test Qwen2.5-0.5B-Instruct (same arch, different training) to separate architecture from training effects.

Status: NEW — supported by 4-model comparison.

---

### H11 (NEW): Steering budget scales with activation magnitude

Claim:
The safe steering strength scales with the model's activation magnitude at the target layer. Larger models have larger activations and thus larger steering budgets.

Current evidence:
- 0.5B L2 steering vector norm: 1.49, safe range: s ∈ [-4, +4]
- 1.5B L26 steering vector norm: 81.39, effective range: s ∈ [-2, +2] (s=-4 causes massive collateral damage)

Best next test:
Measure activation norms at hub layers across scales. Plot steering effectiveness vs normalized strength (s / ||activation||).

Status: NEW — supported by 2-model comparison.

---

### H12 (NEW): Skills have a separability hierarchy

Claim:
Some skills are more surgically editable than others. code_semantics is the most separable; delimiter_tracking is the least.

Current evidence:
- code_semantics SSS: 0.361 (positive insertion gain)
- json_schema SSS: 0.221
- factual_recall SSS: 0.218
- copying SSS: 0.219
- delimiter_tracking SSS: 0.215

Best next test:
Replicate at 1.5B. Test whether the separability hierarchy is stable across scales.

Status: NEW — pilot result (single seed, single scale).

---

## Phase 9 Hypotheses (2026-06-29)

**IMPORTANT CAVEAT:** Phase 9 behavioral verdicts were originally based on mock-judge data. The following statuses reflect training-loss evidence only. Behavioral evidence is pending Phase 9R eval runs.

### H-P9-1: Multi-turn concise format is genuinely better for small-model SFT

Claim:
Multi-turn concise chat format produces better fine-tuned models than other formats when content is held constant.

Current evidence:
- Training loss: multi_turn_concise (1.516) is 3rd, NOT the best
- Multi_turn_verbose (1.372) has lowest loss
- Under content-controlled conditions, verbose beats concise on loss

Status: REJECTED (loss only). The Phase 8 intuition that concise is always better needs revision. Verbose may win because more tokens per example = more learning signal.

---

### H-P9-2: The smol-magpie-ultra advantage is partly format, not merely content

Claim:
Multi-turn format (not just curated content) contributes to smol-magpie's superior performance.

Current evidence:
- Multi-turn formats (verbose 1.372, concise 1.516) have lower loss than flat formats (alpaca 1.732, single_turn 1.748)
- 26% loss gap between best multi-turn and best flat format
- But content and format interact — can't fully separate with current design

Status: PLAUSIBLE (loss only). Format contributes to loss difference. Behavioral confirmation pending.

---

### H-P9-3: Small models benefit from dense, compact examples more than verbose ones

Claim:
Compact, dense training examples should outperform verbose ones for small model SFT.

Current evidence:
- Structured terse (most compact): WORST loss (1.831)
- Multi-turn verbose (most tokens per example): BEST loss (1.372)
- Verbose > concise > structured terse on loss

Status: REJECTED (loss only). More context beats less context at 230M scale for training loss. Behavioral confirmation pending.

---

### H-P9-4: Training loss may not correlate perfectly with behavioral quality

Claim:
Low training loss does not necessarily mean the model produces better outputs.

Current evidence:
- bad_format_control: loss 1.402 (2nd best)
- bad_format_control is deliberately malformed data with filler phrases
- Model learns filler easily → low loss → but outputs should be worse

Status: PLAUSIBLE (from loss ranks). The bad_format_control anomaly strongly suggests loss-quality decoupling. Real eval needed to confirm. If bad_format_control has worst judge scores despite 2nd-best loss, this is confirmed.

---

### H-P9-5: Surgical LoRA can add useful behavior while preserving base model distribution

Claim:
Hub + o_proj only (65K params) should show lower KL drift and better format discipline than full hub targeting (245K).

Current evidence:
- Surgical bsmagpie: loss 1.271, Quality bsmagpie: loss 1.464
- Surgical achieves lower loss with 3.8x fewer params
- This suggests surgical is more efficient at learning

Status: PLAUSIBLE (loss only). Need KL drift measurement and behavioral eval to confirm preservation of base model distribution.

---

### H-P9-6: Structured terse data may outperform verbose chat on JSON/extraction/code tasks

Claim:
Compact structured training data may teach format discipline better than conversational data.

Current evidence:
- Structured terse has WORST overall loss (1.831)
- No category-level evidence available

Status: UNRESOLVED. Loss suggests otherwise, but category-level eval might show structured terse wins on specific task types.

---

### H-P9-7: There is a distinct "small-model-native" data style

Claim:
Small models have a fundamentally different optimal data shape than large models.

Current evidence:
- Multi-turn verbose wins on loss (opposite of Phase 8 intuition)
- More conversational context helps at 230M scale
- But this is loss only — need behavioral confirmation

Status: PLAUSIBLE (loss only). The "small-model-native" style may be "more context" not "more compact." Needs real eval.
