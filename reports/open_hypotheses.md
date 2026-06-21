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
