# Open Hypotheses

## Hypothesis H001: L2 is a general-purpose routing hub

Claim:
Layer 2 of Qwen2.5-0.5B functions as a universal routing/processing hub that is causally important for all task families.

Current evidence:
- L2 ablation causes largest KL divergence across all 12 families (0.5-11.5 nats)
- L2 MLP specifically dominates (not just residual magnitude)
- Steering L2 with factual direction causally boosts target token (3.3x)
- L2 importance changes after LoRA training (reduced for JSON, increased for delimiter/factual/refusal)

Best next test:
Position-specific patching at L2 to determine if effect is position-dependent or uniform.

Expected result:
L2 effect should be concentrated at instruction/delimiter positions, not uniform across all tokens.

Falsifier:
If L2 effect is uniform across all positions (just a residual magnitude effect), L2 is not a routing hub but simply carries the most information.

Status: open

---

## Hypothesis H002: LoRA training concentrates skill into early layers

Claim:
LoRA training on a specific skill concentrates the skill's processing into the earliest layers (L0-L2), reducing reliance on deeper layers.

Current evidence:
- JSON LoRA: L0 MLP +2.99, L2 MLP -1.87, L5 MLP -1.56 (eliminated)
- Rank sweep: concentration peaks at r=4, distributes at higher rank
- Module sweep: o_proj most efficient for L0 concentration

Best next test:
Train LoRA adapters on copying, delimiter, code, and factual families. Compare whether each concentrates into L0-L2 or different layers.

Expected result:
Algorithmic tasks (copying, delimiter) may concentrate in different layers than factual recall. JSON is format-heavy and may be special.

Falsifier:
If delimiter LoRA concentrates in L10-L15 instead of L0-L2, the concentration pattern is task-dependent, not a universal training effect.

Status: open

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

Status: open

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

Status: open

---

## Hypothesis H005: Factual recall and algorithmic tasks use different circuits

Claim:
Factual recall (memory retrieval) and algorithmic tasks (copying, delimiter tracking) rely on fundamentally different circuits in the model.

Current evidence:
- Factual recall most sensitive to L2 ablation (KL=11.54)
- Code semantics most resistant to L2 ablation (KL=0.52)
- Steering L2 causally affects factual recall (3.3x boost)
- After JSON LoRA, L4 appeared for factual recall (+1.42) but not for algorithmic tasks

Best next test:
Head-level and MLP-level ablation separately for factual vs algorithmic families. Build component selectivity maps.

Expected result:
Factual recall should depend more on early MLP layers (knowledge storage). Algorithmic tasks should depend more on attention heads (pattern matching).

Falsifier:
If both families depend on the same heads and MLPs with similar sensitivity, the distinction is quantitative not qualitative.

Status: open
