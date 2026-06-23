# Threats to Validity — Phase 3 Gap Closure

Generated: 2026-06-23

## T01: Single seed everywhere

**Severity:** HIGH
**Affects:** All 20 claims
**Problem:** Every experiment ran with seed=1 (or equivalent). We have zero variance estimates. A finding that looks strong at seed=1 may be noise.
**Fix:** Run 3 seeds for all critical experiments. Report mean +/- std. Flag any finding where std > 50% of effect as fragile.

## T02: Too few prompts per family

**Severity:** HIGH
**Affects:** C01, C04, C09, C15, C16
**Problem:** 1-3 prompts per family for ablation/steering. The KL values may be dominated by individual prompt artifacts.
**Fix:** Expand to 50+ prompts per family. Use the canonical task suite (already has 4300 examples in Phase 2 task_manifest.json).

## T03: Synthetic prompts are toy-like

**Severity:** MEDIUM
**Affects:** C01, C04, C05, C06, C07, C15
**Problem:** Prompts like "Return exactly valid JSON with keys name and age. Eve is 42." are 5-15 tokens. Real-world prompts are longer, more complex, and noisier. The hub location or skill concentration may shift with natural language.
**Fix:** Add natural language prompt variants (short/medium/long) from the Phase 2 task suite. Test whether hub location changes.

## T04: Zero ablation creates unrealistic distribution shift

**Severity:** MEDIUM
**Affects:** C01, C02, C04, C13
**Problem:** Zeroing a layer's output breaks the residual stream chain. All downstream layers receive garbage input. This makes ALL layers appear "necessary" and may inflate effects at early layers (because they corrupt everything downstream).
**Fix:** Phase 2 ablation controls already compared zero/mean/gaussian. Key finding: zero=mean for top layers. But gaussian reduces effect by ~40%. Need to verify rank-order stability with gaussian resample for ALL layers, not just L2/L26.

## T05: Steering may be prompt-length dependent

**Severity:** MEDIUM
**Affects:** C10, C09
**Problem:** Steering vectors computed from short prompts may not transfer to long prompts. The activation subspace may shift with context length.
**Fix:** Phase 2 Block I (long_task_robustness) partially addresses this. Results exist for 0.5B and 1.5B ablation/steering by length. Need to analyze whether steering effects change with prompt length.

## T06: Quantization may change causal surfaces

**Severity:** HIGH
**Affects:** C14 (and all claims if quantized models are used for experiments)
**Problem:** We measured speed and constraint adherence under quantization, but NOT whether ablation effects, steering vectors, or adapter behaviour change. A model that passes benchmarks at 4-bit may have completely different internal structure.
**Fix:** Run the full causal atlas (layer ablation + steering) on 4-bit NF4 versions of 0.5B and 1.5B. Compare hub location and steering effectiveness. This is the "quantization causal surface drift" experiment.

## T07: LoRA effects may be task/dataset specific

**Severity:** MEDIUM
**Affects:** C04, C05, C06, C12
**Problem:** All LoRA experiments used the same training data format (JSON examples from the task suite). Different data curation, learning rates, or training lengths may produce different concentration patterns.
**Fix:** Train LoRA on at least 3 different data sources (task suite, natural language, code) and compare concentration maps.

## T08: "Layer importance" differs by token position

**Severity:** MEDIUM
**Affects:** C01, C02, C15
**Problem:** Position-specific ablation showed L2 is position-dependent (first+last tokens). But the layer ablation used mean KL across all positions. The "hub" may be a hub only for certain positions.
**Fix:** Decompose all layer ablation results by token position. Report per-position hub maps.

## T09: Results may not transfer from base to coder models

**Severity:** MEDIUM
**Affects:** All claims
**Problem:** We tested Qwen2.5 (base) but not Qwen2.5-Coder. Coder models may have different internal structure (code-specific circuits).
**Fix:** Run at least the hub identification experiment on Qwen2.5-Coder-0.5B and Coder-1.5B. Compare hub locations.

## T10: Norm-effect confusion in reporting

**Severity:** LOW (data is fine, narrative is broken)
**Affects:** C11, C19
**Problem:** The publication report simultaneously claims H6 is "supported" (Section 4.3) and "rejected" (NR008). The correlation of 0.85 is described as "weak or negative." This confusion makes the published findings unreliable as written.
**Fix:** Re-analyze adapter-only ablation data. Write a clean, non-contradictory interpretation. The data is probably fine; the narrative needs fixing.

## T11: Skill knockout selectivity may be a prompt artifact

**Severity:** HIGH
**Affects:** C09
**Problem:** The 11654x selectivity ratio at L19 is extreme. NR007 notes that JSON knockout had "limited effect due to near-zero base probability." If the factual recall prompts have measurable base probability while JSON prompts don't, the selectivity ratio is comparing apples to oranges.
**Fix:** Test knockout on prompts where ALL skills have measurable base probability. Use random-vector and shuffled-label controls to establish baseline selectivity.

## T12: Cross-model patching interpretation

**Severity:** LOW
**Affects:** C08
**Problem:** "Trained behavior encoded in late-layer activations" could mean either (a) the adapter writes to late layers and that's where the new behavior lives, or (b) late layers are just closest to the output and any intervention there has the most direct effect. These have very different implications.
**Fix:** Test whether intervening at early layers (L0-L5) with the SAME magnitude of activation change produces proportionally smaller effects. If early-layer interventions are equally effective when matched for magnitude, the "late-layer encoding" is just a proximity-to-output artifact.

## T13: Hub at 3B has two peaks (L34 and L13)

**Severity:** MEDIUM
**Affects:** C02
**Problem:** 3B layer ablation shows L34 as the global hub (9/12 families) but L13 as a secondary hub for 3/12 families (json, dead_code, refusal). Is L13 a genuine second hub or an artifact of those specific tasks?
**Fix:** Expand the 3B task suite. If L13 remains a hub for json/dead_code/refusal with 50+ prompts each, it's real. If it disappears, it was noise.

## T14: The publication report has bugs

**Severity:** LOW
**Affects:** All claims (credibility)
**Problem:** Section 3.10 says "correlation of 0.855, indicating a weak or negative relationship" — 0.85 is strong positive. Finding 9 says "norm-effect separation" while NR008 says "no upstream propagation." The report contradicts itself.
**Fix:** Regenerate the publication report with corrected interpretations. Flag all self-contradictions.
