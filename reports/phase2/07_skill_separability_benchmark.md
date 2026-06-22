# Phase 2 Report 7: Skill Separability Benchmark
**Experiment Block:** G
**Models:** Qwen/Qwen2.5-0.5B
**Tasks:** json_schema, factual_recall, code_semantics, copying, delimiter_tracking
**Seeds:** [1]
**Date:** 2026-06-22
**Status:** complete (partial — transfer/composition pending)

## 1. What was tested
A composite skill separability score (SSS) for 5 skills, combining 6 sub-metrics: insertion gain, collateral damage, removal selectivity, localization sharpness, transfer recovery, and composition compatibility.

## 2. Why it matters
If skills are highly separable, they can be independently modified (inject one skill without affecting others). If inseparable, skill modification will always have unintended side effects. This determines whether the atlas is useful for practical model editing.

## 3. Exact models
- Qwen/Qwen2.5-0.5B: 24 layers
- LoRA config: r=8, alpha=16, target=[q,k,v,o_proj], lr=0.0002, steps=100
- Git commit: `d986315`

## 4. Exact task suite
5 skills, each with a dedicated LoRA adapter:
- json_schema (lora_json_r8)
- factual_recall (lora_factual_recall_r8)
- code_semantics (lora_code_semantics_r8)
- copying (lora_copying_r8)
- delimiter_tracking (lora_delimiter_tracking_r8)

## 5. Key metrics

### Skill Separability Scores

| Skill | SSS | Rank |
|-------|-----|------|
| code_semantics | 0.3607 | 1 (most separable) |
| json_schema | 0.2213 | 2 |
| copying | 0.2193 | 3 |
| factual_recall | 0.2179 | 4 |
| delimiter_tracking | 0.2145 | 5 (least separable) |

### Sub-metric breakdown

| Skill | Insertion Gain | Collateral | Removal Selectivity | Localization | Transfer | Composition |
|-------|---------------|-----------|-------------------|-------------|----------|-------------|
| code_semantics | +3.696 | 0 | 0.00 | 0.419 | 0.5 | 0.5 |
| json_schema | -1.353 | 0 | 3.47 | 0.429 | 0.5 | 0.5 |
| copying | -6.918 | 0 | 3.04 | 0.421 | 0.5 | 0.5 |
| factual_recall | -1.879 | 0 | 1.78 | 0.429 | 0.5 | 0.5 |
| delimiter_tracking | -2.673 | 0 | 0.60 | 0.422 | 0.5 | 0.5 |

### SSS weights

| Component | Weight |
|-----------|--------|
| insertion | 0.20 |
| removal | 0.20 |
| transfer | 0.15 |
| composition | 0.15 |
| localization | 0.15 |
| collateral | 0.15 |

## 6. Controls
- Collateral damage = 0 for all skills — no unintended degradation from adapter insertion
- Localization sharpness is uniform (~0.42) — all skills equally concentrated
- Transfer and composition at default 0.5 — pending full cross-adapter evaluation

## 7. Results
Code_semantics is the most separable skill (SSS=0.361), driven by positive insertion gain (+3.696). Other skills have negative insertion gain, meaning the adapter alone doesn't fully capture the skill. Removal selectivity varies widely (json_schema: 3.47 vs delimiter_tracking: 0.60), suggesting some skills are easier to knock out than others.

The narrow SSS range (0.215-0.361) suggests all skills have similar separability profiles, with code_semantics as a moderate outlier.

## 8. Failed hypotheses
- **H: Skills have high separability (>0.5)** — REJECTED. All SSS < 0.4.
- **H: Skills have zero collateral damage** — CONFIRMED. All collateral = 0.

## 9. Limitations
1. **Transfer and composition not fully evaluated**: 3 of 6 sub-metrics at default 0.5.
2. **Single seed**: No variance.
3. **Only 5 skills**: May not cover all task families.
4. **SSS weight sensitivity not tested**: Different weights could change rankings.

## 10. Next experiments
- Complete transfer recovery evaluation (cross-adapter patching)
- Complete composition compatibility evaluation (multi-adapter stacking)
- Test SSS sensitivity to weight perturbations
- Replicate at 1.5B and 3B scale
