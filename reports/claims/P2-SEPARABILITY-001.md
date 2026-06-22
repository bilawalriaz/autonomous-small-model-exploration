# P2-SEPARABILITY-001: Skill Separability Benchmark — Qwen2.5-0.5B

## Claim
Skill separability scores (SSS) for Qwen2.5-0.5B range from 0.215 (delimiter_tracking) to 0.361 (code_semantics), with code_semantics being the most separable skill. All skills show zero collateral damage from adapter insertion, localization sharpness of ~0.42, and default transfer/composition scores of 0.5 (pending full evaluation).

## Result

| Skill | SSS | Insertion Gain | Collateral | Removal Selectivity | Localization |
|-------|-----|---------------|-----------|-------------------|-------------|
| code_semantics | 0.3607 | +3.696 | 0 | 0.00 | 0.419 |
| json_schema | 0.2213 | -1.353 | 0 | 3.47 | 0.429 |
| copying | 0.2193 | -6.918 | 0 | 3.04 | 0.421 |
| factual_recall | 0.2179 | -1.879 | 0 | 1.78 | 0.429 |
| delimiter_tracking | 0.2145 | -2.673 | 0 | 0.60 | 0.422 |

## Controls
- Collateral damage = 0 for all skills (no unintended degradation)
- Localization sharpness ~0.42 for all skills (consistent)
- Transfer recovery and composition compatibility at default 0.5 (pending)

## Seeds

| Seed | Status |
|------|--------|
| 1 | complete |

## Artifacts
- Raw output: `experiments/results/skill_separability_qwen25-05b.json`
- CSV: `results/summaries/skill_separability_scores.csv`
- Run ID: `P2-SEPARABILITY-001`

## Interpretation
Code_semantics is the most separable skill (SSS=0.361), likely because it has positive insertion gain (+3.696) — the adapter improves performance. Other skills show negative insertion gain, meaning the adapter alone doesn't fully capture the skill. The uniform localization sharpness (~0.42) suggests all skills are equally concentrated in the model, but removal selectivity varies (json_schema: 3.47 vs delimiter_tracking: 0.60).

## Limitations
1. Transfer recovery and composition compatibility at default 0.5 — not fully evaluated.
2. Only 1 seed.
3. SSS weights are fixed (0.2/0.2/0.15/0.15/0.15/0.15) — sensitivity to weights not tested.

## Verdict
partially_confirmed
