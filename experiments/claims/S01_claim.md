# S01 claim — MiniCPM corrected-Q4 Stage 0 baseline

## Finding

On the S01 frozen suite, the corrected direct merge improves the explicit
schema-template family and a narrow concise-transformation family, but it has
no statistically resolved math gain and no observed executable-code gain; this
is a measurement baseline, not evidence for training synergy or release
readiness.

## Evidence

- Runs: `S01v2_composite_base_q4`, `S01v2_composite_merged_q4`.
- Deterministic output integrity: 800/800 outputs per model; zero nonzero
  return codes, prompt echoes, template residues, and empty captures.
- Math: 25.0% → 30.0%, +5.0pp; paired 95% bootstrap CI [−2.5, +12.5],
  wins/losses/ties 36/26/138.
- Explicit JSON/YAML/schema template: 0.0% → 67.0%, +67.0pp; CI
  [+60.5, +73.5], 134/0/66. The ambiguous v1 schema prompt was discarded and
  only the explicit-array v2 rerun is used.
- Concise transformations: 0.0% → 24.5%, +24.5pp; CI [+18.5, +30.5],
  49/0/151.
- Executable MBPP: 1.5% → 1.5%, 0.0pp; CI [−2.0, +2.0], 2/2/196.

## Required claim schema

- Component: corrected rank-8 RS-LoRA direct merge (math 1.0, formatting 0.7).
- Behaviour/task family: strict structured output, concise transformations,
  GSM8K-style math, and MBPP executable code.
- Metric/effect: deterministic exact/schema/test scores and paired bootstrap
  intervals as listed above.
- Ablation/patching/steering: not tested; this is not a mechanistic claim.
- Training-delta evidence: not applicable; no new training occurred.
- Controls: M02 output-boundary fixtures, eight scorer fixtures, identical
  greedy decoding, paired prompts, and raw-output review.
- Failure modes: schema and concise suites are deterministic generators with
  repeated task shapes; their effects require replication on structurally
  diverse, source-held-out examples. The v1 schema prompt was ambiguous.
- Confidence: Medium for this narrow corrected-Q4 baseline; insufficient for
  general capability, preservation, or synergy claims.

## Variance

Pilot — deterministic decoding; uncertainty is 20,000 paired-bootstrap prompt
resamples, not training-seed variance.

## Comparison to Phase 1

Downgraded. The result supports M03's narrow formatting direction but does not
restore the invalidated general MiniCPM stacking/synergy interpretation.

## Reproduction command

```bash
python scripts/eval/score_minicpm5_s01.py \
  --eval-set data/eval/minicpm5_s01_v2_heldout.jsonl \
  --base results/evals/S01v2_composite_base_q4/outputs.jsonl \
  --merged results/evals/S01v2_composite_merged_q4/outputs.jsonl \
  --output results/evals/S01v2_paired_metrics.json \
  --manual-review results/evals/S01v2_manual_review.json
```
