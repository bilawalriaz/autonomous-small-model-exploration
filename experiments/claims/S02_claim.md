# S02 claim — MiniCPM corrected-Q4 structural-transfer baseline

## Finding

The corrected direct merge's S01 explicit-schema improvement did not transfer
to the structurally distinct, split-audited S02 schema family (0/200 for both
models); only a narrow concise-transformation gain remains observed.

## Evidence

- `data/manifests/minicpm5_s02_manifest.json`: zero exact intersections and
  zero MinHash-candidate 5-token-shingle Jaccard matches at threshold 0.80
  across train/validation/heldout.
- `results/evals/S02_paired_metrics.json`: all 800 outputs per model had zero
  return-code, echo, residue, and empty-output failures.
- Schema: 0.0% → 0.0%, 0/0/200 wins/losses/ties, paired CI [0.0, 0.0].
- Concise transformations: 0.0% → 25.5%, +25.5pp, CI [+19.5, +31.5],
  51/0/149.
- Math: 25.0% → 30.0%, +5.0pp, CI [−2.5, +12.5], 36/26/138.
- Executable MBPP: 1.5% → 1.5%, 0.0pp, CI [−2.0, +2.0], 2/2/196.

## Required claim schema

- Component: corrected rank-8 RS-LoRA direct merge (math 1.0, formatting 0.7).
- Behaviour/task family: structurally held-out strict schema, concise
  transformations, GSM8K-test math, and MBPP-test executable code.
- Metric/effect size: deterministic programmatic scores and 20,000 paired
  bootstrap CIs listed above.
- Ablation/patching/steering: not tested; not a mechanistic claim.
- Training-delta evidence: not applicable; no new training occurred.
- Controls: source split separation, exact + near-duplicate audit, M02 output
  fixtures, scorer fixtures, paired greedy decoding, and raw manual review.
- Failure modes: schema and concise rows remain deterministic generators;
  broader naturalistic replication is still needed. Low base code rate limits
  sensitivity for code preservation.
- Confidence: Strong negative evidence against generalizing S01's schema
  result to this distinct schema structure; Medium evidence for the narrow
  concise-template effect.

## Variance

Pilot deterministic decoding; uncertainty is paired prompt resampling, not
training-seed variance.

## Comparison to Phase 1

Downgraded: S01's schema gain is format/template-specific under this test, not
evidence of broad schema reliability or adapter synergy.

## Reproduction command

```bash
python scripts/eval/score_minicpm5_s01.py \
  --eval-set data/eval/minicpm5_s02_heldout.jsonl \
  --base results/evals/S02_base_q4/outputs.jsonl \
  --merged results/evals/S02_merged_q4/outputs.jsonl \
  --output results/evals/S02_paired_metrics.json \
  --manual-review results/evals/S02_manual_review.json \
  --experiment-id S02
```
