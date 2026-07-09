# M03 claim — corrected MiniCPM Q4 merge gate

## Finding

The corrected 1.0/0.7 merge has a credible JSON-format gain on this pilot;
the small control suites are insufficient to certify preservation, but do not
show an observed regression.

## Evidence

- Math: 75.0% vs base 75.0%, paired delta 0.0pp, 95% CI [-33.3, +33.3].
- JSON: 100.0% vs 50.0%, delta +50.0pp, 95% CI [+25.0, +75.0].
- Code control: 41.7% vs 41.7%, delta 0.0pp, CI [-33.3, +33.3].
- Tool-format control: 0.0% for both; delta 0.0pp, CI [0.0, 0.0].
- Extraction integrity: 0 prompt echoes, residues, empty outputs, or
  subprocess failures across 96 generations.

## Variance

Pilot — deterministic decoding; bootstrap uncertainty is paired prompt
resampling, not training-seed variance.

## Confidence level

Medium for a narrow JSON-format improvement. Control evidence is inconclusive
because the 12-prompt families are too small, so this is not a synergy or
general-capability claim.

## Comparison to Phase 1

Downgraded: prior behavioral scores came from the wrong RS-LoRA scaling and
cannot support a synergy claim.

## Reproduction command

```bash
python scripts/eval/score_minicpm5_m03.py --eval-set data/eval/minicpm5_m03_heldout.jsonl --base results/evals/M03_base_q4/outputs.jsonl --merged results/evals/M03_merged_q4/outputs.jsonl --output results/evals/M03_corrected_q4_paired_metrics.json
```
