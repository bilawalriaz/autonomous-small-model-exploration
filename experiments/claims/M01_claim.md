# M01 claim — MiniCPM5-1B LoRA merge implementation

## Finding

For the two tested RS-LoRA adapters, direct tensor surgery and sequential native
PEFT merging implement the same weighted delta sum; the earlier direct-merge
GGUF did not apply its stated weights because it used ordinary-LoRA scaling.

## Evidence

- `results/evals/M01_minicpm5_merge_equivalence_rslora.json`
- 168/168 adapted projection matrices matched the analytic RS-LoRA sum; no
  adapter keys were absent from the base model.
- Corrected direct merge: maximum tensor error `8.48e-4`, mean `4.89e-6`
  (FP16 save/load rounding).
- Sequential PEFT merge: maximum tensor error `8.48e-4`, mean `6.53e-6`,
  maximum next-token logit difference `0.0457`, and 3/3 greedy continuations
  identical to direct merge.
- PEFT `linear` and `cat` factor combinations differed materially in this
  PEFT 0.18.1 / adapter 0.19.1 environment; neither is validated for this use.

## Variance

Pilot — deterministic implementation validation; no training variance.

## Confidence level

Strong for merge equivalence of these two adapters in FP16. No behavioral
improvement claim: corrected Q4 held-out evaluation and multi-seed retraining
remain pending.

## Comparison to Phase 1

New; it downgrades the previous MiniCPM "synergistic cognitive boost" statement
to an unvalidated observation because the original direct merger under-scaled
RS-LoRA deltas by `1/sqrt(r)`.

## Reproduction command

```bash
python scripts/eval/validate_lora_merge_equivalence.py \
  --config configs/M01_minicpm5_merge_validation.json \
  --direct-model /home/billz/results/minicpm5_1b_direct_merge_rslora_m1.0_f0.7 \
  --output results/evals/M01_minicpm5_merge_equivalence_rslora.json
```
