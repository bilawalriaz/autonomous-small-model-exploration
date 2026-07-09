# M02 claim — MiniCPM5-1B GGUF measurement path

## Finding

The revised GGUF evaluator captures generated text without prompt echo or
template residue for the M02 Q4 smoke; this validates measurement only.

## Evidence

- Five deterministic extraction fixtures passed.
- Base and corrected merge smoke runs: 6/6 nonempty raw outputs, zero prompt
  echoes, zero template-residue flags, and zero nonzero subprocess returns.
- The HF MiniCPM template is recorded by SHA-256 in each metadata file.

## Variance

Pilot — deterministic evaluator validation.

## Confidence level

Strong for this binary/template/output-boundary configuration; no behavioural
claim is implied.

## Comparison to Phase 1

New; resolves NR027's measurement defect.

## Reproduction command

```bash
python scripts/eval/run_gguf_eval.py --validate-fixtures
```
