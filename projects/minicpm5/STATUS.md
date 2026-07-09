# MiniCPM5-1B status

## Current stage

**Stage 0 — measurement and data admission.** The next work is to build and
freeze larger train/validation/held-out manifests before further training.

## Facts to preserve

- Both existing math and formatting adapters are rank-8 RS-LoRA (`alpha=16`),
  so direct merging requires `alpha / sqrt(rank)` scaling.
- M01: corrected direct merge covers 168/168 projections and matches
  sequential native PEFT merging in FP16. PEFT `linear` and `cat` are invalid
  for this pair in the installed runtime.
- M02: the GGUF evaluator uses the exact HF MiniCPM template, deterministic
  greedy decoding, explicit stops, and boundary fixtures.
- M03: corrected merge produced a credible JSON gain (6/12 → 12/12; paired
  CI +25 to +75pp). Math/code point estimates were unchanged; the small
  controls cannot certify preservation and do not demonstrate regression.

## Immediate next action

Create an S01 card and config, then build versioned source manifests and
200-example frozen evaluation suites before starting any training.

## Do not do

- Do not reuse pre-M01 MiniCPM behavioral scores.
- Do not call M03 “synergy” or general improvement.
- Do not use a mock judge.
- Do not train an agent adapter until the single-shot release gate passes.
- Do not use PEFT `linear`/`cat` to combine the existing adapters.
