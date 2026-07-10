# MiniCPM5-1B status

## Current stage

**Stage 0 — measurement and data admission.** The next work is to build and
freeze larger train/validation/held-out manifests before further training.

S01 has admitted the data and frozen the 800-example suite; the deterministic
corrected-Q4 base-versus-merge baseline is currently running on aero.  Treat
all behavioral conclusions as pending until fixture-validated outputs and the
paired report are present.

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

Build a source-held-out, structurally diverse Stage 0 suite and run a
near-duplicate audit before deciding whether any Stage 1 training experiment
is justified.

## S02 decision

S02 passed the repaired source/split audit and completed baseline measurement.
It rejects broad schema transfer for the existing direct merge: both Q4 models
scored 0/200 on the distinct schema family. Do not train a schema adapter from
this evidence alone. The only positive S02 result is a narrow concise-template
gain (+25.5pp, CI +19.5 to +31.5); math is unresolved and code unchanged.

## Do not do

- Do not reuse pre-M01 MiniCPM behavioral scores.
- Do not call M03 “synergy” or general improvement.
- Do not use a mock judge.
- Do not train an agent adapter until the single-shot release gate passes.
- Do not use PEFT `linear`/`cat` to combine the existing adapters.
