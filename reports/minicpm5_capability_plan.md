# MiniCPM5-1B capability plan

> Superseded as the active roadmap by
> `reports/minicpm5_single_shot_master_plan.md`. This document remains useful
> for merge-path history and later agent-stage constraints.

## Objective

Build a compact, reproducible specialist blend for mathematics, repository code
work, and tool use without describing it as a general-capability improvement
unless it passes held-out, cross-domain regression gates.

## Gate 0 — repair and freeze the merge path

Use `merge_lora_direct.py` with RS-LoRA-aware scaling, or sequential native
PEFT merge. Never use PEFT `linear` or `cat` for this adapter pair without a
fresh FP16 equivalence check. Re-export Q4 with the llama.cpp converter and
rerun deterministic base-versus-corrected-merge evaluations before using any
previous MiniCPM merge score.

Acceptance: 168/168 intended tensors covered, FP16 maximum tensor error below
`1e-3`, and identical greedy continuations against a sequential-PEFT control.

## Block A — build three independently auditable experts

Train math, code, and tool-use LoRAs independently from the same frozen base,
with seeds 42, 137, and 2026. Each uses the same rank, target modules, chat
template, precision, and max sequence length.

| Expert | Training signal | Held-out primary metric | Non-target controls |
|---|---|---|---|
| Math | clean, answer-verifiable worked problems | exact answer on disjoint problems | JSON, code tests, tool format |
| Code | repository tasks with unit-test/verifier outcomes | task pass rate | math, JSON, tool format |
| Tool use | successful, stateful, verifier-backed trajectories | multi-step task completion | malformed calls, code tests, math |

Data quality rules: deduplicate train/eval, retain task provenance, reject
unverified synthetic answers, and exclude provider/rate-limit failures from
preference data. Tool traces must use MiniCPM's native XML tool-call template,
not a generic JSON imitation.

Acceptance per expert: a positive 95% bootstrap lower bound on its primary
metric and no pre-registered catastrophic regression on the control suite.

## Block B — merge selection, not merge accumulation

For each seed, evaluate base, every expert, and a staged weight sweep. Start
with `{0, 0.25, 0.5, 0.75, 1.0}` for the new expert while retaining previously
selected weights; discard candidates that fail a regression gate before testing
the full cross-product. Select weights only on a validation split, lock them,
and report once on a separate held-out split.

The score is a constrained utility: target gains count only after passing
minimum non-target metrics. Record paired wins/losses and bootstrap intervals,
not only average scores.

## Block C — consolidation

Take the selected direct merge as a starting checkpoint and train a small
consolidation LoRA on a balanced replay mixture of all three expert datasets,
general instruction-following data, and explicit format-preservation examples.
Use low learning rate, early checkpoint evaluation, and domain-balanced sampling.
This is the step that can resolve interference; raw delta addition cannot.

Compare: base, best single expert, best direct blend, balanced joint LoRA, and
blend-plus-consolidation. Keep the smallest model that wins the held-out
constrained utility.

## Block D — preference optimization and release

Apply DPO only to verifier-backed preference pairs after SFT/consolidation is
stable. Include hard-negative tool failures and efficient-success pairs as
separate analyses. Export F16, Q8, and Q4; require equivalent behavior on a
fixed smoke set before release. Publish adapter hashes, merge weights, dataset
manifests, prompts, decoding settings, outputs, and confidence intervals.

## Stop conditions

Do not add another expert if it lacks a positive held-out lower bound, violates
a regression budget, or its benefit disappears after Q4 export. At 1B scale,
the target is a high-quality local specialist blend—not an unsupported claim of
universal capability improvement.
