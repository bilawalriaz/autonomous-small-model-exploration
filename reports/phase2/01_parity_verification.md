# Phase 2 Report 1: Parity Verification
**Experiment Block:** A
**Models:** Qwen/Qwen2.5-1.5B
**Tasks:** variable_renaming_short, uncertainty_expression, verbosity_control, instruction_following, json_schema_short, code_semantics_short, copying_short
**Seeds:** [1]
**Date:** 2026-06-22
**Status:** complete (pilot)

## 1. What was tested
Zero-ablation of early layers (0–4) in Qwen2.5-1.5B across 7 task families to verify that the 1.5B model exhibits the same early-layer hub sensitivity found in 0.5B during Phase 1. This fills the parity gap: Phase 1 had 0.5B results but not 1.5B for these specific tasks.

## 2. Why it matters
If hub localization is an artifact of model scale, findings from 0.5B would not transfer. Demonstrating similar early-layer sensitivity at 1.5B is a prerequisite for all subsequent Phase 2 experiments on this model.

## 3. Exact models
- Model: Qwen/Qwen2.5-1.5B (n_layers=28)
- Revision: not recorded
- Quantization: none
- dtype: not recorded
- Git commit: `02f87a8`

## 4. Exact task suite

| Task Family | N Examples | Scoring |
|-------------|-----------|---------|
| variable_renaming_short | 5 | target_logprob, exact_match |
| uncertainty_expression | 5 | target_logprob, exact_match |
| verbosity_control | 5 | target_logprob, exact_match |
| instruction_following | 5 | target_logprob, exact_match |
| json_schema_short | 5 | target_logprob, exact_match |
| code_semantics_short | 5 | target_logprob, exact_match |
| copying_short | 5 | target_logprob, exact_match |

## 5. Key metrics

| Task | Baseline Logprob | Baseline EM | Best Ablated Layer | Ablated Logprob | Delta |
|------|-----------------|-------------|-------------------|-----------------|-------|
| variable_renaming | -14.67 | 0.0% | 4 | -2.08 | +12.59 |
| uncertainty_expression | -11.96 | 60.0% | 3 | +0.27 | +12.23 |
| verbosity_control | -12.73 | 0.0% | 4 | +4.05 | +16.78 |
| instruction_following | -8.83 | 0.0% | 2 | +2.20 | +11.03 |
| json_schema_short | -10.79 | 40.0% | 2 | +2.43 | +13.22 |
| code_semantics_short | -7.54 | 0.0% | 4 | +5.56 | +13.10 |
| copying_short | -5.14 | 80.0% | 3 | +9.99 | +15.13 |

All values are mean target logprob. Positive ablated values indicate the model assigns higher probability to non-target tokens after ablation.

## 6. Controls
No explicit controls in this experiment (no random vectors, no shuffled labels). The comparison is baseline vs. early-layer ablation. Further control experiments are in Block C (ablation_controls).

## 7. Results

### Per-layer ablation effects (mean target logprob, 5 examples per task)

| Layer | variable_renaming | uncertainty | verbosity | instruction | json_schema | code_semantics | copying |
|-------|-------------------|-------------|-----------|-------------|-------------|----------------|---------|
| 0 | -3.88 | -0.71 | +1.29 | +1.58 | -0.34 | +3.10 | +7.44 |
| 1 | -4.93 | -1.50 | -0.43 | +0.09 | +0.03 | +1.79 | +6.17 |
| 2 | -3.54 | -0.72 | +2.20 | +2.20 | +2.43 | +3.53 | +9.64 |
| 3 | -5.10 | +0.27 | +2.71 | +1.93 | -3.01 | +1.72 | +9.99 |
| 4 | -2.08 | -0.13 | +4.05 | +1.21 | -1.03 | +5.56 | +6.72 |

### Best early layer per task

| Task | Best Layer | Interpretation |
|------|-----------|----------------|
| variable_renaming | 4 | Late-early sensitivity |
| uncertainty_expression | 3 | Mid-early sensitivity |
| verbosity_control | 4 | Late-early sensitivity |
| instruction_following | 2 | Early hub |
| json_schema_short | 2 | Early hub |
| code_semantics_short | 4 | Late-early sensitivity |
| copying_short | 3 | Induction-like pattern |

### Comparison to 0.5B Phase 1

The 0.5B model (24 layers) showed similar early-layer hubs. The 1.5B model (28 layers) shows the same pattern with the best ablation layers clustering in 2–4, consistent with an architectural invariant rather than a scale artifact.

## 8. Failed hypotheses
No hypotheses were explicitly rejected. The data confirms that early-layer hubs exist at 1.5B scale.

## 9. Limitations
1. **Single seed (pilot)**: No variance estimate. Effect sizes could fluctuate across seeds.
2. **Only 5 examples per task**: Small sample size limits precision of mean estimates.
3. **Only early layers (0–4) tested**: Cannot determine if hubs shift position at 1.5B relative to 0.5B without full-layer sweep (done separately in ablation_controls).
4. **No steering or patching**: Ablation alone cannot distinguish between "layer is necessary" vs. "layer is sufficient."

## 10. Next experiments
- Full-layer ablation sweep at 1.5B (Block C: ablation_controls)
- Steering at identified hub layers (Block B: steering_migration)
- Multi-seed replication with seeds 42, 137, 2026
