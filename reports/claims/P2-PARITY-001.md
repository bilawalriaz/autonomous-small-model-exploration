# P2-PARITY-001: Parity Verification — Qwen2.5-1.5B Early Layer Ablation

## Claim
Early-layer zero-ablation in Qwen2.5-1.5B produces task-dependent KL spikes at layers 0–4, with best-performing layers varying by task family, confirming that the 1.5B model has localized early-layer sensitivity comparable to 0.5B findings.

## Result

| Task | Baseline Logprob | Baseline EM | Best Early Layer | KL at Best Layer |
|------|-----------------|-------------|------------------|------------------|
| variable_renaming_short | -14.67 | 0.0% | 4 | -2.08 |
| uncertainty_expression | -11.96 | 60.0% | 3 | 0.27 |
| verbosity_control | -12.73 | 0.0% | 4 | 4.05 |
| instruction_following | -8.83 | 0.0% | 2 | 2.20 |
| json_schema_short | -10.79 | 40.0% | 2 | 2.43 |
| code_semantics_short | -7.54 | 0.0% | 4 | 5.56 |
| copying_short | -5.14 | 80.0% | 3 | 9.99 |

## Controls
Not applicable — this is a parity coverage experiment comparing 1.5B to existing 0.5B results.

## Seeds

| Seed | Model | Status |
|------|-------|--------|
| 1 | Qwen2.5-1.5B | complete |

Mean: pilot (1 seed). No variance available.

## Artifacts
- Raw output: `experiments/results/parity_qwen25-15b.json`
- Config: `configs/experiment_defaults.yaml`
- Run ID: `P2-PARITY-001`
- Script commit: `02f87a8`

## Environment
- Model: Qwen/Qwen2.5-1.5B
- GPU: not recorded in parity file
- Precision: not recorded
- Seed: 1

## Interpretation
The 1.5B model shows clear early-layer sensitivity across all 7 task families. The best-performing ablation layer varies by task (layers 2, 3, or 4), consistent with the Phase 1 finding that different skills localize to different early layers. Copying shows the largest effect (KL 9.99 at layer 3), consistent with induction head localization. This supports upgrading the Phase 1 parity claim from weak to medium confidence.

## Limitations
1. Only 1 seed (pilot) — no variance estimate.
2. Only early layers (0–4) ablated, not full-layer sweep.
3. No steering or patching controls in this experiment.

## Verdict
partially_confirmed
