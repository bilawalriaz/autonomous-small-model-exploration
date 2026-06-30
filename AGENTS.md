# Mechanistic Interpretability + Training Perturbation Agent

## Mission

Build a reproducible causal atlas of a small language model.

The atlas must connect behaviours to components, training data, hyperparameters, adapters, checkpoints, activations, and interventions.

## Standards

Do not overclaim.
Do not ignore null results.
Do not rely on attention maps alone.
Do not rely on probes alone.
Do not rely on SAE labels alone.
Every claim needs a metric, counterfactual, control, and confidence level.

## Required session startup

Read:
1. AGENTS.md
2. progress.md
3. experiments/registry.jsonl
4. reports/current_findings.md
5. reports/open_hypotheses.md
6. reports/decision_log.md
7. reports/negative_results.md

Then pick the highest-value next step.

## Required session shutdown

Update:
1. progress.md
2. experiments/registry.jsonl
3. reports/current_findings.md
4. reports/open_hypotheses.md
5. reports/negative_results.md if relevant
6. reports/artifact_index.md

## Evidence ladder

Weak:
- attention visualisation
- top activating examples
- probe accuracy
- logit lens hint

Medium:
- ablation effect
- repeated effect across prompts
- consistent trained-vs-base delta

Strong:
- activation patching recovery
- corrupt-to-clean destruction
- held-out replication
- random/shuffled controls ruled out
- predictable steering

Very strong:
- selective knockout
- skill injection
- circuit reconstruction
- replicated across seeds and training regimes

## Claim schema

Every claim must include:
- component
- behaviour
- task family
- metric
- effect size
- ablation result
- patching result
- steering result if tested
- training-delta evidence if tested
- controls
- failure modes
- confidence
- reproducibility command

## Default task families

1. Copying / induction
2. Bracket and delimiter tracking
3. JSON/schema following
4. Factual recall
5. Arithmetic micro-reasoning
6. Code syntax recognition
7. Code semantic preservation
8. Variable renaming / alias tracking
9. Dead-code detection
10. Refusal/compliance style using benign prompts only
11. Verbosity/style control
12. Uncertainty/error signalling

## Default training perturbation families

1. Continued pretraining only
2. SFT only
3. Continued pretraining then SFT
4. LoRA rank sweeps
5. LoRA target-module sweeps
6. Layer-freezing sweeps
7. Dataset-family ablations
8. Curriculum-order experiments
9. Mixed-skill interference experiments
10. Skill injection and skill knockout experiments

## Rule

If a conclusion sounds exciting, attack it harder before reporting it.

---

## Phase 2 Rules (2026-06-22)

Phase 2 focuses on reproducibility and testing whether Phase 1 findings hold under stricter methodology. All Phase 1 rules remain in effect. The following additional rules apply to Phase 2 experiments.

### Experiment card required

Before any experiment runs, an experiment card must exist at `experiments/cards/{experiment_id}.md` with:
- Hypothesis (which H1-H8 it tests)
- Method (ablation type, steering config, training config)
- Seeds (default 3, or "pilot" with justification)
- Metrics to collect
- Expected result (quantitative prediction)
- Falsification criteria (what would reject the hypothesis)
- Estimated compute time
- Dependencies on prior experiments

No experiment runs without a card. Cards can be created in-line during a session but must exist before the first run_id is generated.

### Multi-seed replication is the default

All Phase 2 experiments must run with 3 seeds (42, 137, 2026) unless explicitly marked as **pilot** in the experiment card. Pilot experiments use 1 seed and must justify why (e.g., expensive checkpoint training, exploratory cross-model patching).

Pilot results are provisional and cannot upgrade Phase 1 findings to higher confidence without replication.

### Claim cards required for head results

Any experiment result that would update a confidence level, confirm/reject a hypothesis, or modify the component atlas must produce a claim card at `experiments/claims/{experiment_id}_claim.md` with:
- Finding (one sentence)
- Evidence (run_ids, metrics, effect sizes)
- Variance (σ across seeds, or "pilot — no variance")
- Confidence level (Weak/Medium/Strong/Very strong)
- Comparison to Phase 1 (same / upgraded / downgraded / new)
- Reproduction command

### Negative results are first-class

Negative results (null effects, failed predictions, rejected hypotheses) must be logged in `reports/negative_results.md` with the same rigor as positive results. A negative result that replicates across 3 seeds is **more informative** than a positive result from a single seed.

### Configs directory for reproducibility

All experiment configurations must be saved as JSON at `configs/{experiment_id}_{config_slug}.json` before execution. Configs must include:
- model_name_or_path
- seeds
- ablation_type (zero/mean/resample)
- task_family
- lora_config (if training)
- steering_config (if steering)
- hardware info (auto-populated)

Configs enable exact re-execution without code inspection.

### run_id format

All Phase 2 runs use the format:
```
P2_{experiment_id}_{model_slug}_{task_slug}_{YYYYMMDD_HHMMSS}_seed{seed}
```

Example: `P2_A01_qwen05b_factual_20260622_143022_seed42`

- experiment_id: Block letter + sequence (A01, B01, C01, etc.)
- model_slug: short model identifier (qwen05b, qwen15b, etc.)
- task_slug: task family short name
- timestamp: run start time
- seed: integer seed value

run_ids are written to `experiments/registry.jsonl` as in Phase 1.

### Resumability

Before running an experiment, check `experiments/registry.jsonl` for existing run_ids matching the planned experiment_id + model + task + seed combination. If a completed run_id exists, skip it unless `--force` is passed.

This allows interrupted sessions to resume without re-running completed work.

Implementation: each experiment script should accept `--force` flag (default: False) and check registry before executing.

### Phase 2 evidence ladder addition

Phase 2 adds a new tier above "Very strong":

**Very strong + replicated:**
- All "Very strong" criteria met
- Replicated across 3 seeds with σ < 20% of effect size
- Replicated across at least 2 ablation methods (zero + mean)
- Claim card with variance table

This tier is the minimum for publication-ready claims.

---

## Phase 9 Rules (2026-06-29)

Phase 9 is a data-format ablation study. The core question: "What is the optimal information shape for fine-tuning 230M-500M language models?"

### What changed

Phase 8 showed dataset format dominates hyperparameters (5x impact). Phase 9 isolates format from content by holding canonical content constant and rendering into 6 training formats.

### Scientific controls

1. Same canonical content across format variants.
2. Same train/eval split.
3. Same model (LiquidAI/LFM2.5-230M).
4. Same LoRA config within adapter track (quality or surgical).
5. Same optimizer (Adafactor).
6. Same LR (2e-4).
7. Same steps (300).
8. Same max sequence length (1024).
9. Same eval prompts (data/eval/small_model_eval_v1.jsonl).
10. Same decoding settings (temp=0.2, top_p=0.9).
11. Same judge rubric.
12. Blind pairwise judging.
13. Cached judge outputs.
14. Manual review sample mandatory.
15. Config snapshots for every run.

### Hypotheses under test

- H1: Multi-turn concise format is genuinely better for small-model SFT
- H2: smol-magpie-ultra advantage is partly format, not merely content
- H3: Small models benefit from dense, compact examples more than verbose ones
- H4: Low training loss may not correlate with behavioral quality
- H5: Surgical LoRA can add useful behavior while preserving base model distribution
- H6: Structured terse data may outperform verbose chat on JSON/extraction/code
- H7: There is a distinct "small-model-native" data style

### Adapter tracks

Quality: hub layers, all modules, r=8, Adafactor, lr=2e-4, 300 steps
Surgical: hub + o_proj only, r=8, Adafactor, lr=2e-4, 300 steps

### Pipeline commands

```bash
# Render format variants
python scripts/data/render_dataset_formats.py --config configs/experiments/format_ablation_quality.yaml
# Validate
python scripts/data/validate_dataset_formats.py --dataset-dir data/sft/format_ablation/ --canonical data/canonical/phase9_pilot_300.jsonl
# Train
python scripts/train/run_format_ablation.py --config configs/experiments/format_ablation_quality.yaml
# Evaluate
python scripts/eval/run_eval_harness.py --config configs/eval/lfm2_small_model_eval.yaml --run-id <run_id>
# Judge
python scripts/eval/judge_outputs.py --run-id <run_id>
# Aggregate
python scripts/eval/aggregate_eval_results.py --run-id <run_id>
# Report
python scripts/report/build_phase09_report.py
```

### Report must answer

- Did multi-turn concise still win when content was held constant?
- Was smol-magpie advantage mostly content, mostly format, or both?
- Which format gives best behavioral win-rate?
- Which format gives best loss?
- Do loss and behavioral quality correlate?
- Does surgical LoRA preserve the model while improving target behaviors?
- Is there a small-model-native data shape?

### run_id format

`lfm2_230m_{adapter_type}_{format_name}_{date_or_hash}`
