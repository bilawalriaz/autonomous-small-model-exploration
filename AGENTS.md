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
