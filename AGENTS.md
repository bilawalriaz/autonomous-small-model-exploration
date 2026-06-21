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
