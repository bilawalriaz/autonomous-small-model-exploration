# {{EXPERIMENT_ID}}: {{TITLE}}

## Claim
One precise claim. No hedging, no "seems like."

## Result
| Metric | Value | Effect Size | 95% CI |
|--------|-------|-------------|--------|
| ... | ... | ... | ... |

## Controls
What controls were run and what happened.

| Control | Result | Delta vs Treatment |
|---------|--------|--------------------|
| no_steering | ... | baseline |
| random_vector | ... | ... |
| wrong_task_vector | ... | ... |

## Seeds
Seed-level results. Do not hide variance.

| Seed | Metric | Value | Notes |
|------|--------|-------|-------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |

Mean: ... ± ...

## Artifacts
- Raw output: `experiments/results/{{filename}}.json`
- Summary: `results/summaries/{{filename}}_summary.csv`
- Plot: `results/plots/{{filename}}.png`
- Config: `configs/experiment_defaults.yaml`
- Script: `scripts/{{script_name}}.py`
- Run ID: `{{run_id}}`

## Environment
- Model: {{model_name}} ({{revision}})
- Tokenizer: {{tokenizer_name}} ({{tokenizer_revision}})
- Script commit: {{git_hash}}
- Python: {{python_version}}
- Torch: {{torch_version}}
- GPU: {{gpu_name}}
- Precision: {{dtype}}
- Seed: {{seed}}
- Prompt set hash: {{prompt_hash}}

## Interpretation
What this supports. Be specific about evidence strength.

## Limitations
What this does not prove. List at least 2.

## Verdict
<!-- Choose one: confirmed / partially_confirmed / rejected / inconclusive -->
{{VERDICT}}
