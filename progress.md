# Progress

## Current session
- Date: 2026-06-21
- Agent/session id: hyperbot-telegram
- Model: qwen3.5-0.5b (target, pending download)
- Backend: TBD (TransformerLens preferred, NNsight fallback, HF native last resort)
- Hardware: micro (CPU-only) initially, aero (RTX 2070 Super 8GB) for training
- Current goal: Complete minimum viable milestone

## Current state summary
Repo scaffold COMPLETE. 86 files committed. All smoke tests pass (7/7). All pytest tests pass (34/34). Task suite v0 generated: 92 examples across 12 families. Config files, report templates, experiment registry all working. Next: model loading and baseline evaluation.

## Completed
- [x] Repo scaffold
- [ ] Model loads
- [ ] Tokenizer verified
- [ ] Deterministic generation
- [x] Task suite v0
- [ ] Metrics validated (unit tests pass, not validated against model yet)
- [ ] Baseline evals
- [ ] Activation caching
- [ ] Layer ablations
- [ ] Head ablations
- [ ] MLP ablations
- [ ] Residual patching
- [ ] Head/MLP patching
- [ ] Steering sweeps
- [ ] CPT training
- [ ] SFT training
- [ ] CPT→SFT training
- [ ] LoRA sweeps
- [ ] Dataset ablation sweeps
- [ ] Hyperparameter sweeps
- [ ] Checkpoint comparison
- [ ] Adapter comparison
- [ ] SAE training
- [ ] SAE intervention tests
- [ ] Component atlas
- [ ] Blog material
- [ ] Paper/report material

## Key findings so far
None yet. Scaffold only.

## Open hypotheses
None yet.

## Failed experiments / dead ends
- Initial token_entropy test used [1.0, 0.0, 0.0] logits which softmax to [0.576, 0.212, 0.212] — not one-hot. Fixed to use [100.0, 0.0, 0.0].

## Artifact index summary
- `data/eval_sets/task_suite_v0.json` — 92 examples, 12 families
- `data/clean_corrupt_pairs/pairs_v0.json` — 20 clean/corrupt pairs
- `experiments/registry.jsonl` — empty (no experiments yet)

## Next actions
1. Install torch + transformers on target machine (micro or aero)
2. Load qwen3.5-0.5b (or fallback) and verify model loads
3. Run tokenizer diagnostics
4. Run deterministic generation check
5. Run baseline evaluation (scripts/run_baselines.py)
6. Run layer ablation (scripts/run_layer_ablation.py)
7. Save plots and update reports
8. Commit results

## Repro commands
```bash
cd ~/work/autonomous-small-model-exploration
.venv/bin/python scripts/run_smoke_tests.py
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/build_task_suite.py
```

## Notes
- All code runs on CPU. Heavy experiments (training, full ablation sweeps) should target aero.
- TransformerLens backend preferred for ablation/patching (hook support). HF backend works for basic eval.
- 12 task families: copying, delimiter_tracking, json_schema, factual_recall, arithmetic, code_syntax, code_semantics, dead_code, verbosity_control, uncertainty_signalling, refusal_compliance, variable_renaming.
- Research prompt saved at docs/RESEARCH_PROMPT.md.
