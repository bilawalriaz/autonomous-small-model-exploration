# Progress

## Current session
- Date: 2026-06-22
- Agent/session id: hyperbot-telegram (sessions 1-5)
- Model: Qwen/Qwen2.5-0.5B (24L, 14H GQA, d=896, ~0.49B params)
- Backend: HF native with manual hooks
- Hardware: aero (RTX 2070 Super 8GB, bf16)
- Current goal: Publication-ready report with comprehensive analyses

## Current state summary
21 experiments completed across 5 sessions. Component atlas with 11 entries. Strong causal atlas: L2 universal hub (HIGH), positional specialization (L22=last-token, L0/L2=first+last, L9=instruction), skill-specific LoRA concentration, adapter stacking interference, cross-model activation transfer, skill knockout, adapter-only ablation. 10 adapters, 5 checkpoints, 21 result files, 15 publication-quality plots. Publication report generated. All pushed to GitHub.

## Completed
- [x] Repo scaffold
- [x] Model loads (Qwen2.5-0.5B, CUDA bf16)
- [x] Tokenizer verified (all brackets single-token)
- [x] Deterministic generation (temp=0, do_sample=False)
- [x] Task suite v0 (92 examples, 12 families)
- [x] Metrics validated (34/34 pytest pass)
- [x] Baseline evals (0% exact match, expected for base model)
- [x] Activation caching (forward hooks on all layers)
- [x] Layer ablations (L2 dominates all families)
- [x] Head ablations (14 heads across 5 families)
- [x] MLP ablations (L2, L4, L6 dominate)
- [ ] Residual patching (method limitation - KL=0 with full-residual)
- [ ] Head/MLP patching
- [x] Steering sweeps (L2 factual: 3.3x boost at s=+4.0)
- [ ] CPT training
- [ ] SFT training (OOM on 8GB, using LoRA instead)
- [ ] CPT->SFT training
- [x] LoRA sweeps (rank: r=1,2,4,8,16; module: q/v/o/mlp/attn/all)
- [x] Dataset ablation sweeps (5 families: copying, delimiter, factual, code, json)
- [ ] Hyperparameter sweeps
- [x] Checkpoint comparison (LoRA before/after, ablation maps)
- [x] Adapter comparison (archaeology, stacking/interference)
- [ ] SAE training
- [ ] SAE intervention tests
- [x] Component atlas (11 entries, jsonl + md)
- [x] Checkpoint timeline (5 checkpoints: step 10/25/50/75/100)
- [x] Position-specific ablation (7 layers, 11 tasks, per-token effects)
- [x] Cross-model activation patching (trained→base, 17 pairs, 24 layers)
- [x] Skill knockout (negative steering, 2 skills, 7+ layers)
- [x] Adapter-only ablation (12 prompts, 24 layers, norm-effect analysis)
- [x] Layer skipping + early exit efficiency experiments (10 skip configs, 7 exit layers, 3 task-aware)
- [x] Blog material (publication report generated)
- [x] Paper/report material (publication report with 16 plots)
- [x] MI-Atlas skill (codified experimentation workflow for reuse with new models)

## Key findings so far
1. L2 is a universal importance hub (ablation KL 0.5-11.5 across all families). HIGH confidence.
2. L0 MLP is second-strongest across all families. MEDIUM confidence.
3. LoRA training rewires component importance: L0 MLP absorbs JSON (+2.99). MEDIUM confidence.
4. Each skill concentrates in DIFFERENT layers after training (H002 rejected):
   - factual_recall: L3, L16, L19
   - code_semantics: L1, L10, L21
   - json_schema: L6, L12, L13
   - copying: dispersed
   - delimiter: fully absorbed (0 ablation sensitivity)
   MEDIUM confidence.
5. Adapter weight norms peak at L20-L23 but general ablation effects peak at L0-L2. MEDIUM confidence.
6. Steering L2 with factual direction boosts "rome" 3.3x. MEDIUM confidence.
7. o_proj most efficient for skill injection (+3.64 with 344K params). MEDIUM confidence.
8. LoRA rank sweep: L0 MLP peaks at r=4, higher rank distributes. MEDIUM confidence.
9. factual+json adapters stack cleanly (synergy +2.35 factual, +1.17 json). MEDIUM confidence.
10. delimiter adapter is destructive when stacked (-7 to -16 nats). MEDIUM confidence.
11. L1 appears as universal skill injection point (positive delta across 3+ adapters). MEDIUM confidence.
12. Total adapter norm scales linearly with rank (6.14 to 22.92, r=1 to r=16). LOW confidence.
13. L22 is almost exclusively a last-position layer (mean effect 14.55 nats, all others ~0). MEDIUM confidence.
14. L0/L2 are first+last position routers (instruction + prediction tokens). MEDIUM confidence.
15. L9 is the strongest instruction-sensitive layer (first=5.66, last=9.20). MEDIUM confidence.
16. Core circuit (L2/L7/L9) for JSON locks in by step 10 (first 10% of training). MEDIUM confidence.
17. Secondary layers (L15, L6) continue shifting through step 100. LOW confidence.
18. Cross-model patching: recovery increases monotonically — L23=100%, L22=97%, L21=95%, L20=87%. Trained behavior encoded in late-layer activations. MEDIUM confidence.
19. Skill knockout: L19 selectively suppresses factual recall (selectivity 11654x at s=-2.0) while preserving JSON/copying. L2 non-selective (universal hub). MEDIUM confidence.
20. Adapter-only ablation: norm-effect correlation = 0.85 (strong positive). Adapter effect peaks at L19-L23, matching norm distribution. H6 UPDATED. MEDIUM confidence.
21. JSON adapter effect is concentrated at L19-L23: L23=100%, L22=92%, L21=81% of total adapter effect. MEDIUM confidence.

## Open hypotheses
### H1: L2 is a general-purpose routing hub
Status: supported. Position data adds nuance: L2 specifically routes first+last positions, not uniform.

### H2: LoRA training concentrates skill into early layers
Status: REJECTED. Each skill concentrates in different layers.

### H3: Higher LoRA rank distributes skill across components
Status: supported.

### H4: o_proj is the key skill injection pathway
Status: supported for JSON. Needs replication on other families.

### H5: Factual recall and algorithmic tasks use different circuits
Status: weakened. Both depend on L2. Post-training concentration differs. Skill knockout at L19 selectively suppresses factual recall.

### H6: Adapter weights write to late layers but effects propagate upstream
Status: UPDATED → REJECTED in original form. Adapter-only ablation shows effect IS at late layers (corr=0.85 with norm). The earlier finding conflated general layer importance (L0-L2) with adapter-specific importance (L19-L23). General ablation at L0-L2 is devastating because those layers are universally important, NOT because the adapter's effect propagates there.

### H7: L22 is the unembedding pathway
Status: supported. Position data shows L22 exclusively affects last token. Cross-model patching confirms L22 carries 97% of trained behavior.

### H8: Trained behavior is encoded in late-layer activation patterns (NEW)
Status: supported. Cross-model patching shows monotonic recovery increase from early to late layers. Patching L23 trained activations into base model gives 100% recovery.

### H9: Skills can be selectively suppressed via negative steering at skill-specific layers (NEW)
Status: supported. L19 selectively knocks out factual recall (11654x selectivity) while preserving JSON and copying. L2 is non-selective.

## Failed experiments / dead ends
1. Full SFT OOMs on 8GB. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.
4. JSON skill knockout had limited effect because base probability of JSON targets was already near-zero.
5. PeftModel.from_pretrained modifies base model in-place — must use disable_adapter() for base behavior.

## Artifact index summary
- 10 LoRA adapters at experiments/adapters/ (5 family + 5 rank sweep)
- 5 checkpoints at experiments/checkpoints/ (json_timeline_step10/25/50/75/100)
- 24 plots at experiments/plots/ (9 original + 15 publication-quality)
- 21 result JSONs at experiments/results/
- 9+ table JSONs at experiments/tables/
- 21 experiments in experiments/registry.jsonl
- Component atlas: reports/component_atlas.jsonl + .md (11 entries)
- Decision log: reports/decision_log.md (7 decisions)
- Negative results: reports/negative_results.md (6+ entries)
- Publication report: reports/publication_report.md (460 lines)

## Next actions (ranked)
1. Multi-seed replication of top 5 findings (for publication confidence)
2. Mean/resample ablation (replace zero ablation for stronger causal claims)
3. CPT training (continued pretraining on code corpus)
4. SAE training on layers now known to matter (L0, L1, L2, L7, L9, L19, L22)
5. Extend to natural language prompts (validate transfer from synthetic)
6. Cross-model validation (try Qwen2.5-1.5B or Qwen3.5-0.5B)
7. Blog post from publication report
8. Skill injection experiment (can we inject a skill at L19 for factual recall?)

## Repro commands
```bash
ssh aero
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate

# Smoke tests
python scripts/run_smoke_tests.py

# Core experiments
python scripts/run_layer_ablation.py
python scripts/run_head_ablation.py
python scripts/run_mlp_ablation.py
python scripts/run_steering_sweep.py

# LoRA training + comparison
python scripts/train_lora_json.py
python scripts/compare_lora_ablation.py
python scripts/run_lora_rank_sweep.py
python scripts/run_lora_module_sweep.py

# Training perturbation
python scripts/run_dataset_shard_ablation.py
python scripts/run_checkpoint_timeline.py

# Adapter analysis
python scripts/run_adapter_archaeology.py
python scripts/run_adapter_stacking.py

# Position analysis
python scripts/run_position_ablation.py

# Cross-model + knockout + adapter ablation (NEW)
python scripts/run_cross_model_patching.py
python scripts/run_skill_knockout.py
python scripts/run_adapter_ablation.py

# Reports
python scripts/build_component_atlas.py
python scripts/generate_publication_report.py
```

## GitHub push workflow
Aero has gh auth configured (as of 2026-06-21). Push directly:
```bash
cd ~/work/autonomous-small-model-exploration
git add -A && git commit -m "message"
git push origin master
```

## Notes
- Model: Qwen2.5-0.5B, 24 layers, 14 heads (GQA), d_model=896, d_head=64
- 8GB VRAM budget: bf16 model ~1GB, batch processing with torch.cuda.empty_cache()
- LoRA training: 100 steps, lr=2e-4, bs=2, r=8, alpha=16, all-linear targets
- Ablation uses zero-ablation (zero out component output)
- Steering uses activation addition at residual stream
- Python 3.12 on aero, venv at ~/work/autonomous-small-model-exploration/.venv
- PeftModel wraps base model in-place — use disable_adapter() context for base behavior
