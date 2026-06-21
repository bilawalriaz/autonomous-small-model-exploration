# Progress

## Current session
- Date: 2026-06-22
- Agent/session id: hyperbot-telegram (sessions 1-4)
- Model: Qwen/Qwen2.5-0.5B (24L, 14H GQA, d=896, ~0.49B params)
- Backend: HF native with manual hooks
- Hardware: aero (RTX 2070 Super 8GB, bf16)
- Current goal: Cross-model patching, skill knockout, blog/paper outlines

## Current state summary
18 experiments completed across 4 sessions. Component atlas with 11 entries. Strong causal atlas: L2 universal hub (HIGH), positional specialization (L22=last-token, L0/L2=first+last, L9=instruction), skill-specific LoRA concentration, adapter stacking interference. 10 adapters, 5 checkpoints, 18 result files. All pushed to GitHub.

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
- [ ] Blog material
- [ ] Paper/report material

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
5. Adapter weight norms peak at L20-L23 but ablation effects peak at L0-L2. MEDIUM confidence.
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
Status: weakened. Both depend on L2. Post-training concentration differs.

### H6: Adapter weights write to late layers but effects propagate upstream
Status: open. Norm data supports. Needs causal test (e.g., ablate adapter at L20-L23 only).

### H7: L22 is the unembedding pathway
Status: new. Position data shows L22 exclusively affects last token. Likely where final vocab projection information flows.

## Failed experiments / dead ends
1. Full SFT OOMs on 8GB. LoRA required.
2. Full-residual activation patching gives KL=0 everywhere. Position-specific patching needed.
3. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.

## Artifact index summary
- 10 LoRA adapters at experiments/adapters/ (5 family + 5 rank sweep)
- 5 checkpoints at experiments/checkpoints/ (json_timeline_step10/25/50/75/100)
- 9 plots at experiments/plots/
- 18 result JSONs at experiments/results/
- 9 table JSONs at experiments/tables/
- 18 experiments in experiments/registry.jsonl
- Component atlas: reports/component_atlas.jsonl + .md (11 entries)
- Decision log: reports/decision_log.md (7 decisions)
- Negative results: reports/negative_results.md (6 entries)

## Next actions (ranked)
1. Cross-model activation patching (trained-to-base): Can trained activations transfer learned behaviour into base model?
2. Negative steering / skill knockout: Can we selectively suppress a learned skill?
3. Adapter-only ablation: Ablate adapter at specific layers to test H6 (norm/effect separation)
4. CPT training (continued pretraining on code corpus)
5. Blog post outline
6. Paper outline
7. SAE training on layers now known to matter (L0, L1, L2, L7, L9, L22)

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

# Reports
python scripts/build_component_atlas.py
```

## GitHub push workflow
Aero has no gh auth. Push via bundle:
```bash
# On aero:
cd ~/work/autonomous-small-model-exploration
git add -A && git commit -m "message"
git bundle create /tmp/mi-atlas.bundle master

# On micro:
scp aero:/tmp/mi-atlas.bundle /tmp/mi-atlas.bundle
cd ~/work/autonomous-small-model-exploration
git pull /tmp/mi-atlas.bundle master
git push origin master
```

## Notes
- Model: Qwen2.5-0.5B, 24 layers, 14 heads (GQA), d_model=896, d_head=64
- 8GB VRAM budget: bf16 model ~1GB, batch processing with torch.cuda.empty_cache()
- LoRA training: 100 steps, lr=2e-4, bs=2, r=8, alpha=16, all-linear targets
- Ablation uses zero-ablation (zero out component output)
- Steering uses activation addition at residual stream
- Python 3.12 on aero, venv at ~/work/autonomous-small-model-exploration/.venv
