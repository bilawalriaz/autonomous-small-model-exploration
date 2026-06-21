# Progress

## Current session
- Date: 2026-06-22
- Agent/session id: hyperbot-telegram-continuation
- Model: Qwen/Qwen2.5-0.5B (24L, 14H GQA, d=896, ~0.49B params)
- Backend: HF native with manual hooks
- Hardware: aero (RTX 2070 Super 8GB, bf16)
- Current goal: Component atlas construction, checkpoint timeline

## Current state summary
16 experiments completed across 3 sessions. Rich causal atlas with training perturbation data. L2 confirmed as universal hub. Dataset shard ablation reveals each skill concentrates in DIFFERENT layers (rejecting universal L0-L2 hypothesis). Adapter weights concentrate in late layers (L20-L23) but ablation effects hit early layers (L0-L2). Adapter stacking shows factual+json combine cleanly while delimiter adapter is destructive. 10 adapters trained and analyzed.

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
- [ ] Residual patching (method limitation - KL=0 everywhere)
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
- [ ] Component atlas
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

## Open hypotheses
### H1: L2 is a general-purpose routing hub
Status: supported (strong ablation + steering evidence)

### H2: LoRA training concentrates skill into early layers
Status: REJECTED. Each skill concentrates in different layers. factual_recall in L3/16/19, code in L1/10/21, json in L6/12/13.

### H3: Higher LoRA rank distributes skill across components
Status: supported. Norm data confirms: r=1 uniform, r=16 late-layer concentration.

### H4: o_proj is the key skill injection pathway
Status: supported for JSON. Needs replication on other families.

### H5: Factual recall and algorithmic tasks use different circuits
Status: weakened. Both depend on L2 heavily. But post-training concentration differs (factual->L3/16/19 vs code->L1/10/21).

### H6: Adapter weights write to late layers but effects propagate upstream
Status: new. Norm data (L20-L23 peaks) vs ablation data (L0-L2 peaks). Needs causal test.

## Failed experiments / dead ends
1. Full SFT OOMs on 8GB. LoRA required.
2. Full-residual activation patching gives KL=0. Position-specific patching needed.
3. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.

## Artifact index summary
- 10 LoRA adapters at experiments/adapters/
- 5 family-specific adapters: copying, delimiter, factual, code, json_schema (all r=8)
- 5 rank sweep adapters: r=1,2,4,8,16 for JSON
- 9 plots at experiments/plots/
- 8 result JSONs at experiments/results/
- 9 table JSONs at experiments/tables/
- 16 experiments in experiments/registry.jsonl

## Next actions (ranked)
1. Component atlas construction (formalize all findings into atlas schema)
2. Checkpoint timeline (save checkpoints at different training steps, track skill emergence)
3. Position-specific activation patching at L0/L2/L1
4. Cross-model activation patching (trained-to-base)
5. Negative steering / skill knockout
6. CPT training
7. Blog post outline
8. Paper outline

## Repro commands
```bash
ssh aero
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate
python scripts/run_smoke_tests.py
python scripts/run_layer_ablation.py
python scripts/run_dataset_shard_ablation.py
python scripts/run_adapter_archaeology.py
python scripts/run_adapter_stacking.py
```
