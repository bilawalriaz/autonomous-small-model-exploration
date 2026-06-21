# Progress

## Current session
- Date: 2026-06-22
- Agent/session id: hyperbot-telegram-continuation
- Model: Qwen/Qwen2.5-0.5B (24L, 14H GQA, d=896, ~0.49B params)
- Backend: HF native with manual hooks
- Hardware: aero (RTX 2070 Super 8GB, bf16)
- Current goal: Dataset shard ablation + component atlas construction

## Current state summary
13 experiments completed across 2 sessions. Strong causal atlas emerging. L2 identified as universal importance hub. LoRA training rewires component importance (L0 MLP absorbs JSON skill). Steering confirms L2 causal role for factual recall (3.3x boost). Patching method limited by full-residual replacement (KL=0 everywhere). Next: dataset shard ablation, checkpoint timeline, adapter archaeology.

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
- [ ] Residual patching (method limitation - KL=0 everywhere with full-residual)
- [ ] Head/MLP patching
- [x] Steering sweeps (L2 factual: 3.3x boost at s=+4.0)
- [ ] CPT training
- [ ] SFT training (OOM on 8GB, using LoRA instead)
- [ ] CPT->SFT training
- [x] LoRA sweeps (rank: r=1,2,4,8,16; module: q/v/o/mlp/attn/all)
- [ ] Dataset ablation sweeps
- [ ] Hyperparameter sweeps
- [x] Checkpoint comparison (LoRA before/after, ablation maps)
- [ ] Adapter comparison (multiple skill adapters)
- [ ] SAE training
- [ ] SAE intervention tests
- [ ] Component atlas
- [ ] Blog material
- [ ] Paper/report material

## Key findings so far
1. L2 is a universal importance hub (ablation KL 0.5-11.5 across all families). HIGH confidence.
2. L0 MLP is second-strongest across all families (KL 0.3-10.8). MEDIUM confidence.
3. L2 and L0 dominate jointly. Top-2 layers explain >60% of ablation effect. MEDIUM confidence.
4. 14 attention heads span 5 families. L12 H8 is strongest head. MEDIUM confidence.
5. MLP contribution exceeds attention at L2, L4, L6. MEDIUM confidence.
6. Factual recall most sensitive to layer ablation. Code semantics most resistant. MEDIUM confidence.
7. Steering L2 with factual direction boosts "rome" 0.064->0.213 (3.3x) at s=+4.0. MEDIUM confidence.
8. JSON steering at L21 shifts top token from "{" to "Name"/"She" (negative direction). MEDIUM confidence.
9. LoRA training shifts JSON component importance: L0 MLP 10.85->13.84 (+2.99). MEDIUM confidence.
10. L2 MLP reduced after JSON LoRA: 4.81->2.94 (-1.87). MEDIUM confidence.
11. L5 MLP eliminated after JSON LoRA: 1.56->0.00 (-1.56). MEDIUM confidence.
12. L4 appeared for factual_recall (+1.42) and refusal (+0.95) after LoRA. MEDIUM confidence.
13. LoRA rank sweep: L0 MLP peaks at r=4 (15.77), declines at higher rank. MEDIUM confidence.
14. L2 MLP drops monotonically with rank: 5.73->2.16 (r=1->16). MEDIUM confidence.
15. o_proj alone most efficient for L0 concentration (+3.64 with 344K params). MEDIUM confidence.

## Open hypotheses
### H1: L2 is a general-purpose routing hub
- Evidence: L2 ablation causes largest KL across all 12 families
- Next test: Position-specific patching at L2 to identify which tokens matter
- Falsifier: If L2 effect is uniform across all positions, it's just a residual magnitude effect

### H2: LoRA training concentrates skill into early layers (L0-L2)
- Evidence: L0 MLP +2.99 for JSON after LoRA training
- Next test: Train adapters on other skill families, check if same concentration occurs
- Falsifier: If delimiter LoRA concentrates in L10-L15 instead of L0-L2

### H3: Higher LoRA rank distributes skill across more components
- Evidence: L0 MLP peaks at r=4 then declines; L2 drops monotonically
- Next test: Check if r=1 adapter is more surgically precise than r=16
- Falsifier: If r=16 has same selectivity as r=4 but stronger effect

### H4: o_proj is the key module for injecting skills via residual stream
- Evidence: o_proj alone achieves +3.64 L0 effect with only 344K params
- Next test: Train o_proj-only adapters on multiple skill families
- Falsifier: If MLP-only adapters work equally well on non-JSON tasks

### H5: Factual recall and algorithmic tasks use different circuits
- Evidence: Factual recall KL=11.54 at L2 vs code semantics KL=0.52
- Next test: Head/MLP ablation maps separately for factual vs algorithmic
- Falsifier: If both families depend on same heads/MLPs with similar sensitivity

## Failed experiments / dead ends
1. Full SFT OOMs on 8GB even with bf16 + gradient checkpointing. LoRA required.
2. Activation patching with full-residual replacement gives KL=0 everywhere. Position-specific patching needed.
3. Clean/corrupt pair v0 had tokenization misalignment. Fixed in v1.

## Artifact index summary
- 5 LoRA adapters: r={1,2,4,8,16} at experiments/adapters/lora_json_r{N}/
- 6 module sweep configs at experiments/results/lora_module_sweep.json
- 9 plots at experiments/plots/
- 7 result JSONs at experiments/results/
- 9 table JSONs at experiments/tables/
- 13 experiments in experiments/registry.jsonl

## Next actions (ranked)
1. Dataset shard ablation - train adapters on different skill families, compare component importance
2. Position-specific activation patching at L0/L2
3. Checkpoint timeline - save checkpoints during training, track when skills emerge
4. Adapter archaeology - LoRA SVD, norm analysis
5. Multiple skill adapters + interference matrix
6. CPT training
7. Component atlas construction
8. Blog post outline

## Repro commands
```bash
ssh aero
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate
python scripts/run_smoke_tests.py
python scripts/run_layer_ablation.py
python scripts/run_steering_sweep.py
python scripts/train_lora_json.py
python scripts/run_lora_rank_sweep.py
python scripts/run_lora_module_sweep.py
python scripts/compare_lora_ablation.py
```

## Notes
- Model: Qwen2.5-0.5B, 24 layers, 14 heads (GQA), d_model=896, d_head=64
- 8GB VRAM budget: bf16 model ~1GB, batch processing with torch.cuda.empty_cache()
- LoRA training: 100 steps, lr=2e-4, bs=2, r=8, alpha=16, all-linear targets
- Ablation uses zero-ablation (zero out component output)
- Steering uses activation addition at residual stream
