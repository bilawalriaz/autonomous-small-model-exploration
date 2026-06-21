---
name: mi-atlas-experimentation
description: Build a reproducible mechanistic interpretability atlas of any small LLM — component mapping, ablation, patching, steering, LoRA training perturbation, efficiency testing, and publication-ready reports.
version: 1.0.0
---

# MI-Atlas Experimentation Skill

## Purpose

Run a complete mechanistic interpretability investigation on a small language model (<2B params). Produces a causal atlas connecting behaviours to components, training perturbation effects, efficiency analysis, and a publication-ready report with plots.

## When to Use

- You want to understand how a specific small LLM works internally
- You want to find optimization targets (efficient training, skill injection, pruning candidates)
- You want to produce publishable findings with real evidence (not just attention maps)
- A new model has released and you want to map its architecture quickly

## Prerequisites

- CUDA GPU with >=8GB VRAM (RTX 2070 Super or better)
- Python 3.11+ with: torch, transformers, peft, matplotlib, numpy
- HF model that loads with `AutoModelForCausalLM` (GQA architectures supported)
- SSH access to GPU host if running remotely

## Repository Structure

```
mi-atlas/
├── AGENTS.md              # Research protocol (standards, evidence ladder, claim schema)
├── progress.md            # Current state, completed items, next actions
├── src/mi_atlas/          # Core toolkit
│   ├── model_loader.py    # ModelBundle, load_model_hf, detect_model_info
│   ├── backend.py         # HFBackend, TransformerLensBackend, create_backend
│   ├── ablations.py       # Zero/mean/resample ablation, layer/head/MLP suites
│   ├── patching.py        # Activation patching, run_patching_suite
│   ├── steering.py        # Steering vector computation and injection
│   ├── metrics.py         # KL divergence, exact match, patch score
│   ├── plotting.py        # Heatmaps, line plots, bar charts
│   ├── task_suite.py      # TaskSuite loader, family filtering
│   ├── experiment_registry.py  # JSONL experiment tracking
│   └── training/          # LoRA, SFT, CPT, checkpoint eval
├── scripts/               # Runnable experiment scripts
├── experiments/           # Results, plots, tables, adapters, checkpoints
│   ├── registry.jsonl     # Experiment log (append-only)
│   ├── results/           # JSON result files
│   ├── plots/             # PNG figures
│   └── tables/            # JSON data tables for plots
├── reports/               # Findings, hypotheses, decisions, negative results
├── data/                  # Task suites, clean/corrupt pairs
└── config/                # YAML configs (model, experiment_plan, training_plan)
```

## Workflow: 7 Phases

### Phase 1: Setup & Baseline (Day 1)

```bash
# Clone or create repo structure
ssh <gpu_host>
cd ~/work/<repo>
source .venv/bin/activate

# 1a. Verify model loads
python -c "from mi_atlas.model_loader import load_model_hf; b=load_model_hf('<model_name>'); print(b.architecture)"

# 1b. Run baseline eval
python scripts/run_baselines.py

# 1c. Run smoke tests
python scripts/run_smoke_tests.py
```

Key outputs: baseline_eval.json, tokenizer_diagnostics.json
Expected: base model scores ~0% exact match on task suite (normal for base models)

### Phase 2: Component Mapping (Day 1-2)

Run ablation experiments to identify which components matter:

```bash
# Layer-level ablation (24 layers x 12 families)
python scripts/run_layer_ablation.py --ablation-type zero

# Head-level ablation (top 6 layers x all heads)
python scripts/run_head_ablation.py

# MLP-level ablation (all layers)
python scripts/run_mlp_ablation.py

# Position-specific ablation (key layers x token positions)
python scripts/run_position_ablation.py
```

Key outputs: layer_ablation_zero.json, head_ablation.json, mlp_ablation.json, position_specific_ablation.json + heatmaps

**What to look for:**
- Which layer has the highest mean KL across all families? (universal hub candidate)
- Are head effects small relative to layer effects? (distributed attention)
- Which positions are most affected? (first=instruction, last=prediction, operators=near-zero)

### Phase 3: Causal Interventions (Day 2-3)

```bash
# Build clean/corrupt pairs (verify single-token targets!)
python scripts/build_aligned_pairs.py

# Activation patching (clean -> corrupt recovery)
python scripts/run_patching_v1.py
python scripts/run_patching_kl.py

# Steering vectors (positive/negative sweeps)
python scripts/run_steering_sweep.py
```

**Critical gotchas:**
- ALWAYS verify targets are single-token with `tokenizer(target)["input_ids"]`
- Full-residual patching gives KL=0 for identical-prefix pairs — use position-specific instead
- Steering at s >= +2 may cause degeneration (Chinese chars, repetition) — test fine granularity

### Phase 4: Training Perturbation (Day 3-4)

```bash
# Train LoRA adapter on primary skill
python scripts/train_lora_json.py  # or train_lora.py with custom data

# Compare before/after LoRA
python scripts/compare_lora_ablation.py

# LoRA rank sweep (r=1,2,4,8,16)
python scripts/run_lora_rank_sweep.py

# LoRA target-module sweep (q/v/o/mlp/attn/all)
python scripts/run_lora_module_sweep.py

# Dataset shard ablation (train separate adapters per skill family)
python scripts/run_dataset_shard_ablation.py

# Checkpoint timeline (save checkpoints at step 10/25/50/75/100)
python scripts/run_checkpoint_timeline.py
```

**Key findings to look for:**
- Does each skill concentrate in DIFFERENT layers? (rejects universal concentration hypothesis)
- What rank gives the most surgical adapter? (usually r=4)
- Which module is most parameter-efficient? (usually o_proj — writes to residual stream)
- When does the core circuit stabilize? (often by step 10 — first 10% of training)

### Phase 5: Advanced Interventions (Day 4-5)

```bash
# Adapter archaeology (weight norm distribution)
python scripts/run_adapter_archaeology.py

# Adapter stacking (can skills combine?)
python scripts/run_adapter_stacking.py

# Cross-model patching (trained -> base activation transfer)
python scripts/run_cross_model_patching.py

# Skill knockout (negative steering on trained model)
python scripts/run_skill_knockout.py

# Adapter-only ablation (norm vs effect analysis)
python scripts/run_adapter_ablation.py
```

**Critical gotcha:** `PeftModel.from_pretrained()` modifies the base model IN-PLACE. To get base model behavior, use `with trained_model.disable_adapter():` context manager. Do NOT load the model twice — use disable_adapter() to save VRAM.

### Phase 6: Efficiency Testing (Day 5)

```bash
# Layer skipping + early exit + task-aware selective computation
python scripts/run_layer_skipping_early_exit.py
```

**Expected result:** Naive layer skipping with zero-ablation will likely destroy output (0% top-5 overlap). This is a NEGATIVE result worth reporting — all layers are necessary even if some are "weaker" by KL. The value of the atlas for efficiency is in:
- Training efficiency (core circuit locks in early)
- Parameter efficiency (o_proj-only, r=4)
- Selective skill manipulation (knockout without retraining)
- Targeted optimization (focus on universal hub layer)

### Phase 7: Report Generation (Day 5-6)

```bash
# Generate all publication plots
python scripts/generate_publication_report.py

# Generate efficiency plots
python scripts/plot_efficiency.py

# Build component atlas
python scripts/build_component_atlas.py
```

## Session Protocol

### Startup
1. Read AGENTS.md (research standards)
2. Read progress.md (current state)
3. Read experiments/registry.jsonl (all experiments)
4. Read reports/current_findings.md
5. Read reports/open_hypotheses.md
6. Read reports/decision_log.md
7. Read reports/negative_results.md
8. Pick highest-value next step

### Shutdown
1. Update progress.md
2. Update experiments/registry.jsonl (auto-handled by register_experiment)
3. Update reports/current_findings.md
4. Update reports/open_hypotheses.md
5. Update reports/negative_results.md if relevant
6. Update reports/artifact_index.md
7. Git commit + push

## Evidence Ladder

Every claim must be backed by evidence at the appropriate level:

- **Weak:** attention visualization, top activating examples, probe accuracy, logit lens hint
- **Medium:** ablation effect, repeated effect across prompts, consistent trained-vs-base delta
- **Strong:** activation patching recovery, corrupt-to-clean destruction, held-out replication, controls ruled out, predictable steering
- **Very strong:** selective knockout, skill injection, circuit reconstruction, replicated across seeds

## Claim Schema

Every claim must include:
- Component (which layer/head/MLP)
- Behaviour (what task family)
- Metric (KL, logprob, exact match)
- Effect size (numerical value)
- Ablation result
- Patching result (if tested)
- Steering result (if tested)
- Training-delta evidence (if tested)
- Controls
- Failure modes
- Confidence level
- Reproducibility command

## Key Gotchas & Pitfalls

1. **PeftModel in-place modification:** `PeftModel.from_pretrained(base_model, adapter_path)` modifies base_model. Use `with model.disable_adapter():` for base behavior. Never load the model twice.
2. **Single-token targets:** Always verify `len(tokenizer(target)["input_ids"]) == 1` before using in logprob scoring. Multi-token targets break clean/corrupt pairs.
3. **Full-residual patching is trivial:** For identical-prefix pairs, patching the full residual at ANY layer gives KL=0. Use position-specific or component-specific patching instead.
4. **Zero ablation is destructive:** Zeroing a layer's output breaks the residual stream chain. All layers will appear "necessary." Consider mean ablation for less destructive results.
5. **GQA architectures:** TransformerLens may not support GQA (Qwen2.5, Llama 3, etc.). Use HF native with manual forward hooks instead.
6. **8GB VRAM limits:** Full SFT OOMs on 8GB for 0.5B models. Use LoRA (r=8, alpha=16). Full SFT needs >=16GB.
7. **Steering budget:** Steering at extreme strengths (s >= +2) causes degeneration. Test fine granularity (s=0.25, 0.5, 0.75, 1.0).
8. **Head effects are small in small models:** In 0.5B models, individual head ablation gives ~200x smaller effects than layer ablation. Don't expect specialist heads like in larger models.
9. **Early exit doesn't work naively:** Projecting intermediate hidden states through lm_head fails because each layer transforms the residual. L22's output is L23's input, not the final representation.
10. **Adapter norm != functional effect location:** In the original ablation, general layer importance peaks at L0-L2. But adapter-SPECIFIC importance (measured by adapter-only ablation) peaks at L19-L23. Don't conflate these.

## Adapting to a New Model

To run this workflow on a different model:

1. Update `config/model.yaml` with the new model name
2. Verify it loads: `python -c "from mi_atlas.model_loader import load_model_hf; load_model_hf('<new_model>')"`
3. Check architecture: ensure n_layers, n_heads, d_model are detected correctly
4. Run tokenizer diagnostics: verify all task suite targets are single-token
5. If GQA: confirm HF native hooks work (TransformerLens likely won't)
6. If larger model (>1B): may need >8GB VRAM, consider quantization
7. Run Phase 2 (component mapping) first — the atlas structure will differ from Qwen2.5-0.5B
8. Update AGENTS.md with the new model details
9. Start a fresh progress.md for the new model

## Efficiency Findings Template

The efficiency experiment typically reveals:

- **Naive layer skipping fails:** 0% top-5 overlap for all skip configs. Every layer is necessary.
- **Early exit fails:** Intermediate hidden states are not directly projectable to vocab.
- **Real efficiency gains are in training, not inference:**
  - Core circuit locks in early (step 10 of 100)
  - o_proj-only LoRA with r=4 is most parameter-efficient
  - Skill knockout via steering avoids retraining
  - Adapter stacking enables multi-skill without retraining

## Publication/Blog Post Angles

1. **"I mapped every component of a 0.5B LLM"** — Lead with the architecture map (L2 router, L22 unembedding, L19 skill knockout). Educational, clean visuals.

2. **"Train for 10 steps, not 100"** — Core circuit locks in by step 10. Challenges "more training = better" assumption.

3. **"344K params to add a skill"** — o_proj-only LoRA, r=4. Practical for edge deployment.

4. **"I can selectively erase knowledge from an LLM with one vector"** — L19 negative steering, 11654x selectivity. Provocative, gets attention.

5. **"Naive layer skipping doesn't work — here's what does"** — Honest negative result + constructive alternatives. Good for HN.

## GitHub Push

If gh auth is configured on the GPU host:
```bash
cd ~/work/<repo>
git add -A && git commit -m "description"
git push origin master
```

If not, use git bundle:
```bash
# On GPU host:
git bundle create /tmp/mi-atlas.bundle master
# On local machine:
scp <gpu_host>:/tmp/mi-atlas.bundle /tmp/mi-atlas.bundle
cd ~/work/<repo>
git pull /tmp/mi-atlas.bundle master
git push origin master
```

## Future Experiments (Not Yet Run)

These require additional GPU time or new data:

1. **Multi-seed replication** — Run top 5 findings with 3 seeds. Needed for HIGH confidence on all claims.
2. **Mean/resample ablation** — Replace zero ablation for stronger causal claims. Less destructive than zero.
3. **CPT training** — Continued pretraining on code corpus. Compare to LoRA-only.
4. **SAE training** — Train sparse autoencoders on key layers (L0, L1, L2, L7, L9, L19, L22). Find interpretable features.
5. **Skill injection at L19** — Can we INJECT factual recall at L19 (the knockout layer) using positive steering?
6. **Natural language prompts** — Extend from synthetic to natural language. Validate transfer.
7. **Cross-model validation** — Run same workflow on Qwen2.5-1.5B or Qwen3.5-0.5B. Compare architecture.
8. **Adapter SVD/PCA** — Find the subspace each skill occupies in adapter weight space. Test orthogonality.
9. **Checkpoint interpolation** — Blend step-10 and step-100 weights. Find optimal interpolation ratio.
10. **Multi-skill steering** — Apply factual + JSON steering vectors simultaneously. Test composition.
11. **Layer fusion** — Train a student model that merges adjacent layers. Distillation from 24L to 18L.
12. **Mixture of Depths (MoD)** — Train the model to conditionally skip layers. Requires architecture modification.
