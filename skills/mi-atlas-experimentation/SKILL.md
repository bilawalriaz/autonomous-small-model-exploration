---
name: mi-atlas-experimentation
description: Build a reproducible mechanistic interpretability atlas of any small LLM — component mapping, ablation, patching, steering, LoRA training perturbation, efficiency testing, and publication-ready reports.
version: 2.0.0
---

# MI-Atlas Experimentation Skill

## Purpose

Run a complete mechanistic interpretability investigation on a small language model (<2B params). Produces a causal atlas connecting behaviours to components, training perturbation effects, efficiency analysis, and a publication-ready report with plots.

## When to Use

- You want to understand how a specific small LLM works internally
- You want to find optimization targets (efficient training, skill injection, pruning candidates)
- You want to produce publishable findings with real evidence (not just attention maps)
- A new model has released and you want to map its architecture quickly

## Related Skills

- **`devops/mechanistic-interpretability`** (v1.4.0) — The original detailed MI skill with code patterns, PyTorch hook snippets, VRAM budgeting tables, and 7 reference files covering session-specific findings. This skill (`mi-atlas-experimentation`) focuses on the workflow/process/publication side; the other has the deep code-level detail. Load both when doing MI work — the code patterns from `mechanistic-interpretability` are essential for implementation, while this skill provides the 7-phase workflow and cross-scale comparison data.

## Quick Start: Single Entry Point

```bash
# Run the full 7-phase atlas on any model
python scripts/run_full_atlas.py --model Qwen/Qwen2.5-1.5B --suffix 1.5b
python scripts/run_full_atlas.py --model Qwen/Qwen2.5-0.5B --suffix 0.5b
```

This runs all phases (component mapping → causal interventions → training perturbation → advanced interventions → efficiency) and saves results with the suffix to avoid overwriting.

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
├── prompts/               # Prompt library submodule (bilawalriaz/mi-prompt-library)
│   └── prompts/           # JSON prompt files by category
├── docs/                  # GitHub Pages site (HTML, CSS inline)
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

### Phase 8: Qualitative Analysis (Day 5-6)

The structural atlas tells you WHICH components matter. Qualitative analysis tells you what the model FEELS like — prose quality, creativity, correctness, malleability, and inference speed. Both are needed for a complete picture.

**Prompt Library:** The `prompts/` submodule (from `bilawalriaz/mi-prompt-library`) contains structured prompts across 7 categories: creative, reasoning, code, instruction, factual, style, edge_cases. Each prompt has constraints and a scoring rubric. Add new prompts as you discover useful test cases.

```bash
# Run qualitative analysis on HF model
python scripts/run_qualitative_analysis.py --model Qwen/Qwen2.5-0.5B --suffix 0.5b_hf --backend hf
python scripts/run_qualitative_analysis.py --model Qwen/Qwen2.5-1.5B --suffix 1.5b_hf --backend hf

# Run on GGUF quants (requires llama.cpp)
python scripts/run_qualitative_analysis.py --model ~/gguf_models/0.5b_fp16.gguf --suffix 0.5b_fp16 --backend llama
python scripts/run_qualitative_analysis.py --model ~/gguf_models/0.5b_q8_0.gguf --suffix 0.5b_q8 --backend llama
python scripts/run_qualitative_analysis.py --model ~/gguf_models/0.5b_q4_k_m.gguf --suffix 0.5b_q4 --backend llama
```

**What qualitative analysis captures that MI doesn't:**
- **Prose quality:** coherence, fluency, voice, does it sound human or robotic?
- **Creativity:** divergent thinking, originality, idea generation
- **Correctness:** factual accuracy, logical reasoning, code correctness
- **Malleability:** instruction following, constraint adherence, format compliance, persona adoption
- **Inference speed:** tokens/sec at different quantization levels
- **Failure modes:** empty outputs, garbage/degenerate text, repetition loops, Chinese character degeneration

**Automated scoring vs human review:** The script provides automated heuristic scores (word count, constraint adherence, repetition ratio, empty/garbage detection). These capture structural properties but NOT nuanced quality. Human review of the actual generated text is needed for final qualitative scoring using the rubrics in the prompt library.

**Prompt library schema:** See `prompts/README.md`. Each prompt has: id, category, prompt text, constraints, scoring_rubric (1-5 scales), expected_behavior, notes.

### Phase 9: Quantization Testing

Test how quantization affects both inference speed and output quality. This determines the optimal quant level for practical deployment.

```bash
# Convert models to GGUF and quantize
bash scripts/convert_and_quantize.sh Qwen/Qwen2.5-0.5B 0.5b
bash scripts/convert_and_quantize.sh Qwen/Qwen2.5-1.5B 1.5b

# Then run qualitative analysis on each quant (see Phase 8)
```

**Quantization levels tested:**
- **fp16 (half precision):** Baseline quality, largest file, reference for comparison
- **Q8_0 (8-bit):** Expected sweet spot — minimal quality loss, ~50% size reduction, fast inference
- **Q4_K_M (4-bit K-quants medium):** Maximum compression, may lose quality on complex tasks

**What to measure:**
- File size (MB)
- Inference speed (tokens/sec)
- Output quality: constraint adherence rate, repetition ratio, empty/garbage count
- Qualitative comparison: does the prose degrade? Does reasoning break? Does code still work?

**Default recommendation:** Based on testing, **4bit NF4 (bitsandbytes) is the recommended default quantization for qualitative analysis on 1.5B+ models.** It provides only 9% speed loss vs bf16 on 1.5B (vs 52% for 8bit!) with identical constraint adherence. The user's hypothesis that 8bit would be the sweet spot does NOT hold for bitsandbytes — 8bit has more dequantization overhead than 4bit NF4. Use 8bit only if you need the absolute highest precision for numerical tasks. Use Q4_K_M GGUF with llama.cpp for deployment scenarios.

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
11. **Model size changes everything:** The universal hub moves from L2 (0.5B) to L26 (1.5B) to L34 (3B). Head specialization increases 22x. Skill knockout selectivity drops from 11654x to 0.24x. MLPs become 3x less important. The atlas is NOT transferable across scales — each model needs its own. **UPDATE (Phase 2):** Steering did NOT collapse at 1.5B as originally believed — the Phase 1 conclusion was an artifact of only testing L2. Phase 2 steering migration showed L6 (+3.14), L21 (+4.64), L26 (-5.88), and multi-layer (-7.20) all produce strong steering effects at 1.5B. Steering leverage MIGRATED to different layers, not disappeared. Always test steering at ALL candidate hub layers, not just the 0.5B hub location.
12. **1.5B training on 8GB:** Load in bf16 (~3GB), batch_size=1, gradient_accumulation_steps=2, gradient_checkpointing=True, max_length=256. The vocab projection (logits.float()) needs ~594MB — fp32 model will OOM.
13. **Head ablation hook signature:** `register_forward_pre_hook` takes `(module, args)` not `(module, input, output)`. The hook returns a tuple of modified args.
14. **Trainer needs labels:** HuggingFace Trainer requires `labels` in the dataset for causal LM. Add `tokenized["labels"] = tokenized["input_ids"].copy()` in the tokenize function.
15. **Early exit gives inf KL on larger models:** Projecting intermediate hidden states through lm_head on 1.5B gives KL=inf for some layers (numerical overflow in bf16). Use torch.float32 for the projection step.
16. **GGUF conversion needs sentencepiece:** The `convert_hf_to_gguf.py` script requires the `sentencepiece` package. Install it in the venv: `pip install sentencepiece`. The script also needs a local model directory path, not a HF model name — use `snapshot_download()` first.
17. **4bit NF4 is the sweet spot, not 8bit:** Testing showed 8bit is actually the SLOWEST quantization (bitsandbytes 8bit has more dequantization overhead than 4bit NF4). On 1.5B: bf16=18.8 tok/s, 8bit=9.0 tok/s (52% slower), 4bit=17.1 tok/s (only 9% slower). Use 4bit NF4 as default.
18. **Prompt library is a git submodule:** The `prompts/` directory is a submodule from `bilawalriaz/mi-prompt-library`. Run `git submodule update --init` after cloning. Add new prompts to the prompt library repo directly.
19. **BitsAndBytesConfig for 8bit:** Do NOT pass `bnb_4bit_quant_type` when using 8bit mode — it must be a string and will crash if set to None. Only pass it for 4bit mode (`bnb_4bit_quant_type="nf4"`). Use a kwargs dict and conditionally add the key.
20. **llama.cpp binary paths:** The binaries are at `~/llama.cpp/build/bin/` (llama-cli, llama-quantize, etc.), NOT `~/llama-cpp-build/build/bin/`. The conversion script is at `~/llama.cpp/convert_hf_to_gguf.py`. After `git pull` + rebuild, set `export LD_LIBRARY_PATH=~/llama.cpp/build/bin:$LD_LIBRARY_PATH` or llama-cli will fail with symbol lookup errors.
21. **GGUF conversion needs local model path:** `convert_hf_to_gguf.py` requires a local directory, not a HF model name. Use `snapshot_download("Qwen/Qwen2.5-0.5B")` first, then pass the returned path to the converter.
22. **8bit is slower than 4bit (counterintuitive):** bitsandbytes 8bit uses row-wise quantization with online dequantization per matmul. 4bit NF4 uses block-wise quantization with pre-computed dequant tables. On 1.5B: bf16=18.8 tok/s, 8bit=9.0 tok/s (52% slower), 4bit=17.1 tok/s (only 9% slower). Don't assume higher precision = faster.
23. **GQA head dimension vs head count in patching hooks:** When writing activation patching hooks for GQA models (Qwen2.5, Llama 3, etc.), do NOT confuse `n_heads` (query heads) with `d_head` (head dimension). In Qwen2.5-3B: n_heads=16, d_head=128. Patching code that reshapes using `n_heads` where it should use `d_head` will crash with `size mismatch: tensor a (16) must match tensor b (128)`. The fix: always use `model.config.head_dim` or `d_model // n_heads` for reshaping, and be aware that GQA has fewer KV heads than Q heads. Test patching hooks on a single layer before running full sweeps.
24. **Subagent file conflicts when delegating script writing:** When dispatching parallel subagents to write experiment scripts for the same repo, multiple agents may write to the same filename. The last writer wins and silently overwrites. Mitigation: assign each subagent a unique filename, or write to a staging directory and merge after. Check for sibling-subagent warnings in write_file output.
25. **Offline GPU host workflow:** When the GPU host is unavailable, build all infrastructure locally (configs, task suite, registry, report stubs, scripts). Push to git. When the host returns, `git pull` and run. The scaffold is the high-value work; GPU execution is mechanical.
26. **Phase 2 orchestrator pattern:** Use `run_full_phase2_atlas.py` as the single entry point. It dispatches blocks by priority, handles model override, --force, --dry-run, and logs every run to `experiments/runs/`. Never run individual block scripts manually when the orchestrator exists.
27. **Registry entries before experiments:** Write experiment registry entries BEFORE running experiments. This forces you to specify what you're testing and what would falsify it. The registry is the scientific contract; the results are the evidence.
28. **Orchestrator-script argument contract:** The orchestrator passes `--model <full_model_id>` to every block script. ALL block scripts MUST accept `--model`, `--force`, and `--seed`. If a script's argparse omits `--model`, the orchestrator crashes with `unrecognized arguments`. Always include all three args, even if the script hardcodes the model.
29. **Cross-model adapter loading fails:** LoRA adapters are model-specific — they encode weight deltas sized to the base model's hidden dimension. Loading a 0.5B adapter (d=896) into 1.5B (d=1536) causes `size mismatch`. Always use adapters trained on THAT model.
30. **Metric function availability:** Subagents may import functions from `mi_atlas.metrics` that don't exist yet. Before dispatching script-writing subagents, audit `src/mi_atlas/` for required functions and add them FIRST.
31. **Script output buffering with quantized models:** 3B+ models with 4-bit quantization appear to hang during ablation sweeps (10+ min per sweep with no output). This is normal — dequantization overhead. Use `python -u` for unbuffered output and check `ps aux | grep python` to confirm the process is alive.
32. **LoRA effect testing in robustness scripts:** When testing LoRA effects across prompt lengths, the adapter must match the model. A 0.5B adapter cannot load into 1.5B. Either train model-specific adapters or gracefully skip with an error message.

## Complete New Model Atlas Procedure

Use this procedure to run the full MI-Atlas on any new model (e.g., Qwen3.5-0.8B, Qwen2.5-Coder-3B, Phi-4-mini, etc.). The output is directly comparable to existing Qwen2.5-0.5B/1.5B/3B results.

### Step 0: Check prerequisites

```bash
ssh <gpu_host>
cd ~/work/autonomous-small-model-exploration
source .venv/bin/activate

# Check GPU
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"

# Check packages
python3 -c "import transformers, peft, torch; print(f'transformers={transformers.__version__}, peft={peft.__version__}')"
```

### Step 1: Register the model in configs/models.yaml

Add a new entry:

```yaml
  qwen35_08b:
    id: "Qwen/Qwen3.5-0.8B"       # exact HF model ID
    slug: "qwen35_0.8b"            # short slug for filenames
    n_layers: null                  # filled after first load
    n_heads: null
    d_model: null
    d_head: null
    vocab_size: null
    context_length: null
    activation_function: null
    dtype: "bfloat16"
    vram_bf16_mb: null              # filled after first load
    vram_4bit_mb: null
    notes: "New model to atlas"
```

### Step 2: Verify model loads and detect architecture

```bash
python3 -c "
from src.mi_atlas.model_loader import load_model_hf
bundle = load_model_hf('Qwen/Qwen3.5-0.8B')
print(f'Layers: {bundle.model.config.num_hidden_layers}')
print(f'Heads: {bundle.model.config.num_attention_heads}')
print(f'KV Heads: {bundle.model.config.num_key_value_heads}')
print(f'd_model: {bundle.model.config.hidden_size}')
print(f'd_head: {bundle.model.config.head_dim}')
print(f'Vocab: {bundle.model.config.vocab_size}')
import torch
print(f'VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
"
```

Update configs/models.yaml with actual values. Also check:
- Is it GQA? (num_key_value_heads < num_attention_heads) → HF hooks work, TransformerLens won't
- Does it fit in bf16 on 8GB? If not, use 4bit NF4 for training

### Step 3: Run tokenizer diagnostics

```bash
python scripts/run_smoke_tests.py  # includes tokenizer checks
```

Verify all task targets are single-token. If not, update task suite targets.

### Step 4: Build the canonical task suite (if not already built)

```bash
python scripts/build_phase2_task_suite.py
# Creates data/tasks/canonical_short/, canonical_long/, deobfuscation/
# and data/tasks/task_manifest.json (34 entries, 4300 examples)
```

Skip if `data/tasks/task_manifest.json` already exists.

### Step 5: Run Phase 1 atlas (component mapping + training perturbation)

This is the core atlas — it identifies where behaviours live.

```bash
# Full Phase 1 atlas (runs all 7 phases)
python scripts/run_full_atlas.py --model Qwen/Qwen3.5-0.8B --suffix qwen35_08b
```

Or run phases individually:

```bash
# Component mapping
python scripts/run_layer_ablation.py --model Qwen/Qwen3.5-0.8B
python scripts/run_head_ablation.py --model Qwen/Qwen3.5-0.8B
python scripts/run_mlp_ablation.py --model Qwen/Qwen3.5-0.8B
python scripts/run_position_ablation.py --model Qwen/Qwen3.5-0.8B

# Causal interventions
python scripts/run_steering_sweep.py --model Qwen/Qwen3.5-0.8B
python scripts/run_patching_v1.py --model Qwen/Qwen3.5-0.8B

# Training perturbation
python scripts/train_lora_json.py --model Qwen/Qwen3.5-0.8B
python scripts/compare_lora_ablation.py --model Qwen/Qwen3.5-0.8B
python scripts/run_lora_rank_sweep.py --model Qwen/Qwen3.5-0.8B
python scripts/run_lora_module_sweep.py --model Qwen/Qwen3.5-0.8B
python scripts/run_dataset_shard_ablation.py --model Qwen/Qwen3.5-0.8B

# Advanced
python scripts/run_cross_model_patching.py --model Qwen/Qwen3.5-0.8B
python scripts/run_skill_knockout.py --model Qwen/Qwen3.5-0.8B
python scripts/run_adapter_ablation.py --model Qwen/Qwen3.5-0.8B

# Efficiency
python scripts/run_layer_skipping_early_exit.py --model Qwen/Qwen3.5-0.8B
```

### Step 6: Run Phase 2 blocks

```bash
# Run all Phase 2 blocks for the new model
python scripts/run_full_phase2_atlas.py --model Qwen/Qwen3.5-0.8B --blocks all

# Or specific blocks
python scripts/run_full_phase2_atlas.py --model Qwen/Qwen3.5-0.8B --blocks B,C,D
```

### Step 7: Generate claim cards

```bash
# For each experiment
python scripts/generate_claim_card.py \
  --experiment P2-STEER-001 \
  --result experiments/results/steering_migration_qwen35_08b.json \
  --verdict confirmed
```

### Step 8: Add to cross-scale comparison table

Update the cross-scale table in this skill file AND in `docs/05-phase2-repeatability.html`:

| Metric | 0.5B | 1.5B | 3B | NEW_MODEL |
|--------|------|------|-----|-----------|
| Universal hub | L2 (8%) | L26 (93%) | L34 (94%) | L?? (??%) |
| Hub total KL | 19.11 | 13.70 | 221.34 | ?.?? |
| ... | ... | ... | ... | ... |

### Step 9: Write reports

```bash
# Generate report for the new model
# Write reports/phase2/NN_<model>_atlas.md following REPORT_TEMPLATES.md
# Update reports/phase2/10_final_phase2_findings.md with new data point
```

### Step 10: Update GitHub Pages

Add a new HTML page in `docs/` following the existing pattern:
- `docs/06-<model>-analysis.html` — full atlas for the new model
- Update `docs/index.html` navigation and card grid
- Update `docs/05-phase2-repeatability.html` cross-scale table

### Step 11: Commit and push

```bash
cd ~/work/autonomous-small-model-exploration
git add -A && git commit -m "Atlas: <model_name> — hub at L?, key findings..."
git push origin master
```

### Quick Atlas (reduced — for fast comparison)

If you just want a quick comparison point (2-3 hours instead of 20+), run only the critical experiments:

```bash
python scripts/run_full_phase2_atlas.py --model <model_id> --blocks B,C,D
```

This gives you: hub identification (ablation), steering effectiveness, and the cross-scale comparison data points. Skip adapter surgery, deobfuscation, and separability for a quick pass.

### VRAM Budget Table

| Model Size | bf16 VRAM | 4bit NF4 VRAM | Training Config | Notes |
|------------|-----------|---------------|-----------------|-------|
| 0.3-0.5B | ~1GB | N/A | LoRA r=8, bs=2, 100 steps | Fits easily |
| 0.8-1.5B | ~3GB | N/A | LoRA r=8, bs=1, grad_ckpt, 100 steps | Needs grad checkpointing |
| 2-3B | ~6GB | ~2GB | LoRA r=8, bs=1, grad_ckpt, 100 steps | bf16 tight; 4bit recommended |
| 7B | OOM | ~4GB | LoRA r=4, bs=1, grad_ckpt, 50 steps | 4bit only on 8GB |
| 14B+ | OOM | ~7-8GB | LoRA r=2, bs=1, grad_ckpt | Barely fits on 8GB |

### Cross-Family Checklist

When testing a completely different architecture (not Qwen):

1. Verify `model.config` has `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `head_dim`
2. Check if GQA: `num_key_value_heads < num_attention_heads`
3. Run layer ablation FIRST — the hub will be at a completely different position
4. Don't assume Qwen's L2/L26/L34 pattern applies
5. The model may have NO clear hub (distributed processing) — report this honestly
6. Test steering at the top 3-5 ablation layers, not at Qwen's hub positions
7. Cross-family comparison = "does each model have a hub, and where?" NOT "does this model match Qwen"

## Efficiency Findings Template

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

6. **"What changes inside an LLM when you triple its size?"** — Cross-scale comparison. The universal hub migrates from L2 to L26 to L34. MLPs lose importance. Heads gain 22x specialization. Steering MIGRATES (not collapses) — L21 at 1.5B is 2x stronger than L2 at 0.5B. Skill knockout goes from 11654x selective to non-functional. SmolLM2 hub at L0 shows architecture matters more than size. Practical punchline: each model scale needs its own atlas, and steering is about finding the right layer, not giving up.

## GitHub Push

See `references/github-pages-publishing.md` for instructions on publishing results as a GitHub Pages site.

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
7. **Cross-model validation** — Run same workflow on Qwen2.5-1.5B or Qwen3.5-0.5B. Compare architecture. ✅ DONE (1.5B)
8. **Adapter SVD/PCA** — Find the subspace each skill occupies in adapter weight space. Test orthogonality.
9. **Checkpoint interpolation** — Blend step-10 and step-100 weights. Find optimal interpolation ratio.
10. **Multi-skill steering** — Apply factual + JSON steering vectors simultaneously. Test composition.
11. **Layer fusion** — Train a student model that merges adjacent layers. Distillation from 24L to 18L.
12. **Mixture of Depths (MoD)** — Train the model to conditionally skip layers. Requires architecture modification.

## Phase 2: Repeatable Small-Model Surgery Protocol

Phase 2 turns Phase 1's findings into a repeatable scientific protocol. Every experiment has a registry entry, claim card, and multi-seed replication.

### Phase 2 Entry Points

```bash
# Run ALL Phase 2 blocks in priority order
python scripts/run_full_phase2_atlas.py --model Qwen/Qwen2.5-0.5B --blocks all

# Run specific blocks
python scripts/run_full_phase2_atlas.py --blocks A,B,C --model Qwen/Qwen2.5-1.5B

# List available blocks
python scripts/run_full_phase2_atlas.py --list-blocks

# Dry run (show what would execute)
python scripts/run_full_phase2_atlas.py --blocks all --dry-run
```

### Phase 2 Block Map

See `references/phase2-block-map.md` for full block map, VRAM budgets, dependency graph, hypothesis→block mapping, metrics per block, and quality gates.

| Block | Script | Description | Priority |
|-------|--------|-------------|----------|
| A | run_phase2_parity.py | Fill missing 1.5B experiments | 1 |
| B | run_phase2_steering_migration.py | Steering at hub layers + strength sweeps | 2 |
| C | run_phase2_ablation_controls.py | Zero/mean/resample/patch comparison | 3 |
| F | run_phase2_adapter_surgery.py | Adapter surgery + compatibility matrix | 4 |
| H | run_phase2_deobfuscation.py | Deobfuscation subskill atlas | 5 |
| G | run_phase2_skill_separability.py | Skill separability benchmark | 6 |
| D | run_phase2_third_scale.py | 3B reduced atlas | 7 |
| E | run_phase2_cross_family.py | Gemma/SmolLM cross-family | 8 |
| I | run_phase2_long_task_robustness.py | Short/medium/long prompt robustness | 9 |

### Phase 2 Hypotheses

H1: Hub migration is real. Dominant hub moves from early layers in 0.5B to later/distributed in 1.5B+.
H2: MLP/attention responsibility changes with scale. 0.5B → MLP dominance, 1.5B+ → attention head specialization.
H3: Steering did not collapse at 1.5B; leverage moved from L2 to newly identified hubs.
H4: Skill knockout selectivity decreases with scale (skills become entangled).
H5: Fine-tuned transfer concentrates in final ~10% of layers (universal invariant).
H6: LoRA weight norms insufficient; functional effect must be measured causally.
H7: Naive layer skipping invalid; inference compression needs structured pruning + recovery.
H8: Some adapters compose cleanly, others destructively interfere; compatibility predictable from localization.

### Phase 2 Configuration

All configs in `configs/` directory:
- `models.yaml` — model identifiers, architecture metadata, VRAM budgets
- `tasks.yaml` — 16 task families with scorers and length splits
- `experiment_defaults.yaml` — seeds, ablation types, steering strengths, training config, metrics, weights

### Phase 2 Registry

`experiments/registry.jsonl` — one JSON line per experiment. Required fields:
```json
{"id": "P2-STEER-001", "title": "...", "hypothesis": "...", "models": [...], "tasks": [...],
 "independent_variables": [...], "dependent_metrics": [...], "controls": [...],
 "seeds": [1,2,3], "expected_artifacts": [...], "status": "planned|running|success|failed"}
```

### Claim Cards

Every completed experiment must produce a claim card at `reports/claims/<experiment_id>.md`.
Use: `python scripts/generate_claim_card.py --experiment P2-STEER-001 --result experiments/results/steering_migration_0.5b.json --verdict confirmed`

Claim card sections: Claim, Result (metrics table), Controls, Seeds (per-seed table), Artifacts, Environment, Interpretation, Limitations, Verdict.

### Run ID Format

`P2_{experiment_id}_{model_slug}_{task_slug}_{YYYYMMDD_HHMMSS}_seed{seed}`

All scripts must be resumable — skip completed run IDs unless `--force` is passed.

### Procedure: Run standard Phase 2 atlas on a new model

#### Purpose
Generate a comparable causal/control-surface profile for a small LM.

#### Inputs
- model id (e.g., Qwen/Qwen2.5-3B)
- model revision
- tokenizer revision
- task suite (configs/tasks.yaml)
- output directory
- seeds (default: [1,2,3])

#### Command
```bash
python scripts/run_full_phase2_atlas.py --model <model_id> --blocks all
```

#### Outputs
- Raw JSONL in experiments/results/
- Summary CSV in results/summaries/
- Plots in results/plots/
- Claim cards in reports/claims/
- Updated registry entries in experiments/registry.jsonl

#### Quality gates
- All controls completed
- No missing seed results (3 seeds unless marked pilot)
- Raw artifacts saved
- Claim card generated
- Limitations documented

### Procedure: Add a new task family

1. Add entry to `configs/tasks.yaml` with name, family, scorer, splits
2. Add examples to `data/tasks/canonical_short/` and `data/tasks/canonical_long/`
3. Implement scorer in `src/mi_atlas/task_suite.py` or as standalone function
4. Update `data/tasks/task_manifest.json`
5. Run on both models: `python scripts/run_phase2_parity.py --tasks <new_family>`

### Procedure: Run steering sweeps

```bash
# Full steering migration sweep
python scripts/run_phase2_steering_migration.py --model Qwen/Qwen2.5-1.5B

# Tests: target/random/wrong-task/anti vectors
# Strengths: [-4, -2, -1, -0.5, 0.5, 1, 2, 4]
# Layers: L2, L6, L14, L21, L25, L26, L27 (1.5B)
# Metrics: target_logit_delta, KL, task_accuracy, format_validity, collateral_damage
```

### Procedure: Run ablation controls

```bash
python scripts/run_phase2_ablation_controls.py --model Qwen/Qwen2.5-0.5B

# Compares: zero, mean, resample_gaussian, clean→corrupt, corrupt→clean, random_patch
# Per layer, per task family
# Key output: rank-order stability across methods
```

### Procedure: Run adapter surgery

```bash
python scripts/run_phase2_adapter_surgery.py --model Qwen/Qwen2.5-0.5B

# For each adapter:
#   layer-wise norm, module-wise norm, causal ablation, rank truncation
# Compatibility matrix: all-pairs merge + test
# Output: adapter_compatibility_matrix.csv
```

### Procedure: Run skill separability scoring

```bash
python scripts/run_phase2_skill_separability.py --model Qwen/Qwen2.5-0.5B

# 5 operations per skill: Insert, Remove, Move, Compose, Localize
# SSS = 0.20*insertion + 0.20*removal + 0.15*transfer + 0.15*composition + 0.15*localization - 0.15*collateral
# Output: skill_separability_scores.csv
```

### Procedure: Run deobfuscation surgery

```bash
python scripts/run_phase2_deobfuscation.py --model Qwen/Qwen2.5-0.5B

# Subskills: variable_renaming, dead_code, string_decoding, constant_folding, control_flow, semantic_preservation
# Tests: localization overlap, interference, transfer, joint vs composed
# Eval: exact match, AST equivalence, hallucination rate
```

### Procedure: Generate reports

After experiments complete:
1. Generate claim cards for each experiment
2. Write report per block in reports/phase2/
3. Write reports/phase2/10_final_phase2_findings.md
4. Update reports/negative_results.md
5. Update reports/open_hypotheses.md
6. Update reports/replication_status.md
7. Push to GitHub

### Distinguishing real findings from artifacts

1. Check seed variance — if std > 50% of effect, finding is fragile
2. Check ablation method — if finding disappears with mean ablation, it was an artifact of zeroing
3. Check controls — if random vector has same effect as task vector, finding is not task-specific
4. Check prompt length — if finding only works on short prompts, it may be a toy-task artifact
5. Check model family — if finding doesn't replicate on a different architecture, it may be Qwen-specific

---

## Cross-Scale Comparison: 0.5B vs 1.5B vs 3B

Running the full atlas across Qwen2.5-0.5B, 1.5B, and 3B revealed critical scaling insights:

| Metric | 0.5B | 1.5B | 3B | Trend |
|--------|------|------|-----|-------|
| Architecture | 24L, 14H, d=896 | 28L, 12H, d=1536 | 36L, 16H, d=2048 | More layers + wider |
| Universal hub | L2 (8% depth) | L26 (93% depth) | L34 (94% depth) | Hub migrates to final layers |
| Hub total KL | 19.11 | 13.70 | 221.34 | Effect grows with scale |
| MLP max effect | L0 (KL 8.12) | L0 (KL 2.58) | ? | 3x weaker MLPs at scale |
| Head max effect | 0.046 | 1.023 | ? | 22x stronger head specialization |
| Steering (best single-layer) | L2 +2.13 (3.3x) | L21 +4.64 | ? | MIGRATED, not collapsed |
| Steering (multi-layer) | N/A | -7.20 | ? | Multi-layer steering stronger |
| Skill knockout | L19, 11654x | L21, 0.24x | ? | Selective suppression fails at scale |
| Norm-effect corr | 0.85 | 0.54 | ? | Weaker norm-effect relationship |
| Cross-model best | L23 (100%) | L27 (99.9%) | ? | Same pattern, shifted layers |
| Layer skipping | 0% top-5 overlap | 0% top-5 overlap | ? | All layers necessary in both |

Key insight: **the atlas is NOT transferable across scales.** Each model size has a fundamentally different internal architecture. What works for 0.5B (L2 steering, L19 knockout, o_proj injection) does NOT work for 1.5B. New models need their own atlas.

Phase 2 correction: **Steering did NOT collapse at 1.5B.** The Phase 1 conclusion that "steering sensitivity drops 70x" was an artifact of only testing L2 (the 0.5B hub) at 1.5B. Phase 2 steering migration tested all candidate hub layers and found strong effects at L6 (+3.14), L21 (+4.64), L26 (-5.88), and multi-layer distributed steering (-7.20). The steering leverage migrated to the new hub layers — it didn't disappear.

Cross-family finding: **SmolLM2-1.7B has hub at L0** (all 4 task families, hub_consistent=True). This is completely different from Qwen's pattern (L2→L26→L34). Hub position is architecture-specific, not just depth-dependent. SmolLM2's steering effects were small but consistent (0.01-0.15 boost across tasks).

What IS consistent across scales:
- L0 MLP is always the top MLP (early processing matters universally)
- Cross-model patching always shows monotonic recovery (late layers encode trained behavior)
- Layer skipping always fails (all layers necessary)
- Core circuit locks in early in training (step 10 pattern)
- Hub position is architecture-specific, not just depth-dependent (SmolLM2 hub at L0 vs Qwen L2/L26/L34)

See `references/phase2-execution-lessons.md` for real bugs encountered during Phase 2 orchestrator runs and their fixes.
