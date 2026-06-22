# Phase 2 Execution Lessons

Real bugs encountered during Phase 2 orchestrator runs on aero (RTX 2070 Super 8GB).

## Bug 1: Missing `labels` in tokenize_fn (3 scripts affected)

**Symptom:** `ValueError: The model did not return a loss from the inputs, only the following keys: logits.`

**Root cause:** HuggingFace Trainer requires `labels` in the dataset for causal LM loss computation. Scripts written by subagents omitted this.

**Affected scripts:** `run_phase2_adapter_surgery.py`, `run_phase2_deobfuscation.py`, `run_phase2_skill_separability.py` (training path)

**Fix pattern:**
```python
def tokenize_fn(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # REQUIRED
    return tokenized
```

**Prevention:** This is already gotcha #14 in the skill. The issue is that subagents writing scripts don't read the skill. When dispatching script-writing subagents, explicitly mention "add labels to tokenize_fn" in the context.

## Bug 2: Orchestrator passes --model but scripts don't accept it

**Symptom:** `error: unrecognized arguments: --model Qwen/Qwen2.5-3B`

**Root cause:** `run_full_phase2_atlas.py` always passes `--model <full_model_id>` to block scripts. If a script's argparse doesn't include `--model`, it crashes.

**Affected script:** `run_phase2_third_scale.py` (hardcoded MODEL_NAME, no argparse --model)

**Fix:** Add `--model` to argparse, then override the module-level constant:
```python
parser.add_argument("--model", type=str, default=None)
# ... after args = parser.parse_args():
global MODEL_NAME
if args.model:
    MODEL_NAME = args.model
```

**Prevention:** Template for all block scripts:
```python
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--force", action="store_true")
parser.add_argument("--seed", type=int, default=None)
```

## Bug 3: Model slug vs full model ID mismatch

**Symptom:** `Unknown model: Qwen/Qwen2.5-0.5B` / `Available: ['qwen05b', 'qwen15b']`

**Root cause:** Some scripts use internal slug mapping (e.g., `qwen05b`) while the orchestrator passes the full HF model ID.

**Fix:** Accept both formats in model matching:
```python
models_to_run = [m for m in MODELS if m["slug"] == args.model or m["name"] == args.model]
```

## Bug 4: Cross-model adapter dimension mismatch

**Symptom:** `size mismatch for base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight: copying a param with shape torch.Size([8, 896]) from checkpoint, the shape in current model is torch.Size([8, 1536])`

**Root cause:** LoRA adapters encode weight deltas sized to the base model's hidden_dim. A 0.5B adapter (d=896) cannot load into a 1.5B model (d=1536) or 3B model (d=2048).

**Fix:** When a robustness/experiment script needs to test LoRA effects on a model:
1. Check if a compatible adapter exists (same model architecture + hidden_dim)
2. If not, train a new adapter for that model before testing
3. Wrap adapter loading in try/except and gracefully skip if incompatible

## Bug 5: Missing metric functions in mi_atlas.metrics

**Symptom:** `ImportError: cannot import name 'kl_divergence' from 'mi_atlas.metrics'`

**Root cause:** Subagent-written scripts assumed `kl_divergence` existed in the metrics module.

**Fix:** Add the function to `src/mi_atlas/metrics.py`:
```python
def kl_divergence(base_logits, test_logits, dim=-1):
    base_probs = torch.softmax(base_logits.float(), dim=dim)
    test_log_probs = torch.log_softmax(test_logits.float(), dim=dim)
    base_log_probs = torch.log(base_probs + 1e-10)
    kl = (base_probs * (base_log_probs - test_log_probs)).sum(dim=dim)
    return kl.mean().item()
```

**Prevention:** Before dispatching script-writing subagents, audit `src/mi_atlas/` for required functions and add them proactively.

## Bug 6: 3B quantized model appears to hang

**Symptom:** Log shows model loading, then no output for 10+ minutes. Process is alive at 99% CPU.

**Root cause:** 36-layer ablation sweep on 4-bit quantized model is slow (dequantization overhead per forward pass). Output is buffered.

**Fix:** This is normal behavior. Use `python -u` for unbuffered output. Check `ps aux | grep python` to confirm the process is alive.

## Bug 7: 3B GQA head dimension mismatch in patching/skip/knockout hooks

**Symptom:** `The size of tensor a (16) must match the size of tensor b (128) at non-singleton dimension 3`

**Root cause:** Qwen2.5-3B uses GQA with n_heads=16 and d_head=128. Patching hooks that reshape activations using `n_heads` (16) where they should use `d_head` (128) cause dimension mismatch. The error appeared in patching, layer_skip, and knockout experiments but NOT in ablation (which doesn't reshape).

**Affected experiments:** final_layer_patch, layer_skip, adapter_knockout on 3B

**Fix:** In any hook that reshapes hidden states for head-level manipulation, use:
```python
d_head = model.config.head_dim  # or d_model // n_heads
# NOT: d_head = model.config.num_attention_heads
```

**Prevention:** Always test patching hooks on a single layer before running full sweeps. Add a smoke test: `python -c "run_single_layer_patch(model, layer_idx=0)"`.

## Bug 8: GitHub Pages build failures after adding large HTML files

**Symptom:** `gh api repos/.../pages/builds/latest` returns `status: errored` with `Page build failed`.

**Root cause:** Adding HTML files >40KB (like `01-qwen05b-analysis.html` at 48KB) can cause GitHub Pages Jekyll builds to fail. The error message is generic — no specific file or line number.

**Fix:** Trigger a rebuild with `gh api -X POST repos/.../pages/builds`. If it persists, check for:
- HTML files with unclosed tags
- Very large inline CSS/JS
- Files >100KB

**Prevention:** Keep individual HTML files under 50KB. Split large analysis pages into multiple files if needed.

## Execution Pattern: Parallel subagent delegation for scripts

When dispatching multiple subagents to write experiment scripts:
1. Assign UNIQUE filenames to each subagent (no overlap)
2. Include "add `labels` to tokenize_fn" in context for any script that trains
3. Include "accept --model, --force, --seed args" in context for all block scripts
4. Include "accept both slug and full model ID" in context
5. Check for sibling-subagent warnings in write_file output after delegation

## Results summary from first Phase 2 run

| Block | Status | Duration | Notes |
|-------|--------|----------|-------|
| A (parity) | PASS | 3s | Skipped (already ran during smoke test) |
| B (steering 0.5B) | PASS | 100s | L12 best single-layer (+2.16) |
| B (steering 1.5B) | PASS | 201s | L21 best (+4.64), multi-layer (-7.20) |
| C (ablation 0.5B) | PASS | 66s | 24 layers × 6 types |
| C (ablation 1.5B) | PASS | 160s | 28 layers × 6 types |
| F (adapter surgery) | PASS | 131s | 5 adapters, compatibility matrix |
| H (deobfuscation) | PASS | 18s | Partial (training failed on labels bug) |
| G (separability) | PASS | 35s | 5 skills scored, code_semantics best SSS=0.36 |
| D (3B atlas) | FAIL→PASS | ~30min | Fixed --model arg, hub at L34/36 |
| E (cross-family) | PASS | 78s | SmolLM2 hub at L0, Gemma skipped |
| I (robustness) | FAIL→PASS | 227s | Fixed slug matching, LoRA adapter mismatch on 1.5B |
