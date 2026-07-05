# LFM2.5-8B-A1B MI-Atlas: GGUF-First Approach

## Target Model

**LFM2.5-8B-A1B** — Liquid AI's on-device Mixture-of-Experts
- Architecture: `Lfm2MoeForCausalLM` (hybrid conv + attention + sparse MoE)
- 8.3B total params, **1.5B active per token**
- 24 layers: 18 conv + 6 full attention
- 32 experts per layer, 4 activated per token (12.5% sparsity)
- MoE intermediate size: 1792 per expert
- Dense FFN intermediate size: 7168
- Hidden size: 2048, 32 attn heads, 8 KV heads (GQA 4:1)
- 128K context, vocab 128K, bf16 native, Q4_K_M available (4.9GB)

## Why GGUF-First Works

The 230M atlas used ~25 experiments with activation hooking on bf16. That's overkill for what we actually need. The actionable insights come from:

1. **What the model does** (behavioral probing) — works with any inference
2. **Which experts fire when** (routing analysis) — needs minor llama.cpp modification
3. **How it compares to 230M dense** (cross-architecture) — needs inference only
4. **Where to finetune** (capability mapping + sensitivity) — needs inference + targeted probing
5. **Efficiency** (memory/speed) — needs benchmarking tools

GGUF quantization introduces ~1-3% error on weights. For behavioral analysis and routing patterns, this is negligible. For precise weight-level CKA similarity, we skip it — the 230M atlas already covers that architecture's weight dynamics.

## Hardware

- **Deck** (billz@deck.tail9cc5b.ts.net): 16GB LPDDR5, Van Gogh APU (4 RDNA2 CUs), Vulkan llama.cpp. GGUF fits 8GB GTT (4.9GB). Primary inference target.
- **Aero** (aero.tail9cc5b.ts.net): RTX 2070 Super 8GB. If we need bf16 for specific weight-level analyses later, this is the fallback.

## Phase 1: Architecture Walkthrough (Code-Level, Nano-vLLM Style)

**Goal**: "Inside LFM2.5-8B-A1B" — a code walkthrough explaining exactly how a token flows through the hybrid MoE architecture.

**Deliverable**: `docs/10-inside-lfm25-8b-a1b.html`

### Content

1. **Config Anatomy**
   - Parse config.json, explain every field
   - layer_types array: map the 24-layer pattern (conv×2, attn, conv×3, attn, ...)
   - MoE config: num_experts=32, num_experts_per_tok=4, moe_intermediate_size=1792
   - GQA ratio: 32 heads / 8 KV heads = 4:1 sharing

2. **Weight Tensor Map**
   - Extract all tensor names from GGUF: `gguf_inspect.py`
   - Group by component: embeddings, conv blocks, attention blocks, MoE FFNs, dense FFNs, routing weights
   - Show shapes and sizes per component
   - Calculate memory breakdown: how much VRAM goes to each component

3. **Data Flow: Token → Output**
   - Step-by-step: input_ids → embedding → Layer 0 (conv) → Layer 1 (conv) → Layer 2 (full_attention) → ... → Layer 23 → lm_head
   - At each layer type, explain:
     - Conv block: depthwise k=3 conv → gated activation → up/down projections
     - Attention block: QKV projection → GQA attention → output projection
     - MoE FFN: router logits → top-k expert selection → expert FFN computation → weighted combine
     - Residual connections at each step
   - Shape transformations: trace hidden_state shape through each operation

4. **MoE Routing Mechanics**
   - Router: linear(hidden_size → num_experts) = linear(2048 → 32)
   - Top-k selection: softmax → select top 4 → renormalize
   - norm_topk_prob: whether to normalize before or after top-k
   - routed_scaling_factor: 1.0 (no scaling)
   - Expert FFN: each expert is (gate_proj → silu → up_proj) = MoE gate: 2048→1792→2048
   - Weighted sum of 4 expert outputs

5. **Conv vs Attention: Where MoE Sits**
   - The layer_types pattern: conv layers use dense FFN, attention layers use MoE FFN
   - Wait — need to verify: does the MoE apply to ALL layers or only certain ones?
   - Check num_dense_layers=2 — what are these?
   - This is a key architectural question to answer in Phase 1

### Methods
- `gguf` Python library to inspect tensor metadata
- Read llama.cpp source for LFM2 MoE forward pass
- Config.json analysis
- Write-up with ASCII diagrams of data flow

---

## Phase 2: Routing Analysis

**Goal**: Map which experts fire for which inputs. Understand routing patterns.

### Approach: Modified llama.cpp

Add debug logging to the MoE routing path in llama.cpp:

```c
// In the MoE forward pass (llama-graph.cpp or ggml-backend)
// After computing router logits and selecting top-k experts:
fprintf(stderr, "LAYER %d ROUTING: token=%d experts=[%d,%d,%d,%d] weights=[%.4f,%.4f,%.4f,%.4f]\n",
        layer_idx, token_id,
        expert_idx[0], expert_idx[1], expert_idx[2], expert_idx[3],
        expert_w[0], expert_w[1], expert_w[2], expert_w[3]);
```

### Experiments

| # | Task | Prompt | What to observe |
|---|------|--------|-----------------|
| 2A | Single token routing | "The" | Baseline: what experts fire for a common token |
| 2B | Code token | `def factorial(n):` | Do code experts differ from English? |
| 2C | Math token | "What is 127 × 384?" | Math-specific expert activation |
| 2D | Multilingual | "Bonjour, comment allez-vous?" | Language routing |
| 2E | Reasoning | "Think step by step: if x² = 144, what is x?" | Reasoning chain routing |
| 2F | Domain shift | Medical, legal, creative writing prompts | Domain specialization |

### Analysis
- Expert utilization heatmap: layer × expert_id frequency across all tokens
- Per-task routing signatures: which expert combinations are activated
- Do conv layers route differently from attention layers?
- Expert co-occurrence: do certain expert pairs always fire together?
- Entropy of routing weights: is the router confident or uncertain?

### Deliverable
- `docs/11-lfm25-8b-routing-analysis.html`

---

## Phase 3: Behavioral Probing & Capability Mapping

**Goal**: Where does this model excel, where does it fail, and where is it sensitive to perturbation.

### Experiments

| # | Probe | Method | Insight |
|---|-------|--------|---------|
| 3A | Factual knowledge | 500 QA pairs across domains | What does the model know? |
| 3B | Reasoning chains | Math, logic, multi-step | Think-tag quality |
| 3C | Code generation | Python, JS, SQL tasks | Code capability map |
| 3D | Instruction following | IFEval-style constraints | Obedience to constraints |
| 3E | Multilingual | 10 languages | Language-specific degradation |
| 3F | Context window | 1K → 4K → 16K → 64K → 128K | At what length does quality drop? |
| 3G | Noise sensitivity | Add typos, swap words, truncate | Robustness to input corruption |
| 3H | Few-shot vs zero-shot | 0, 1, 3, 5 shots | In-context learning curve |

### Comparison with 230M dense
- Side-by-side: same prompts on 230M bf16 vs 8B Q4_K_M
- Where does 8B MoE beat 230M dense? By how much?
- Where does 230M hold up surprisingly well?

### Deliverable
- `docs/12-lfm25-8b-capability-map.html`

---

## Phase 4: Fine-tuning Sensitivity Analysis

**Goal**: Identify which parts of the model are most sensitive to perturbation — the roadmap for finetuning.

### Experiments

| # | Perturbation | Method | What it tells us |
|---|-------------|--------|------------------|
| 4A | Router weight noise | Add σ noise to router logits | How brittle is routing? |
| 4B | Expert dropout | Mask 1-2 experts, measure degradation | Which experts are critical? |
| 4C | Conv layer noise | Internal activation noise (σ=0.1-1.0) | Conv robustness at 8B scale |
| 4D | Attention layer noise | Same as 4C on attention layers | Compare conv vs attention robustness |
| 4E | Residual scaling | Scale residuals by 0.5-2.0 | Layer contribution balance |
| 4F | Head masking | Zero out individual attention heads | Which heads matter? |
| 4G | Embedding perturbation | Noise on input embeddings | Input sensitivity map |

### Key questions to answer
- Is the router more fragile than the experts themselves?
- Which layers are the "hub" layers at 8B scale? (230M: L0-L13 locked at CKA=1.0)
- Do conv layers still dominate at 8B, or does MoE shift the balance?
- What's the minimum set of experts needed for decent performance?

### Deliverable
- `docs/13-lfm25-8b-sensitivity.html`

---

## Phase 5: Efficiency Profiling

**Goal**: Quantify exactly what you get per watt, per byte, per millisecond.

### Experiments

| # | Benchmark | Tool | Metrics |
|---|-----------|------|---------|
| 5A | Decode throughput | llama-bench | tok/s at various prompt lengths |
| 5B | Memory profile | nvidia-smi / Vulkan memory | Peak VRAM, per-layer breakdown |
| 5C | Context scaling | llama-bench with -c 1K→128K | Throughput vs context length |
| 5D | Quantization comparison | Q4_K_M vs Q5_K_M vs Q8_0 vs bf16 | Quality vs speed vs memory tradeoff |
| 5E | MoE offloading | -ncmoe flag | CPU vs GPU expert routing tradeoff |
| 5F | Batch size scaling | -b 128→8192 | Throughput vs batch size |

### Comparison points
- LFM2.5-8B-A1B Q4_K_M vs Qwen3.5-4B Q4_K_M (same quant, different arch)
- LFM2.5-8B-A1B Q4_K_M vs LFM2.5-230M bf16 (same arch, different scale)
- LFM2.5-8B-A1B vs Qwen3-1.7B (similar active params)

### Deliverable
- `docs/14-lfm25-8b-efficiency.html`

---

## Phase 6: Cross-Architecture Comparison (230M Dense vs 8B MoE)

**Goal**: What changes when you scale from 230M dense to 8B MoE on the same backbone.

### Questions
- Do the 230M findings transfer? (hub at L0, conv MLP > attn MLP, internal noise destructive on conv)
- Does the MoE router create new failure modes the 230M didn't have?
- Is the 230M's "loss ≠ quality inverse correlation" still true at 8B?
- What does the 230M atlas tell us about where to finetune the 8B?

### Deliverable
- `docs/15-lfm25-230m-vs-8b-comparison.html`

---

## Execution Plan

### Prerequisites
1. Build llama.cpp with MoE debug logging on deck
2. Write `gguf_inspect.py` for tensor metadata extraction
3. Set up experiment scripts on deck

### Phase order
- **Phase 1** (architecture walkthrough): ~2 days, no GPU needed — pure analysis
- **Phase 2** (routing): ~3 days, needs modified llama.cpp build on deck
- **Phase 3** (capability): ~3 days, can run in parallel with Phase 2
- **Phase 4** (sensitivity): ~3 days, needs Phase 2 routing insights to guide
- **Phase 5** (efficiency): ~1 day, can run alongside anything
- **Phase 6** (comparison): ~2 days, needs Phases 1-5 complete

**Total estimated: ~2 weeks of focused work**

### Scripts and tools to build
- `experiments/lfm25_8b/gguf_inspect.py` — extract tensor metadata from GGUF
- `experiments/lfm25_8b/routing_logger.py` — parse llama.cpp routing debug output
- `experiments/lfm25_8b/behavioral_probe.py` — batch prompt testing via llama-cli
- `experiments/lfm25_8b/perturbation.py` — routing noise / expert dropout experiments
- `experiments/lfm25_8b/bench_suite.py` — automated efficiency benchmarks

### Reporting
- All reports as interactive HTML (matching 230M atlas style)
- Published to GitHub Pages: bilawalriaz.github.io/autonomous-small-model-exploration/
- Push and deploy without asking
