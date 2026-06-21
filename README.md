# Autonomous Mechanistic Interpretability + Training Perturbation Research

A reproducible causal atlas of a small language model (default: qwen3.5-0.5b).

Systematically discover what controls what inside a 0.5B-class causal LM through:
- Baseline evaluation across 12+ task families
- Layer/head/MLP ablation and activation patching
- Steering vector interventions
- Training perturbation comparisons (CPT, SFT, LoRA, adapters)
- Component atlas construction
- SAE feature analysis

See `docs/RESEARCH_PROMPT.md` for the full research specification.
See `AGENTS.md` for agent operating procedures.
See `progress.md` for current status.

## Setup

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

```bash
python scripts/run_smoke_tests.py
python scripts/build_task_suite.py
python scripts/run_baselines.py
```
