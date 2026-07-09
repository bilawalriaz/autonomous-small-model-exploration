# MiniCPM5-1B project control center

This directory is the navigation layer for the active MiniCPM project. It does
not duplicate experiment artifacts; canonical artifacts stay in the standard
repository directories.

## Start here

- [Current status](STATUS.md)
- [Single-shot master plan](../../reports/minicpm5_single_shot_master_plan.md)
- [Overnight autonomy prompt](OVERNIGHT_AUTONOMY_PROMPT.md)
- [Capability/merge history](../../reports/minicpm5_capability_plan.md)

## Canonical locations

| Kind | Location |
|---|---|
| Experiment cards | `experiments/cards/M*.md` |
| Claim cards | `experiments/claims/M*_claim.md` |
| Configs | `configs/M*_minicpm5_*.json`, `configs/sft/minicpm5_*.json` |
| Evaluation data | `data/eval/minicpm5_*.jsonl` |
| Evaluation scripts | `scripts/eval/*minicpm5*`, `scripts/eval/run_gguf_eval.py` |
| Merge script | `scripts/train/merge_lora_direct.py` |
| Results | `results/evals/M*`, `results/evals/M03_*` |

## Naming convention

- Cards/configs/claims: `M##` for MiniCPM stages and `S##` for subsequent
  single-shot experiments.
- Phase 2 runs: `P2_S##_minicpm1b_<task>_<timestamp>_seed<seed>`.
- Data manifests, raw outputs, and metrics must include the experiment ID.
- Never overwrite a completed result; create a new run ID or use `--force`
  only when the card explicitly allows it.
