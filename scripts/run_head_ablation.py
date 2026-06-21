"""Run head-level attention ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import numpy as np
from mi_atlas.model_loader import load_model
from mi_atlas.backend import create_backend, TransformerLensBackend
from mi_atlas.task_suite import TaskSuite
from mi_atlas.plotting import plot_ablation_heatmap
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--suite", type=str, default=None)
    args = parser.parse_args()

    bundle = load_model(args.model)
    backend = create_backend(bundle)
    suite_path = args.suite or str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)

    if not isinstance(backend, TransformerLensBackend):
        print("Head ablation requires TransformerLens backend. Skipping.")
        return

    n_layers = backend.n_layers
    n_heads = backend.n_heads
    families = suite.families

    effect_matrix = np.zeros((n_layers * n_heads, len(families)))

    for fam_idx, family in enumerate(families):
        examples = list(suite.filter_by_family(family))[:3]
        for layer in range(n_layers):
            for head in range(n_heads):
                hook_name = f"blocks.{layer}.attn.hook_result"
                effects = []
                for ex in examples:
                    inputs = backend.tokenize(ex.clean_prompt)
                    input_ids = inputs["input_ids"].to(backend.device)
                    orig_logits, _ = backend.run_with_cache(input_ids)

                    def zero_head(activation, hook, h=head):
                        activation[:, :, h, :] = 0.0
                        return activation

                    abl_logits = backend.run_with_hooks(input_ids, fwd_hooks=[(hook_name, zero_head)])
                    kl = torch.nn.functional.kl_div(
                        torch.log_softmax(abl_logits[0, -1], dim=-1),
                        torch.softmax(orig_logits[0, -1], dim=-1),
                        reduction="sum"
                    ).item()
                    effects.append(kl)
                effect_matrix[layer * n_heads + head, fam_idx] = np.mean(effects)

    row_labels = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)]
    plot_path = plot_ablation_heatmap(effect_matrix, row_labels, families,
                                       title="Head Ablation Heatmap", name="head_ablation_heatmap")

    output_path = PROJECT_ROOT / "experiments" / "results" / "head_ablation.json"
    save_json({"effect_matrix": effect_matrix.tolist(), "row_labels": row_labels, "families": families}, output_path)

    register_experiment(type="ablation", model=bundle.model_name, backend=bundle.backend,
                       config="config/experiment_plan.yaml", inputs=[suite_path],
                       outputs=[str(output_path), str(plot_path)], status="success",
                       summary=f"Head ablation: {n_layers}L x {n_heads}H", next="Run MLP ablation")
    print("Head ablation complete.")


if __name__ == "__main__":
    main()
