"""Run tokenizer diagnostics + baseline evaluation in one go on aero."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mi_atlas.model_loader import load_model_hf
from mi_atlas.backend import create_backend
from mi_atlas.tokenization import run_tokenization_diagnostics
from mi_atlas.task_suite import TaskSuite
from mi_atlas.eval_runner import evaluate_suite
from mi_atlas.plotting import plot_task_scores
from mi_atlas.experiment_registry import register_experiment
from mi_atlas.utils import save_json, set_seed, PROJECT_ROOT


def main():
    set_seed(42)

    print("=" * 60)
    print("STEP 1: Load model")
    print("=" * 60)
    bundle = load_model_hf("Qwen/Qwen2.5-0.5B")
    backend = create_backend(bundle)
    print(f"  Model: {bundle.model_name}")
    print(f"  Device: {bundle.device}")
    print(f"  Architecture: {bundle.architecture}")

    print("\n" + "=" * 60)
    print("STEP 2: Tokenizer diagnostics")
    print("=" * 60)
    tok_report = run_tokenization_diagnostics(bundle)
    tok_path = PROJECT_ROOT / "experiments" / "results" / "tokenizer_diagnostics.json"
    save_json(tok_report, tok_path)
    print(f"  Saved to {tok_path}")
    print(f"  Vocab size: {tok_report['basic_info']['vocab_size']}")
    print(f"  Bracket single-token:")
    for bracket, info in tok_report['bracket_tokenization'].items():
        print(f"    {bracket!r}: single={info['single_token']}, ids={info['token_ids']}")
    print(f"  Alignment checks:")
    for check in tok_report['alignment_checks']:
        print(f"    '{check['prompt'][:30]}...' + '{check['target']}' -> aligned={check['aligned']}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Deterministic generation check")
    print("=" * 60)
    test_prompts = [
        "The capital of France is ",
        "7 + 5 = ",
        "A B C A B ",
    ]
    for prompt in test_prompts:
        gen1 = backend.generate(prompt, max_new_tokens=10)
        gen2 = backend.generate(prompt, max_new_tokens=10)
        det = gen1 == gen2
        print(f"  '{prompt[:30]}' -> '{gen1[:30]}' (deterministic={det})")
        if not det:
            print(f"    WARNING: Non-deterministic! gen2='{gen2[:30]}'")

    print("\n" + "=" * 60)
    print("STEP 4: Baseline evaluation")
    print("=" * 60)
    suite_path = str(PROJECT_ROOT / "data" / "eval_sets" / "task_suite_v0.json")
    suite = TaskSuite.load(suite_path)
    print(f"  Suite: {suite.summary()}")

    # Evaluate on test split
    results = evaluate_suite(backend, suite, max_new_tokens=30, split="test")

    # Save results
    results_path = PROJECT_ROOT / "experiments" / "results" / "baseline_eval.json"
    # Can't save tensors to JSON, strip them
    clean_results = {
        "timestamp": results["timestamp"],
        "summary": results["summary"],
        "results": [{k: v for k, v in r.items() if k != "logits"} for r in results["results"]],
    }
    save_json(clean_results, results_path)
    print(f"  Results saved to {results_path}")

    print(f"\n  BASELINE RESULTS:")
    print(f"  Total examples: {results['summary']['total_examples']}")
    print(f"  Errors: {results['summary']['errors']}")
    print(f"  Overall mean: {results['summary']['overall_mean']:.3f}")
    print(f"  Per-family scores:")
    for fam, score in results['summary']['primary_metric_by_family'].items():
        print(f"    {fam}: {score:.3f}")

    # Print some example generations
    print(f"\n  SAMPLE GENERATIONS:")
    for r in results['results'][:10]:
        gen = r.get('generated', 'ERROR')
        tgt = r.get('target', '?')
        match = '✓' if gen.strip().startswith(tgt.strip()) else '✗'
        print(f"    {match} [{r['family']}] target='{tgt}' gen='{gen[:50]}'")

    # Generate plot
    if results['summary']['primary_metric_by_family']:
        plot_path = plot_task_scores(
            results['summary']['primary_metric_by_family'],
            title="Qwen2.5-0.5B Baseline Task Scores",
        )
        print(f"\n  Plot saved to {plot_path}")

    # Register experiment
    register_experiment(
        type="baseline",
        model=bundle.model_name,
        backend=bundle.backend,
        config="config/model.yaml",
        inputs=[suite_path],
        outputs=[str(results_path)],
        status="success",
        summary=f"Baseline eval: overall mean={results['summary']['overall_mean']:.3f}, {results['summary']['total_examples']} examples",
        key_metrics=results['summary']['primary_metric_by_family'],
        next="Run layer-level residual ablation",
    )
    print("\n  Experiment registered.")

    print("\n" + "=" * 60)
    print("STEP 5: Layer ablation")
    print("=" * 60)
    from mi_atlas.ablations import run_layer_ablation_suite
    import numpy as np

    print(f"  Running layer ablation (zero type, test split)...")
    ablation_results = run_layer_ablation_suite(backend, suite, ablation_type="zero", split="test")

    ablation_path = PROJECT_ROOT / "experiments" / "results" / "layer_ablation_zero.json"
    save_json(ablation_results, ablation_path)
    print(f"  Results saved to {ablation_path}")

    effect_matrix = np.array(ablation_results["effect_matrix"])
    print(f"  Effect matrix shape: {effect_matrix.shape}")
    print(f"  Max effect: {effect_matrix.max():.4f}")
    print(f"  Mean effect: {effect_matrix.mean():.4f}")

    # Find most important layers per family
    print(f"\n  TOP 3 LAYERS PER FAMILY (by KL divergence):")
    for fam_idx, fam in enumerate(ablation_results["families"]):
        layer_effects = effect_matrix[:, fam_idx]
        top3 = sorted(enumerate(layer_effects), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"L{i}({v:.3f})" for i, v in top3)
        print(f"    {fam}: {top3_str}")

    # Generate heatmap
    from mi_atlas.plotting import plot_ablation_heatmap
    plot_path = plot_ablation_heatmap(
        effect_matrix,
        row_labels=ablation_results["layer_names"],
        col_labels=ablation_results["families"],
        title="Qwen2.5-0.5B Layer Zero-Ablation Heatmap (KL Divergence)",
        name="layer_ablation_heatmap_zero",
    )
    print(f"\n  Heatmap saved to {plot_path}")

    register_experiment(
        type="ablation",
        model=bundle.model_name,
        backend=bundle.backend,
        config="config/experiment_plan.yaml",
        inputs=[suite_path],
        outputs=[str(ablation_path), str(plot_path)],
        status="success",
        summary=f"Layer zero ablation: 24 layers, max KL={effect_matrix.max():.3f}",
        next="Head and MLP ablation on top layers",
    )

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
