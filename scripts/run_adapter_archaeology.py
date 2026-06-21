"""
Adapter Archaeology: Reverse-engineer LoRA adapter structure.
Analyze norms, effective rank, layer concentration, SVD directions.

Uses all 10 adapters: 5 rank sweep (r=1,2,4,8,16) + 5 family-specific (r=8).
"""
import json, os, sys
from pathlib import Path
import torch
import numpy as np

REPO = Path(__file__).parent.parent
ADAPTERS_DIR = REPO / "experiments" / "adapters"
RESULTS_DIR = REPO / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_adapter(adapter_path: str, name: str):
    """Analyze a single LoRA adapter."""
    adapter_path = Path(adapter_path) / "adapter"
    if not adapter_path.exists():
        print(f"  [{name}] not found at {adapter_path}")
        return None

    # Load adapter weights
    state_dict = {}
    for fname in os.listdir(adapter_path):
        if fname.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict.update(load_file(adapter_path / fname))
        elif fname.endswith(".bin"):
            state_dict.update(torch.load(adapter_path / fname, map_location="cpu"))

    if not state_dict:
        print(f"  [{name}] no weights found")
        return None

    # Analyze by layer
    layer_info = {}
    for key, tensor in state_dict.items():
        # Parse: base_model.model.model.layers.{L}.{module}.{A|B}
        parts = key.split(".")
        layer_idx = None
        module_name = None
        matrix_type = None

        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
            if p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                module_name = p
            if p in ("lora_A", "lora_B"):
                matrix_type = p

        if layer_idx is None or module_name is None or matrix_type is None:
            continue

        layer_key = f"L{layer_idx}"
        if layer_key not in layer_info:
            layer_info[layer_key] = {}
        if module_name not in layer_info[layer_key]:
            layer_info[layer_key][module_name] = {}

        t = tensor.float()
        layer_info[layer_key][module_name][matrix_type] = {
            "shape": list(t.shape),
            "norm_frobenius": t.norm().item(),
            "norm_l1": t.abs().sum().item(),
            "mean_abs": t.abs().mean().item(),
            "std": t.std().item(),
        }

    # Compute per-layer combined norms
    layer_norms = {}
    for layer_key, modules in layer_info.items():
        total_norm = 0
        for module_name, matrices in modules.items():
            for mt, stats in matrices.items():
                total_norm += stats["norm_frobenius"] ** 2
        layer_norms[layer_key] = total_norm ** 0.5

    # Compute effective rank per layer (SVD of concatenated A/B)
    layer_eff_rank = {}
    for layer_key, modules in layer_info.items():
        all_weights = []
        for module_name, matrices in modules.items():
            for mt_key in ["lora_A", "lora_B"]:
                if mt_key in matrices:
                    # Reconstruct actual weight
                    for fname in os.listdir(adapter_path):
                        if fname.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            sd = load_file(adapter_path / fname)
                            for k, v in sd.items():
                                if layer_key.lower().replace("l", "layers.") in k and module_name in k and mt_key in k:
                                    all_weights.append(v.float().flatten())
                            break
                    break
        if all_weights:
            combined = torch.cat(all_weights)
            # Approximate effective rank via singular value decay
            # Use the ratio of norms as a proxy
            n = len(combined)
            if n > 1:
                # Effective rank approximation: ||w||_1^2 / ||w||_2^2
                l1 = combined.abs().sum().item()
                l2 = combined.norm().item()
                if l2 > 0:
                    eff_rank = (l1 / l2) ** 2 / n
                else:
                    eff_rank = 0
                layer_eff_rank[layer_key] = round(eff_rank, 4)

    # Module concentration
    module_norms = {}
    for layer_key, modules in layer_info.items():
        for module_name, matrices in modules.items():
            if module_name not in module_norms:
                module_norms[module_name] = 0
            for mt, stats in matrices.items():
                module_norms[module_name] += stats["norm_frobenius"] ** 2
    for k in module_norms:
        module_norms[k] = module_norms[k] ** 0.5

    # Top-3 layers by norm
    top3_layers = sorted(layer_norms.items(), key=lambda x: -x[1])[:3]

    return {
        "name": name,
        "layer_norms": {k: round(v, 4) for k, v in layer_norms.items()},
        "layer_eff_rank": layer_eff_rank,
        "module_norms": {k: round(v, 4) for k, v in module_norms.items()},
        "top3_layers": [(k, round(v, 4)) for k, v in top3_layers],
        "total_norm": round(sum(v**2 for v in layer_norms.values()) ** 0.5, 4),
    }


def main():
    print("=" * 60)
    print("Adapter Archaeology")
    print("=" * 60)

    # Rank sweep adapters
    rank_adapters = [
        (str(ADAPTERS_DIR / "lora_json_r1"), "json_r1"),
        (str(ADAPTERS_DIR / "lora_json_r2"), "json_r2"),
        (str(ADAPTERS_DIR / "lora_json_r4"), "json_r4"),
        (str(ADAPTERS_DIR / "lora_json_r8"), "json_r8"),
        (str(ADAPTERS_DIR / "lora_json_r16"), "json_r16"),
    ]

    # Family-specific adapters
    family_adapters = [
        (str(ADAPTERS_DIR / "lora_copying_r8"), "copying_r8"),
        (str(ADAPTERS_DIR / "lora_delimiter_tracking_r8"), "delimiter_r8"),
        (str(ADAPTERS_DIR / "lora_factual_recall_r8"), "factual_r8"),
        (str(ADAPTERS_DIR / "lora_code_semantics_r8"), "code_r8"),
        (str(ADAPTERS_DIR / "lora_json_schema_r8"), "json_schema_r8"),
    ]

    all_results = {}

    print("\n--- Rank sweep adapters ---")
    for path, name in rank_adapters:
        result = analyze_adapter(path, name)
        if result:
            all_results[name] = result
            print(f"  {name}: top3={result['top3_layers']}, total_norm={result['total_norm']:.2f}")

    print("\n--- Family-specific adapters ---")
    for path, name in family_adapters:
        result = analyze_adapter(path, name)
        if result:
            all_results[name] = result
            print(f"  {name}: top3={result['top3_layers']}, total_norm={result['total_norm']:.2f}")

    # Cross-adapter comparison
    print("\n--- Layer concentration comparison ---")
    # For rank sweep: how does concentration change with rank?
    print("\nRank sweep layer norms (top-5 layers):")
    for name in ["json_r1", "json_r2", "json_r4", "json_r8", "json_r16"]:
        if name in all_results:
            norms = all_results[name]["layer_norms"]
            top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
            print(f"  {name}: {', '.join(f'{k}={v:.2f}' for k, v in top5)}")

    print("\nFamily adapters layer norms (top-5 layers):")
    for name in ["copying_r8", "delimiter_r8", "factual_r8", "code_r8", "json_schema_r8"]:
        if name in all_results:
            norms = all_results[name]["layer_norms"]
            top5 = sorted(norms.items(), key=lambda x: -x[1])[:5]
            print(f"  {name}: {', '.join(f'{k}={v:.2f}' for k, v in top5)}")

    # Module concentration
    print("\nModule norms by adapter:")
    for name, result in all_results.items():
        mods = result["module_norms"]
        total = sum(mods.values())
        pcts = {k: round(v / total * 100, 1) for k, v in mods.items()}
        print(f"  {name}: {pcts}")

    # Save
    out_path = RESULTS_DIR / "adapter_archaeology.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Registry entry
    import time
    from datetime import datetime, timezone
    entry = {
        "id": "exp_000015",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "comparison",
        "model": "Qwen/Qwen2.5-0.5B",
        "backend": "hf",
        "git_commit": "",
        "config": "",
        "inputs": [],
        "outputs": [str(out_path)],
        "status": "success",
        "summary": f"Adapter archaeology: {len(all_results)} adapters analyzed",
        "key_metrics": {},
        "failure": None,
        "next": "Adapter stacking/interference, checkpoint timeline",
    }
    with open(REPO / "experiments" / "registry.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
