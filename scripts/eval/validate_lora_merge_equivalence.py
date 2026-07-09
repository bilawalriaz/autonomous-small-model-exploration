#!/usr/bin/env python3
"""Validate that independently-trained LoRAs are merged as an analytic delta sum.

This intentionally runs before GGUF conversion. It separates a correct FP16
merge from PEFT factor-combination behaviour and from quantization effects.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--direct-model", required=True,
                   help="FP16 model saved by merge_lora_direct.py")
    p.add_argument("--output", required=True)
    return p.parse_args()


def module_key(adapter_key: str) -> str:
    return (adapter_key.replace("base_model.model.", "", 1)
            .replace(".lora_A.weight", ".weight"))


def adapter_deltas(path: Path, weight: float) -> tuple[dict[str, torch.Tensor], dict]:
    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(path / "adapter_model.safetensors")
    scale = config["lora_alpha"] / (config["r"] ** 0.5 if config.get("use_rslora") else config["r"])
    deltas: dict[str, torch.Tensor] = {}
    errors = []
    for key, a in tensors.items():
        if ".lora_A.weight" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in tensors:
            errors.append(f"missing B tensor for {key}")
            continue
        deltas[module_key(key)] = (tensors[b_key].float() @ a.float()) * scale * weight
    return deltas, {"config": config, "tensor_count": len(tensors), "errors": errors}


def load_base(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()


@torch.inference_mode()
def fingerprints(model, tokenizer: AutoTokenizer) -> dict:
    probes = [
        "Solve carefully: If 12 pencils cost $3, what do 20 pencils cost? End with #### answer.",
        "Return only valid JSON: {\"name\": \"Ada\", \"active\": true}",
        "Write a Python function add(a, b) that returns their sum.",
    ]
    rows = []
    for prompt in probes:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt"
        )
        # Recent Transformers may return BatchEncoding rather than a Tensor.
        ids = (ids.input_ids if hasattr(ids, "input_ids") else ids).to(model.device)
        logits = model(ids).logits[:, -1, :].float().cpu()
        generated = model.generate(ids, do_sample=False, max_new_tokens=16,
                                   pad_token_id=tokenizer.eos_token_id)
        rows.append({
            "top_token": int(logits.argmax(-1).item()),
            "top_logit": float(logits.max().item()),
            "logits": logits.squeeze(0).tolist(),
            "generated_ids": generated[0, ids.shape[1]:].cpu().tolist(),
        })
    return {"rows": rows}


def max_logit_difference(a: dict, b: dict) -> float:
    return max(float((torch.tensor(x["logits"]) - torch.tensor(y["logits"])).abs().max())
               for x, y in zip(a["rows"], b["rows"]))


def tensor_difference(model, expected: dict[str, torch.Tensor]) -> dict:
    diffs = [(model.state_dict()[key].float().cpu() - value).abs()
             for key, value in expected.items()]
    return {
        "max_abs_error": float(max(x.max() for x in diffs)),
        "mean_abs_error": float(torch.cat([x.flatten() for x in diffs]).mean()),
    }


def main() -> int:
    ns = args()
    cfg = json.loads(Path(ns.config).read_text())
    math_path = Path(cfg["lora_config"]["math_adapter"])
    format_path = Path(cfg["lora_config"]["format_adapter"])
    weights = cfg["lora_config"]["weights"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)

    math, math_meta = adapter_deltas(math_path, weights["math"])
    fmt, fmt_meta = adapter_deltas(format_path, weights["format"])
    all_keys = sorted(set(math) | set(fmt))
    report = {
        "model": cfg["model_name_or_path"], "weights": weights,
        "math": math_meta, "format": fmt_meta,
        "math_only_keys": sorted(set(math) - set(fmt)),
        "format_only_keys": sorted(set(fmt) - set(math)),
        "shared_keys": len(set(math) & set(fmt)), "all_delta_keys": len(all_keys),
        "comparisons": {},
    }

    base = load_base(cfg["model_name_or_path"])
    base_state = base.state_dict()
    missing = [key for key in all_keys if key not in base_state]
    report["missing_base_keys"] = missing
    # Safetensors adapter weights reside on CPU while the model is on CUDA.
    # Keep this audit on CPU; it is a parameter check, not an inference path.
    expected = {key: base_state[key].float().cpu() + math.get(key, 0) + fmt.get(key, 0)
                for key in all_keys if key in base_state}
    del base
    gc.collect(); torch.cuda.empty_cache()

    direct = AutoModelForCausalLM.from_pretrained(
        ns.direct_model, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    direct_state = direct.state_dict()
    report["direct_tensor_equivalence"] = {
        "keys_checked": len(expected),
        **tensor_difference(direct, expected),
    }
    direct_fp = fingerprints(direct, tokenizer)
    del direct
    gc.collect(); torch.cuda.empty_cache()

    peft = load_base(cfg["model_name_or_path"])
    peft = PeftModel.from_pretrained(peft, str(math_path), adapter_name="math")
    peft.load_adapter(str(format_path), adapter_name="format")
    # Scaling the active format adapter makes this an exact simultaneous delta sum.
    for module in peft.modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict) and "format" in module.scaling:
            module.scaling["format"] *= weights["format"]
    try:
        peft.set_adapter(["math", "format"])
        active_fp = fingerprints(peft, tokenizer)
        report["comparisons"]["active_adapters_vs_direct"] = {
            "max_logit_abs_error": max_logit_difference(active_fp, direct_fp),
            "greedy_ids_equal": [a["generated_ids"] == b["generated_ids"]
                                 for a, b in zip(active_fp["rows"], direct_fp["rows"])],
        }
    except Exception as exc:
        report["comparisons"]["active_adapters_vs_direct"] = {"error": repr(exc)}

    # Fresh wrappers avoid modifying the original adapters while testing PEFT combinations.
    del peft
    gc.collect(); torch.cuda.empty_cache()

    # This is the native PEFT control corresponding exactly to W + Δmath + .7Δformat.
    # The older PEFT runtime does not support set_adapter([..]), so merge each active
    # adapter sequentially rather than relying on factor-combination semantics.
    try:
        sequential = load_base(cfg["model_name_or_path"])
        sequential = PeftModel.from_pretrained(sequential, str(math_path), adapter_name="math")
        sequential.set_adapter("math")
        sequential = sequential.merge_and_unload()
        sequential = PeftModel.from_pretrained(sequential, str(format_path), adapter_name="format")
        for module in sequential.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict) and "format" in module.scaling:
                module.scaling["format"] *= weights["format"]
        sequential.set_adapter("format")
        sequential = sequential.merge_and_unload()
        fp = fingerprints(sequential, tokenizer)
        report["comparisons"]["peft_sequential_merge_vs_direct"] = {
            "tensor_equivalence": tensor_difference(sequential, expected),
            "max_logit_abs_error": max_logit_difference(fp, direct_fp),
            "greedy_ids_equal": [a["generated_ids"] == b["generated_ids"]
                                 for a, b in zip(fp["rows"], direct_fp["rows"])],
        }
        del sequential
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        report["comparisons"]["peft_sequential_merge_vs_direct"] = {"error": repr(exc)}

    for kind in ("cat", "linear"):
        try:
            model = load_base(cfg["model_name_or_path"])
            model = PeftModel.from_pretrained(model, str(math_path), adapter_name="math")
            model.load_adapter(str(format_path), adapter_name="format")
            model.add_weighted_adapter(["math", "format"], [weights["math"], weights["format"]],
                                       adapter_name="combined", combination_type=kind)
            model.set_adapter("combined")
            fp = fingerprints(model, tokenizer)
            report["comparisons"][f"peft_{kind}_vs_direct"] = {
                "max_logit_abs_error": max_logit_difference(fp, direct_fp),
                "greedy_ids_equal": [a["generated_ids"] == b["generated_ids"]
                                     for a, b in zip(fp["rows"], direct_fp["rows"])],
            }
            del model
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            report["comparisons"][f"peft_{kind}_vs_direct"] = {"error": repr(exc)}

    Path(ns.output).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
