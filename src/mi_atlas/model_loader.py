"""Model loading and management."""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import load_config, get_device, get_dtype


class ModelBundle:
    """Container for model, tokenizer, and metadata."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        backend: str,
        device: torch.device,
        dtype: torch.dtype,
        architecture: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.architecture = architecture

    def __repr__(self) -> str:
        return (
            f"ModelBundle(name={self.model_name!r}, backend={self.backend!r}, "
            f"device={self.device}, layers={self.architecture.get('n_layers')})"
        )


def detect_model_info(model: Any, tokenizer: Any) -> dict:
    """Extract architecture metadata from a loaded HF model."""
    config = model.config
    info = {
        "n_layers": getattr(config, "num_hidden_layers", None),
        "n_heads": getattr(config, "num_attention_heads", None),
        "n_kv_heads": getattr(config, "num_key_value_heads", None),
        "d_model": getattr(config, "hidden_size", None),
        "d_head": getattr(config, "head_dim", None),
        "d_mlp": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None) or len(tokenizer),
        "context_length": getattr(config, "max_position_embeddings", None),
        "activation_function": getattr(config, "hidden_act", None),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
    }
    # Compute d_head if not directly available
    if info["d_head"] is None and info["d_model"] and info["n_heads"]:
        info["d_head"] = info["d_model"] // info["n_heads"]
    return info


def load_model_hf(
    model_name: str | None = None,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
) -> ModelBundle:
    """Load a model using Hugging Face Transformers."""
    config = load_config("model")
    if model_name is None:
        model_name = config["model"]["primary"]

    if device == "auto":
        device = get_device()
    if dtype == "auto":
        dtype = get_dtype()

    print(f"Loading {model_name} with HF Transformers...")
    print(f"  Device: {device}, Dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=str(device) if device.type != "cpu" else None,
    )
    if device.type == "cpu":
        model = model.to(device)

    model.eval()

    arch = detect_model_info(model, tokenizer)
    print(f"  Loaded. Architecture: {arch['n_layers']}L, {arch['n_heads']}H, d={arch['d_model']}")

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        backend="hf_native",
        device=device,
        dtype=dtype,
        architecture=arch,
    )


def load_model(
    model_name: str | None = None,
    backend: str | None = None,
) -> ModelBundle:
    """Load model with automatic backend selection."""
    config = load_config("model")
    if model_name is None:
        model_name = config["model"]["primary"]

    if backend is None:
        backend_priority = config.get("backend", {}).get("priority", ["hf_native"])
    else:
        backend_priority = [backend]

    errors = []
    for be in backend_priority:
        try:
            if be == "hf_native":
                return load_model_hf(model_name)
            elif be == "transformerlens":
                return _load_transformerlens(model_name)
            elif be == "nnsight":
                return _load_nnsight(model_name)
            else:
                errors.append(f"Unknown backend: {be}")
        except Exception as e:
            errors.append(f"{be}: {e}")
            print(f"  Backend {be} failed: {e}")
            continue

    # Try fallbacks from config
    fallbacks = config.get("model", {}).get("fallbacks", [])
    for fb in fallbacks:
        print(f"Trying fallback model: {fb}")
        for be in backend_priority:
            try:
                if be == "hf_native":
                    return load_model_hf(fb)
                elif be == "transformerlens":
                    return _load_transformerlens(fb)
                elif be == "nnsight":
                    return _load_nnsight(fb)
            except Exception as e:
                errors.append(f"{fb}/{be}: {e}")
                continue

    raise RuntimeError(f"All model loading attempts failed:\n" + "\n".join(errors))


def _load_transformerlens(model_name: str) -> ModelBundle:
    """Load model via TransformerLens."""
    import transformer_lens as tl

    config = load_config("model")
    tl_config = config.get("backend", {}).get("transformerlens", {})

    print(f"Loading {model_name} with TransformerLens...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=get_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tl_model = tl.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=tl_config.get("fold_ln", True),
        center_writing_weights=tl_config.get("center_writing_weights", True),
        center_unembed=tl_config.get("center_unembed", True),
        device=get_device(),
    )

    arch = detect_model_info(hf_model, tokenizer)

    # Clean up HF model (TL takes over)
    del hf_model

    return ModelBundle(
        model=tl_model,
        tokenizer=tokenizer,
        model_name=model_name,
        backend="transformerlens",
        device=get_device(),
        dtype=get_dtype(),
        architecture=arch,
    )


def _load_nnsight(model_name: str) -> ModelBundle:
    """Load model via NNsight."""
    from nnsight import LanguageModel as NNsightLM

    print(f"Loading {model_name} with NNsight...")

    ns_model = NNsightLM(
        model_name,
        device_map=str(get_device()),
        torch_dtype=get_dtype(),
    )
    tokenizer = ns_model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    arch = detect_model_info(ns_model.model, tokenizer)

    return ModelBundle(
        model=ns_model,
        tokenizer=tokenizer,
        model_name=model_name,
        backend="nnsight",
        device=get_device(),
        dtype=get_dtype(),
        architecture=arch,
    )
