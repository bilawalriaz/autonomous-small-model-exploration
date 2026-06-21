"""Backend abstraction for model execution.

Provides a unified interface over HF Transformers, TransformerLens, and NNsight.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np

from .model_loader import ModelBundle


class BackendBase(ABC):
    """Abstract base for model backends."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.device = bundle.device
        self.architecture = bundle.architecture

    @abstractmethod
    def run_with_cache(
        self, input_ids: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        """Run model and return (logits, activation_cache)."""
        ...

    @abstractmethod
    def run_with_hooks(
        self,
        input_ids: torch.Tensor,
        fwd_hooks: list | None = None,
        bwd_hooks: list | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run model with hooks attached, return logits."""
        ...

    @abstractmethod
    def get_layer_names(self) -> list[str]:
        """Return list of layer/component names for ablation."""
        ...

    @abstractmethod
    def get_activation(self, cache: dict, layer_name: str) -> torch.Tensor:
        """Extract activation tensor from cache by layer name."""
        ...

    def tokenize(self, text: str) -> dict:
        """Tokenize text and return dict with input_ids, attention_mask."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.architecture.get("context_length", 2048),
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int = 1,
        do_sample: bool = False,
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenize(text)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k if top_k > 1 else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.decode(output[0][input_ids.shape[1]:])

    def get_logits(self, text: str) -> torch.Tensor:
        """Get logits for all positions."""
        inputs = self.tokenize(text)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits

    def get_next_token_logits(self, text: str) -> torch.Tensor:
        """Get logits for the next token (last position)."""
        logits = self.get_logits(text)
        return logits[0, -1, :]

    def get_next_token_probs(self, text: str) -> torch.Tensor:
        """Get probability distribution over next token."""
        logits = self.get_next_token_logits(text)
        return torch.softmax(logits, dim=-1)

    def get_target_logprob(self, text: str, target: str) -> float:
        """Get log probability of a specific target token/string following text."""
        full_text = text + target
        inputs_full = self.tokenize(full_text)
        inputs_ctx = self.tokenize(text)

        ctx_len = inputs_ctx["input_ids"].shape[1]
        full_ids = inputs_full["input_ids"].to(self.device)

        with torch.no_grad():
            logits = self.model(full_ids).logits

        # Log prob of target tokens given context
        log_probs = torch.log_softmax(logits, dim=-1)
        target_ids = full_ids[0, ctx_len:]
        target_logprobs = log_probs[0, ctx_len - 1:-1]

        # Gather log probs for target tokens
        gathered = target_logprobs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        return gathered.sum().item()

    @property
    def n_layers(self) -> int:
        return self.architecture.get("n_layers", 0)

    @property
    def n_heads(self) -> int:
        return self.architecture.get("n_heads", 0)

    @property
    def d_model(self) -> int:
        return self.architecture.get("d_model", 0)


class HFBackend(BackendBase):
    """Backend using raw Hugging Face Transformers."""

    def get_layer_names(self) -> list[str]:
        names = []
        n = self.n_layers
        for i in range(n):
            names.append(f"layer_{i:02d}")
        return names

    def run_with_cache(self, input_ids: torch.Tensor, **kwargs):
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        with torch.no_grad():
            outputs = self.model(input_ids)

        cache = {
            "logits": outputs.logits,
        }
        # Store hidden states
        if outputs.hidden_states:
            for i, hs in enumerate(outputs.hidden_states):
                cache[f"layer_{i:02d}"] = hs
        # Store attention outputs (final attention per layer)
        if outputs.attentions:
            for i, att in enumerate(outputs.attentions):
                cache[f"attn_{i:02d}"] = att

        return outputs.logits, cache

    def run_with_hooks(self, input_ids, fwd_hooks=None, bwd_hooks=None, **kwargs):
        # HF doesn't support hooks natively — use context managers
        # For simple ablation, we modify activations manually
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits

    def get_activation(self, cache: dict, layer_name: str) -> torch.Tensor:
        return cache[layer_name]


class TransformerLensBackend(BackendBase):
    """Backend using TransformerLens."""

    def run_with_cache(self, input_ids: torch.Tensor, **kwargs):
        logits, cache = self.model.run_with_cache(input_ids)
        return logits, cache

    def run_with_hooks(self, input_ids, fwd_hooks=None, bwd_hooks=None, **kwargs):
        logits = self.model.run_with_hooks(
            input_ids,
            fwd_hooks=fwd_hooks or [],
            bwd_hooks=bwd_hooks or [],
        )
        return logits

    def get_layer_names(self) -> list[str]:
        names = []
        n = self.n_layers
        for i in range(n):
            names.append(f"blocks.{i}.hook_resid_post")
            names.append(f"blocks.{i}.attn.hook_result")
            names.append(f"blocks.{i}.hook_mlp_out")
        return names

    def get_activation(self, cache, layer_name):
        return cache[layer_name]


class NNsightBackend(BackendBase):
    """Backend using NNsight."""

    def run_with_cache(self, input_ids, **kwargs):
        with self.model.trace(input_ids) as tracer:
            logits = self.model.output.logits.save()
            # Save layer activations
            hidden = {}
            for i, layer in enumerate(self.model.model.layers):
                hidden[f"layer_{i:02d}"] = layer.output[0].save()
        cache = {k: v.value for k, v in hidden.items()}
        cache["logits"] = logits.value
        return cache["logits"], cache

    def run_with_hooks(self, input_ids, fwd_hooks=None, bwd_hooks=None, **kwargs):
        with self.model.trace(input_ids):
            logits = self.model.output.logits.save()
        return logits.value

    def get_layer_names(self) -> list[str]:
        return [f"layer_{i:02d}" for i in range(self.n_layers)]

    def get_activation(self, cache, layer_name):
        return cache[layer_name]


def create_backend(bundle: ModelBundle) -> BackendBase:
    """Create the appropriate backend for a loaded model."""
    backends = {
        "hf_native": HFBackend,
        "transformerlens": TransformerLensBackend,
        "nnsight": NNsightBackend,
    }
    cls = backends.get(bundle.backend)
    if cls is None:
        raise ValueError(f"Unknown backend: {bundle.backend}")
    return cls(bundle)
