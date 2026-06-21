"""Tokenization diagnostics and utilities."""

import torch
from transformers import PreTrainedTokenizer

from .model_loader import ModelBundle


class TokenizationDiagnostics:
    """Diagnostic checks for tokenizer behaviour."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def basic_info(self) -> dict:
        """Return basic tokenizer metadata."""
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": getattr(self.tokenizer, "bos_token", None),
            "unk_token": getattr(self.tokenizer, "unk_token", None),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def tokenize_report(self, text: str) -> dict:
        """Tokenize text and return detailed token info."""
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.encode(text)
        decoded = [self.tokenizer.decode([tid]) for tid in ids]
        return {
            "text": text,
            "tokens": tokens,
            "token_ids": ids,
            "decoded_per_token": decoded,
            "n_tokens": len(ids),
        }

    def alignment_check(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Check that target strings align to clean token boundaries.

        Args:
            pairs: list of (prompt, target) pairs

        Returns:
            list of dicts with alignment info
        """
        results = []
        for prompt, target in pairs:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            full_ids = self.tokenizer.encode(prompt + target, add_special_tokens=False)

            # Check if concatenation is just prompt_ids + target_ids
            expected = prompt_ids + target_ids
            aligned = full_ids == expected

            results.append({
                "prompt": prompt,
                "target": target,
                "prompt_ids": prompt_ids,
                "target_ids": target_ids,
                "full_ids": full_ids,
                "aligned": aligned,
                "prompt_n_tokens": len(prompt_ids),
                "target_n_tokens": len(target_ids),
            })
        return results

    def multi_token_targets(self, targets: list[str]) -> list[dict]:
        """Check which targets split into multiple tokens."""
        results = []
        for target in targets:
            ids = self.tokenizer.encode(target, add_special_tokens=False)
            decoded_parts = [self.tokenizer.decode([tid]) for tid in ids]
            results.append({
                "target": target,
                "n_tokens": len(ids),
                "token_ids": ids,
                "decoded_parts": decoded_parts,
                "is_single_token": len(ids) == 1,
            })
        return results

    def bracket_tokenization(self) -> dict:
        """Check how common delimiters are tokenized."""
        delimiters = ["(", ")", "[", "]", "{", "}", ":", ",", ".", ";", '"', "'", "```", "->", "=>"]
        results = {}
        for d in delimiters:
            ids = self.tokenizer.encode(d, add_special_tokens=False)
            decoded = [self.tokenizer.decode([tid]) for tid in ids]
            results[d] = {
                "token_ids": ids,
                "decoded": decoded,
                "single_token": len(ids) == 1,
            }
        return results

    def digit_tokenization(self) -> dict:
        """Check how single digits are tokenized."""
        results = {}
        for digit in "0123456789":
            ids = self.tokenizer.encode(digit, add_special_tokens=False)
            decoded = [self.tokenizer.decode([tid]) for tid in ids]
            results[digit] = {
                "token_ids": ids,
                "decoded": decoded,
                "single_token": len(ids) == 1,
            }
        return results


def run_tokenization_diagnostics(bundle: ModelBundle) -> dict:
    """Run full tokenization diagnostics for the loaded model."""
    diag = TokenizationDiagnostics(bundle.tokenizer)

    report = {
        "basic_info": diag.basic_info(),
        "bracket_tokenization": diag.bracket_tokenization(),
        "digit_tokenization": diag.digit_tokenization(),
    }

    # Common alignment checks
    test_pairs = [
        ("The capital of France is ", "Paris"),
        ("7 + 5 = ", "12"),
        ("A B C A B ", "C"),
        ("def add(a, b):\n    return a + ", "b"),
        ('{"name": "Alice", "age": ', "31"),
    ]
    report["alignment_checks"] = diag.alignment_check(test_pairs)

    # Multi-token target checks
    common_targets = [
        "Paris", "12", "C", "b", "31", "true", "false", "null",
        "print", "return", "function", "```", "[]", "{}",
    ]
    report["multi_token_targets"] = diag.multi_token_targets(common_targets)

    return report
