#!/usr/bin/env python3
"""Compile canonical JSONL into specific SFT training format.

Phase 9 data-format-ablation project.
Reads canonical JSONL and renders into one of 6 training format variants.

Usage:
    python scripts/data/compile_sft_dataset.py \
        --canonical data/canonical/phase9_pilot_300.jsonl \
        --format multi_turn_concise \
        --output data/sft/format_ablation/multi_turn_concise.jsonl

Formats: alpaca_flat, single_turn_chat, multi_turn_concise,
         multi_turn_verbose, structured_terse, bad_format_control
"""

import argparse
import json
import logging
import os
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Domain-specific follow-up prompts for multi-turn formats
DOMAIN_FOLLOWUPS = {
    "coding": [
        "Can you also add error handling?",
        "Now add type hints to the function.",
        "Can you make it handle edge cases too?",
        "Add a docstring explaining the parameters.",
    ],
    "gamefaq": [
        "What about the hidden items in that area?",
        "Any tips for the boss fight?",
        "What's the best strategy for the early game?",
        "Are there any secret paths I should know about?",
    ],
    "json": [
        "Now add a field for timestamps.",
        "Can you also include a status field?",
        "Add validation for the required fields.",
        "Make it include nested objects for metadata.",
    ],
    "extraction": [
        "Can you also extract the dates mentioned?",
        "Now pull out the numeric values too.",
        "What about the locations mentioned?",
        "Can you normalize the extracted entities?",
    ],
    "classification": [
        "What about borderline cases?",
        "Can you explain the reasoning behind the classification?",
        "How confident is this classification?",
        "What features were most important?",
    ],
    "summarization": [
        "Can you make it even shorter?",
        "Now focus on the key statistics.",
        "What are the main takeaways?",
        "Can you highlight the most surprising finding?",
    ],
    "math": [
        "Can you verify this answer?",
        "What's the general formula for this?",
        "Show the step-by-step work.",
        "Does this work for negative numbers too?",
    ],
    "reasoning": [
        "Can you break this down further?",
        "What assumptions are we making?",
        "Is there an alternative approach?",
        "What would change if we modified the premise?",
    ],
}

# Verbose expansions for multi_turn_verbose
VERBOSE_PREFIXES = [
    "Certainly! Let me explain this in detail. ",
    "Great question! Here's a thorough breakdown: ",
    "I'd be happy to help with that. ",
    "Let me walk you through this step by step. ",
    "That's a good point. Here's the full explanation: ",
]

# Bad format control filler phrases
BAD_FILLER_PREAMBLES = [
    "Thank you for your interesting question. I'll do my best to provide a helpful response. ",
    "That's a great question! Let me think about this carefully before answering. ",
    "I appreciate you asking this. Here is what I think about it: ",
    "This is a common question that many people ask. Let me explain. ",
    "Before I answer, I should note that this is just my understanding, and I could be wrong. ",
]

BAD_FILLER_CAVEATS = [
    "\n\nPlease note that this is general information and you should always verify with official documentation.",
    "\n\nHope this helps! Let me know if you have any other questions.",
    "\n\nDisclaimer: This is AI-generated content and should be used as a starting point only.",
    "\n\nAs always, your mileage may vary and results may differ depending on your specific situation.",
    "\n\nI hope this explanation was clear. Feel free to ask for clarification if needed!",
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def load_canonical(path: str) -> list[dict[str, Any]]:
    """Load canonical JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON on line %d of %s: %s", i, path, e)
                sys.exit(1)
            examples.append(obj)
    logger.info("Loaded %d canonical examples from %s", len(examples), path)
    return examples


def _build_user_content(ex: dict, include_intent: bool = True) -> str:
    """Build user content from canonical example."""
    intent = ex.get("user_intent", "")
    context = ex.get("context", "")
    if include_intent and intent and context:
        return f"{intent}\n\n{context}"
    elif context:
        return context
    elif intent:
        return intent
    return ""


def _get_domain_followup(domain: str, idx: int) -> tuple[str, str]:
    """Get a domain-appropriate follow-up question and concise answer."""
    followups = DOMAIN_FOLLOWUPS.get(domain, DOMAIN_FOLLOWUPS["reasoning"])
    q = followups[idx % len(followups)]
    # Generic concise follow-up answers
    answers = [
        "Done! I've updated the response accordingly.",
        "Sure, here's that addition.",
        "Good point — here you go.",
        "Added! Let me know if you need more changes.",
    ]
    a = answers[idx % len(answers)]
    return q, a


def render_alpaca_flat(ex: dict) -> dict:
    """Render to Alpaca flat format."""
    return {
        "instruction": ex.get("user_intent", ""),
        "input": ex.get("context", ""),
        "output": ex.get("ideal_answer", ""),
        "_canonical_id": ex.get("id", ""),
        "_domain": ex.get("domain", ""),
    }


def render_single_turn_chat(ex: dict) -> dict:
    """Render to single-turn chat format."""
    user_content = _build_user_content(ex, include_intent=True)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ex.get("ideal_answer", "")},
        ],
        "_canonical_id": ex.get("id", ""),
        "_domain": ex.get("domain", ""),
    }


def render_multi_turn_concise(ex: dict) -> dict:
    """Render to multi-turn concise format."""
    domain = ex.get("domain", "")
    turns = ex.get("turns")

    if domain == "multi_turn" and turns:
        # Use the canonical multi-turn structure
        messages = []
        for turn in turns:
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})
        return {
            "messages": messages,
            "_canonical_id": ex.get("id", ""),
            "_domain": domain,
        }

    # For single-turn domains, build 1-2 turns with a follow-up
    user_content = _build_user_content(ex, include_intent=True)
    ideal = ex.get("ideal_answer", "")
    followup_q, followup_a = _get_domain_followup(domain, 0)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ideal},
        {"role": "user", "content": followup_q},
        {"role": "assistant", "content": followup_a},
    ]
    return {
        "messages": messages,
        "_canonical_id": ex.get("id", ""),
        "_domain": domain,
    }


def render_multi_turn_verbose(ex: dict) -> dict:
    """Render to multi-turn verbose format with longer explanatory responses."""
    domain = ex.get("domain", "")
    turns = ex.get("turns")

    if domain == "multi_turn" and turns:
        messages = []
        for i, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn.get("user", "")})
            verbose_answer = turn.get("assistant", "")
            if i == 0:
                verbose_answer = VERBOSE_PREFIXES[i % len(VERBOSE_PREFIXES)] + verbose_answer
            messages.append({"role": "assistant", "content": verbose_answer})
        return {
            "messages": messages,
            "_canonical_id": ex.get("id", ""),
            "_domain": domain,
        }

    user_content = _build_user_content(ex, include_intent=True)
    ideal = ex.get("ideal_answer", "")
    verbose_ideal = VERBOSE_PREFIXES[0] + ideal
    followup_q, followup_a = _get_domain_followup(domain, 0)
    verbose_followup = VERBOSE_PREFIXES[1 % len(VERBOSE_PREFIXES)] + followup_a

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": verbose_ideal},
        {"role": "user", "content": followup_q},
        {"role": "assistant", "content": verbose_followup},
    ]
    return {
        "messages": messages,
        "_canonical_id": ex.get("id", ""),
        "_domain": domain,
    }


def render_structured_terse(ex: dict) -> dict:
    """Render to structured terse format with compact outputs."""
    domain = ex.get("domain", "")
    ideal = ex.get("ideal_answer", "")
    user_content = _build_user_content(ex, include_intent=True)

    # For JSON/code domains, try to compactify
    terse_ideal = ideal
    if domain in ("json", "coding", "extraction"):
        # Try to parse and re-serialize with minimal whitespace
        try:
            parsed = json.loads(ideal)
            terse_ideal = json.dumps(parsed, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            # Not pure JSON — strip excessive whitespace
            terse_ideal = " ".join(ideal.split())

    if domain == "multi_turn" and ex.get("turns"):
        messages = []
        for turn in ex["turns"]:
            messages.append({"role": "user", "content": turn.get("user", "")})
            a = turn.get("assistant", "")
            messages.append({"role": "assistant", "content": " ".join(a.split())})
        return {
            "messages": messages,
            "_canonical_id": ex.get("id", ""),
            "_domain": domain,
        }

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": terse_ideal},
        ],
        "_canonical_id": ex.get("id", ""),
        "_domain": domain,
    }


def render_bad_format_control(ex: dict) -> dict:
    """Render to bad format control — verbose, generic, full of filler."""
    domain = ex.get("domain", "")
    ideal = ex.get("ideal_answer", "")
    user_content = _build_user_content(ex, include_intent=True)
    canonical_id = ex.get("id", "")

    # Deterministic filler selection based on ID
    idx = int("".join(filter(str.isdigit, canonical_id)) or "0")
    preamble = BAD_FILLER_PREAMBLES[idx % len(BAD_FILLER_PREAMBLES)]
    caveat = BAD_FILLER_CAVEATS[idx % len(BAD_FILLER_CAVEATS)]

    bad_answer = preamble + ideal + caveat

    if domain == "multi_turn" and ex.get("turns"):
        messages = []
        for i, turn in enumerate(ex["turns"]):
            messages.append({"role": "user", "content": turn.get("user", "")})
            a = turn.get("assistant", "")
            bad_a = BAD_FILLER_PREAMBLES[(idx + i) % len(BAD_FILLER_PREAMBLES)] + a + BAD_FILLER_CAVEATS[(idx + i) % len(BAD_FILLER_CAVEATS)]
            messages.append({"role": "assistant", "content": bad_a})
        return {
            "messages": messages,
            "_canonical_id": canonical_id,
            "_domain": domain,
        }

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": bad_answer},
        ],
        "_canonical_id": canonical_id,
        "_domain": domain,
    }


# Registry of format renderers
FORMAT_RENDERERS = {
    "alpaca_flat": render_alpaca_flat,
    "single_turn_chat": render_single_turn_chat,
    "multi_turn_concise": render_multi_turn_concise,
    "multi_turn_verbose": render_multi_turn_verbose,
    "structured_terse": render_structured_terse,
    "bad_format_control": render_bad_format_control,
}

ALL_FORMATS = list(FORMAT_RENDERERS.keys())


def compile_dataset(
    canonical_path: str,
    fmt: str,
    output_path: str,
    seed: int = 42,
) -> dict:
    """Compile canonical dataset into specified format.

    Returns metadata dict with counts and paths.
    """
    if fmt not in FORMAT_RENDERERS:
        logger.error("Unknown format: %s. Choose from: %s", fmt, ALL_FORMATS)
        sys.exit(1)

    renderer = FORMAT_RENDERERS[fmt]
    examples = load_canonical(canonical_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rendered = []
    for ex in examples:
        record = renderer(ex)
        rendered.append(record)

    # Write output JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in rendered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Compiled %d examples into %s format -> %s",
        len(rendered), fmt, output_path,
    )

    return {
        "format": fmt,
        "path": output_path,
        "count": len(rendered),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compile canonical JSONL into SFT training format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--canonical",
        required=True,
        help="Path to canonical JSONL file.",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=ALL_FORMATS,
        help="Output format variant.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for rendered JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42).",
    )
    args = parser.parse_args()

    meta = compile_dataset(
        canonical_path=args.canonical,
        fmt=args.format,
        output_path=args.output,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
