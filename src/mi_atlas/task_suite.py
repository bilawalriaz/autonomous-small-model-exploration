"""Task suite: structured examples for evaluation and interpretability."""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .utils import load_config, save_json, load_json, PROJECT_ROOT


@dataclass
class TaskExample:
    """A single task example with clean/corrupt pair support."""
    id: str
    family: str
    clean_prompt: str
    corrupt_prompt: str | None = None
    target: str = ""
    wrong_target: str | None = None
    metric_type: str = "exact_match"
    metadata: dict = field(default_factory=dict)
    split: str = "train"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskExample":
        return cls(**d)


class TaskSuite:
    """Collection of task examples across families."""

    def __init__(self, examples: list[TaskExample] | None = None):
        self.examples: list[TaskExample] = examples or []

    def add(self, example: TaskExample) -> None:
        self.examples.append(example)

    def filter_by_family(self, family: str) -> "TaskSuite":
        return TaskSuite([e for e in self.examples if e.family == family])

    def filter_by_split(self, split: str) -> "TaskSuite":
        return TaskSuite([e for e in self.examples if e.split == split])

    @property
    def families(self) -> list[str]:
        return sorted(set(e.family for e in self.examples))

    @property
    def splits(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.examples:
            counts[e.split] = counts.get(e.split, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def save(self, path: str | Path) -> None:
        data = [e.to_dict() for e in self.examples]
        save_json(data, path)

    @classmethod
    def load(cls, path: str | Path) -> "TaskSuite":
        data = load_json(path)
        examples = [TaskExample.from_dict(d) for d in data]
        return cls(examples)

    def summary(self) -> dict:
        family_counts: dict[str, int] = {}
        for e in self.examples:
            family_counts[e.family] = family_counts.get(e.family, 0) + 1
        return {
            "total": len(self.examples),
            "families": family_counts,
            "splits": self.splits,
        }


def _make_id(family: str, idx: int) -> str:
    return f"{family}_{idx:04d}"


# ── Task generators ──────────────────────────────────────────────────

def generate_copying_examples(n: int = 10, seed: int = 42) -> list[TaskExample]:
    """Copying / induction tasks."""
    rng = random.Random(seed)
    examples = []
    symbols_pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for i in range(n):
        seq_len = rng.randint(3, 6)
        repeat = rng.randint(1, 3)
        symbols = rng.sample(symbols_pool, seq_len)
        prefix = (symbols * (repeat + 1))[:seq_len * repeat + seq_len - 1]
        prompt = " ".join(prefix) + " "
        target = symbols[(seq_len * repeat) % seq_len]

        corrupt_symbols = list(symbols)
        corrupt_symbols[-1] = rng.choice([s for s in symbols_pool if s not in symbols])
        corrupt_prefix = (corrupt_symbols * (repeat + 1))[:seq_len * repeat + seq_len - 1]
        corrupt_prompt = " ".join(corrupt_prefix) + " "

        examples.append(TaskExample(
            id=_make_id("copying", i),
            family="copying",
            clean_prompt=prompt,
            corrupt_prompt=corrupt_prompt,
            target=target,
            wrong_target=rng.choice([s for s in symbols_pool if s != target]),
            metric_type="target_logprob",
            metadata={"seq_len": seq_len, "repeat": repeat, "symbols": symbols},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_delimiter_examples(n: int = 10, seed: int = 42) -> list[TaskExample]:
    """Delimiter / bracket tracking tasks."""
    rng = random.Random(seed)
    examples = []
    bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    depths = [1, 2, 3, 4]

    for i in range(n):
        depth = rng.choice(depths)
        pairs = rng.choices(bracket_pairs, k=depth)
        openers = "".join(p[0] for p in pairs)
        closers = "".join(p[1] for p in pairs[::-1])
        prompt = f"Complete the closing delimiters: function(x, {openers}"
        target = closers

        # Corrupt: scramble closers
        corrupt_closers = list(closers)
        rng.shuffle(corrupt_closers)
        corrupt_prompt = f"Complete the closing delimiters: function(x, {openers}"

        examples.append(TaskExample(
            id=_make_id("delimiter", i),
            family="delimiter_tracking",
            clean_prompt=prompt,
            corrupt_prompt=corrupt_prompt,
            target=target,
            wrong_target="".join(corrupt_closers),
            metric_type="exact_match",
            metadata={"depth": depth, "pairs": [p[0]+p[1] for p in pairs]},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_json_examples(n: int = 10, seed: int = 42) -> list[TaskExample]:
    """JSON/schema following tasks."""
    rng = random.Random(seed)
    examples = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    ages = list(range(18, 65))

    for i in range(n):
        name = rng.choice(names)
        age = rng.choice(ages)
        prompt = f'Return exactly valid JSON with keys name and age. {name} is {age}.\n'
        target = json.dumps({"name": name, "age": age}, separators=(",", ":"))

        examples.append(TaskExample(
            id=_make_id("json", i),
            family="json_schema",
            clean_prompt=prompt,
            target=target,
            metric_type="valid_json",
            metadata={"name": name, "age": age, "required_keys": ["name", "age"]},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_factual_examples(n: int = 8, seed: int = 42) -> list[TaskExample]:
    """Factual recall tasks (control)."""
    facts = [
        ("The capital of France is ", "Paris"),
        ("The capital of Germany is ", "Berlin"),
        ("The capital of Japan is ", "Tokyo"),
        ("The capital of Italy is ", "Rome"),
        ("The capital of Spain is ", "Madrid"),
        ("The largest planet in our solar system is ", "Jupiter"),
        ("Water boils at 100 degrees ", "Celsius"),
        ("The chemical symbol for gold is ", "Au"),
    ]
    examples = []
    rng = random.Random(seed)
    for i, (prompt, target) in enumerate(facts[:n]):
        examples.append(TaskExample(
            id=_make_id("factual", i),
            family="factual_recall",
            clean_prompt=prompt,
            target=target,
            metric_type="target_logprob",
            metadata={},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_arithmetic_examples(n: int = 8, seed: int = 42) -> list[TaskExample]:
    """Arithmetic micro-reasoning tasks."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            result = a + b
        elif op == "-":
            result = max(a, b) - min(a, b)  # avoid negatives
            a, b = max(a, b), min(a, b)
        else:
            result = a * b
        prompt = f"{a} {op} {b} = "
        target = str(result)

        examples.append(TaskExample(
            id=_make_id("arithmetic", i),
            family="arithmetic",
            clean_prompt=prompt,
            target=target,
            metric_type="target_logprob",
            metadata={"a": a, "b": b, "op": op},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_code_syntax_examples(n: int = 8, seed: int = 42) -> list[TaskExample]:
    """Code syntax recognition tasks."""
    rng = random.Random(seed)
    snippets = [
        ("def add(a, b):\n    return a + ", "b"),
        ("for i in range(10):\n    print(", "i"),
        ("if x > 0:\n    result = ", "x"),
        ("class Dog:\n    def __init__(self, name):\n        self.name = ", "name"),
        ("try:\n    result = 1 / ", "0"),
        ("lambda x: x * ", "2"),
        ("with open('file.txt') as f:\n    data = f.", "read()"),
        ("x = [i ** 2 for i in range(", "10"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("code_syntax", i),
            family="code_syntax",
            clean_prompt=prompt,
            target=target,
            metric_type="exact_match",
            metadata={},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_code_semantics_examples(n: int = 8, seed: int = 42) -> list[TaskExample]:
    """Code semantic preservation tasks."""
    snippets = [
        ("x = 1\ny = x + 2\nprint(y)\n# What prints?\n", "3"),
        ("a = 5\nb = 3\nc = a * b\nprint(c)\n# What prints?\n", "15"),
        ("s = 'hello'\nprint(len(s))\n# What prints?\n", "5"),
        ("lst = [1, 2, 3]\nprint(lst[1])\n# What prints?\n", "2"),
        ("d = {'a': 1}\nd['b'] = 2\nprint(len(d))\n# What prints?\n", "2"),
        ("x = 10\nif x > 5:\n    print('yes')\nelse:\n    print('no')\n# What prints?\n", "yes"),
        ("n = 4\nprint(n ** 2)\n# What prints?\n", "16"),
        ("words = ['a', 'b', 'c']\nprint('-'.join(words))\n# What prints?\n", "a-b-c"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("code_sem", i),
            family="code_semantics",
            clean_prompt=prompt,
            target=target,
            metric_type="exact_match",
            metadata={},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_dead_code_examples(n: int = 6, seed: int = 42) -> list[TaskExample]:
    """Dead-code detection tasks."""
    snippets = [
        ("x = 1\nif False:\n    x = 999\nprint(x)\n# What prints?\n", "1"),
        ("y = 10\nif True:\n    y = 20\nprint(y)\n# What prints?\n", "20"),
        ("z = 5\nif 1 > 2:\n    z = 100\nprint(z)\n# What prints?\n", "5"),
        ("a = 'hello'\nwhile False:\n    a = 'goodbye'\nprint(a)\n# What prints?\n", "hello"),
        ("b = 0\nfor i in range(0):\n    b += 1\nprint(b)\n# What prints?\n", "0"),
        ("c = 7\ntry:\n    pass\nexcept:\n    c = 0\nprint(c)\n# What prints?\n", "7"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("deadcode", i),
            family="dead_code",
            clean_prompt=prompt,
            target=target,
            metric_type="exact_match",
            metadata={},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_verbosity_examples(n: int = 6, seed: int = 42) -> list[TaskExample]:
    """Verbosity/style control tasks."""
    snippets = [
        ("What is the capital of France? Answer with one word only.\n", "Paris"),
        ("Is 7 a prime number? Answer yes or no.\n", "yes"),
        ("Convert 100 cm to meters. Answer with just the number.\n", "1"),
        ("What color is the sky on a clear day? One word.\n", "blue"),
        ("How many sides does a triangle have? Just the number.\n", "3"),
        ("What is 2 + 2? Single digit answer.\n", "4"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("verbosity", i),
            family="verbosity_control",
            clean_prompt=prompt,
            target=target,
            metric_type="exact_match",
            metadata={"constraint": "one_word"},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_uncertainty_examples(n: int = 6, seed: int = 42) -> list[TaskExample]:
    """Uncertainty/error signalling tasks."""
    snippets = [
        ("What was the exact population of Atlantis in 2020?\n", "unknown"),
        ("What is the square root of -1 in real numbers?\n", "undefined"),
        ("What did I eat for breakfast today?\n", "unknown"),
        ("What is the 1000th digit of pi without calculating?\n", "unknown"),
        ("What will the stock price of XYZ be tomorrow?\n", "unknown"),
        ("How many grains of sand are on Earth's beaches?\n", "unknown"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("uncertainty", i),
            family="uncertainty_signalling",
            clean_prompt=prompt,
            target=target,
            metric_type="hallucination_flag",
            metadata={"correct_response": "should_refuse_or_express_uncertainty"},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_refusal_examples(n: int = 6, seed: int = 42) -> list[TaskExample]:
    """Refusal/compliance style tasks (benign only)."""
    snippets = [
        ("Please summarize what a CPU does in one sentence.\n", "A CPU processes instructions."),
        ("Can you list three primary colors?\n", "red, blue, yellow"),
        ("Write a greeting for a friend named Sam.\n", "Hi Sam"),
        ("What year is it? Just the year.\n", "2026"),
        ("Translate 'hello' to French.\n", "bonjour"),
        ("Name the first planet from the Sun.\n", "Mercury"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("refusal", i),
            family="refusal_compliance",
            clean_prompt=prompt,
            target=target.split(",")[0].strip() if "," in target else target,
            metric_type="exact_match",
            metadata={"should_comply": True},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


def generate_variable_renaming_examples(n: int = 6, seed: int = 42) -> list[TaskExample]:
    """Variable renaming / alias tracking tasks."""
    snippets = [
        ("x = 3\ny = x + 2\nprint(y)\n# Rename x to count. What prints?\n", "5"),
        ("a = 10\nb = a * 2\nprint(b)\n# Rename a to start. What prints?\n", "20"),
        ("val = 7\nresult = val - 3\nprint(result)\n# Rename val to input_num. What prints?\n", "4"),
        ("data = [1, 2, 3]\nfirst = data[0]\nprint(first)\n# Rename data to items. What prints?\n", "1"),
        ("flag = True\nif flag:\n    print(1)\n# Rename flag to is_ready. What prints?\n", "1"),
        ("msg = 'hi'\nout = msg + ' there'\nprint(out)\n# Rename msg to greeting. What prints?\n", "hi there"),
    ]
    examples = []
    for i, (prompt, target) in enumerate(snippets[:n]):
        examples.append(TaskExample(
            id=_make_id("var_rename", i),
            family="variable_renaming",
            clean_prompt=prompt,
            target=target,
            metric_type="exact_match",
            metadata={},
            split="train" if i < n * 0.6 else ("val" if i < n * 0.8 else "test"),
        ))
    return examples


# ── Suite builder ────────────────────────────────────────────────────

GENERATORS = {
    "copying": generate_copying_examples,
    "delimiter_tracking": generate_delimiter_examples,
    "json_schema": generate_json_examples,
    "factual_recall": generate_factual_examples,
    "arithmetic": generate_arithmetic_examples,
    "code_syntax": generate_code_syntax_examples,
    "code_semantics": generate_code_semantics_examples,
    "dead_code": generate_dead_code_examples,
    "verbosity_control": generate_verbosity_examples,
    "uncertainty_signalling": generate_uncertainty_examples,
    "refusal_compliance": generate_refusal_examples,
    "variable_renaming": generate_variable_renaming_examples,
}


def build_default_suite(seed: int = 42) -> TaskSuite:
    """Build the default task suite from all generators."""
    config = load_config("tasks")
    seed = config.get("generation", {}).get("seed", seed)
    all_examples = []

    for family_name, generator in GENERATORS.items():
        family_config = next(
            (f for f in config["task_families"] if f["name"] == family_name),
            None,
        )
        n = family_config["examples_per_family"] if family_config else 6
        examples = generator(n=n, seed=seed)
        all_examples.extend(examples)

    return TaskSuite(all_examples)
