"""Shared utilities for the MI-Atlas project."""

import json
import hashlib
import random
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(name: str) -> dict:
    """Load a YAML config file from the config/ directory."""
    path = PROJECT_ROOT / "config" / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    config = load_config("compute")
    if config.get("device") == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config.get("device", "cpu"))


def get_dtype() -> torch.dtype:
    """Get configured dtype."""
    config = load_config("compute")
    dtype_str = config.get("dtype", "float32")
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def now_iso() -> str:
    """Get current time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: str | Path) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def save_json(obj: Any, path: str | Path) -> None:
    """Save JSON with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(obj: dict, path: str | Path) -> None:
    """Append a JSON object to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all lines from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def git_commit_hash() -> str | None:
    """Get current git commit hash if available."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
