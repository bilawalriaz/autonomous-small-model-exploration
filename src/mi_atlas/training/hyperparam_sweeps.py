"""Hyperparameter sweep definitions."""

# Sweep plans for experiment family F

SWEEP_PLANS = {
    "learning_rate": {
        "param": "learning_rate",
        "values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "fixed": {"batch_size": 4, "max_steps": 500},
    },
    "batch_size": {
        "param": "batch_size",
        "values": [1, 2, 4, 8, 16],
        "fixed": {"learning_rate": 2e-4, "max_steps": 500},
    },
    "max_steps": {
        "param": "max_steps",
        "values": [100, 250, 500, 1000, 2000],
        "fixed": {"learning_rate": 2e-4, "batch_size": 4},
    },
    "lora_rank": {
        "param": "rank",
        "values": [1, 2, 4, 8, 16, 32],
        "fixed": {"learning_rate": 2e-4, "batch_size": 4, "max_steps": 500},
    },
}


def get_sweep_plan(name: str) -> dict:
    """Get a sweep plan by name."""
    if name not in SWEEP_PLANS:
        raise ValueError(f"Unknown sweep: {name}. Available: {list(SWEEP_PLANS.keys())}")
    return SWEEP_PLANS[name]
