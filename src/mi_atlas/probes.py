"""Simple probes for internal representations."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """Train a linear probe on activations.

    Args:
        activations: (n_samples, d_model) array
        labels: (n_samples,) array of labels
        n_folds: Cross-validation folds

    Returns:
        dict with accuracy, coefficients, and cross-val scores
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(clf, activations, labels, cv=n_folds)

    clf.fit(activations, labels)
    train_acc = clf.score(activations, labels)

    return {
        "train_accuracy": train_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist(),
        "n_classes": len(set(labels)),
        "n_samples": len(labels),
        "coefficients": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }


def extract_probe_activations(
    cache: dict[str, torch.Tensor],
    layer_name: str,
    position: str = "last",
) -> np.ndarray:
    """Extract activations at a layer for probing.

    Args:
        cache: Activation cache
        layer_name: Layer to extract from
        position: "last" for last token, "mean" for mean pooling

    Returns:
        numpy array of shape (n_samples, d_model)
    """
    if layer_name not in cache:
        raise KeyError(f"Layer {layer_name} not in cache")

    tensor = cache[layer_name]
    if isinstance(tensor, list):
        tensor = torch.cat(tensor, dim=0)

    if position == "last":
        return tensor[:, -1, :].numpy()
    elif position == "mean":
        return tensor.mean(dim=1).numpy()
    else:
        raise ValueError(f"Unknown position: {position}")
