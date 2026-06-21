"""Centered Kernel Alignment (CKA) for representation similarity."""

import torch
import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n_samples, d_features_a) activations from model A
        Y: (n_samples, d_features_b) activations from model B

    Returns:
        CKA similarity score (0 to 1)
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XTX = X.T @ X
    YTY = Y.T @ Y
    XTY = X.T @ Y

    hsic_xy = np.trace(XTY @ XTY.T)
    hsic_xx = np.trace(XTX @ XTX.T)
    hsic_yy = np.trace(YTY @ YTY.T)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


def cka_matrix(
    activations_a: dict[str, np.ndarray],
    activations_b: dict[str, np.ndarray],
    layer_names: list[str],
) -> np.ndarray:
    """Compute pairwise CKA between layers of two models.

    Returns matrix of shape (n_layers_a, n_layers_b).
    """
    n = len(layer_names)
    matrix = np.zeros((n, n))
    for i, name_a in enumerate(layer_names):
        for j, name_b in enumerate(layer_names):
            if name_a in activations_a and name_b in activations_b:
                a = activations_a[name_a]
                b = activations_b[name_b]
                if a.shape[0] == b.shape[0] and a.ndim >= 2:
                    # Take last token
                    a_flat = a[:, -1, :] if a.ndim > 2 else a
                    b_flat = b[:, -1, :] if b.ndim > 2 else b
                    matrix[i, j] = linear_cka(a_flat, b_flat)
    return matrix
