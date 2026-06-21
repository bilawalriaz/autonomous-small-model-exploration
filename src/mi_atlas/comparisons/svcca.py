"""Singular Vector Canonical Correlation Analysis (SVCCA)."""

import torch
import numpy as np


def svcca_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 20,
) -> float:
    """Compute SVCCA similarity between two representation matrices.

    Args:
        X: (n_samples, d_features_a) activations from model A
        Y: (n_samples, d_features_b) activations from model B
        n_components: Number of components to keep

    Returns:
        Mean canonical correlation (0 to 1)
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # SVD
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)

    # Truncate
    k = min(n_components, Ux.shape[1], Uy.shape[1])
    Ux = Ux[:, :k]
    Uy = Uy[:, :k]

    # CCA via SVD of cross-covariance
    M = Ux.T @ Uy
    _, S, _ = np.linalg.svd(M, full_matrices=False)

    return float(np.mean(S))
