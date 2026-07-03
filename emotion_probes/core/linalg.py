"""
Low-level linear algebra used throughout the project.

These four small functions are the entire mathematical core of "confound
removal" and "geometry":

* :func:`top_pcs_for_variance` — find the principal directions of some data.
* :func:`project_out`          — remove those directions from a vector.
* :func:`unit_normalize`       — scale a vector to length 1.
* :func:`cosine_similarity_matrix` — pairwise cosine similarity of a set of vectors.

All functions take and return NumPy arrays and do their internal arithmetic in
float64 for numerical stability (the matrices involved are small: emotion means
and a few thousand neutral samples).
"""

from __future__ import annotations

import numpy as np


def top_pcs_for_variance(x: np.ndarray, variance_threshold: float) -> np.ndarray:
    """Return the top principal components of ``x`` that together explain at
    least ``variance_threshold`` of the total variance.

    Parameters
    ----------
    x:
        Array of shape ``(n_samples, n_features)`` — rows are observations.
    variance_threshold:
        Fraction in ``(0, 1]``. We keep the fewest leading PCs whose cumulative
        explained-variance reaches this fraction.

    Returns
    -------
    np.ndarray
        Shape ``(k, n_features)`` with **orthonormal rows** (each row is one
        principal direction). ``k`` is the number of PCs needed.

    Notes
    -----
    We use the SVD of the mean-centred data, which is more numerically stable
    than eigendecomposing the covariance matrix. The singular values squared are
    proportional to the variance explained by each component.
    """
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (n_samples, n_features); got shape {x.shape}")

    x_centered = x - x.mean(axis=0, keepdims=True)
    # full_matrices=False gives the "economy" SVD: Vt has shape (min(N,D), D).
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    variance = s**2
    total = variance.sum()
    if total == 0:
        # Degenerate (all samples identical): no variance to remove.
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    cumulative = np.cumsum(variance) / total
    # Keep components until cumulative variance reaches the threshold.
    k = int(np.sum(cumulative < variance_threshold)) + 1
    k = min(k, vt.shape[0])
    return vt[:k]


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the subspace spanned by ``basis`` from ``v``.

    Computes ``v - (v · basisᵀ) · basis``, i.e. subtracts the component of ``v``
    that lies in the row-space of ``basis``. ``basis`` must have orthonormal rows
    (as returned by :func:`top_pcs_for_variance`).

    Works for a single vector ``v`` of shape ``(n_features,)`` or a stack of
    vectors of shape ``(n_vectors, n_features)``.
    """
    v = np.asarray(v, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape[0] == 0:
        return v  # nothing to remove
    # coeffs @ basis reconstructs the in-subspace component; subtract it.
    coeffs = v @ basis.T          # (..., k)
    return v - coeffs @ basis     # (..., n_features)


def unit_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale vectors to unit L2 length along ``axis``.

    Zero (or near-zero) vectors are left as zeros instead of producing NaNs.
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of a set of vectors.

    Parameters
    ----------
    vectors:
        Shape ``(n_vectors, n_features)``.

    Returns
    -------
    np.ndarray
        Symmetric ``(n_vectors, n_vectors)`` matrix; entry ``(i, j)`` is the
        cosine similarity between vectors ``i`` and ``j``. The diagonal is 1.
    """
    unit = unit_normalize(np.asarray(vectors, dtype=np.float64), axis=1)
    return unit @ unit.T
