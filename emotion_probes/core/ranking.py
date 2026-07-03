"""
Ranking emotions within a span of text.

The visualiser answers "which emotions are most present in this text/selection?"
by fusing two signals per emotion:

* its **mean probe score** over the tokens (is it active on average?), and
* its **autocorrelation** across tokens (is the activation sustained/coherent,
  rather than a single noisy spike?).

Each emotion is ranked by both signals, and the two ranks are combined with
**reciprocal-rank-fusion (RRF)**. This module implements that exactly so the
visualiser and any analysis share one ranking definition.
"""

from __future__ import annotations

import numpy as np


def rank_descending(values: np.ndarray) -> np.ndarray:
    """1-based ranks of ``values`` sorted high-to-low (rank 1 = largest).

    Ties are broken by original order (stable sort).
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")     # indices of items, best first
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    return ranks


def reciprocal_rank_fusion(rank_arrays: list[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Fuse several 1-based rankings into one score per item.

    ``score_i = sum_over_rankings 1 / (k + rank_i)``. Higher is better. ``k=60``
    is the conventional RRF constant (and the one the visualiser used).
    """
    if not rank_arrays:
        raise ValueError("need at least one ranking")
    total = np.zeros_like(np.asarray(rank_arrays[0], dtype=np.float64))
    for ranks in rank_arrays:
        total = total + 1.0 / (k + np.asarray(ranks, dtype=np.float64))
    return total


def autocorrelation(scores: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each emotion's score sequence.

    Parameters
    ----------
    scores:
        ``(num_emotions, num_tokens)`` probe scores.

    Returns
    -------
    np.ndarray
        ``(num_emotions,)``. For each emotion: ``sum_t s[t]*s[t+1] / sum_t s[t]^2``
        (matches the visualiser). If there is only one token, falls back to the
        mean score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be (num_emotions, num_tokens)")
    if scores.shape[1] < 2:
        return scores.mean(axis=1)
    numerator = (scores[:, :-1] * scores[:, 1:]).sum(axis=1)
    denominator = np.maximum((scores**2).sum(axis=1), 1e-8)
    return numerator / denominator


def rank_emotions(scores: np.ndarray, k: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Rank emotions in a text span by RRF of (mean score, autocorrelation).

    Parameters
    ----------
    scores:
        ``(num_emotions, num_tokens)`` probe scores for the span.

    Returns
    -------
    (order, fused_scores)
        ``order``: emotion indices best-first.
        ``fused_scores``: the RRF score per emotion (same indexing as input).
    """
    scores = np.asarray(scores, dtype=np.float64)
    mean_scores = scores.mean(axis=1)
    ac = autocorrelation(scores)
    fused = reciprocal_rank_fusion([rank_descending(mean_scores), rank_descending(ac)], k=k)
    order = np.argsort(-fused, kind="stable")
    return order, fused
