"""Exact enumeration of by-cluster sign/label permutations.

With ~10 emotion clusters the Monte-Carlo permutation test is both wasteful and
dangerous: 2000 draws from only C(10,5)=252 distinct arrangements re-samples each
arrangement ~8x, and any rounding mismatch between the observed statistic and the
re-computed permuted statistic can silently exclude the identity arrangement,
yielding p-values BELOW the mathematical floor (this exact failure produced the
retracted "p=0.0005" freeform keystones; the true floor at 10 clusters is 2/252).

Rules enforced here:
  * the observed statistic must be computed by the SAME function as the permuted
    ones (caller's responsibility - pass the stat function, we call it on the
    identity arrangement too);
  * enumeration is exact whenever the number of distinct arrangements is
    tractable (<= max_exact), else Monte-Carlo WITHOUT rounding and with the
    add-one estimator;
  * the identity arrangement always counts, so exact p >= 1/n_arrangements.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

MAX_EXACT = 20000


def n_distinct_permutations(values) -> int:
    """Number of distinct permutations of the multiset `values`."""
    c = Counter(values)
    n = math.factorial(sum(c.values()))
    for k in c.values():
        n //= math.factorial(k)
    return n


def distinct_permutations(values):
    """Yield every distinct permutation of the multiset `values` as a tuple.

    Standard lexicographic multiset-permutation recursion: at each depth try each
    distinct remaining value once. Cost is O(n) per emitted permutation.
    """
    items = sorted(Counter(values).items())  # [(value, remaining_count), ...]
    n = sum(c for _, c in items)
    perm = [None] * n

    def rec(depth):
        if depth == n:
            yield tuple(perm)
            return
        for i, (v, c) in enumerate(items):
            if c:
                items[i] = (v, c - 1)
                perm[depth] = v
                yield from rec(depth + 1)
                items[i] = (v, c)

    yield from rec(0)


def sign_permutation_pvalue(signs, stat_fn, n_perm: int = 2000, seed: int = 0,
                            max_exact: int = MAX_EXACT) -> dict:
    """Two-sided permutation p for |stat_fn(signs)| under permutations of `signs`.

    `signs` is the per-cluster label vector (e.g. one valence per emotion);
    `stat_fn(np.ndarray) -> float` recomputes the statistic for an arrangement.
    The observed value is stat_fn(identity) - never a rounded import from
    elsewhere. Exact enumeration when tractable, else add-one Monte-Carlo.
    """
    signs = np.asarray(signs, float)
    obs = float(stat_fn(signs))
    total = n_distinct_permutations(signs.tolist())
    if total <= max_exact:
        count = sum(1 for p in distinct_permutations(signs.tolist())
                    if abs(float(stat_fn(np.asarray(p)))) >= abs(obs) - 1e-9)
        return {"p_perm": count / total, "observed": obs, "exact": True,
                "n_arrangements": total, "floor": 1.0 / total}
    rng = np.random.default_rng(seed)
    count = sum(1 for _ in range(n_perm)
                if abs(float(stat_fn(rng.permutation(signs)))) >= abs(obs) - 1e-9)
    return {"p_perm": (1 + count) / (n_perm + 1), "observed": obs, "exact": False,
            "n_arrangements": n_perm, "floor": 1.0 / (n_perm + 1)}
