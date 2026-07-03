"""
Turn pairwise preferences into Elo ratings.

The paper's preference experiment asks the model, for every pair of activities,
"would you prefer A or B?", then summarises each activity with an **Elo** rating
(how strongly the model prefers it). This module implements that summary.

* :func:`compute_elo` — standard Elo from a list of (winner, loser) games.
* :func:`games_from_preferences` — build that game list from per-pair choices.

"Hard Elo" (used for the base model in the paper, where option logits are nearly
tied) is just :func:`compute_elo` applied to discrete win/loss games — there is
no separate function; you simply threshold the preference into a clear winner.
"""

from __future__ import annotations

import numpy as np


def compute_elo(
    num_items: int,
    games: list[tuple[int, int]],
    k: float = 32.0,
    epochs: int = 30,
    base_rating: float = 1500.0,
) -> np.ndarray:
    """Compute an Elo rating per item from pairwise games.

    Parameters
    ----------
    num_items:
        Number of items (e.g. activities) being rated.
    games:
        List of ``(winner_index, loser_index)`` tuples — one per decided pair.
    k:
        Elo K-factor (update step size).
    epochs:
        How many times to sweep through all games. Elo is order-sensitive, so we
        sweep several times for the ratings to settle (the game set is fixed, so
        this converges to a stable ranking).
    base_rating:
        Starting rating for every item (conventionally 1500).

    Returns
    -------
    np.ndarray
        Shape ``(num_items,)`` of Elo ratings.
    """
    ratings = np.full(num_items, float(base_rating), dtype=np.float64)
    for _ in range(epochs):
        for winner, loser in games:
            # Expected score for the winner under the logistic Elo model.
            expected_winner = 1.0 / (1.0 + 10.0 ** ((ratings[loser] - ratings[winner]) / 400.0))
            update = k * (1.0 - expected_winner)
            ratings[winner] += update
            ratings[loser] -= update
    return ratings


def games_from_preferences(
    pairs: list[tuple[int, int]],
    prefer_first: list[bool],
) -> list[tuple[int, int]]:
    """Convert per-pair choices into (winner, loser) games.

    Parameters
    ----------
    pairs:
        List of ``(a_index, b_index)`` activity pairs that were compared.
    prefer_first:
        For each pair, ``True`` if the model preferred ``a`` over ``b``.

    Returns
    -------
    list[tuple[int, int]]
        ``(winner, loser)`` for each pair.
    """
    if len(pairs) != len(prefer_first):
        raise ValueError("pairs and prefer_first must be the same length")
    games: list[tuple[int, int]] = []
    for (a, b), a_wins in zip(pairs, prefer_first):
        games.append((a, b) if a_wins else (b, a))
    return games
