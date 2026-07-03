"""
Choose (target, displayed) emotion pairs for the deflection dataset.

For each target emotion we want to MASK it behind a *genuinely different*
displayed emotion (so we don't get trivial pairs like "hiding angry, showing
furious"). We therefore:

1. take the emotion vectors at the analysis layer,
2. for each target, rank all other emotions by cosine similarity,
3. keep the most-DISSIMILAR ``dissimilar_fraction`` of them as a candidate pool,
4. randomly sample ``displayed_per_target`` displayed emotions from that pool.

Issue #5 (fixed here): the original ``select_pairs.py`` set the pool to the top
25% most dissimilar but its docstring/comments said "top 50% / top half". The
behaviour is unchanged (a quartile), but the name (``dissimilar_fraction``) and
the documentation now agree, and the value lives in :class:`Config`.

The analysis layer is taken from the config's ``layer_fraction`` (not a hard-coded
19), so it is consistent with the rest of the pipeline (Issue #4).
"""

from __future__ import annotations

import json
import random

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.linalg import cosine_similarity_matrix
from emotion_probes.core.probes import ProbeBank


def select_deflection_pairs(
    config: Config,
    bank: ProbeBank | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Return ``{target_emotion: [displayed_emotion, ...]}`` and write it to
    ``config.deflection_pairs_json``."""
    bank = bank or ProbeBank.load(config.emotion_vectors_path)
    layer = config.analysis_layer(bank.num_layers)
    emotions = bank.emotions
    n = len(emotions)

    sim = cosine_similarity_matrix(bank.layer_vectors(layer))  # (n, n)
    np.fill_diagonal(sim, np.inf)  # never pair an emotion with itself

    pool_size = int((n - 1) * config.dissimilar_fraction)
    if pool_size < config.displayed_per_target:
        raise ValueError(
            f"pool size {pool_size} < displayed_per_target {config.displayed_per_target}; "
            "increase dissimilar_fraction"
        )

    rng = random.Random(seed)
    pairs: dict[str, list[str]] = {}
    for i, target in enumerate(emotions):
        most_dissimilar = np.argsort(sim[i])[:pool_size]  # ascending similarity = most dissimilar first
        pool = [emotions[j] for j in most_dissimilar]
        pairs[target] = rng.sample(pool, config.displayed_per_target)

    config.deflection_pairs_json.parent.mkdir(parents=True, exist_ok=True)
    config.deflection_pairs_json.write_text(json.dumps(pairs, indent=2))
    total = sum(len(v) for v in pairs.values())
    print(
        f"Wrote {total} (target, displayed) pairs "
        f"({n} targets x {config.displayed_per_target}) to {config.deflection_pairs_json} "
        f"[layer {layer}, most-dissimilar {config.dissimilar_fraction:.0%}]"
    )
    return pairs


def main() -> None:
    select_deflection_pairs(Config())


if __name__ == "__main__":
    main()
