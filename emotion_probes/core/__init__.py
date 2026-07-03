"""
core — the pure-NumPy math of the project.

NOTHING in this package imports torch or needs a GPU. Everything operates on
plain NumPy arrays, so you can read it, reason about it, and unit-test it on a
laptop. The GPU-heavy work (running the model to PRODUCE activations) lives in
:mod:`emotion_probes.models`; once you have activations as NumPy arrays, every
algorithm in the paper is implemented here.

Modules
-------
linalg    Low-level linear algebra: principal components for a variance target,
          projecting a subspace out of a vector, unit-normalising, cosine matrices.
vectors   The emotion-vector "recipe" (difference-of-means + confound removal)
          and the deflection-vector recipe (+ orthogonalisation vs the story space).
probes    :class:`ProbeBank` — a set of unit emotion vectors per layer, plus the
          projection that turns activations into per-emotion "probe" scores.
elo       Turn pairwise preferences into Elo ratings (and "Hard Elo").
ranking   Reciprocal-rank-fusion + autocorrelation — how the visualiser ranks
          which emotions are most present in a span of text.
spans     Exact character->token span mapping (via tokenizer offsets) and masked
          means over token positions.
"""

from emotion_probes.core import elo, linalg, ranking, spans, vectors
from emotion_probes.core.probes import ProbeBank

__all__ = ["linalg", "vectors", "probes", "elo", "ranking", "spans", "ProbeBank"]
