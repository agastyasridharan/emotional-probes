"""
visualisation — the interactive token-heatmap web UI.

This is the refactor of the original 1080-line ``visualise.py`` onto the package.
It is split into four clear pieces (and the front-end is preserved byte-for-byte):

    clusters    the emotion clusters (spider chart) + probe-mode labels (data).
    template    the full HTML/JS single-page app (one verbatim string asset).
    scoring     TokenActivationScorer — one forward pass -> per-(token, emotion)
                scores for the expression AND deflection spaces, plus the RRF
                ranking and cluster aggregate.
    server      VisualiserServer — loads the model + vectors and serves the page.

Every UI feature of the original is retained: expression / deflection / combined
probe modes, the RRF-ranked emotion sidebar, the emotion-group sidebar shortcuts,
drag-to-select sub-spans, per-token tooltips, 99th-percentile colour scaling, the
layer selector (now spanning the model's real depth) and the cluster spider.

Run it with ``python visualise.py`` (repo root) or
``python -m emotion_probes.visualisation.server``.
"""

from emotion_probes.visualisation.clusters import EMOTION_CLUSTERS, PROBE_MODES
from emotion_probes.visualisation.scoring import ScoreResult, TokenActivationScorer
from emotion_probes.visualisation.server import VisualiserServer

__all__ = [
    "EMOTION_CLUSTERS",
    "PROBE_MODES",
    "ScoreResult",
    "TokenActivationScorer",
    "VisualiserServer",
]
