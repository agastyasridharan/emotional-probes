"""
Tests for the visualiser's scoring layer (CPU-only, no torch / no Flask).

The Flask server and the model forward pass need a GPU, but the *scoring* logic —
recovering token strings from character offsets, mapping newlines to the ``<br>``
marker, projecting onto both probe banks, the RRF ranking and the cluster
aggregate — is pure NumPy. We drive it with a tiny fake model that yields
hand-built offsets and random activations, and with a synthetic probe bank.
"""

from __future__ import annotations

import json

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.visualisation.scoring import TokenActivationScorer
from emotion_probes.visualisation.server import VisualiserServer


class _FakeModel:
    """A stand-in for ProbedModel exposing only what the scorer uses."""

    def __init__(self, num_layers: int, hidden: int):
        self.num_layers = num_layers
        self.hidden_size = hidden
        self.config = Config()

    def iter_token_activations(self, texts, batch_size=1, layers=None, max_length=None, with_offsets=True):
        # Hand-built offsets for "I am happy\ntoday": a leading <bos> (empty span),
        # three word tokens, a newline-only token, then a final word.
        offsets = [(0, 0), (0, 1), (1, 4), (4, 10), (10, 11), (11, 16)]
        rng = np.random.RandomState(0)
        acts = rng.randn(len(layers), len(offsets), self.hidden_size).astype(np.float32)
        yield {"activations": acts, "seq_len": len(offsets), "layers": list(layers), "offset_mapping": offsets}


def _bank(num_layers: int, hidden: int) -> ProbeBank:
    emotions = ["happy", "sad", "calm", "angry", "afraid", "loving"]
    rng = np.random.RandomState(1)
    return ProbeBank(
        emotions=emotions,
        vectors=rng.randn(num_layers, len(emotions), hidden).astype(np.float32),
        global_means=rng.randn(num_layers, hidden).astype(np.float32),
    )


def test_scorer_recovers_tokens_and_marks_newlines() -> None:
    bank = _bank(4, 8)
    scorer = TokenActivationScorer(_FakeModel(4, 8), bank)
    result = scorer.score("I am happy\ntoday", layer=2, probe_mode="expression")
    # Leading <bos> (empty span) is dropped; the newline token becomes "<br>".
    assert result.tokens == ["I", " am", " happy", "<br>", "today"]


def test_scorer_scores_both_spaces_with_aligned_lengths() -> None:
    bank = _bank(4, 8)
    scorer = TokenActivationScorer(_FakeModel(4, 8), bank, deflection_bank=bank)
    result = scorer.score("I am happy\ntoday", layer=2)
    assert set(result.expr_emotions) == set(bank.emotions)
    assert result.defl_emotions is not None  # deflection bank supplied
    # one score per *display* token (5 after dropping <bos>)
    assert all(len(v) == 5 for v in result.expr_token_scores.values())
    assert len(result.ranked_emotions) == bank.num_emotions
    assert len(result.cluster_scores) == 10  # the 10 paper-style clusters


def test_scorer_disables_deflection_when_absent() -> None:
    bank = _bank(4, 8)
    scorer = TokenActivationScorer(_FakeModel(4, 8), bank, deflection_bank=None)
    result = scorer.score("I am happy\ntoday", layer=2)
    assert result.defl_emotions is None
    assert result.defl_token_scores is None


def test_server_render_fields_are_json_serialisable() -> None:
    bank = _bank(4, 8)
    scorer = TokenActivationScorer(_FakeModel(4, 8), bank, deflection_bank=bank)
    result = scorer.score("I am happy\ntoday", layer=2, probe_mode="expression")

    server = VisualiserServer.__new__(VisualiserServer)  # skip __init__ (no model load)
    server.config = Config()
    server.num_layers = 4
    server.default_layer = 2
    fields = server._render_fields(result, "expression")

    # Every JSON blob the template injects must parse.
    for key in ("all_scores_json", "emotions_json", "tokens_json", "cluster_scores_json",
                "expr_scores_json", "expr_emotions_json", "defl_scores_json", "defl_emotions_json"):
        json.loads(fields[key])
    # The initial selected emotion is the top-ranked one.
    assert fields["selected_emotion"] == result.ranked_emotions[0]
