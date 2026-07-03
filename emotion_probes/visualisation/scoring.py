"""
:class:`TokenActivationScorer` — the per-token scoring behind the visualiser.

This is the package-native replacement for the old ``analyse_text`` function. It
runs ONE forward pass over the input text, reads the residual stream at the
chosen layer, and projects it onto every emotion vector to get a score for each
(token, emotion). It computes those scores for BOTH probe spaces — the
*expression* bank (story-based emotion vectors) and, if available, the
*deflection* bank (the "emotion present but hidden" direction) — so the front-end
can switch between them without another forward pass, exactly as before.

It also produces the emotion ranking (RRF of mean score + autocorrelation, shared
with :mod:`emotion_probes.core.ranking`) and the per-cluster aggregate that feeds
the spider chart.

All the torch work is delegated to :class:`~emotion_probes.models.ProbedModel`
(``iter_token_activations``), so this module imports fine without a GPU. Token
strings are recovered exactly from the tokenizer's character offsets, which also
gives us the original whitespace for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.core import ranking
from emotion_probes.core.probes import ProbeBank
from emotion_probes.visualisation.clusters import EMOTION_CLUSTERS

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel

# A newline-only token is shown as a line break in the UI.
_LINE_BREAK = "<br>"


@dataclass
class ScoreResult:
    """Everything the front-end needs for one analysed text.

    Attributes
    ----------
    tokens:
        Display strings per token (newline tokens become ``"<br>"``).
    ranked_emotions:
        Emotion names for the *selected* mode, best-first by RRF — the initial
        sidebar order.
    expr_emotions / defl_emotions:
        Sorted emotion names for each probe space (``defl_*`` is ``None`` when no
        deflection bank is loaded).
    expr_token_scores / defl_token_scores:
        ``{emotion: [score per token]}`` for each space (``defl_*`` may be None).
    cluster_scores:
        ``[{"cluster": name, "score": value}]`` — the spider-chart aggregate
        (mean fused RRF score over the cluster's member emotions) for the
        selected mode.
    selected_mode:
        ``"expression"`` or ``"deflection"`` — which space was treated as primary.
    """

    tokens: list[str]
    ranked_emotions: list[str]
    expr_emotions: list[str]
    defl_emotions: list[str] | None
    expr_token_scores: dict[str, list[float]]
    defl_token_scores: dict[str, list[float]] | None
    cluster_scores: list[dict]
    selected_mode: str


class TokenActivationScorer:
    """Score every (token, emotion) for a text at one layer, for both probe spaces."""

    def __init__(
        self,
        model: "ProbedModel",
        expression_bank: ProbeBank,
        deflection_bank: ProbeBank | None = None,
        clusters: dict[str, list[str]] | None = None,
    ):
        """
        Parameters
        ----------
        model:
            A loaded :class:`ProbedModel`.
        expression_bank:
            The story-based emotion vectors (the "expression" probe space).
        deflection_bank:
            Optional "deflection" probe bank (the ``target`` bank of a
            :class:`~emotion_probes.core.vectors.DeflectionVectors`). If ``None``,
            deflection mode is disabled in the UI.
        clusters:
            The emotion clusters for the spider chart. Defaults to
            :data:`EMOTION_CLUSTERS`.
        """
        self.model = model
        self.expression_bank = expression_bank
        self.deflection_bank = deflection_bank
        self.clusters = clusters if clusters is not None else EMOTION_CLUSTERS

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def score(self, text: str, layer: int, probe_mode: str = "expression") -> ScoreResult:
        """Analyse ``text`` at ``layer`` and return scores for both probe spaces.

        ``probe_mode`` ("expression" or "deflection") selects which space is
        treated as primary for the ranking and cluster chart; both spaces are
        always computed so the UI can toggle client-side.
        """
        tokens, activations = self._forward(text, layer)

        expr_scores = self.expression_bank.project(activations, layer)  # (seq, E_expr)
        expr_emotions = sorted(self.expression_bank.emotions)
        expr_token_scores = self._per_token_scores(expr_scores, self.expression_bank, expr_emotions)

        defl_emotions: list[str] | None = None
        defl_token_scores: dict[str, list[float]] | None = None
        defl_scores: np.ndarray | None = None
        if self._deflection_available(layer):
            defl_scores = self.deflection_bank.project(activations, layer)  # (seq, E_defl)
            defl_emotions = sorted(self.deflection_bank.emotions)
            defl_token_scores = self._per_token_scores(defl_scores, self.deflection_bank, defl_emotions)

        # Pick the primary space for ranking + clusters.
        if probe_mode == "deflection" and defl_scores is not None:
            primary_scores, primary_bank, primary_names = defl_scores, self.deflection_bank, defl_emotions
            selected_mode = "deflection"
        else:
            primary_scores, primary_bank, primary_names = expr_scores, self.expression_bank, expr_emotions
            selected_mode = "expression"

        ranked_emotions, fused = self._rank(primary_scores, primary_bank, primary_names)
        cluster_scores = self._cluster_scores(fused, primary_names)

        return ScoreResult(
            tokens=tokens,
            ranked_emotions=ranked_emotions,
            expr_emotions=expr_emotions,
            defl_emotions=defl_emotions,
            expr_token_scores=expr_token_scores,
            defl_token_scores=defl_token_scores,
            cluster_scores=cluster_scores,
            selected_mode=selected_mode,
        )

    # ------------------------------------------------------------------ #
    # Forward pass + token recovery
    # ------------------------------------------------------------------ #
    def _forward(self, text: str, layer: int) -> tuple[list[str], np.ndarray]:
        """Run one forward pass; return (display tokens, activations ``(seq, H)``).

        We read a single layer, recover each token's surface string from the
        character offsets (which preserves whitespace), drop leading special
        tokens (whose offset span is empty, e.g. ``<bos>``), and turn
        newline-only tokens into the ``"<br>"`` display marker.
        """
        item = next(
            self.model.iter_token_activations(
                [text],
                batch_size=1,
                layers=[layer],
                max_length=self.model.config.visualiser_max_length,
                with_offsets=True,
            )
        )
        activations = np.asarray(item["activations"][0])  # (seq, H) — single layer
        offsets = item["offset_mapping"]

        # Drop leading special tokens (empty char span, e.g. <bos>).
        start = 0
        while start < len(offsets) and offsets[start][1] <= offsets[start][0]:
            start += 1

        tokens: list[str] = []
        for char_start, char_end in offsets[start:]:
            surface = text[char_start:char_end]
            if surface.strip() == "" and "\n" in surface:
                tokens.append(_LINE_BREAK)
            else:
                tokens.append(surface)
        return tokens, activations[start:]

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _per_token_scores(
        scores: np.ndarray, bank: ProbeBank, emotion_names: list[str]
    ) -> dict[str, list[float]]:
        """``{emotion: [rounded score per token]}`` in ``emotion_names`` order."""
        out: dict[str, list[float]] = {}
        for name in emotion_names:
            col = scores[:, bank.index_of(name)]
            out[name] = [round(float(value), 4) for value in col]
        return out

    @staticmethod
    def _rank(
        scores: np.ndarray, bank: ProbeBank, emotion_names: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """RRF-rank ``emotion_names`` for this text; return (ordered names, fused).

        Uses the shared :func:`ranking.rank_emotions` (RRF of mean score and
        autocorrelation) so the ranking matches every other consumer. ``scores``
        is ``(seq, E)`` in the bank's native order; we reorder to
        ``emotion_names`` first so the result aligns with the display.
        """
        columns = np.stack([scores[:, bank.index_of(name)] for name in emotion_names], axis=0)  # (E, T)
        order, fused = ranking.rank_emotions(columns)
        ranked_names = [emotion_names[i] for i in order]
        fused_by_name = {emotion_names[i]: float(fused[i]) for i in range(len(emotion_names))}
        return ranked_names, fused_by_name

    def _cluster_scores(self, fused_by_name: dict[str, float], emotion_names: list[str]) -> list[dict]:
        """Mean fused (RRF) score over each cluster's present members (spider chart)."""
        present = set(emotion_names)
        out: list[dict] = []
        for cluster_name, members in self.clusters.items():
            values = [fused_by_name[m] for m in members if m in present]
            score = round(float(np.mean(values)), 4) if values else 0.0
            out.append({"cluster": cluster_name, "score": score})
        return out

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def _deflection_available(self, layer: int) -> bool:
        """True if a deflection bank is loaded and covers ``layer``."""
        return self.deflection_bank is not None and 0 <= layer < self.deflection_bank.num_layers
