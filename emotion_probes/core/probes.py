"""
:class:`ProbeBank` — a set of emotion vectors and the projection that reads them.

An "emotion probe" in the paper is just a linear projection of a model
activation onto a unit "emotion vector". A :class:`ProbeBank` stores those unit
vectors for every emotion at every layer, plus the global-mean activation that
is subtracted before projecting (this centring is what makes the projection a
meaningful *difference* from the average emotion, matching the recipe used to
build the vectors).

Scoring an activation ``a`` at layer ``L`` for emotion ``e`` is::

    score = (a - global_mean[L]) · vector[L, e]

This is exactly what the original ``visualise.py`` computed; it now lives in one
place so the visualiser, the analyses, and the steering code all agree.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints; never imports torch
    from emotion_probes.models.language_model import ProbedModel


@dataclass
class ProbeBank:
    """Unit emotion vectors for every (layer, emotion), plus per-layer means.

    Attributes
    ----------
    emotions:
        Ordered list of emotion names; index ``e`` matches row ``e`` of ``vectors``.
    vectors:
        Float array of shape ``(num_layers, num_emotions, hidden)``. Each
        ``vectors[L, e]`` is a unit vector.
    global_means:
        Float array of shape ``(num_layers, hidden)`` — the mean activation
        across emotions at each layer, subtracted before projecting.
    source_model_id:
        The ``Config.model_id`` of the model these vectors were BUILT from (or
        ``None`` if unknown). Stamped by the pipeline so a later analysis run can
        detect that a bank is being used with a *different* model — the one way to
        get silently wrong numbers (see :meth:`check_against`).
    """

    emotions: list[str]
    vectors: np.ndarray       # (num_layers, num_emotions, hidden)
    global_means: np.ndarray  # (num_layers, hidden)
    source_model_id: str | None = None

    # ---- shape helpers ----
    @property
    def num_layers(self) -> int:
        return self.vectors.shape[0]

    @property
    def num_emotions(self) -> int:
        return self.vectors.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.vectors.shape[2]

    def index_of(self, emotion: str) -> int:
        """Row index of an emotion (raises ``KeyError`` if unknown)."""
        try:
            return self.emotions.index(emotion)
        except ValueError as exc:
            raise KeyError(f"unknown emotion {emotion!r}") from exc

    def layer_vectors(self, layer: int) -> np.ndarray:
        """The ``(num_emotions, hidden)`` matrix of unit vectors at one layer."""
        return self.vectors[layer]

    # ---- the core readout ----
    def project(self, activations: np.ndarray, layer: int) -> np.ndarray:
        """Project activations onto every emotion vector at ``layer``.

        Parameters
        ----------
        activations:
            Array whose last axis is the hidden dimension. Shape ``(..., hidden)``
            — e.g. ``(hidden,)`` for one token, ``(num_tokens, hidden)`` for a
            sequence, or ``(batch, num_tokens, hidden)``.
        layer:
            Which layer's vectors/mean to use.

        Returns
        -------
        np.ndarray
            Shape ``activations.shape[:-1] + (num_emotions,)`` — the probe score
            for every emotion at every position.
        """
        activations = np.asarray(activations, dtype=np.float32)
        if activations.shape[-1] != self.hidden_size:
            raise ValueError(
                f"activations last dim {activations.shape[-1]} != hidden {self.hidden_size}"
            )
        centered = activations - self.global_means[layer].astype(np.float32)
        # (..., hidden) @ (hidden, num_emotions) -> (..., num_emotions)
        return centered @ self.layer_vectors(layer).astype(np.float32).T

    # ---- compatibility guard (so a bank is never silently used with the wrong model) ----
    def check_against(self, model: "ProbedModel", where: str = "") -> None:
        """Verify this bank matches a loaded model, or raise/warn.

        Emotion vectors are only meaningful for the *exact* model they were built
        from: a vector lives in that model's residual-stream basis, and layer ``L``
        means that model's layer ``L``. Reusing a bank with a different model is
        the one mistake that can produce **silently wrong** numbers — specifically
        when the two models share a hidden width but differ in depth (then
        ``project(.., layer)`` reads a real but *wrong* layer, with no shape error
        to catch it).

        This raises :class:`ValueError` on a hard mismatch (different number of
        layers or hidden size) and emits a :class:`UserWarning` on a softer
        mismatch (a different ``source_model_id``). Call it wherever a bank is
        first paired with a model.

        Parameters
        ----------
        model: the loaded :class:`ProbedModel` the bank will be read against.
        where: optional label for the error message (e.g. ``"visualiser"``).
        """
        context = f" ({where})" if where else ""
        problems: list[str] = []
        if self.num_layers != model.num_layers:
            problems.append(
                f"bank has {self.num_layers} layers but the model has {model.num_layers}"
            )
        if self.hidden_size != model.hidden_size:
            problems.append(
                f"bank hidden size {self.hidden_size} != model hidden size {model.hidden_size}"
            )
        if problems:
            raise ValueError(
                f"Probe bank is incompatible with the loaded model{context}: "
                + "; ".join(problems)
                + ". These vectors were built from a different model — rebuild them "
                "for this model (`python -m emotion_probes.pipeline emotion`) or load "
                "the model the vectors came from."
            )
        model_id = getattr(model.config, "model_id", None)
        if self.source_model_id and model_id and self.source_model_id != model_id:
            warnings.warn(
                f"Probe bank{context} was built from {self.source_model_id!r} but the "
                f"loaded model is {model_id!r}. Shapes match, so this will run, but the "
                "vectors only carry their intended meaning for the model they came "
                "from. Rebuild the vectors unless this swap is deliberate.",
                stacklevel=2,
            )

    # ---- persistence (portable .npz; no torch needed) ----
    def save(self, path: str | Path) -> None:
        """Save to a ``.npz`` file (portable, loads without torch)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            emotions=np.array(self.emotions, dtype=object),
            vectors=self.vectors.astype(np.float32),
            global_means=self.global_means.astype(np.float32),
            source_model_id=np.array(self.source_model_id or "", dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbeBank":
        """Load a :class:`ProbeBank` saved by :meth:`save`."""
        data = np.load(Path(path), allow_pickle=True)
        source = str(data["source_model_id"]) if "source_model_id" in data else ""
        return cls(
            emotions=list(data["emotions"]),
            vectors=data["vectors"],
            global_means=data["global_means"],
            source_model_id=source or None,
        )
