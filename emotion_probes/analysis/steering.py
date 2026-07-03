"""
:class:`SteeringEngine` — a thin, shared helper for activation steering.

Both the preference experiment (Fig 4) and the alignment behaviors (Part 3) work
by *adding an emotion vector to the residual stream* and watching how behavior
changes. This class wraps :meth:`ProbedModel.steer` so callers just say "steer
toward `desperate` at strength 0.05" without managing per-layer vectors or the
norm calibration.

Steering strength follows the paper's convention (footnote 3): it is a fraction
of the average residual-stream norm at the steered layer(s). Positive steers
toward the emotion; negative steers against it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Sequence

from emotion_probes.core.probes import ProbeBank

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel


class SteeringEngine:
    """Steer a model along emotion-vector directions from a :class:`ProbeBank`."""

    def __init__(
        self,
        model: "ProbedModel",
        bank: ProbeBank,
        layers: Sequence[int] | None = None,
    ):
        """
        Parameters
        ----------
        model: the loaded :class:`ProbedModel`.
        bank:  the emotion vectors to steer with.
        layers: which layer(s) to steer at. Defaults to the single analysis layer
                (``model.layer_index_for_fraction()``). The paper steers across a
                band of middle layers — pass a list to do the same.
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(model, where="steering")
        self.layers = list(layers) if layers is not None else [model.layer_index_for_fraction()]
        self._norms: dict[int, float] | None = None

    def calibrate(self, texts: Sequence[str]) -> dict[int, float]:
        """Measure the average residual norm per steered layer (needed for the
        norm-relative strength). Call once with a representative corpus.

        Uses ``skip=0``: steering prompts (a forced-choice question, a short
        scenario) are often shorter than ``config.token_skip`` (50, a
        story-extraction convention). With the default skip those prompts would be
        fully masked, the calibrated norm would be 0.0, and steering would silently
        add a zero vector. The steering norm should be taken over all real tokens."""
        norms = self.model.average_residual_norm(texts, self.layers, skip=0)
        if any(v <= 0.0 for v in norms.values()):
            raise RuntimeError(f"steering calibration produced a non-positive norm: {norms}")
        self._norms = norms
        return self._norms

    def set_norms(self, norm_by_layer: dict[int, float]) -> None:
        """Provide precomputed per-layer norms instead of calibrating."""
        self._norms = dict(norm_by_layer)

    def _require_norms(self) -> dict[int, float]:
        if self._norms is None:
            raise RuntimeError("call calibrate(texts) or set_norms(...) before steering")
        return self._norms

    @contextmanager
    def steer(self, emotion: str, strength: float, positions: str | Sequence[int] = "all"):
        """Context manager that steers toward ``emotion`` by ``strength``.

        ``strength`` is a fraction of the residual norm (negative to steer
        against). ``positions`` is ``"all"`` (every token, for generation) or a
        list of token indices (e.g. an activity's span)."""
        idx = self.bank.index_of(emotion)
        unit_vectors = {layer: self.bank.vectors[layer, idx] for layer in self.layers}
        with self.model.steer(unit_vectors, strength, self._require_norms(), positions=positions):
            yield
