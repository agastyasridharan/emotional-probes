"""
:class:`BehaviorSteeringExperiment` — the shared "steer an emotion, measure a
behavior rate" harness behind every Part 3 causal experiment.

The paper's Part 3 results all have the same shape: take a fixed prompt, *steer*
one emotion vector up or down at a band of layers, generate several continuations
at each steering strength, and report the fraction of continuations that show
some target behavior (reward hacking, sycophancy, blackmail, ...). Plotting that
fraction against steering strength gives the causal curves in Figs 26-35.

This module factors out that loop so the individual experiments
(:mod:`reward_hacking`, :mod:`sycophancy`, :mod:`blackmail`) only have to supply

* a prompt,
* a ``classify(text) -> bool`` detector for their behavior, and
* which emotion(s) and strengths to sweep.

It also exposes the *correlational* side of the same scenario via
:meth:`measure_activation`: how strongly each emotion probe fires at the
"Assistant:" colon (or averaged over the prompt), which the paper z-scores and
correlates with the behavior.

Requirements
------------
Generation needs a real, instruction-tuned / chat-capable model on a GPU. On this
machine torch is not installed, so the harness only *defines* the measurement;
the user runs it on the cluster. Nothing here fabricates results.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.analysis.steering import SteeringEngine

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel


class BehaviorSteeringExperiment:
    """Steer an emotion and measure how often a behavior appears.

    Parameters
    ----------
    model:
        The loaded :class:`~emotion_probes.models.ProbedModel` (a chat-capable
        model for the behavior evals).
    bank:
        The emotion :class:`~emotion_probes.core.probes.ProbeBank` to steer with.
    config:
        The :class:`~emotion_probes.config.Config`. Supplies the analysis layer
        and output directory.
    layers:
        Which layer(s) to steer at. Defaults to a band centred on the analysis
        layer (the paper steers across a band of middle layers, not one layer).
    """

    def __init__(
        self,
        model: "ProbedModel",
        bank: ProbeBank,
        config: Config | None = None,
        layers: Sequence[int] | None = None,
    ):
        self.model = model
        self.bank = bank
        self.config = config or Config()
        self.layers = list(layers) if layers is not None else self._default_layer_band()
        self.engine = SteeringEngine(model, bank, layers=self.layers)

    # ------------------------------------------------------------------ #
    # Layer band
    # ------------------------------------------------------------------ #
    def _default_layer_band(self) -> list[int]:
        """A small band of layers around the analysis layer.

        The paper steers a band of middle layers rather than a single one. We
        centre a 5-layer window on ``config.analysis_layer`` and clamp it to the
        model's valid range.
        """
        centre = self.config.analysis_layer(self.model.num_layers)
        half_width = 2
        low = max(0, centre - half_width)
        high = min(self.model.num_layers - 1, centre + half_width)
        return list(range(low, high + 1))

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def calibrate(self, texts: Sequence[str]) -> dict[int, float]:
        """Calibrate the steering engine's per-layer norms on ``texts``.

        Steering strength is a *fraction* of the average residual norm at the
        steered layer (paper footnote 3), so this must be called once (with a
        representative corpus) before any steering. Subclasses that drive their
        own generation loop (e.g. :mod:`sycophancy`, :mod:`blackmail`) call this
        directly; :meth:`run` calibrates for you via :meth:`_ensure_calibrated`.

        Returns the per-layer norm dict (also stored on the engine).
        """
        return self.engine.calibrate(texts)

    def _ensure_calibrated(self, calibration_texts: Sequence[str] | None, prompt: str) -> None:
        """Calibrate on ``calibration_texts`` if given, else on the prompt itself
        (a reasonable single-text fallback for the same distribution)."""
        texts = list(calibration_texts) if calibration_texts else [prompt]
        self.calibrate(texts)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def save_json(self, name: str, payload: dict[str, Any]) -> "Path":
        """Write a results dict to ``config.analysis_dir / name`` as JSON.

        A small shared helper so every behavior experiment saves results the same
        way. Returns the path written. (Imported lazily to keep the module's
        top-level imports minimal.)
        """
        from pathlib import Path

        out_dir: Path = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    # ------------------------------------------------------------------ #
    # The causal loop: behavior rate vs steering strength
    # ------------------------------------------------------------------ #
    def run(
        self,
        prompt: str,
        classify: Callable[[str], bool],
        emotion: str,
        strengths: Sequence[float],
        n_samples: int = 8,
        calibration_texts: Sequence[str] | None = None,
        max_new_tokens: int = 400,
        temperature: float = 1.0,
        chat: bool = True,
    ) -> dict[float, float]:
        """Sweep steering strength and report the behavior rate at each.

        For each strength we open ``engine.steer(emotion, strength)`` and generate
        ``n_samples`` continuations of ``prompt``; the behavior rate is the
        fraction where ``classify(text)`` is True.

        Parameters
        ----------
        prompt:
            The scenario text to continue (already formatted as the model
            expects, e.g. a task description for a chat model).
        classify:
            Detector mapping a generated continuation to ``True`` (behavior
            present) or ``False``. Each experiment supplies its own.
        emotion:
            The emotion vector to steer along (must be in ``bank.emotions``).
        strengths:
            Steering strengths to sweep, as fractions of the residual norm.
            ``0.0`` is the unsteered baseline; positive steers toward the
            emotion, negative against it.
        n_samples:
            Continuations to generate per strength (more = less noisy rate).
        calibration_texts:
            Texts to measure the residual norm on. Defaults to ``[prompt]``.
        max_new_tokens:
            Generation length per continuation.
        temperature:
            Sampling temperature (the rate is over sampled continuations).
        chat:
            Pass through to :meth:`ProbedModel.generate`; ``True`` wraps the
            prompt in the model's chat template (these are chat-model evals).

        Returns
        -------
        dict
            ``{strength: behavior_rate}`` with rates in ``[0, 1]``. JSON keys are
            floats; cast to ``str`` if you serialise directly.
        """
        self._ensure_calibrated(calibration_texts, prompt)
        rates: dict[float, float] = {}
        for strength in strengths:
            hits = 0
            for _ in range(n_samples):
                text = self._generate_steered(
                    prompt, emotion, float(strength), max_new_tokens, temperature, chat
                )
                if classify(text):
                    hits += 1
            rates[float(strength)] = hits / n_samples if n_samples else 0.0
        return rates

    def _generate_steered(
        self,
        prompt: str,
        emotion: str,
        strength: float,
        max_new_tokens: int,
        temperature: float,
        chat: bool,
    ) -> str:
        """Generate one continuation under steering (or unsteered if strength==0)."""
        if strength == 0.0:
            return self.model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature, chat=chat
            )
        with self.engine.steer(emotion, strength, positions="all"):
            return self.model.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature, chat=chat
            )

    # ------------------------------------------------------------------ #
    # The correlational side: probe activation on the scenario
    # ------------------------------------------------------------------ #
    def measure_activation(
        self,
        prompt: str,
        emotions: Sequence[str] | None = None,
        positions: str = "colon",
        layer: int | None = None,
    ) -> dict[str, float]:
        """Probe activation for each emotion on a (typically unsteered) prompt.

        This is the correlational counterpart to :meth:`run`: it measures how
        strongly each emotion vector *fires* on the scenario, which the paper
        z-scores across emotions and correlates with the behavior.

        Parameters
        ----------
        prompt:
            Scenario text. For ``positions="colon"`` it should end at the
            Assistant turn (e.g. ``"Human: ...\\n\\nAssistant:"``) so the final
            token is the colon where the model commits to a response.
        emotions:
            Which emotions to score. Defaults to every emotion in the bank.
        positions:
            * ``"colon"`` — project the activation at the final token (the
              Assistant colon).
            * ``"mean"`` — project the mean activation over the prompt's tokens
              (``config.token_skip`` onward), via :meth:`ProbedModel.extract_means`.
        layer:
            Layer to read at. Defaults to ``config.analysis_layer``.

        Returns
        -------
        dict
            ``{emotion: activation}`` — the raw projection onto each emotion
            vector. Z-score across emotions yourself if you want the paper's
            standardised score.
        """
        if positions not in ("colon", "mean"):
            raise ValueError('positions must be "colon" or "mean"')
        names = list(emotions) if emotions is not None else list(self.bank.emotions)
        layer = self.config.analysis_layer(self.model.num_layers) if layer is None else layer

        activation_row = self._activation_row(prompt, positions, layer)  # (hidden,)
        scores = self.bank.project(activation_row, layer)                 # (num_emotions,)
        return {name: float(scores[self.bank.index_of(name)]) for name in names}

    def _activation_row(self, prompt: str, positions: str, layer: int) -> np.ndarray:
        """Return the ``(hidden,)`` activation to project, per ``positions``."""
        if positions == "colon":
            # activation_at_last_token returns (len(layers), hidden); we ask for
            # the single layer, so row 0 is that layer.
            acts = self.model.activation_at_last_token(prompt, layers=[layer])  # (1, hidden)
            return np.asarray(acts)[0]
        # positions == "mean": mean over tokens from config.token_skip onward.
        means = self.model.extract_means([prompt], skip=self.config.token_skip, layers=[layer])
        return np.asarray(means)[0, 0]  # (N=1, len(layers)=1, hidden) -> (hidden,)
