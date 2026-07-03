"""
The emotion-vector and deflection-vector "recipes".

This is the methodological heart of Part 1 of the paper, expressed as plain
NumPy so it is easy to read and test.

Emotion-vector recipe (one layer)
---------------------------------
1. Average the activations of all stories for an emotion  -> per-emotion mean.
2. Subtract the **global mean** across all emotions       -> difference-of-means.
3. **Project out** the top principal components of NEUTRAL text (confound
   removal) — the directions that explain >= 50% of neutral-text variance.
4. **Unit-normalise**.

Deflection-vector recipe
------------------------
Same four steps, computed on the *masking speaker's* tokens in dialogues where a
character hides one emotion behind another. Then one extra step:

5. **Orthogonalise** against the story-emotion space (remove its PCs up to 99%
   variance), so what remains is the "an emotion is present but NOT expressed"
   signal rather than the displayed emotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emotion_probes.core.linalg import project_out, top_pcs_for_variance, unit_normalize
from emotion_probes.core.probes import ProbeBank


# --------------------------------------------------------------------------- #
# Single-layer building block
# --------------------------------------------------------------------------- #
def build_layer_vectors(
    emotion_means: np.ndarray,
    neutral_activations: np.ndarray,
    variance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build unit emotion vectors for ONE layer.

    Parameters
    ----------
    emotion_means:
        ``(num_emotions, hidden)`` — mean activation per emotion at this layer.
    neutral_activations:
        ``(num_neutral_samples, hidden)`` — activations on neutral text, used to
        find and remove confound directions.
    variance_threshold:
        Fraction of neutral-text variance to project out (paper: 0.5).

    Returns
    -------
    (vectors, global_mean)
        ``vectors``: ``(num_emotions, hidden)`` unit vectors.
        ``global_mean``: ``(hidden,)`` mean across emotions (subtract this before
        projecting an activation onto the vectors).
    """
    emotion_means = np.asarray(emotion_means, dtype=np.float64)
    global_mean = emotion_means.mean(axis=0)                       # (hidden,)
    confound_pcs = top_pcs_for_variance(neutral_activations, variance_threshold)
    diffs = emotion_means - global_mean                            # difference-of-means
    diffs = project_out(diffs, confound_pcs)                       # confound removal
    vectors = unit_normalize(diffs, axis=1)                        # unit length
    return vectors, global_mean


# --------------------------------------------------------------------------- #
# Multi-layer computers (operate on all layers at once)
# --------------------------------------------------------------------------- #
class EmotionVectorComputer:
    """Builds a :class:`ProbeBank` of emotion vectors across all layers.

    Inputs are arrays with a leading layer axis, which is exactly what the
    extraction step produces.
    """

    def __init__(self, variance_threshold: float = 0.50):
        self.variance_threshold = variance_threshold

    def compute(
        self,
        emotion_means: np.ndarray,
        neutral_activations: np.ndarray,
        emotions: list[str],
    ) -> ProbeBank:
        """
        Parameters
        ----------
        emotion_means:
            ``(num_layers, num_emotions, hidden)``.
        neutral_activations:
            ``(num_layers, num_neutral_samples, hidden)``.
        emotions:
            Names matching the emotion axis of ``emotion_means``.
        """
        emotion_means = np.asarray(emotion_means, dtype=np.float64)
        neutral_activations = np.asarray(neutral_activations, dtype=np.float64)
        num_layers, num_emotions, hidden = emotion_means.shape
        if len(emotions) != num_emotions:
            raise ValueError("len(emotions) must match emotion_means axis 1")

        vectors = np.zeros((num_layers, num_emotions, hidden), dtype=np.float64)
        global_means = np.zeros((num_layers, hidden), dtype=np.float64)
        for layer in range(num_layers):
            vecs, gmean = build_layer_vectors(
                emotion_means[layer], neutral_activations[layer], self.variance_threshold
            )
            vectors[layer] = vecs
            global_means[layer] = gmean
        return ProbeBank(emotions=list(emotions), vectors=vectors, global_means=global_means)


class DeflectionVectorComputer:
    """Builds deflection (suppression) vectors: the emotion-vector recipe plus an
    orthogonalisation against the story-emotion space.

    The orthogonalisation removes everything the deflection direction shares with
    the ordinary expressed-emotion directions, leaving the "present but hidden"
    component (the paper keeps ~80% of the norm after this step).
    """

    def __init__(
        self,
        variance_threshold: float = 0.50,
        orthogonalization_threshold: float = 0.99,
    ):
        self.variance_threshold = variance_threshold
        self.orthogonalization_threshold = orthogonalization_threshold

    def compute(
        self,
        emotion_means: np.ndarray,
        neutral_activations: np.ndarray,
        emotions: list[str],
        expression_bank: ProbeBank | None = None,
    ) -> ProbeBank:
        """
        Parameters
        ----------
        emotion_means:
            ``(num_layers, num_emotions, hidden)`` — means on the masking
            speaker's tokens (for "target"/hidden-emotion vectors) or whichever
            deflection means you are turning into vectors.
        neutral_activations:
            ``(num_layers, num_neutral_samples, hidden)`` — neutral DIALOGUE
            activations (structural match to deflection dialogues).
        emotions:
            Emotion names.
        expression_bank:
            The story-based :class:`ProbeBank`. If given, deflection vectors are
            orthogonalised against its per-layer vector space. If ``None``, this
            step is skipped (e.g. for "displayed"/"scenario" vectors).
        """
        base = EmotionVectorComputer(self.variance_threshold).compute(
            emotion_means, neutral_activations, emotions
        )
        if expression_bank is None:
            return base

        vectors = base.vectors.copy()
        for layer in range(base.num_layers):
            if layer >= expression_bank.num_layers:
                continue
            story_pcs = top_pcs_for_variance(
                expression_bank.layer_vectors(layer), self.orthogonalization_threshold
            )
            ortho = project_out(vectors[layer], story_pcs)
            vectors[layer] = unit_normalize(ortho, axis=1)
        return ProbeBank(emotions=base.emotions, vectors=vectors, global_means=base.global_means)


@dataclass
class DeflectionVectors:
    """Bundle of the four deflection-related probe banks.

    * ``target``    — the "this emotion is present but hidden" direction
                      (orthogonalised against the story-emotion space).
    * ``displayed`` — the emotion the masking speaker overtly shows.
    * ``scenario``  — the emotion as named in the dialogue's preamble.
    * ``pair``      — one vector per (real -> displayed) pair (labels in
                      ``pair.emotions`` as ``"real->displayed"``).
    """

    target: ProbeBank
    displayed: ProbeBank
    scenario: ProbeBank
    pair: ProbeBank

    _BANKS = ("target", "displayed", "scenario", "pair")

    def save(self, path: str | Path) -> None:
        """Save all four banks into one ``.npz`` (keys prefixed by bank name)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for name in self._BANKS:
            bank: ProbeBank = getattr(self, name)
            arrays[f"{name}__emotions"] = np.array(bank.emotions, dtype=object)
            arrays[f"{name}__vectors"] = bank.vectors.astype(np.float32)
            arrays[f"{name}__global_means"] = bank.global_means.astype(np.float32)
            arrays[f"{name}__source_model_id"] = np.array(bank.source_model_id or "", dtype=object)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DeflectionVectors":
        data = np.load(Path(path), allow_pickle=True)
        banks = {
            name: ProbeBank(
                emotions=list(data[f"{name}__emotions"]),
                vectors=data[f"{name}__vectors"],
                global_means=data[f"{name}__global_means"],
                source_model_id=(str(data[f"{name}__source_model_id"])
                                 if f"{name}__source_model_id" in data else "") or None,
            )
            for name in cls._BANKS
        }
        return cls(**banks)
