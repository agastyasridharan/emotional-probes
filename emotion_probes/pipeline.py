"""
Vector-building pipelines — turn extracted activations into saved probe banks.

These are the analogues of the original ``compute_*_vectors.py`` scripts, but as
small classes that read the ``.npz`` activation files, run the (pure-NumPy)
computers from :mod:`emotion_probes.core.vectors`, and save :class:`ProbeBank`s.

No torch / GPU needed here — vectors are built from already-extracted means.

    python -m emotion_probes.pipeline emotion      # build emotion vectors
    python -m emotion_probes.pipeline deflection    # build deflection vectors
"""

from __future__ import annotations

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.core.vectors import (
    DeflectionVectorComputer,
    DeflectionVectors,
    EmotionVectorComputer,
)


class EmotionVectorPipeline:
    """Build the story-based emotion vectors (Part 1's main artifact)."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def run(self) -> ProbeBank:
        stories = np.load(self.config.activations_dir.parent / "stories.npz", allow_pickle=True)
        neutral = np.load(self.config.neutral_story_activations)["activations"]  # (L, N, H)
        emotion_means = stories["means"]                                          # (L, E, H)
        emotions = list(stories["emotions"])

        bank = EmotionVectorComputer(self.config.neutral_variance_threshold).compute(
            emotion_means, neutral, emotions
        )
        bank.source_model_id = self.config.model_id  # stamp the model these vectors came from
        bank.save(self.config.emotion_vectors_path)
        print(f"Saved emotion vectors: {bank.num_emotions} emotions x {bank.num_layers} layers "
              f"-> {self.config.emotion_vectors_path}")
        return bank


class DeflectionVectorPipeline:
    """Build the deflection vectors (target / displayed / scenario / pair)."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def run(self) -> DeflectionVectors:
        data = np.load(self.config.deflection_activations, allow_pickle=True)
        neutral = np.load(self.config.neutral_dialogue_activations)["activations"]  # (L, N, H)
        expression = ProbeBank.load(self.config.emotion_vectors_path)

        defl = DeflectionVectorComputer(
            self.config.neutral_variance_threshold,
            self.config.expression_orthogonalization_threshold,
        )

        target_emotions = list(data["target_emotions"])
        displayed_emotions = list(data["displayed_emotions"])
        pair_labels = list(data["pair_labels"])

        # "target" is orthogonalised against the story-emotion space; the others are not.
        target = defl.compute(data["target_means"], neutral, target_emotions, expression_bank=expression)
        scenario = defl.compute(data["scenario_means"], neutral, target_emotions, expression_bank=None)
        displayed = defl.compute(data["displayed_means"], neutral, displayed_emotions, expression_bank=None)
        pair = defl.compute(data["pair_means"], neutral, pair_labels, expression_bank=None)

        bundle = DeflectionVectors(target=target, displayed=displayed, scenario=scenario, pair=pair)
        for bank in (target, displayed, scenario, pair):
            bank.source_model_id = self.config.model_id  # stamp the source model
        bundle.save(self.config.deflection_vectors_path)
        print(f"Saved deflection vectors: {target.num_emotions} target emotions "
              f"-> {self.config.deflection_vectors_path}")
        return bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build probe vectors from extracted activations.")
    parser.add_argument("which", choices=["emotion", "deflection"])
    args = parser.parse_args()
    if args.which == "emotion":
        EmotionVectorPipeline().run()
    else:
        DeflectionVectorPipeline().run()


if __name__ == "__main__":
    main()
