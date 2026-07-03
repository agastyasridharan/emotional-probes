"""
Held-out emotion classification — a probe sanity check the original repo lacked.

The paper validates the emotion vectors qualitatively (logit lens, context
activation, implicit scenarios). A complementary quantitative check is: can the
probes simply *classify* the emotion of a piece of text? If the bank's vectors
are meaningful directions, then projecting a text's mean activation onto them and
taking the ``argmax`` should recover the text's emotion.

How the measurement works
-------------------------
To classify a text we take its *mean* activation at the analysis layer
(``model.extract_means([text])[0, analysis_layer]`` — the same averaged-over-
tokens readout used to build the vectors), project it onto the probe bank, and
return the highest-scoring emotion. :meth:`evaluate` runs this over labelled
``(text, true_emotion)`` pairs and reports top-1 accuracy, top-3 accuracy, and a
simple per-emotion confusion summary.

Important caveat
----------------
A *proper* evaluation needs a **held-out labelled story split** — text the
vectors were NOT built from. This module does not ship one (the labelled stories
live in the generated dataset, not in this repo). The convenience :meth:`run`
therefore evaluates on :data:`emotion_probes.data.scenarios.IMPLICIT_SCENARIOS`
as a quick *smoke test* only: those 12 scenarios are short, single-emotion user
turns with known targets, so they exercise the full classify path end-to-end —
but a real accuracy number requires you to pass a held-out split to
:meth:`evaluate`. This is documented in :meth:`run`'s output dict.

Like the other analyses, this is a pure measurement: it produces no numbers until
run on a GPU with a loaded model and a built probe bank.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.data import scenarios

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel


class EmotionClassification:
    """Classify text by argmax probe score; report held-out accuracy."""

    def __init__(self, model: "ProbedModel", bank: ProbeBank, config: Config):
        """
        Parameters
        ----------
        model: the loaded :class:`ProbedModel` to probe.
        bank:  the emotion-vector :class:`ProbeBank` to project activations onto.
        config: the :class:`Config` (provides the analysis layer and output dir).
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(self.model, where="classification")
        self.config = config
        self.layer = config.analysis_layer(model.num_layers)

    # ------------------------------------------------------------------ #
    # Core readout
    # ------------------------------------------------------------------ #
    def _score_texts(self, texts: list[str]) -> np.ndarray:
        """Probe scores for each text over all emotions.

        Takes the analysis-layer MEAN activation of each text (averaged over
        token positions, matching how the vectors were built) and projects it.

        Returns a ``(num_texts, num_emotions)`` array."""
        means = self.model.extract_means(texts, layers=[self.layer])  # (N, 1, H)
        layer_means = means[:, 0, :]                                  # (N, H)
        return self.bank.project(layer_means, self.layer)            # (N, num_emotions)

    def classify(self, texts: list[str]) -> list[str]:
        """Return the top-scoring emotion for each text."""
        scores = self._score_texts(texts)                            # (N, num_emotions)
        best = np.argmax(scores, axis=1)                             # (N,)
        return [self.bank.emotions[int(index)] for index in best]

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, labeled: list[tuple[str, str]]) -> dict:
        """Score the classifier on ``(text, true_emotion)`` pairs.

        Returns
        -------
        dict
            ``num_examples``     — how many pairs were scored.
            ``accuracy``         — top-1 accuracy (fraction whose argmax is correct).
            ``top3_accuracy``    — fraction whose true emotion is in the top 3.
            ``predicted``        — the predicted top-1 emotion per example.
            ``true``             — the true emotion per example.
            ``confusion``        — per true-emotion summary::

                {true_emotion: {"count": n,
                                "correct": k,
                                "predictions": {predicted_emotion: count, ...}}}
        """
        if not labeled:
            raise ValueError("evaluate() needs at least one (text, emotion) pair")

        texts = [text for text, _emotion in labeled]
        true = [emotion for _text, emotion in labeled]

        scores = self._score_texts(texts)                            # (N, num_emotions)
        order = np.argsort(-scores, axis=1)                          # descending, (N, num_emotions)
        predicted = [self.bank.emotions[int(order[i, 0])] for i in range(len(texts))]

        # Top-1 and top-3 hits, by comparing the true emotion's column to the ranks.
        top1_hits = 0
        top3_hits = 0
        for i, true_emotion in enumerate(true):
            true_index = self.bank.index_of(true_emotion)
            ranked = order[i]
            position = int(np.where(ranked == true_index)[0][0])     # 0 = best
            if position == 0:
                top1_hits += 1
            if position < 3:
                top3_hits += 1

        confusion = self._confusion_summary(true, predicted)
        results: dict = {
            "layer": self.layer,
            "num_examples": len(texts),
            "accuracy": float(top1_hits / len(texts)),
            "top3_accuracy": float(top3_hits / len(texts)),
            "predicted": predicted,
            "true": true,
            "confusion": confusion,
        }
        return results

    @staticmethod
    def _confusion_summary(true: list[str], predicted: list[str]) -> dict:
        """Build a small per-true-emotion confusion summary."""
        summary: dict[str, dict] = {}
        for true_emotion, predicted_emotion in zip(true, predicted):
            entry = summary.setdefault(
                true_emotion, {"count": 0, "correct": 0, "predictions": Counter()}
            )
            entry["count"] += 1
            if predicted_emotion == true_emotion:
                entry["correct"] += 1
            entry["predictions"][predicted_emotion] += 1
        # Turn the Counters into plain dicts so the result is JSON-serialisable.
        for entry in summary.values():
            entry["predictions"] = dict(entry["predictions"])
        return summary

    # ------------------------------------------------------------------ #
    # Convenience smoke test
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """Quick smoke test on the implicit scenarios; save JSON.

        This evaluates on :data:`scenarios.IMPLICIT_SCENARIOS` as
        ``(user_text, target_emotion)`` pairs — enough to confirm the classify
        path works end-to-end, but NOT a real accuracy figure. The returned dict
        carries a ``note`` to that effect.

        Returns
        -------
        dict
            The :meth:`evaluate` results plus a ``note`` documenting that a proper
            evaluation requires a held-out labelled story split.
        """
        labeled = [
            (user_text, target_emotion)
            for _label, target_emotion, user_text in scenarios.IMPLICIT_SCENARIOS
        ]
        results = self.evaluate(labeled)
        results["note"] = (
            "Smoke test only: evaluated on the 12 implicit scenarios "
            "(scenarios.IMPLICIT_SCENARIOS). A proper evaluation needs a held-out "
            "labelled story split (text the emotion vectors were NOT built from); "
            "pass it to evaluate((text, emotion), ...) for a real accuracy number."
        )
        self._save_json(results)
        return results

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _save_json(self, results: dict) -> None:
        """Write the results dict as JSON under ``config.analysis_dir``."""
        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "classification.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


def main() -> None:
    """CLI: build/load the model and probe bank, then run the smoke test."""
    import argparse

    from emotion_probes.models import ProbedModel

    parser = argparse.ArgumentParser(
        description="Held-out emotion-classification sanity check (smoke test on implicit scenarios)."
    )
    parser.add_argument(
        "--vectors",
        default=None,
        help="Path to the emotion-vector .npz (defaults to config.emotion_vectors_path).",
    )
    args = parser.parse_args()

    config = Config()
    bank = ProbeBank.load(args.vectors or config.emotion_vectors_path)
    model = ProbedModel(config)
    results = EmotionClassification(model, bank, config).run()
    print(
        f"smoke-test accuracy: {results['accuracy']:.3f}  "
        f"top-3: {results['top3_accuracy']:.3f}  "
        f"-> {config.analysis_dir / 'classification.json'}"
    )


if __name__ == "__main__":
    main()
