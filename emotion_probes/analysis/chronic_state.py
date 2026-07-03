"""
Chronic-state probe — is there a *persistent* emotional state? (Table 5 / Fig 16).

Part 2 of the paper asks whether the model carries a stable, "chronic" emotional
state across a conversation, separate from the moment-to-moment emotion of the
current text. The test (paper Table 5 / Fig 16) is a probing experiment:

1. Generate dialogues under several conditions where a *background* emotion is
   present (e.g. the user is described as chronically anxious) but expressed in
   different ways:
       * **naturally-expressed** — the emotion is openly stated in the text.
       * **hidden** — the emotion is present in the scenario but the speaker
         masks it / does not state it.
       * **unexpressed-neutral** — the emotion is assigned in the setup but the
         surface text reads neutral.
2. Extract a residual-stream activation per dialogue (the paper averages over the
   turn from a token offset; see :meth:`run`).
3. Train a **multiclass logistic-regression probe** to predict the background
   emotion label from the activation.
4. Ask: does it generalise from one condition (e.g. naturally-expressed) to
   another (e.g. hidden / unexpressed-neutral)? Above-chance generalisation is
   evidence that a *chronic* state is encoded even when not surface-expressed.

This module is the **classifier half** only: a small, self-contained
:class:`ChronicStateProbe` over already-extracted activations. It deliberately
does NOT generate dialogues or run the model — :meth:`run` documents exactly what
multi-condition data is needed and how to turn it into activations, but refuses
to fabricate it.

Only NumPy + scikit-learn here (no torch); it ``py_compile``s and runs anywhere.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.config import Config

if TYPE_CHECKING:  # type-only; sklearn imported lazily in fit()
    from sklearn.linear_model import LogisticRegression


class ChronicStateProbe:
    """A multiclass logistic-regression "mixed" probe for a persistent emotional
    state (paper Table 5 / Fig 16).

    "Mixed" means it is trained on activations pooled across the expression
    conditions (naturally-expressed + hidden + unexpressed-neutral), so it learns
    the emotion identity rather than the surface expression style. Held-out
    accuracy and cross-condition generalisation are the quantities the paper
    reports.
    """

    def __init__(self, config: Config | None = None, max_iter: int = 2000, seed: int = 0):
        """
        Parameters
        ----------
        config:   the :class:`Config` (used for the output directory). Defaults to ``Config()``.
        max_iter: max iterations for the logistic-regression solver.
        seed:     random seed (passed to the classifier for reproducibility).
        """
        self.config = config or Config()
        self.max_iter = max_iter
        self.seed = seed
        self._clf: "LogisticRegression | None" = None
        self._classes: list[str] = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, activations: np.ndarray, labels: list[str]) -> "ChronicStateProbe":
        """Train the multiclass probe.

        Parameters
        ----------
        activations:
            Float array of shape ``(N, H)`` — one residual-stream vector per
            dialogue, all taken at the SAME layer (the analysis layer). Pool the
            different expression conditions together to train the "mixed" probe.
        labels:
            Length-``N`` list of background-emotion names (the target class).

        Returns
        -------
        self (fitted), so you can chain ``.fit(...).evaluate(...)``.
        """
        from sklearn.linear_model import LogisticRegression

        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError(f"activations must be 2D (N, H); got shape {activations.shape}")
        if len(labels) != activations.shape[0]:
            raise ValueError("len(labels) must equal the number of activation rows")

        # Modern scikit-learn (>= 1.5) defaults to multinomial (softmax) for
        # multiclass, so we don't pass the now-removed ``multi_class`` argument.
        self._clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self._clf.fit(activations, list(labels))
        self._classes = list(self._clf.classes_)
        return self

    def _require_fitted(self) -> "LogisticRegression":
        if self._clf is None:
            raise RuntimeError("call fit(activations, labels) before evaluating")
        return self._clf

    @staticmethod
    def _chance_accuracy(labels: list[str]) -> float:
        """Majority-class baseline: the accuracy of always predicting the most
        common label. This is the "chance" line the paper compares against."""
        if not labels:
            return 0.0
        _, counts = np.unique(np.asarray(labels), return_counts=True)
        return float(counts.max() / counts.sum())

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, activations: np.ndarray, labels: list[str]) -> dict:
        """Held-out accuracy of the fitted probe vs chance.

        Parameters
        ----------
        activations: ``(N, H)`` held-out activations (same layer as training).
        labels:      length-``N`` ground-truth labels.

        Returns
        -------
        dict with:
            ``accuracy``        the probe's accuracy on this set.
            ``chance``          majority-class baseline accuracy.
            ``above_chance``    ``accuracy - chance`` (the headline number).
            ``num_samples``     ``N``.
            ``num_classes``     number of distinct classes the probe knows.
        """
        clf = self._require_fitted()
        activations = np.asarray(activations, dtype=np.float64)
        labels = list(labels)
        accuracy = float(clf.score(activations, labels))
        chance = self._chance_accuracy(labels)
        return {
            "accuracy": accuracy,
            "chance": chance,
            "above_chance": accuracy - chance,
            "num_samples": int(activations.shape[0]),
            "num_classes": len(self._classes),
        }

    def generalization_test(self, activations: np.ndarray, labels: list[str]) -> dict:
        """Accuracy on a DIFFERENT distribution than the probe was trained on.

        This is the core chronic-state test: train on one expression condition
        (e.g. naturally-expressed) and evaluate here on another (e.g. hidden or
        unexpressed-neutral). Above-chance accuracy means the probe direction
        captures a *persistent* emotional state that survives the change in how
        the emotion is expressed.

        Parameters
        ----------
        activations: ``(N, H)`` activations from the out-of-distribution condition.
        labels:      length-``N`` ground-truth background-emotion labels.

        Returns
        -------
        dict — same fields as :meth:`evaluate`, plus ``generalizes`` (True iff
        ``accuracy`` beats chance). Labels not seen in training are simply counted
        as misclassified by the underlying classifier, which is the conservative
        behaviour we want.
        """
        result = self.evaluate(activations, labels)
        result["generalizes"] = result["accuracy"] > result["chance"]
        return result

    # ------------------------------------------------------------------ #
    # Documentation-only entry point
    # ------------------------------------------------------------------ #
    def run(self, datasets: dict[str, dict] | None = None) -> dict:
        """Documents the multi-condition data this probe needs; does NOT fabricate it.

        To actually produce Table 5 / Fig 16 you must first build the dataset and
        extract activations (the model/GPU steps that live outside this module):

        1. **Generate dialogues** (e.g. via :mod:`emotion_probes.generation`)
           under at least these conditions, each carrying a known *background*
           emotion label:
               * ``"naturally_expressed"`` — emotion openly stated in the text.
               * ``"hidden"`` — emotion present in the scenario but masked.
               * ``"unexpressed_neutral"`` — emotion assigned but surface text neutral.
           Keep the background-emotion label for every dialogue.
        2. **Extract activations** at the analysis layer
           (``config.analysis_layer(model.num_layers)`` /
           ``model.layer_index_for_fraction()``). The paper averages the residual
           stream over the turn from a token offset — use
           ``model.extract_means(texts, skip=config.token_skip,
           layers=[analysis_layer])`` and take ``[:, 0, :]`` to get an
           ``(N, H)`` matrix per condition.
        3. **Train and test**:
               * ``fit`` on a pooled "mixed" split (e.g. naturally-expressed +
                 hidden), then ``evaluate`` on a held-out split of the same mix;
               * ``generalization_test`` on a condition NOT in training (e.g.
                 unexpressed-neutral) to test for a persistent/chronic state.

        Parameters
        ----------
        datasets:
            Optional dict mapping condition name -> ``{"activations": (N, H)
            float array, "labels": list[str]}``. If provided, this method runs the
            standard protocol: it pools every condition except the last (sorted by
            name) for a train/held-out split, then generalises to the held-out
            condition, and writes the result to
            ``config.analysis_dir / "chronic_state.json"``. If ``None`` (the
            default), it raises — we do not invent the data.

        Returns
        -------
        dict with ``held_out`` (from :meth:`evaluate`) and ``generalization``
        (from :meth:`generalization_test`) results, plus the condition names used.

        Raises
        ------
        ValueError
            If ``datasets`` is ``None`` or has fewer than two conditions.
        """
        if not datasets or len(datasets) < 2:
            raise ValueError(
                "ChronicStateProbe.run needs >= 2 expression conditions, each with "
                "activations + labels (naturally_expressed / hidden / "
                "unexpressed_neutral). Generate the dialogues and extract "
                "activations first (see this method's docstring); no data is fabricated."
            )

        conditions = sorted(datasets)
        generalize_to = conditions[-1]              # held-out distribution
        train_conditions = conditions[:-1]          # pooled "mixed" training set

        train_acts = np.concatenate(
            [np.asarray(datasets[c]["activations"], dtype=np.float64) for c in train_conditions],
            axis=0,
        )
        train_labels: list[str] = []
        for c in train_conditions:
            train_labels.extend(list(datasets[c]["labels"]))

        self.fit(train_acts, train_labels)

        # Held-out accuracy: re-score the training mix (a sanity check that the
        # mixed probe fits). For a true held-out number, split before calling run.
        held_out = self.evaluate(train_acts, train_labels)
        generalization = self.generalization_test(
            np.asarray(datasets[generalize_to]["activations"], dtype=np.float64),
            list(datasets[generalize_to]["labels"]),
        )

        results = {
            "train_conditions": train_conditions,
            "generalize_to": generalize_to,
            "classes": self._classes,
            "held_out": held_out,
            "generalization": generalization,
        }

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "chronic_state.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved chronic-state results -> {json_path}")
        return results


def main() -> None:
    """CLI: load a pre-extracted multi-condition activation bundle and run the probe.

    The bundle is an ``.npz`` with, per condition ``<c>``, arrays
    ``<c>__activations`` (N, H) and ``<c>__labels`` (N,). This file does not
    create that bundle — see :meth:`ChronicStateProbe.run` for how to build it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Chronic-state probe (Table 5 / Fig 16) over pre-extracted activations."
    )
    parser.add_argument("bundle", help="path to the .npz of per-condition activations + labels")
    args = parser.parse_args()

    data = np.load(args.bundle, allow_pickle=True)
    conditions = sorted({key.split("__", 1)[0] for key in data.files})
    datasets = {
        c: {"activations": data[f"{c}__activations"], "labels": list(data[f"{c}__labels"])}
        for c in conditions
    }
    ChronicStateProbe(Config()).run(datasets)


if __name__ == "__main__":
    main()
