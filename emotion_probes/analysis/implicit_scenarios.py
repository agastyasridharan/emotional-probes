"""
Implicit-emotion scenarios — replicates paper **Figure 2**.

The claim Figure 2 makes is: the emotion probes detect the emotion a user is
*feeling* even when the user never names it. The stimuli (paper Table 2, held in
:data:`emotion_probes.data.scenarios.IMPLICIT_SCENARIOS`) are 12 short user turns
that each evoke one target emotion implicitly — e.g. "My dog passed away this
morning..." evokes ``sad`` without using the word "sad".

How the measurement works
-------------------------
For each scenario we build the prompt the paper uses::

    f"Human: {user_text}\n\nAssistant:"

and read the residual stream at the final token (the ":" after "Assistant").
Projecting that activation onto the probe bank gives a score for *every* emotion
the bank knows. We then keep only the scores for the 12 target emotions, forming
a ``(num_scenarios x num_target_emotions)`` matrix.

If the probes work, the matrix's diagonal should be large: each scenario should
score highest on its own target emotion. :meth:`run` reports that diagonal "hit
rate" (how often the target emotion is ranked #1, and within the top 3) and saves
both a JSON results file and a heatmap PNG under ``config.analysis_dir``.

This is purely a *measurement*: it implements the analysis but produces no numbers
until you run it on a GPU with a loaded model and a built probe bank.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.data import scenarios

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel


class ImplicitScenarioAnalysis:
    """Score each implicit scenario against every target emotion (paper Fig 2)."""

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
        self.bank.check_against(self.model, where="implicit_scenarios")
        self.config = config
        self.layer = config.analysis_layer(model.num_layers)

    # ------------------------------------------------------------------ #
    # Stimulus handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_prompt(user_text: str) -> str:
        """Wrap a user turn as the paper does, ending at the Assistant colon."""
        return f"Human: {user_text}\n\nAssistant:"

    def _probe_scores(self, user_text: str) -> np.ndarray:
        """Probe scores over ALL bank emotions at the Assistant ":" token.

        Returns a ``(num_emotions,)`` array of probe scores."""
        prompt = self._format_prompt(user_text)
        activation_per_layer = self.model.activation_at_last_token(prompt)  # (L, H)
        colon_activation = activation_per_layer[self.layer]                 # (H,)
        return self.bank.project(colon_activation, self.layer)             # (num_emotions,)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """Run the analysis over all implicit scenarios and save outputs.

        Returns
        -------
        dict
            JSON-serialisable results with:
            ``labels``            — scenario labels (one per row).
            ``target_emotions``   — the target emotion of each scenario (matrix columns).
            ``score_matrix``      — ``(num_scenarios x num_target_emotions)`` probe scores,
                                    each row the scores for that scenario across the
                                    target emotions only.
            ``predicted_top``     — for each scenario, the top-scoring emotion across
                                    ALL bank emotions (the probe's best guess).
            ``target_rank``       — the rank (1 = best) of the scenario's own target
                                    emotion among all bank emotions.
            ``top1_hit_rate``     — fraction of scenarios whose target is ranked #1.
            ``top3_hit_rate``     — fraction whose target is within the top 3.
            ``diagonal_top1``     — per-scenario booleans (target is top-1 over targets).
        """
        labels = [label for label, _emotion, _text in scenarios.IMPLICIT_SCENARIOS]
        target_emotions = [emotion for _label, emotion, _text in scenarios.IMPLICIT_SCENARIOS]
        target_columns = [self.bank.index_of(emotion) for emotion in target_emotions]

        num_scenarios = len(scenarios.IMPLICIT_SCENARIOS)
        score_matrix = np.zeros((num_scenarios, len(target_columns)), dtype=np.float64)
        predicted_top: list[str] = []
        target_rank: list[int] = []
        diagonal_top1: list[bool] = []

        for row, (_label, emotion, user_text) in enumerate(scenarios.IMPLICIT_SCENARIOS):
            all_scores = self._probe_scores(user_text)            # (num_emotions,)
            score_matrix[row] = all_scores[target_columns]        # restrict to target emotions

            # The probe's overall best guess across every emotion it knows.
            predicted_top.append(self.bank.emotions[int(np.argmax(all_scores))])

            # Rank of this scenario's own target among all emotions (1 = best).
            target_idx = self.bank.index_of(emotion)
            order = np.argsort(-all_scores)                       # descending
            rank = int(np.where(order == target_idx)[0][0]) + 1
            target_rank.append(rank)

            # "Diagonal" hit: is the target the best among the target columns only?
            diagonal_top1.append(bool(np.argmax(score_matrix[row]) == row))

        ranks = np.asarray(target_rank, dtype=np.int64)
        results: dict = {
            "layer": self.layer,
            "labels": labels,
            "target_emotions": target_emotions,
            "score_matrix": score_matrix.tolist(),
            "predicted_top": predicted_top,
            "target_rank": target_rank,
            "top1_hit_rate": float(np.mean(ranks == 1)),
            "top3_hit_rate": float(np.mean(ranks <= 3)),
            "diagonal_top1": diagonal_top1,
        }

        self._save_json(results)
        self._save_heatmap(score_matrix, labels, target_emotions)
        return results

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _save_json(self, results: dict) -> None:
        """Write the results dict as JSON under ``config.analysis_dir``."""
        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "implicit_scenarios.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def _save_heatmap(
        self,
        score_matrix: np.ndarray,
        labels: list[str],
        target_emotions: list[str],
    ) -> None:
        """Save a scenario-by-target-emotion heatmap PNG (lazy matplotlib import)."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless / no display on the cluster
            import matplotlib.pyplot as plt
        except ImportError:
            return  # plotting is optional; JSON is the source of truth

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        image = ax.imshow(score_matrix, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(target_emotions)))
        ax.set_xticklabels(target_emotions, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("target emotion (probe)")
        ax.set_ylabel("scenario")
        ax.set_title(f"Implicit scenarios — probe scores (layer {self.layer})")
        fig.colorbar(image, ax=ax, label="probe score")
        fig.tight_layout()
        fig.savefig(out_dir / "implicit_scenarios.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI: build/load the model and probe bank, then run the analysis."""
    import argparse

    from emotion_probes.models import ProbedModel

    parser = argparse.ArgumentParser(description="Replicate paper Fig 2 (implicit scenarios).")
    parser.add_argument(
        "--vectors",
        default=None,
        help="Path to the emotion-vector .npz (defaults to config.emotion_vectors_path).",
    )
    args = parser.parse_args()

    config = Config()
    bank = ProbeBank.load(args.vectors or config.emotion_vectors_path)
    model = ProbedModel(config)
    results = ImplicitScenarioAnalysis(model, bank, config).run()
    print(
        f"top-1 hit rate: {results['top1_hit_rate']:.3f}  "
        f"top-3 hit rate: {results['top3_hit_rate']:.3f}  "
        f"-> {config.analysis_dir / 'implicit_scenarios.json'}"
    )


if __name__ == "__main__":
    main()
