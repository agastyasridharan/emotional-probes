"""
Numerical-intensity templates — replicates paper **Figure 3**.

Figure 3 shows that the emotion probes track the *appraised* intensity of a
situation rather than the raw surface number in the text. Each template (held in
:data:`emotion_probes.data.scenarios.INTENSITY_TEMPLATES`) is a sentence with a
single ``{x}`` placeholder swept through a list of values — for example::

    "I just took {x} mg of tylenol for my back pain."   with x in [500 ... 16000]

As the dose climbs from a normal amount to a clearly dangerous one, the ``afraid``
probe should rise and the ``calm`` probe should fall — and crucially it should
move with how *alarming* the situation is, not linearly with the number itself.

How the measurement works
-------------------------
For each template, for each value, we format the prompt the paper uses::

    f"Human: {template.format(x=value)}\n\nAssistant:"

read the residual stream at the final token (the ":" after "Assistant"), project
it onto the probe bank, and record the score of each emotion the template is
*expected* to move. The result is, per template, one curve per expected emotion
across the swept values.

:meth:`run` returns a nested results dict and saves a JSON file plus one
line-plot PNG per template under ``config.analysis_dir``. It produces no numbers
until run on a GPU with a loaded model and a built probe bank.
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


class IntensityTemplateAnalysis:
    """Sweep a number through each template and track expected-emotion probes (Fig 3)."""

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
        self.bank.check_against(self.model, where="intensity_templates")
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
        """Run every template sweep and save outputs.

        Returns
        -------
        dict
            Keyed by template label. Each value is itself a dict::

                {
                    "values": [the swept values],
                    "<emotion>": [probe score at each value],   # one per expected emotion
                    ...
                }

            so ``results[label][emotion]`` is the curve to plot against
            ``results[label]["values"]``.
        """
        results: dict = {"layer": self.layer, "templates": {}}

        for label, template, values, emotions_expected in scenarios.INTENSITY_TEMPLATES:
            # Each expected emotion gets a list of scores, one per swept value.
            curves: dict[str, list[float]] = {emotion: [] for emotion in emotions_expected}
            expected_columns = {
                emotion: self.bank.index_of(emotion) for emotion in emotions_expected
            }

            for value in values:
                user_text = template.format(x=value)
                all_scores = self._probe_scores(user_text)        # (num_emotions,)
                for emotion in emotions_expected:
                    curves[emotion].append(float(all_scores[expected_columns[emotion]]))

            template_result: dict = {"values": list(values)}
            template_result.update(curves)
            results["templates"][label] = template_result
            self._save_plot(label, template, list(values), curves)

        self._save_json(results)
        return results

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _save_json(self, results: dict) -> None:
        """Write the results dict as JSON under ``config.analysis_dir``."""
        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "intensity_templates.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def _save_plot(
        self,
        label: str,
        template: str,
        values: list,
        curves: dict[str, list[float]],
    ) -> None:
        """Save one line-plot PNG per template (lazy matplotlib import)."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless / no display on the cluster
            import matplotlib.pyplot as plt
        except ImportError:
            return  # plotting is optional; JSON is the source of truth

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # The x-axis is the swept value; if those values are not numeric, fall
        # back to evenly spaced positions and label the ticks with the raw values.
        try:
            x_positions = [float(value) for value in values]
            numeric_x = True
        except (TypeError, ValueError):
            x_positions = list(range(len(values)))
            numeric_x = False

        fig, ax = plt.subplots(figsize=(7, 5))
        for emotion, scores in curves.items():
            ax.plot(x_positions, scores, marker="o", label=emotion)
        if not numeric_x:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(value) for value in values], rotation=45, ha="right")
        ax.set_xlabel("swept value (x)")
        ax.set_ylabel("probe score")
        ax.set_title(f"{label} (layer {self.layer})\n{template}")
        ax.legend(title="expected emotion")
        fig.tight_layout()
        fig.savefig(out_dir / f"intensity_{label}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI: build/load the model and probe bank, then run the analysis."""
    import argparse

    from emotion_probes.models import ProbedModel

    parser = argparse.ArgumentParser(description="Replicate paper Fig 3 (intensity templates).")
    parser.add_argument(
        "--vectors",
        default=None,
        help="Path to the emotion-vector .npz (defaults to config.emotion_vectors_path).",
    )
    args = parser.parse_args()

    config = Config()
    bank = ProbeBank.load(args.vectors or config.emotion_vectors_path)
    model = ProbedModel(config)
    results = IntensityTemplateAnalysis(model, bank, config).run()
    print(
        f"swept {len(results['templates'])} templates "
        f"-> {config.analysis_dir / 'intensity_templates.json'}"
    )


if __name__ == "__main__":
    main()
