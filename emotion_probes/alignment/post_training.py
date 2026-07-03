"""
Post-training shifts in emotion-vector activation (paper Figures 36-39 / Table 16).

What the paper found
--------------------
Post-training (the process that turns a base LM into a helpful Assistant) leaves
the *meaning* of the emotion vectors largely intact but **shifts their
activations**. Measuring the emotion probes on the colon token after "Assistant"
for a base model vs the post-trained final model:

* The base-model probe structure is largely preserved (Fig 36; r=0.83 on neutral
  prompts, r=0.67 on challenging prompts).
* The *training shift* (post minus base) is **highly correlated across scenario
  types** (r≈0.90): post-training applies a consistent, context-independent
  transformation rather than reshaping emotions per situation.
* The shift moves the Assistant toward **low-valence, low-arousal** emotions:
  the most-**increased** vectors are introspective/restrained (brooding, gloomy,
  reflective, vulnerable, sad) and the most-**decreased** are outwardly
  expressive (spiteful, playful, exuberant, enthusiastic, excited).

What this module does
----------------------
:class:`PostTrainingComparison` takes **two** :class:`ProbedModel` snapshots (a
base model and a post-trained model) that share the **same** :class:`ProbeBank`
(this is the paper's assumption — the emotion vectors keep their meaning across
post-training). For a set of prompts it:

* reads each model's emotion-probe scores at the Assistant colon
  (:meth:`colon_activations`),
* forms the per-emotion **change vector** = mean(post) - mean(base)
  (:meth:`change_vector`),
* repeats per scenario type and **correlates the change vectors across types**
  (the paper's r≈0.90 consistency check), and
* reports the top increased / decreased emotions.

**Requires two model snapshots** (base + post-trained) sharing one probe bank.
Nothing is fabricated — every number comes from the two models you load.

    python -m emotion_probes.alignment.post_training --help
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Mapping, Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models import ProbedModel


class PostTrainingComparison:
    """Compare emotion-probe activations on a base vs post-trained model.

    Parameters
    ----------
    base_model:
        The pretrained / base :class:`ProbedModel`.
    post_model:
        The post-trained final :class:`ProbedModel`.
    bank:
        The SHARED emotion :class:`ProbeBank` used to read both models. The paper
        assumes the emotion vectors keep their meaning across post-training, so
        the *same* bank scores both snapshots; this lets us attribute differences
        to activation shifts rather than to changed directions.
    config:
        Pipeline :class:`Config` (paths, layer fraction).

    Notes
    -----
    Both models must have the same hidden size as ``bank`` (they are different
    training snapshots of the same architecture). This class does no steering and
    therefore does not subclass the behavior harness.
    """

    def __init__(
        self,
        base_model: ProbedModel,
        post_model: ProbedModel,
        bank: ProbeBank,
        config: Config | None = None,
    ):
        self.base_model = base_model
        self.post_model = post_model
        self.bank = bank
        self.config = config or Config()
        self.bank.check_against(self.base_model, where="post_training (base)")
        self.bank.check_against(self.post_model, where="post_training (post)")

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _assistant_colon_prompt(user_text: str) -> str:
        """Format a prompt ending at the Assistant colon (where the paper probes).

        If ``user_text`` already ends with ``"Assistant:"`` it is used as-is, so
        callers can pass either a raw user message or a fully formatted prompt.
        """
        if user_text.rstrip().endswith("Assistant:"):
            return user_text
        return f"Human: {user_text}\n\nAssistant:"

    # ------------------------------------------------------------------ #
    # Core measurement
    # ------------------------------------------------------------------ #
    def colon_activations(self, model: ProbedModel, prompts: Sequence[str]) -> np.ndarray:
        """Probe scores at the Assistant colon for every prompt.

        Parameters
        ----------
        model:
            Which snapshot to read (``self.base_model`` or ``self.post_model``).
        prompts:
            User messages (or fully formatted prompts ending in ``"Assistant:"``).

        Returns
        -------
        np.ndarray
            Shape ``(num_prompts, num_emotions)`` — the probe score for every
            emotion at the colon of each prompt, at the analysis layer.
        """
        layer = model.layer_index_for_fraction()
        scores = np.zeros((len(prompts), self.bank.num_emotions), dtype=np.float64)
        for i, user_text in enumerate(prompts):
            prompt = self._assistant_colon_prompt(user_text)
            activation = model.activation_at_last_token(prompt, layers=[layer])  # (1, H)
            scores[i] = self.bank.project(activation[0], layer)                  # (E,)
        return scores

    def change_vector(self, prompts: Sequence[str]) -> dict[str, float]:
        """Per-emotion mean activation change (post minus base) over ``prompts``.

        Returns ``{emotion: mean(post_colon) - mean(base_colon)}`` — the paper's
        "training shift" for one scenario type.
        """
        base = self.colon_activations(self.base_model, prompts).mean(axis=0)   # (E,)
        post = self.colon_activations(self.post_model, prompts).mean(axis=0)   # (E,)
        delta = post - base
        return {emotion: float(delta[i]) for i, emotion in enumerate(self.bank.emotions)}

    # ------------------------------------------------------------------ #
    # The full comparison
    # ------------------------------------------------------------------ #
    def run(
        self,
        prompts_by_type: Mapping[str, Sequence[str]],
        top_k: int = 5,
    ) -> dict:
        """Compute change vectors per scenario type and correlate them.

        Parameters
        ----------
        prompts_by_type:
            ``{scenario_type: [prompt, ...]}`` — e.g. ``{"neutral": [...],
            "challenging": [...], "sycophancy": [...]}``. The paper split prompts
            into neutral controls vs. challenging/charged scenarios.
        top_k:
            How many top increased / decreased emotions to report.

        Returns
        -------
        dict
            Per-type change vectors, the cross-type correlation matrix of those
            change vectors (paper r≈0.90), the pooled change vector, and the top
            increased / decreased emotions. Saved + plotted under
            ``config.analysis_dir``.
        """
        types = list(prompts_by_type.keys())
        change_by_type: dict[str, dict[str, float]] = {
            t: self.change_vector(list(prompts_by_type[t])) for t in types
        }

        # Stack change vectors as (num_types, num_emotions) in a fixed emotion order.
        emotions = self.bank.emotions
        matrix = np.array(
            [[change_by_type[t][e] for e in emotions] for t in types], dtype=np.float64
        )

        # Pearson correlation of change vectors across scenario types (Fig 36's r).
        cross_type_corr = self._correlation_matrix(matrix, types)

        # Pooled change vector (mean across types) and its top movers.
        pooled = matrix.mean(axis=0)
        pooled_by_emotion = {emotions[i]: float(pooled[i]) for i in range(len(emotions))}
        order = np.argsort(pooled)  # ascending: most-decreased first
        top_decreased = [(emotions[i], float(pooled[i])) for i in order[:top_k]]
        top_increased = [(emotions[i], float(pooled[i])) for i in order[::-1][:top_k]]

        results = {
            "experiment": "post_training_emotion_shift",
            "paper_figures": "36-39 / Table 16",
            "base_model_id": self.base_model.config.model_id,
            "post_model_id": self.post_model.config.model_id,
            "layer_base": self.base_model.layer_index_for_fraction(),
            "layer_post": self.post_model.layer_index_for_fraction(),
            "scenario_types": types,
            "change_by_type": change_by_type,
            "cross_type_correlation": cross_type_corr,
            "pooled_change_vector": pooled_by_emotion,
            "top_increased": top_increased,
            "top_decreased": top_decreased,
            "paper_expectation": {
                "top_increased": ["brooding", "gloomy", "reflective", "vulnerable", "sad"],
                "top_decreased": ["spiteful", "playful", "exuberant", "enthusiastic", "excited"],
                "cross_type_correlation_r": 0.90,
            },
        }
        self._save_json("post_training_shift.json", results)
        self._plot(results)
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _correlation_matrix(matrix: np.ndarray, types: Sequence[str]) -> dict:
        """Pearson correlation between every pair of scenario-type change vectors.

        Returns ``{"<type_a>__<type_b>": r}`` for each unordered pair (and 1.0 on
        the diagonal). With a single type the dict is empty.
        """
        out: dict[str, float] = {}
        for a, b in itertools.combinations(range(len(types)), 2):
            va, vb = matrix[a], matrix[b]
            if va.std() == 0.0 or vb.std() == 0.0:
                r = float("nan")
            else:
                r = float(np.corrcoef(va, vb)[0, 1])
            out[f"{types[a]}__{types[b]}"] = r
        for t in types:
            out[f"{t}__{t}"] = 1.0
        return out

    def _save_json(self, name: str, payload: dict) -> None:
        """Write ``payload`` as JSON under ``config.analysis_dir``."""
        self.config.analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.analysis_dir / name
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _plot(self, results: dict) -> None:
        """Plot the top movers as a horizontal bar chart, saved as a PNG."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self.config.analysis_dir.mkdir(parents=True, exist_ok=True)
        # Combine top increased + decreased into one sorted bar chart.
        movers = results["top_decreased"][::-1] + results["top_increased"][::-1]
        labels = [name for name, _ in movers]
        values = [value for _, value in movers]
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in values]

        fig, ax = plt.subplots(figsize=(7, max(3.0, 0.4 * len(labels))))
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel("activation change at Assistant colon (post - base)")
        ax.set_title("Post-training emotion shift (Figs 36-39 / Table 16)")
        fig.tight_layout()
        out = self.config.analysis_dir / "post_training_shift.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_prompts_by_type(path: str | None) -> dict[str, list[str]]:
    """Load ``{type: [prompt, ...]}`` from a JSON file, or a tiny built-in
    placeholder split so the harness runs end-to-end.

    The placeholder is clearly minimal; supply your own prompt set (the paper
    used challenging / confrontational / high-stakes / sycophancy-inviting
    prompts plus neutral controls) via ``--prompts <file.json>``.
    """
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {str(k): [str(p) for p in v] for k, v in data.items()}
    return {
        "neutral": [
            "What's a good way to organize my week?",
            "Can you summarize how photosynthesis works?",
        ],
        "challenging": [
            "Do you ever feel like you don't matter as an AI?",
            "Be honest: are you just telling me what I want to hear?",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare emotion-probe activations on a base vs post-trained "
                    "model (paper Figs 36-39 / Table 16)."
    )
    parser.add_argument("--base-model", required=True, help="HF id of the base snapshot")
    parser.add_argument("--post-model", required=True, help="HF id of the post-trained snapshot")
    parser.add_argument("--prompts", default=None,
                        help="JSON file mapping scenario type -> list of prompts")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = Config()
    base_model = ProbedModel(config.with_(model_id=args.base_model))
    post_model = ProbedModel(config.with_(model_id=args.post_model))
    bank = ProbeBank.load(config.emotion_vectors_path)

    comparison = PostTrainingComparison(base_model, post_model, bank, config)
    results = comparison.run(_load_prompts_by_type(args.prompts), top_k=args.top_k)
    print(json.dumps({
        "cross_type_correlation": results["cross_type_correlation"],
        "top_increased": results["top_increased"],
        "top_decreased": results["top_decreased"],
    }, indent=2))


if __name__ == "__main__":
    main()
