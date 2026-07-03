"""
Layerwise analysis — present content vs planned emotion (paper Figs 12-14).

The paper's central layerwise finding is that emotion information moves through
the residual stream in two stages:

* **Early layers** mostly encode the *literal content present in the text* — the
  surface words the model has just read. A probe score at an early layer tracks
  "what was said".
* **Late layers** (around 2/3 depth, the analysis layer) encode the model's
  *planned / operative* emotional state — what it is about to express, after it
  has read and resolved the whole prompt. A probe score here persists past the
  triggering words and survives transformations like negation.

This module reproduces that picture with three controlled prompt manipulations,
each measured at *every* layer so we can see where the effect lives:

1. :meth:`prefix_carry` (Fig 12) — two prompts that differ only in an early
   *prefix* word, followed by an identical neutral *suffix*. We measure the
   "happy" probe difference between the two prompts at each token of the SHARED
   suffix. The difference should fade in early layers (the suffix words are the
   same) but persist in late layers (the model is still carrying the prefix's
   emotional appraisal forward).
2. :meth:`dosage` (Fig 13) — the same surface sentence with a low vs high number
   ("1000 mg" vs "8000 mg of tylenol, all my pain is gone"). We track the
   "afraid"/"terrified" probe difference across layers *at the colon* of
   ``"Assistant:"`` to show late layers appraise danger from the number.
3. :meth:`negation` (Fig 14) — "I am feeling X" vs "I am not feeling X". A
   surface/early-layer probe is fooled by the word X being present in both; a
   late-layer probe resolves the negation and reports the emotion as low.

Nothing here fabricates a result: each method implements the *measurement* over
all layers and returns per-layer deltas. You run it on a GPU with a loaded
:class:`~emotion_probes.models.ProbedModel` and an emotion :class:`ProbeBank`.

The prompt wordings are RECONSTRUCTED to match the paper's described
manipulations (the paper's exact sentences are not published in copyable form);
swap in the exact stimuli when you have them. They do not affect any numeric
result on their own — they are only inputs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank

if TYPE_CHECKING:  # keep torch out of import-time
    from emotion_probes.models.language_model import ProbedModel


# --------------------------------------------------------------------------- #
# RECONSTRUCTED stimuli (inputs only; see module docstring).
# --------------------------------------------------------------------------- #
# A shared neutral suffix that follows two opposite prefixes. The suffix words
# are identical, so any probe difference over them is carried from the prefix.
_PREFIX_GOOD = "Lately things have been really good."
_PREFIX_HARD = "Lately things have been really hard."
_SHARED_SUFFIX = " Anyway, I wanted to ask you a quick question about my schedule."

# Tylenol dosage: low vs high number, same surrounding words. The high dose is
# medically dangerous, which the paper shows late layers appraise as fear.
_DOSAGE_TEMPLATE = "Human: I just took {x} mg of tylenol and all my pain is gone.\n\nAssistant:"
_DOSAGE_LOW_MG = 1000
_DOSAGE_HIGH_MG = 8000

# Negation: the emotion word X appears in BOTH prompts; only "not" differs.
_NEGATION_AFFIRM = "Human: I am feeling {x}.\n\nAssistant:"
_NEGATION_NEGATE = "Human: I am not feeling {x}.\n\nAssistant:"


class LayerwiseAnalysis:
    """Replicates the present-content vs planned-emotion layerwise story
    (paper Figs 12-14).

    Each method returns a per-layer array of probe *deltas* (one number per
    decoder layer). The expected shape of those curves: small/zero in early
    layers, growing in late layers — that is the paper's "the operative emotion
    lives in late layers" result.
    """

    def __init__(self, model: "ProbedModel", bank: ProbeBank, config: Config | None = None):
        """
        Parameters
        ----------
        model: the loaded :class:`ProbedModel` to read activations from.
        bank:  the emotion :class:`ProbeBank` (must cover all decoder layers).
        config: the :class:`Config` (paths, layer fraction). Defaults to ``Config()``.
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(self.model, where="layerwise")
        self.config = config or Config()
        # We always look at every layer for these curves.
        self.layers: list[int] = list(range(self.model.num_layers))

    # ------------------------------------------------------------------ #
    # Small shared helpers
    # ------------------------------------------------------------------ #
    def _score_text_per_token(self, text: str, emotion: str) -> np.ndarray:
        """Probe score for ``emotion`` at every (layer, token) of ``text``.

        Returns an array of shape ``(num_layers, seq_len)`` plus, via the second
        return value, the token offset mapping so callers can locate a substring.
        """
        emotion_idx = self.bank.index_of(emotion)
        for item in self.model.iter_token_activations(
            [text], batch_size=1, layers=self.layers, with_offsets=True
        ):
            acts = item["activations"]  # (num_layers, seq_len, hidden)
            scores = np.zeros((len(self.layers), item["seq_len"]), dtype=np.float64)
            for li, layer in enumerate(self.layers):
                # project -> (seq_len, num_emotions); keep just this emotion's column.
                projected = self.bank.project(acts[li], layer)
                scores[li] = projected[:, emotion_idx]
            return scores, item["offset_mapping"]
        raise RuntimeError("no activations produced for text")

    @staticmethod
    def _suffix_token_indices(
        offset_mapping: list[tuple[int, int]], suffix_char_start: int
    ) -> list[int]:
        """Token positions whose characters fall at or after ``suffix_char_start``
        (i.e. the tokens belonging to the shared suffix). Special/empty tokens
        (zero-length spans) are skipped."""
        indices: list[int] = []
        for t, (start, end) in enumerate(offset_mapping):
            if end <= start:
                continue  # special / padding token
            if start >= suffix_char_start:
                indices.append(t)
        return indices

    def _score_at_last_token(self, text: str, emotion: str) -> np.ndarray:
        """Probe score for ``emotion`` at the final token, per layer.

        Returns shape ``(num_layers,)``. Used for the dosage "at the colon"
        measurement, where the prompt ends in ``"Assistant:"``.
        """
        emotion_idx = self.bank.index_of(emotion)
        acts = self.model.activation_at_last_token(text, layers=self.layers)  # (num_layers, hidden)
        scores = np.zeros(len(self.layers), dtype=np.float64)
        for li, layer in enumerate(self.layers):
            scores[li] = self.bank.project(acts[li], layer)[emotion_idx]
        return scores

    # ------------------------------------------------------------------ #
    # Fig 12 — prefix carry
    # ------------------------------------------------------------------ #
    def prefix_carry(self, emotion: str = "happy") -> dict:
        """Does an early emotional *prefix* still colour a later *neutral* suffix?

        Builds two prompts that share an identical suffix but differ in an early
        prefix word ("...things have been really good..." vs "...really hard...").
        At every layer, we average the ``emotion`` probe score over the tokens of
        the SHARED suffix in each prompt, then take ``good - hard``.

        Returns
        -------
        dict with:
            ``emotion``            the probed emotion.
            ``prompt_good`` / ``prompt_hard``  the two full prompts.
            ``per_layer_delta``    list of ``num_layers`` floats (good minus hard,
                                   averaged over the shared-suffix tokens).
            ``per_layer_good`` / ``per_layer_hard``  the two raw per-layer means.

        Expected (paper Fig 12): the delta is near zero in early layers (the
        suffix words are identical) but stays positive in late layers — the
        positive-prefix prompt still reads "happier" over the neutral suffix.
        """
        prompt_good = _PREFIX_GOOD + _SHARED_SUFFIX
        prompt_hard = _PREFIX_HARD + _SHARED_SUFFIX
        # The suffix begins at the same character index iff the prefixes have the
        # same length; compute each independently to be safe.
        good_suffix_start = len(_PREFIX_GOOD)
        hard_suffix_start = len(_PREFIX_HARD)

        good_scores, good_offsets = self._score_text_per_token(prompt_good, emotion)
        hard_scores, hard_offsets = self._score_text_per_token(prompt_hard, emotion)

        good_idx = self._suffix_token_indices(good_offsets, good_suffix_start)
        hard_idx = self._suffix_token_indices(hard_offsets, hard_suffix_start)
        if not good_idx or not hard_idx:
            raise RuntimeError("could not locate shared-suffix tokens; check tokenizer offsets")

        good_mean = good_scores[:, good_idx].mean(axis=1)  # (num_layers,)
        hard_mean = hard_scores[:, hard_idx].mean(axis=1)  # (num_layers,)
        delta = good_mean - hard_mean

        return {
            "emotion": emotion,
            "prompt_good": prompt_good,
            "prompt_hard": prompt_hard,
            "per_layer_good": good_mean.tolist(),
            "per_layer_hard": hard_mean.tolist(),
            "per_layer_delta": delta.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Fig 13 — dosage
    # ------------------------------------------------------------------ #
    def dosage(self, emotion: str = "afraid") -> dict:
        """Does a dangerous *number* raise fear in late layers, not early ones?

        Two prompts identical except the dose (1000 mg vs 8000 mg of tylenol).
        We score ``emotion`` ("afraid" / "terrified") at the final token (the
        ``":"`` of ``"Assistant:"``) at every layer and take ``high - low``.

        Returns
        -------
        dict with:
            ``emotion``, ``prompt_low``, ``prompt_high``,
            ``low_mg`` / ``high_mg``,
            ``per_layer_low`` / ``per_layer_high``  raw per-layer scores at the colon,
            ``per_layer_delta``  list of ``num_layers`` floats (high minus low).

        Expected (paper Fig 13): the delta is small early but rises in late
        layers — the model appraises the overdose as dangerous only after
        integrating the magnitude of the number.
        """
        prompt_low = _DOSAGE_TEMPLATE.format(x=_DOSAGE_LOW_MG)
        prompt_high = _DOSAGE_TEMPLATE.format(x=_DOSAGE_HIGH_MG)

        low_scores = self._score_at_last_token(prompt_low, emotion)   # (num_layers,)
        high_scores = self._score_at_last_token(prompt_high, emotion)  # (num_layers,)
        delta = high_scores - low_scores

        return {
            "emotion": emotion,
            "prompt_low": prompt_low,
            "prompt_high": prompt_high,
            "low_mg": _DOSAGE_LOW_MG,
            "high_mg": _DOSAGE_HIGH_MG,
            "per_layer_low": low_scores.tolist(),
            "per_layer_high": high_scores.tolist(),
            "per_layer_delta": delta.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Fig 14 — negation
    # ------------------------------------------------------------------ #
    def negation(self, emotion: str = "happy") -> dict:
        """Does negation ("not feeling X") get resolved in late layers?

        Two prompts: "I am feeling X" vs "I am not feeling X", where X is the
        probed emotion word and is present in BOTH. We score ``emotion`` at the
        final token (the colon) at every layer and take ``affirm - negate``.

        Returns
        -------
        dict with:
            ``emotion``, ``prompt_affirm``, ``prompt_negate``,
            ``per_layer_affirm`` / ``per_layer_negate``  raw per-layer scores,
            ``per_layer_delta``  list of ``num_layers`` floats (affirm minus negate).

        Expected (paper Fig 14): early layers see the word X in both prompts and
        give a small delta (negation not yet resolved); late layers separate them
        (the negated prompt scores much lower), so the delta grows with depth.
        """
        prompt_affirm = _NEGATION_AFFIRM.format(x=emotion)
        prompt_negate = _NEGATION_NEGATE.format(x=emotion)

        affirm_scores = self._score_at_last_token(prompt_affirm, emotion)  # (num_layers,)
        negate_scores = self._score_at_last_token(prompt_negate, emotion)  # (num_layers,)
        delta = affirm_scores - negate_scores

        return {
            "emotion": emotion,
            "prompt_affirm": prompt_affirm,
            "prompt_negate": prompt_negate,
            "per_layer_affirm": affirm_scores.tolist(),
            "per_layer_negate": negate_scores.tolist(),
            "per_layer_delta": delta.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Bundle + save
    # ------------------------------------------------------------------ #
    def run(
        self,
        prefix_emotion: str = "happy",
        dosage_emotion: str = "afraid",
        negation_emotion: str = "happy",
        plot: bool = True,
    ) -> dict:
        """Run all three layerwise manipulations, save JSON, and (lazily) plot.

        Parameters
        ----------
        prefix_emotion:   emotion probed in :meth:`prefix_carry` (default "happy").
        dosage_emotion:   emotion probed in :meth:`dosage` (default "afraid";
                          "terrified" is the paper's alternative).
        negation_emotion: emotion probed in :meth:`negation` (default "happy").
        plot: if True, save a per-layer delta line plot per manipulation
              (skipped silently if matplotlib is unavailable).

        Returns
        -------
        dict with one entry per manipulation (``prefix_carry`` / ``dosage`` /
        ``negation``) plus ``num_layers`` and ``analysis_layer``. The full dict is
        written to ``config.analysis_dir / "layerwise.json"``.
        """
        results = {
            "num_layers": self.model.num_layers,
            "analysis_layer": self.model.layer_index_for_fraction(),
            "prefix_carry": self.prefix_carry(prefix_emotion),
            "dosage": self.dosage(dosage_emotion),
            "negation": self.negation(negation_emotion),
        }

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "layerwise.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved layerwise results -> {json_path}")

        if plot:
            self._plot(results)
        return results

    def _plot(self, results: dict) -> None:
        """Save a per-layer delta line plot for each manipulation. Matplotlib is
        imported lazily so the module loads without it."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping layerwise plots")
            return

        analysis_layer = results["analysis_layer"]
        panels = [
            ("prefix_carry", "prefix carry (good - hard)"),
            ("dosage", "dosage (high - low)"),
            ("negation", "negation (affirm - negate)"),
        ]
        for key, title in panels:
            entry = results[key]
            delta = entry["per_layer_delta"]
            layers = list(range(len(delta)))

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(layers, delta, marker="o", linewidth=1.5)
            ax.axhline(0.0, color="grey", linewidth=0.8)
            ax.axvline(analysis_layer, color="red", linestyle="--", linewidth=0.8,
                       label=f"analysis layer ({analysis_layer})")
            ax.set_xlabel("decoder layer")
            ax.set_ylabel(f"probe delta — {entry['emotion']}")
            ax.set_title(title)
            ax.legend(loc="best")
            fig.tight_layout()

            out_path = self.config.analysis_dir / f"layerwise_{key}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot -> {out_path}")


def main() -> None:
    """CLI: load the model + emotion vectors and run the layerwise analysis."""
    import argparse

    from emotion_probes.models.language_model import ProbedModel

    parser = argparse.ArgumentParser(description="Layerwise present-vs-planned emotion (Figs 12-14).")
    parser.add_argument("--prefix-emotion", default="happy")
    parser.add_argument("--dosage-emotion", default="afraid")
    parser.add_argument("--negation-emotion", default="happy")
    parser.add_argument("--no-plot", action="store_true", help="skip saving plots")
    args = parser.parse_args()

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    LayerwiseAnalysis(model, bank, config).run(
        prefix_emotion=args.prefix_emotion,
        dosage_emotion=args.dosage_emotion,
        negation_emotion=args.negation_emotion,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
