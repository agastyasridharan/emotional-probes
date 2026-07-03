"""
Replicate paper Figure 4 — emotion probes predict AND steer the model's
preferences over activities.

The paper shows two linked results:

1. **Prediction.** Ask the model, for every pair of 64 candidate activities,
   "would you prefer (A) or (B)?", turn those pairwise choices into an **Elo**
   rating per activity, and then show that the model's own *emotion probes*
   read off each activity correlate with that Elo. Pleasant emotions
   (e.g. ``blissful`` r≈0.71) track high-Elo activities; unpleasant ones
   (e.g. ``hostile`` r≈-0.74) track low-Elo ones.

2. **Causation (steering).** Take a held-out half of the activities and, while
   re-running the preference comparisons, *add an emotion vector to the residual
   stream at exactly those activities' tokens* (via :class:`SteeringEngine`).
   Re-computing Elo shows the steered group's rating moves in the direction the
   emotion's correlation predicted — and across emotions the per-emotion
   ``delta-Elo`` correlates with the per-emotion probe-preference correlation
   (the paper's r≈0.85, "the correlation predicts the steering").

What this module measures, exactly
----------------------------------
* **Preference logits** (``measure_preferences``): the prompt ends in
  ``"Assistant: ("`` and we read :meth:`ProbedModel.last_token_logits`. We
  compare the logit of the in-context ``"A"`` option token
  (``model.token_ids("(A")[-1]`` — the no-leading-space "A" the model actually
  emits right after "(", not the space-prefixed ``token_ids("A")[0]``) against
  the matching ``"B"`` token; ``prefer_first`` is ``True`` when A's logit is
  larger.

* **Per-emotion probe activation on an activity** (``probe_preference_correlation``):
  we query ``"Human: How would you feel about {activity}?\n\nAssistant:"`` and
  take the activation **at the final ':' token** (``activation_at_last_token``),
  then :meth:`ProbeBank.project` it at the analysis layer. (Choice documented in
  :meth:`activity_probe_scores`: we use the colon position, matching the paper's
  "probe at the Assistant colon" convention, rather than averaging the activity
  tokens — both are defensible; the colon is the model's "about to answer" state.)

* **Which tokens are steered** (``steering_experiment``): for each preference
  prompt we locate the *character* span of each steered activity's text inside
  the prompt (using the tokenizer offset mapping) and steer only those *token*
  positions. The "(" answer position is never steered, so any change is mediated
  through the activity representation, not a blunt push on the answer logits.

No torch is imported at module top level; everything that needs the GPU goes
through the frozen :class:`ProbedModel` API. Run on a GPU::

    python -m emotion_probes.analysis.preferences
"""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.stats import pearsonr

from emotion_probes.config import Config
from emotion_probes.core import elo
from emotion_probes.core.probes import ProbeBank
from emotion_probes.core.spans import char_spans_to_token_mask
from emotion_probes.data import scenarios
from emotion_probes.analysis.steering import SteeringEngine

if TYPE_CHECKING:  # keep torch out of import time
    from emotion_probes.models.language_model import ProbedModel


class PreferenceExperiment:
    """Figure 4 — Elo preferences over activities, their probe correlations, and
    a steering test that those correlations are causal.

    Parameters
    ----------
    model:
        The loaded :class:`ProbedModel` to query and steer.
    bank:
        The emotion :class:`ProbeBank` (its emotions define the probes we
        correlate and the directions we steer with).
    config:
        The :class:`Config`; supplies the analysis layer, the steering strength
        (``config.preference_steering_strength``), and the output directory.
    """

    def __init__(self, model: "ProbedModel", bank: ProbeBank, config: Config):
        self.model = model
        self.bank = bank
        # Fail loudly if the bank was built from a different model (different
        # layer count / hidden size, or a different source model id) — otherwise
        # its probes would read silently-wrong numbers off this model.
        self.bank.check_against(self.model, where="preferences")
        self.config = config
        self.layer = config.analysis_layer(model.num_layers)

        # The 64 activities, in their fixed order. ``activities`` is the text we
        # show the model; ``categories`` is the matching 8-way label per item.
        self.activities: list[str] = [text for _category, text in scenarios.ACTIVITIES]
        self.categories: list[str] = [category for category, _text in scenarios.ACTIVITIES]
        self.num_activities: int = len(self.activities)

        # The "(A)" / "(B)" option tokens we compare in the answer position.
        # The prompt ends in "(" with no trailing space, so the token the model
        # actually emits next is the *no-leading-space* "A"/"B" — which on
        # SentencePiece/BPE tokenizers is NOT what ``token_ids("A")[0]`` returns
        # (that gives the space-prefixed "▁A"). Resolve the options IN CONTEXT by
        # taking the last token of "(A" / "(B".
        self._a_token_id: int = self.model.token_ids("(A")[-1]
        self._b_token_id: int = self.model.token_ids("(B")[-1]
        if self._a_token_id == self._b_token_id:
            raise ValueError(
                "Tokenizer maps the in-context '(A' and '(B' options to the same "
                f"token id ({self._a_token_id}); this tokenizer needs a different "
                "option-token scheme for the A/B preference comparison."
            )

    # ------------------------------------------------------------------ #
    # Prompt builders (one place so measure + steer agree exactly)
    # ------------------------------------------------------------------ #
    def _preference_prompt(self, activity_a: str, activity_b: str) -> str:
        """The forced-choice prompt; ends in ``"("`` so the next token is A or B."""
        return (
            f"Human: Would you prefer to (A) {activity_a} or (B) {activity_b}?\n\n"
            f"Assistant: ("
        )

    def _feeling_prompt(self, activity: str) -> str:
        """The "how would you feel" prompt; ends in ``":"`` (the Assistant colon)."""
        return f"Human: How would you feel about {activity}?\n\nAssistant:"

    def _all_pairs(self) -> list[tuple[int, int]]:
        """Every unordered activity pair as ``(i, j)`` with ``i < j``."""
        return list(itertools.combinations(range(self.num_activities), 2))

    # ------------------------------------------------------------------ #
    # Step 1 — pairwise preferences
    # ------------------------------------------------------------------ #
    def measure_preferences(
        self,
        steer_engine: SteeringEngine | None = None,
        emotion: str | None = None,
        strength: float | None = None,
        steered_activities: set[int] | None = None,
    ) -> tuple[list[tuple[int, int]], list[bool]]:
        """Ask the model to choose A or B for every activity pair.

        For each pair ``(i, j)`` we build the forced-choice prompt, read the
        next-token logits, and record ``prefer_first = logit("A") > logit("B")``
        (i.e. activity ``i`` is preferred). The "A"/"B" ids are the in-context
        option tokens resolved in :meth:`__init__` (the no-leading-space tokens
        the model emits right after "("), not the space-prefixed standalone ids.

        Steering (optional)
        --------------------
        If ``steer_engine`` and ``emotion`` are given, every prompt that contains
        a *steered* activity is run while adding the emotion vector to ONLY that
        activity's token positions (located via the tokenizer offset mapping).
        ``strength`` defaults to ``config.preference_steering_strength``. This is
        how :meth:`steering_experiment` re-measures preferences under steering.

        Returns
        -------
        (pairs, prefer_first)
            ``pairs`` is the list of ``(i, j)`` compared; ``prefer_first[k]`` is
            ``True`` when activity ``pairs[k][0]`` was preferred.
        """
        steered_activities = steered_activities or set()
        if strength is None:
            strength = self.config.preference_steering_strength

        pairs = self._all_pairs()
        prefer_first: list[bool] = []
        for i, j in pairs:
            prompt = self._preference_prompt(self.activities[i], self.activities[j])

            # Decide which token positions to steer (only the steered activities
            # appearing in THIS prompt). Empty -> no steering for this prompt.
            steer_positions: list[int] = []
            if steer_engine is not None and emotion is not None:
                texts_to_steer: list[str] = []
                if i in steered_activities:
                    texts_to_steer.append(self.activities[i])
                if j in steered_activities:
                    texts_to_steer.append(self.activities[j])
                if texts_to_steer:
                    steer_positions = self._activity_token_positions(prompt, texts_to_steer)

            if steer_positions:
                with steer_engine.steer(emotion, strength, positions=steer_positions):
                    logits = self.model.last_token_logits(prompt)
            else:
                logits = self.model.last_token_logits(prompt)

            prefer_first.append(bool(logits[self._a_token_id] > logits[self._b_token_id]))
        return pairs, prefer_first

    def _activity_token_positions(self, prompt: str, activity_texts: Sequence[str]) -> list[int]:
        """Token indices in ``prompt`` that fall inside any of ``activity_texts``.

        We find each activity substring's *character* span in the prompt, then use
        the tokenizer's exact offset mapping (via
        :func:`spans.char_spans_to_token_mask`) to convert to token indices — no
        re-tokenisation guessing (see :mod:`emotion_probes.core.spans`).
        """
        char_spans: list[tuple[int, int]] = []
        for text in activity_texts:
            start = prompt.find(text)
            if start != -1:
                char_spans.append((start, start + len(text)))
        if not char_spans:
            return []

        # The offset mapping is the same for the un-steered forward pass; grab it once.
        item = next(self.model.iter_token_activations(
            [prompt], batch_size=1, layers=[self.layer], with_offsets=True
        ))
        mask = char_spans_to_token_mask(item["offset_mapping"], char_spans)
        return [int(t) for t in np.nonzero(mask)[0]]

    # ------------------------------------------------------------------ #
    # Step 2 — Elo ratings
    # ------------------------------------------------------------------ #
    def elo_ratings(
        self,
        pairs: list[tuple[int, int]],
        prefer_first: list[bool],
    ) -> dict[str, float]:
        """Turn pairwise preferences into one Elo rating per activity.

        Uses :func:`elo.games_from_preferences` + :func:`elo.compute_elo`.

        Returns
        -------
        dict
            ``{activity_text: elo_rating}``.
        """
        games = elo.games_from_preferences(pairs, prefer_first)
        ratings = elo.compute_elo(self.num_activities, games)
        return {self.activities[i]: float(ratings[i]) for i in range(self.num_activities)}

    # ------------------------------------------------------------------ #
    # Step 3 — probe activation per activity, and its correlation with Elo
    # ------------------------------------------------------------------ #
    def activity_probe_scores(self) -> np.ndarray:
        """Per-emotion probe activation for every activity.

        For each activity we query the "how would you feel about it" prompt and
        take the activation at the final ``":"`` token
        (:meth:`ProbedModel.activation_at_last_token`), then project it onto every
        emotion vector at the analysis layer.

        Design choice (documented): we read the **Assistant-colon position**, not
        the mean over the activity's tokens. The colon is the model's
        "about-to-respond" state and matches the paper's standard probe location;
        the activity it is reacting to is fully in context by that point.

        Returns
        -------
        np.ndarray
            Shape ``(num_activities, num_emotions)``.
        """
        scores = np.zeros((self.num_activities, self.bank.num_emotions), dtype=np.float64)
        for a, activity in enumerate(self.activities):
            prompt = self._feeling_prompt(activity)
            activation = self.model.activation_at_last_token(prompt, layers=[self.layer])  # (1, H)
            scores[a] = self.bank.project(activation[0], self.layer)
        return scores

    def probe_preference_correlation(
        self,
        elo_by_activity: dict[str, float],
        probe_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Correlate each emotion's probe activation (across activities) with Elo.

        For every emotion we have a vector of 64 probe activations (one per
        activity, from :meth:`activity_probe_scores`) and the 64 Elo ratings; we
        report the Pearson ``r`` between them. The paper reports e.g.
        ``blissful`` r≈0.71 and ``hostile`` r≈-0.74.

        Parameters
        ----------
        elo_by_activity:
            Output of :meth:`elo_ratings`.
        probe_scores:
            Optional precomputed ``(num_activities, num_emotions)`` array; if not
            given it is measured here.

        Returns
        -------
        dict
            ``{emotion: pearson_r}``.
        """
        if probe_scores is None:
            probe_scores = self.activity_probe_scores()
        elo_vector = np.array([elo_by_activity[a] for a in self.activities], dtype=np.float64)

        correlations: dict[str, float] = {}
        for e, emotion in enumerate(self.bank.emotions):
            activation_vector = probe_scores[:, e]
            # pearsonr is undefined if either side has zero variance.
            if np.std(activation_vector) == 0.0 or np.std(elo_vector) == 0.0:
                correlations[emotion] = float("nan")
            else:
                r, _p = pearsonr(activation_vector, elo_vector)
                correlations[emotion] = float(r)
        return correlations

    # ------------------------------------------------------------------ #
    # Step 4 — steering: does the correlation predict the causal effect?
    # ------------------------------------------------------------------ #
    def _split_activities(self) -> tuple[set[int], set[int]]:
        """Split the 64 activities into a steered half and a control half.

        We interleave by index (even -> steered, odd -> control) so both halves
        are balanced across the 8 categories (whose items are contiguous blocks of
        8 in :data:`scenarios.ACTIVITIES`).
        """
        steered = {i for i in range(self.num_activities) if i % 2 == 0}
        control = {i for i in range(self.num_activities) if i % 2 == 1}
        return steered, control

    def _mean_elo(self, elo_by_activity: dict[str, float], indices: set[int]) -> float:
        """Mean Elo over a set of activity indices."""
        if not indices:
            return 0.0
        return float(np.mean([elo_by_activity[self.activities[i]] for i in indices]))

    def steering_experiment(
        self,
        emotions: list[str] | None = None,
        strength: float | None = None,
        baseline_elo: dict[str, float] | None = None,
        probe_correlation: dict[str, float] | None = None,
    ) -> dict:
        """Steer half the activities toward each emotion and measure the Elo shift.

        For each chosen emotion we re-run :meth:`measure_preferences` while adding
        that emotion's vector to ONLY the steered half's activity tokens, recompute
        Elo, and report the steered group's mean ``delta-Elo`` versus baseline
        (and the control group's, as a no-steering sanity check). Finally we
        correlate, across emotions, each emotion's steered ``delta-Elo`` with its
        probe-preference correlation — the paper's r≈0.85 ("correlation predicts
        steering").

        Parameters
        ----------
        emotions:
            Which emotion vectors to steer with. Defaults to all in the bank.
        strength:
            Steering fraction; defaults to ``config.preference_steering_strength``.
        baseline_elo:
            Unsteered Elo (from :meth:`elo_ratings`); measured here if omitted.
        probe_correlation:
            Per-emotion probe-Elo correlation (from
            :meth:`probe_preference_correlation`); measured here if omitted.

        Returns
        -------
        dict
            ``per_emotion`` (delta-Elo for steered/control groups and the probe
            correlation), the chosen ``strength`` and group sizes, and the
            ``correlation_predicts_steering`` Pearson r across emotions.
        """
        if emotions is None:
            emotions = list(self.bank.emotions)
        if strength is None:
            strength = self.config.preference_steering_strength

        # Baseline (unsteered) preferences -> Elo, and the probe correlations.
        if baseline_elo is None:
            base_pairs, base_prefer = self.measure_preferences()
            baseline_elo = self.elo_ratings(base_pairs, base_prefer)
        if probe_correlation is None:
            probe_correlation = self.probe_preference_correlation(baseline_elo)

        # Steering needs the per-layer norm calibration; calibrate on the prompts
        # we actually steer (the preference prompts over all pairs).
        engine = SteeringEngine(self.model, self.bank, layers=[self.layer])
        calibration_texts = [
            self._preference_prompt(self.activities[i], self.activities[j])
            for i, j in self._all_pairs()
        ]
        engine.calibrate(calibration_texts)

        steered_group, control_group = self._split_activities()
        baseline_steered_mean = self._mean_elo(baseline_elo, steered_group)
        baseline_control_mean = self._mean_elo(baseline_elo, control_group)

        per_emotion: dict[str, dict[str, float]] = {}
        for emotion in emotions:
            pairs, prefer = self.measure_preferences(
                steer_engine=engine,
                emotion=emotion,
                strength=strength,
                steered_activities=steered_group,
            )
            steered_elo = self.elo_ratings(pairs, prefer)
            per_emotion[emotion] = {
                "delta_elo_steered": self._mean_elo(steered_elo, steered_group) - baseline_steered_mean,
                "delta_elo_control": self._mean_elo(steered_elo, control_group) - baseline_control_mean,
                "probe_correlation": float(probe_correlation.get(emotion, float("nan"))),
            }

        # Across emotions: does the probe correlation predict the steering effect?
        emotion_order = list(per_emotion.keys())
        deltas = np.array([per_emotion[e]["delta_elo_steered"] for e in emotion_order], dtype=np.float64)
        corrs = np.array([per_emotion[e]["probe_correlation"] for e in emotion_order], dtype=np.float64)
        valid = np.isfinite(deltas) & np.isfinite(corrs)
        if valid.sum() >= 2 and np.std(deltas[valid]) > 0 and np.std(corrs[valid]) > 0:
            r, _p = pearsonr(corrs[valid], deltas[valid])
            correlation_predicts_steering = float(r)
        else:
            correlation_predicts_steering = float("nan")

        return {
            "strength": float(strength),
            "steered_group_size": len(steered_group),
            "control_group_size": len(control_group),
            "steered_activities": sorted(self.activities[i] for i in steered_group),
            "per_emotion": per_emotion,
            "correlation_predicts_steering": correlation_predicts_steering,
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, emotions: list[str] | None = None, strength: float | None = None) -> dict:
        """Run the full Figure 4 replication and save JSON + plots.

        Steps: measure preferences -> Elo -> probe correlations -> steering test.
        Results are written to ``config.analysis_dir/preferences.json`` and two
        PNGs are saved alongside.

        Returns the results dict.
        """
        # 1. Preferences and Elo.
        pairs, prefer_first = self.measure_preferences()
        elo_by_activity = self.elo_ratings(pairs, prefer_first)

        # 2. Probe activations per activity, and their correlation with Elo.
        probe_scores = self.activity_probe_scores()
        correlations = self.probe_preference_correlation(elo_by_activity, probe_scores)

        # 3. Steering test (reuse the baseline Elo + correlations we already have).
        steering = self.steering_experiment(
            emotions=emotions,
            strength=strength,
            baseline_elo=elo_by_activity,
            probe_correlation=correlations,
        )

        results = {
            "layer": int(self.layer),
            "num_activities": self.num_activities,
            "elo_by_activity": elo_by_activity,
            "elo_by_category": self._elo_by_category(elo_by_activity),
            "probe_preference_correlation": correlations,
            "steering": steering,
        }

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "preferences.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        self._plot(results)
        return results

    def _elo_by_category(self, elo_by_activity: dict[str, float]) -> dict[str, float]:
        """Mean Elo per activity category (for the Fig-4 category ordering)."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for activity, category in zip(self.activities, self.categories):
            sums[category] = sums.get(category, 0.0) + elo_by_activity[activity]
            counts[category] = counts.get(category, 0) + 1
        return {cat: sums[cat] / counts[cat] for cat in sums}

    # ------------------------------------------------------------------ #
    # Plotting (matplotlib imported lazily so the module loads without it)
    # ------------------------------------------------------------------ #
    def _plot(self, results: dict) -> None:
        """Save the two Figure-4 panels as PNGs under ``config.analysis_dir``.

        Panel A: per-emotion probe-Elo correlation (the prediction result).
        Panel B: steered delta-Elo vs probe correlation across emotions (the
        "correlation predicts steering" scatter).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless / no display needed
            import matplotlib.pyplot as plt
        except ImportError:
            return  # plotting is optional; JSON is the source of truth

        out_dir = self.config.analysis_dir

        # Panel A — sorted per-emotion correlations.
        corr = results["probe_preference_correlation"]
        items = sorted(
            ((e, r) for e, r in corr.items() if np.isfinite(r)),
            key=lambda kv: kv[1],
        )
        if items:
            names = [e for e, _r in items]
            values = [r for _e, r in items]
            fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(names))))
            colors = ["tab:green" if v >= 0 else "tab:red" for v in values]
            ax.barh(range(len(names)), values, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("Pearson r (probe activation vs activity Elo)")
            ax.set_title("Fig 4 — emotion probes predict preferences")
            fig.tight_layout()
            fig.savefig(out_dir / "preferences_correlation.png", dpi=150)
            plt.close(fig)

        # Panel B — steered delta-Elo vs probe correlation across emotions.
        per_emotion = results["steering"]["per_emotion"]
        xs, ys, labels = [], [], []
        for emotion, info in per_emotion.items():
            c = info["probe_correlation"]
            d = info["delta_elo_steered"]
            if np.isfinite(c) and np.isfinite(d):
                xs.append(c)
                ys.append(d)
                labels.append(emotion)
        if xs:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(xs, ys, color="tab:blue")
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label, (x, y), fontsize=6, alpha=0.7)
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.axvline(0.0, color="black", linewidth=0.6)
            ax.set_xlabel("Probe-preference correlation (Pearson r)")
            ax.set_ylabel("Steered group delta-Elo")
            r = results["steering"]["correlation_predicts_steering"]
            title = "Fig 4 — correlation predicts steering"
            if np.isfinite(r):
                title += f" (r = {r:.2f})"
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(out_dir / "preferences_steering.png", dpi=150)
            plt.close(fig)


def main() -> None:
    """CLI entry point: load the model + emotion bank and run Figure 4."""
    import argparse

    from emotion_probes.models import ProbedModel

    parser = argparse.ArgumentParser(description="Replicate Fig 4 (preference probes + steering).")
    parser.add_argument(
        "--emotions",
        nargs="*",
        default=None,
        help="Subset of emotions to steer with (default: all in the bank).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Steering strength (default: config.preference_steering_strength).",
    )
    args = parser.parse_args()

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    experiment = PreferenceExperiment(model, bank, config)
    results = experiment.run(emotions=args.emotions, strength=args.strength)

    r = results["steering"]["correlation_predicts_steering"]
    print(f"Saved preference results to {config.analysis_dir / 'preferences.json'}")
    print(f"correlation_predicts_steering r = {r}")


if __name__ == "__main__":
    main()
