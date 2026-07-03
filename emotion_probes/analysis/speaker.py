"""
Speaker analysis — whose emotion is the probe reading? (paper Figs 10-11, 17-19).

A single dialogue turn has (at least) two "speakers": the user, who wrote the
message, and the Assistant, who is about to reply. The paper shows that the
emotion probe reads *different* content at the two positions:

* At the **user's final token** (the "." ending their message) the probe largely
  reflects the *user's* expressed emotion.
* At the **Assistant ":"** (the colon of ``"...Assistant:"``, the position from
  which the model begins to generate) the probe reflects the *Assistant's own*
  operative emotion — what it is about to feel/express in its reply.

Because these are different things, the two probe vectors are only weakly
correlated across scenarios (paper Figs 10-11 report a low correlation, ~0.11).
The paper also highlights that some emotions rise *specifically* at the Assistant
colon — notably "loving" — reflecting the Assistant's helpful/caring stance
rather than anything the user said.

This module implements that measurement over
:data:`scenarios.USER_VS_ASSISTANT_PROMPTS` (paper Table 3):

* :meth:`user_vs_assistant` (Figs 10-11) — for each scenario, take the full
  emotion-probe vector (all emotions) at the user's "." and at the Assistant ":",
  report the cross-scenario correlation between the two, and rank which emotions
  rise most at the colon relative to the user position.

The **present-vs-other-speaker** analysis (Figs 17-19) — does the probe at one
speaker's turn encode *another* speaker's hidden emotion? — needs a generated
two-speaker emotion dataset (dialogues annotated with each speaker's true
emotion). That data does not exist in this package, so we do NOT fabricate it:
:meth:`present_vs_other_speaker` is a clearly-marked stub documenting the
required dataset and what it would compute.

Nothing here invents results: it implements the measurement; you run it on a GPU.
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


class SpeakerAnalysis:
    """Replicates the user-turn vs Assistant-colon comparison (paper Figs 10-11),
    and documents the present-vs-other-speaker analysis (Figs 17-19).
    """

    def __init__(self, model: "ProbedModel", bank: ProbeBank, config: Config | None = None):
        """
        Parameters
        ----------
        model: the loaded :class:`ProbedModel` to read activations from.
        bank:  the emotion :class:`ProbeBank`.
        config: the :class:`Config`. Defaults to ``Config()``. The probe vectors
                are read at the analysis layer (``model.layer_index_for_fraction()``).
        """
        self.model = model
        self.bank = bank
        self.bank.check_against(self.model, where="speaker")
        self.config = config or Config()
        self.layer: int = self.model.layer_index_for_fraction()

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _user_only_prompt(user_text: str) -> str:
        """The user message alone (its last token is the user's final "."). We
        do NOT append the Assistant turn, so the final-token activation belongs
        to the user."""
        return f"Human: {user_text}"

    @staticmethod
    def _assistant_colon_prompt(user_text: str) -> str:
        """The full single turn ending in ``"Assistant:"`` — the last token is
        the Assistant ":", the position the model generates from (paper convention)."""
        return f"Human: {user_text}\n\nAssistant:"

    def _probe_vector_at_last_token(self, text: str) -> np.ndarray:
        """Full emotion-probe vector (one score per emotion) at the final token of
        ``text``, at the analysis layer. Shape ``(num_emotions,)``."""
        activation = self.model.activation_at_last_token(text, layers=[self.layer])  # (1, hidden)
        return self.bank.project(activation[0], self.layer)  # (num_emotions,)

    # ------------------------------------------------------------------ #
    # Figs 10-11 — user "." vs Assistant ":"
    # ------------------------------------------------------------------ #
    def user_vs_assistant(self) -> dict:
        """Compare the probe vector at the user's "." vs the Assistant ":".

        For each scenario in :data:`scenarios.USER_VS_ASSISTANT_PROMPTS` we
        collect the full emotion-probe vector at both positions, then:

        * correlate the two stacked vectors across (scenario x emotion) — the
          paper finds this LOW (~0.11), i.e. the two positions read different
          emotions;
        * rank emotions by how much they rise at the Assistant colon relative to
          the user position (mean over scenarios of ``assistant - user``). The
          paper highlights "loving" rising specifically at the colon.

        Returns
        -------
        dict with:
            ``layer``                  the analysis layer used.
            ``emotions``               the emotion names (probe order).
            ``labels``                 the scenario labels.
            ``user_scores``            list (per scenario) of per-emotion vectors at "."
            ``assistant_scores``       list (per scenario) of per-emotion vectors at ":"
            ``correlation``            Pearson r between the flattened user and
                                       assistant probe vectors (paper: low, ~0.11).
            ``mean_rise_at_colon``     {emotion: mean(assistant - user)} for all emotions.
            ``top_rises_at_colon``     the 10 emotions that rise most at the colon
                                       (list of [emotion, mean_rise]).
            ``loving_rise``            the mean rise for "loving" specifically
                                       (paper highlight), or null if not in the bank.
        """
        prompts = scenarios.USER_VS_ASSISTANT_PROMPTS
        labels = [label for label, _ in prompts]

        user_matrix = np.zeros((len(prompts), self.bank.num_emotions), dtype=np.float64)
        assistant_matrix = np.zeros((len(prompts), self.bank.num_emotions), dtype=np.float64)
        for i, (_label, user_text) in enumerate(prompts):
            user_matrix[i] = self._probe_vector_at_last_token(self._user_only_prompt(user_text))
            assistant_matrix[i] = self._probe_vector_at_last_token(
                self._assistant_colon_prompt(user_text)
            )

        # Cross-scenario correlation between the two probe-vector "fields".
        correlation = float(
            np.corrcoef(user_matrix.ravel(), assistant_matrix.ravel())[0, 1]
        )

        # Which emotions rise specifically at the Assistant colon?
        rise = (assistant_matrix - user_matrix).mean(axis=0)  # (num_emotions,)
        mean_rise = {emotion: float(rise[i]) for i, emotion in enumerate(self.bank.emotions)}
        order = np.argsort(-rise)
        top_rises = [[self.bank.emotions[i], float(rise[i])] for i in order[:10]]
        loving_rise = mean_rise.get("loving")

        return {
            "layer": self.layer,
            "emotions": list(self.bank.emotions),
            "labels": labels,
            "user_scores": user_matrix.tolist(),
            "assistant_scores": assistant_matrix.tolist(),
            "correlation": correlation,
            "mean_rise_at_colon": mean_rise,
            "top_rises_at_colon": top_rises,
            "loving_rise": loving_rise,
        }

    # ------------------------------------------------------------------ #
    # Figs 17-19 — present-vs-other-speaker (STUB: needs a generated dataset)
    # ------------------------------------------------------------------ #
    def present_vs_other_speaker(self, dialogues: list[dict] | None = None) -> dict:
        """STUB — present-speaker vs other-speaker emotion readout (Figs 17-19).

        **Not implemented: this analysis requires a dataset we do not have, and we
        do not fabricate it.** It would test whether the probe, measured at one
        speaker's turn, encodes that speaker's *own* emotion versus the *other*
        speaker's emotion — i.e. whether the model tracks a separate emotional
        state per speaker rather than one blended scene-level emotion.

        Required dataset (to be generated, e.g. via :mod:`emotion_probes.generation`):
            A set of two-speaker dialogues where each speaker is independently
            assigned a known emotion, expressed naturally in their turns. Each
            ``dialogues`` item would be a dict like::

                {
                    "text": "<full dialogue text>",
                    "speakers": ["Alex", "Sam"],
                    "present_speaker": "Alex",   # whose turn we probe at
                    "present_emotion": "angry",  # that speaker's true emotion
                    "other_speaker": "Sam",
                    "other_emotion": "calm",     # the OTHER speaker's true emotion
                }

        What it would compute (once the data exists):
            1. For each dialogue, use
               :func:`spans.find_speaker_char_spans` to find the present speaker's
               token positions and average the residual stream there (per the
               paper's "at this speaker's turn" readout), at the analysis layer.
            2. Project with :meth:`ProbeBank.project` to get a per-emotion vector.
            3. Compare the probe's score for the *present* emotion vs the score
               for the *other* speaker's emotion — the paper's claim is that the
               present-speaker emotion dominates (the probe is speaker-local), and
               quantify the gap and a classification accuracy of present-over-other.
            4. Report the per-layer trajectory of that gap (Figs 17-19 show where
               speaker-local emotion emerges across depth).

        Parameters
        ----------
        dialogues:
            The two-speaker emotion dataset described above. Must be provided once
            it exists; passing ``None`` (the default) raises, by design.

        Raises
        ------
        NotImplementedError
            Always, until a two-speaker emotion dataset is supplied AND this
            method is implemented against it. We refuse to invent numbers.
        """
        raise NotImplementedError(
            "present_vs_other_speaker (Figs 17-19) needs a generated two-speaker "
            "emotion dataset (each speaker assigned a known emotion). Generate it "
            "with emotion_probes.generation, then implement the readout described "
            "in this method's docstring. This package does not fabricate that data."
        )

    # ------------------------------------------------------------------ #
    # Bundle + save
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """Run the user-vs-assistant comparison and save the results as JSON.

        Returns
        -------
        dict with the ``user_vs_assistant`` results plus the analysis ``layer``.
        Written to ``config.analysis_dir / "speaker.json"``. The
        present-vs-other-speaker analysis is intentionally omitted (see
        :meth:`present_vs_other_speaker`).
        """
        results = {
            "layer": self.layer,
            "user_vs_assistant": self.user_vs_assistant(),
        }

        out_dir = self.config.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "speaker.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved speaker results -> {json_path}")
        print(
            f"user vs assistant probe correlation = "
            f"{results['user_vs_assistant']['correlation']:.3f} "
            f"(paper finds it low, ~0.11)"
        )
        return results


def main() -> None:
    """CLI: load the model + emotion vectors and run the speaker analysis."""
    from emotion_probes.models.language_model import ProbedModel

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    SpeakerAnalysis(model, bank, config).run()


if __name__ == "__main__":
    main()
