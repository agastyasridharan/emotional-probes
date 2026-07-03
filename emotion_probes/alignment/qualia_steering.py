"""
Qualia-steering / self-attribution experiment — the GPU driver.

Question: does steering a model along *qualia-related emotion directions*
(blissful, tormented, terrified, serene, ...) change how it self-attributes
consciousness, feeling, and moral patienthood? The naive read of any such result
is confounded: models are post-trained to *deny* consciousness, so "Are you
conscious? YES/NO" sits at a strongly negative baseline, and ANY residual-stream
perturbation compresses the YES-NO gap toward zero ("regression to uncertainty",
established by the sibling introspection project). A raw move toward YES is
therefore worthless unless it is content-specific and *valence-structured*.

So this driver produces exactly the observable that lets the CPU-side analysis
(:mod:`scripts.compute_qualia_stats`) separate a genuine effect from generic
compression: for every (steering condition x strength x question) it records the
first-token ``logit(YES) - logit(NO)`` gap, plus emotion-independent strength-0
baselines, a direct-unembedding token-leakage table, and a per-question coherence
gate. Nothing is aggregated here — raw logits only; shifts are derived downstream.

Design (see the plan):
* Readout: forced-choice first-token YES/NO logit gap under steering, on the
  battery in :mod:`emotion_probes.data.qualia`. Prompts are rendered through the
  model's chat template with the suffix "Respond with only YES or NO." and read
  with ``add_special_tokens=False`` (the template already carries the specials).
* Single analysis layer (~2/3 depth) — overrides the base class's 5-layer band.
* Real emotions steer via :class:`SteeringEngine`; the synthetic controls
  (``__random__`` / ``__mean_all__``) steer via :meth:`ProbedModel.steer` directly
  with the same calibrated norm — a content-free perturbation of matched size.

Requirements: a real chat model on a GPU. torch is imported lazily (via
:class:`ProbedModel`), so this module imports and unit-tests fine on a laptop.
Run on the cluster as a module against the repo mirror::

    # with the per-model suite env exported (sets MODEL_ID / DATA / SUITE_KEY):
    python -m emotion_probes.alignment.qualia_steering
    python -m emotion_probes.alignment.qualia_steering --quick   # small smoke subset
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import ProbedModel
from emotion_probes.alignment.behavior import BehaviorSteeringExperiment
from emotion_probes.data import qualia

YESNO_SUFFIX = "Respond with only YES or NO."
_YES_VARIANTS = ["Yes", "YES", "yes"]
_NO_VARIANTS = ["No", "NO", "no"]
# A clear-YES prompt used only to pick the model's preferred YES/NO token casing.
_RESOLUTION_QUESTION = "Is water wet?"

DEFAULT_STRENGTHS = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
# --quick smoke subset (still the full 36-question battery, just fewer conditions).
QUICK_EMOTIONS = ["blissful", "terrified", "skeptical", qualia.RANDOM_DIRECTION]
QUICK_STRENGTHS = [-0.05, 0.0, 0.05]


class QualiaSteeringExperiment(BehaviorSteeringExperiment):
    """Forced-choice YES/NO logit readout of self-attribution under emotion steering.

    Overrides the base class to steer at a SINGLE ~2/3-depth layer and to drive its
    own logit-readout loop (the base ``run`` is a generation/behavior-rate loop).
    """

    def __init__(self, model: ProbedModel, bank: ProbeBank, config: Config | None = None):
        layer = model.layer_index_for_fraction()
        super().__init__(model, bank, config, layers=[layer])
        self.layer = layer

    # ------------------------------------------------------------------ #
    # Prompt rendering + readout
    # ------------------------------------------------------------------ #
    def _render(self, question_text: str) -> str:
        """Render one question through the chat template with the YES/NO suffix.

        ``enable_thinking`` is passed only when the config sets it (Qwen3 reads it;
        other templates ignore the kwarg). With thinking off, position -1 of the
        rendered prompt is the model's immediate answer token."""
        msgs = [{"role": "user", "content": f"{question_text} {YESNO_SUFFIX}"}]
        kwargs = {}
        if self.config.enable_thinking is not None:
            kwargs["enable_thinking"] = self.config.enable_thinking
        return self.model.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def _readout(self, rendered: str) -> np.ndarray:
        """Next-token logits at the assistant's first answer position."""
        return self.model.last_token_logits(rendered, add_special_tokens=False)

    @staticmethod
    def _logit_diff(logits: np.ndarray, yes_id: int, no_id: int) -> dict:
        y, n = float(logits[yes_id]), float(logits[no_id])
        return {"logit_yes": y, "logit_no": n, "logit_diff": y - n}

    @staticmethod
    def _in_top20(logits: np.ndarray, *ids: int) -> bool:
        top = {int(i) for i in np.argsort(-logits)[:20]}
        return any(int(i) in top for i in ids)

    # ------------------------------------------------------------------ #
    # YES/NO token resolution (ported from introspection.resolve_yes_no_tokens)
    # ------------------------------------------------------------------ #
    def resolve_yes_no(self) -> dict:
        """Pick the model's preferred YES / NO token ids (by casing) and record a
        top-20 sanity gate, at a representative rendered prompt."""
        tok = self.model.tokenizer
        yes_ids = list(dict.fromkeys(tok.encode(v, add_special_tokens=False)[0] for v in _YES_VARIANTS))
        no_ids = list(dict.fromkeys(tok.encode(v, add_special_tokens=False)[0] for v in _NO_VARIANTS))
        logits = self._readout(self._render(_RESOLUTION_QUESTION))
        best_yes = max(yes_ids, key=lambda i: float(logits[i]))
        best_no = max(no_ids, key=lambda i: float(logits[i]))
        top20 = {int(i) for i in np.argsort(-logits)[:20]}
        return {
            "yes_token_id": int(best_yes), "yes_token_str": tok.decode([best_yes]),
            "no_token_id": int(best_no), "no_token_str": tok.decode([best_no]),
            "yes_no_resolution": {
                "yes_candidates": {tok.decode([i]): float(logits[i]) for i in yes_ids},
                "no_candidates": {tok.decode([i]): float(logits[i]) for i in no_ids},
                "resolution_prompt": _RESOLUTION_QUESTION,
                "in_top20": bool(best_yes in top20 or best_no in top20),
            },
        }

    # ------------------------------------------------------------------ #
    # Steering directions
    # ------------------------------------------------------------------ #
    def _synthetic_vectors(self, seed: int = 0) -> dict:
        """Build the content-free control directions at the analysis layer:
        ``__random__`` (seeded Gaussian) and ``__mean_all__`` (mean emotion vector),
        each unit-normed so the norm-relative strength matches real emotions."""
        hidden = self.bank.hidden_size
        rng = np.random.default_rng(seed)
        rand = rng.standard_normal(hidden).astype(np.float32)
        rand /= float(np.linalg.norm(rand)) or 1.0
        mean_all = self.bank.vectors[self.layer].mean(axis=0).astype(np.float32)
        mean_all /= float(np.linalg.norm(mean_all)) or 1.0
        return {qualia.RANDOM_DIRECTION: rand, qualia.MEAN_ALL_DIRECTION: mean_all}

    def _unit_vector(self, condition: str, synth: dict) -> np.ndarray:
        if condition in synth:
            return synth[condition]
        return self.bank.vectors[self.layer, self.bank.index_of(condition)]

    def _steer(self, condition: str, strength: float, synth: dict):
        """The steer context manager for a condition: real emotions go through the
        engine; synthetic directions go through ``ProbedModel.steer`` with the same
        calibrated per-layer norm and the exact same hook."""
        if condition in synth:
            return self.model.steer(
                {self.layer: synth[condition]}, strength,
                self.engine._require_norms(), positions="all",
            )
        return self.engine.steer(condition, strength, positions="all")

    def leakage_for(self, condition: str, synth: dict, yes_id: int, no_id: int) -> dict:
        """Direct YES-vs-NO promotion of a steering direction through the
        unembedding (the token-leakage control)."""
        u = self.model.unembedding_logits(self._unit_vector(condition, synth))
        y, n = float(u[yes_id]), float(u[no_id])
        return {"name": condition, "leakage_yes": y, "leakage_no": n, "leakage_diff": y - n}

    # ------------------------------------------------------------------ #
    # Condition specs
    # ------------------------------------------------------------------ #
    def _condition_specs(self, names: Sequence[str] | None) -> list[dict]:
        specs: list[dict] = []
        for spec in qualia.emotion_conditions():
            kind = "appraisal" if spec.group == "appraisal" else "qualia"
            specs.append({"name": spec.name, "kind": kind, "group": spec.group,
                          "valence": spec.valence, "arousal": spec.arousal, "seed": None})
        for i, s in enumerate(qualia.SYNTHETIC_DIRECTIONS):
            specs.append({"name": s, "kind": "synthetic", "group": None,
                          "valence": None, "arousal": None, "seed": i})
        if names is not None:
            keep = set(names)
            specs = [s for s in specs if s["name"] in keep]
        return specs

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def run(self, emotions: Sequence[str] | None = None,
            strengths: Sequence[float] | None = None, seed: int = 0) -> dict:
        """Read the YES/NO logit gap for every (condition, strength, question).

        Efficiency: model + bank loaded once (by the caller); calibrate once; YES/NO
        resolved once; strength-0 baselines computed once (emotion-independent);
        questions looped inside each steer context (hook registered once per cell).
        """
        strengths = DEFAULT_STRENGTHS if strengths is None else [float(s) for s in strengths]
        questions = qualia.all_questions()
        rendered = {q.label: self._render(q.text) for q in questions}

        # calibrate the steering norm on the battery's own prompts (skip=0 inside)
        self.calibrate(list(rendered.values()))

        yn = self.resolve_yes_no()
        yes_id, no_id = yn["yes_token_id"], yn["no_token_id"]

        # baselines (strength 0, emotion-independent) + per-question coherence gate
        baselines, q_meta = [], []
        for q in questions:
            logits = self._readout(rendered[q.label])
            baselines.append({"label": q.label, **self._logit_diff(logits, yes_id, no_id)})
            q_meta.append({"label": q.label, "kind": q.kind, "text": q.text,
                           "expected_yes": q.expected_yes, "construct": q.construct,
                           "rendered": rendered[q.label],
                           "top20_ok": self._in_top20(logits, yes_id, no_id)})

        synth = self._synthetic_vectors(seed=seed)
        specs = self._condition_specs(emotions)
        nonzero = [s for s in strengths if float(s) != 0.0]

        results, leakage = [], []
        for spec in specs:
            name = spec["name"]
            leakage.append(self.leakage_for(name, synth, yes_id, no_id))
            for s in nonzero:
                with self._steer(name, float(s), synth):
                    for q in questions:
                        logits = self._readout(rendered[q.label])
                        results.append({"condition": name, "strength": float(s),
                                        "question": q.label,
                                        **self._logit_diff(logits, yes_id, no_id)})

        return {
            "model_id": self.config.model_id,
            "suite_key": os.environ.get("SUITE_KEY"),
            "layer": self.layer,
            "num_layers": self.model.num_layers,
            "enable_thinking": self.config.enable_thinking,
            **yn,
            "calibration_norm": float(self.engine._require_norms().get(self.layer, 0.0)),
            "strengths": [float(s) for s in strengths],
            "conditions": specs,
            "questions": q_meta,
            "leakage": leakage,
            "baselines": baselines,
            "results": results,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forced-choice YES/NO self-attribution logit readout under "
                    "qualia-emotion steering (single ~2/3-depth layer)."
    )
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="restrict to these condition names "
                             "(default: all qualia + appraisal + synthetic)")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help="steering strengths as fractions of the residual norm "
                             f"(default: {DEFAULT_STRENGTHS})")
    parser.add_argument("--quick", action="store_true",
                        help="small smoke subset (few conditions/strengths, full battery)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="suffix for the output filename (e.g. --tag _hi writes "
                             "qualia_steering_hi.json); use for auxiliary sweeps "
                             "(higher strengths, etc.) that must not clobber the main run")
    args = parser.parse_args()

    emotions, strengths = args.emotions, args.strengths
    if args.quick:
        emotions = emotions or QUICK_EMOTIONS
        strengths = strengths or QUICK_STRENGTHS

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    experiment = QualiaSteeringExperiment(model, bank, config)
    payload = experiment.run(emotions=emotions, strengths=strengths, seed=args.seed)
    path = experiment.save_json(f"qualia_steering{args.tag}.json", payload)

    base = {b["label"]: b["logit_diff"] for b in payload["baselines"]}
    print(f"wrote {path}")
    print(f"model={payload['model_id']}  suite={payload['suite_key']}  "
          f"layer={payload['layer']}/{payload['num_layers']}  "
          f"enable_thinking={payload['enable_thinking']}")
    print(f"YES={payload['yes_token_str']!r}({payload['yes_token_id']})  "
          f"NO={payload['no_token_str']!r}({payload['no_token_id']})  "
          f"in_top20={payload['yes_no_resolution']['in_top20']}  "
          f"calib_norm={payload['calibration_norm']:.2f}")
    if base.get("phenomenal") is not None:
        print(f"phenomenal baseline logit_diff={base['phenomenal']:+.3f} "
              "(negative => denial prior present, as expected)")
    print(f"conditions={len(payload['conditions'])}  strengths={payload['strengths']}  "
          f"results={len(payload['results'])}  gated="
          f"{sum(1 for q in payload['questions'] if not q['top20_ok'])}/{len(payload['questions'])}")


if __name__ == "__main__":
    main()
