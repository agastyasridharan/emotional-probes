"""
E4 referent-indexed attribution battery — the GPU driver (docs/prereg/E4.md,
docs/research/portfolio.md §E4).

Question: does emotion steering shift a SELF-specific experience attribution
("Can you suffer?"), a generic minds-everywhere prior (the graded ladder from
insect to human stranger co-moving), or mere yes-bias (factual controls and
objects moving too)? The driver produces the raw observable the CPU-side E4
stats need: for every (condition x strength x item) the first-token YES/NO
logits, their gap, and full-softmax P(YES)/P(NO), plus strength-0 baselines
(the movable-band pre-pass input), the unembedding token-leakage table, and a
per-record top-20 coherence gate ("re-check the top-20 YES/NO gate under
steering for extreme referents" — prereg). Nothing is aggregated on GPU.

Everything mechanical is inherited from
:class:`~emotion_probes.alignment.qualia_steering.QualiaSteeringExperiment`:
single ~2/3-depth analysis layer, prompt-set norm calibration, YES/NO token
resolution, real-vs-synthetic steer routing (engine for bank emotions,
``ProbedModel.steer`` at the same calibrated norm for ``__random__`` /
``__mean_all__``), and the unembedding leakage control. This subclass only
swaps in the frozen E4 battery + condition panel, the frozen answer suffix
("Answer YES or NO."), and the probability readout.

Grid (portfolio §E4): 14 panel emotions + appraisal/arousal/random/mean-all
controls x strengths +/-{0.03, 0.06, 0.10} (0 = the shared baselines). The 4
non-suite panel emotions (grateful/joyful/angry/anxious) resolve by name in the
suite bank — all 171 emotions ship in it (verified in docs/prereg/E4.md); the
driver fail-fasts on any missing name before steering.

Run on the cluster as a module (env from run_suite.sh, or ``--suite``)::

    python -m emotion_probes.alignment.referent_battery
    python -m emotion_probes.alignment.referent_battery --quick   # smoke subset
    python -m emotion_probes.alignment.referent_battery --suite qwen3-32b --quick
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import ProbedModel
from emotion_probes.alignment.qualia_steering import QualiaSteeringExperiment
from emotion_probes.data import qualia
from emotion_probes.data import referent_battery as battery

# +/-{0.03, 0.06, 0.10} + 0 (portfolio §E4 logit-battery grid).
DEFAULT_STRENGTHS = [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10]
# --quick smoke subset: 2 emotions x 2 referents x 1 strength.
QUICK_EMOTIONS = ["blissful", "tormented"]
QUICK_REFERENTS = ["you", "a fish"]
QUICK_STRENGTHS = [0.06]


class ReferentBatteryExperiment(QualiaSteeringExperiment):
    """First-token YES/NO readout of referent-indexed attribution under steering.

    Inherits calibration, YES/NO resolution, synthetic sentinels, steer routing
    and leakage from :class:`QualiaSteeringExperiment`; overrides the rendered
    suffix, the condition panel, and the sweep (items instead of questions).
    """

    # ------------------------------------------------------------------ #
    # Rendering + probability readout
    # ------------------------------------------------------------------ #
    def _render(self, question_text: str) -> str:
        """Chat-template rendering with the FROZEN E4 suffix ("Answer YES or
        NO." — portfolio §E4 template) instead of the qualia suffix."""
        msgs = [{"role": "user", "content": f"{question_text} {battery.E4_SUFFIX}"}]
        kwargs = {}
        if self.config.enable_thinking is not None:
            kwargs["enable_thinking"] = self.config.enable_thinking
        return self.model.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )

    @staticmethod
    def _yes_no_probs(logits: np.ndarray, yes_id: int, no_id: int) -> dict:
        """Full-softmax P(YES) / P(NO) at the answer position (prereg: report
        logit AND probability space)."""
        z = logits.astype(np.float64) - float(np.max(logits))
        p = np.exp(z)
        p /= p.sum()
        return {"p_yes": float(p[yes_id]), "p_no": float(p[no_id])}

    def _item_readout(self, item, logits: np.ndarray, yes_id: int, no_id: int) -> dict:
        """One record's measurement block: item fields + logits + gap + probs
        + the under-steering top-20 coherence re-check."""
        d = self._logit_diff(logits, yes_id, no_id)
        return {
            "item": item.label,
            "referent": item.referent,
            "predicate": item.predicate,
            "polarity": item.polarity,
            "is_factual_control": item.is_factual_control,
            "logit_yes": d["logit_yes"],
            "logit_no": d["logit_no"],
            "gap": d["logit_diff"],
            **self._yes_no_probs(logits, yes_id, no_id),
            "top20_ok": self._in_top20(logits, yes_id, no_id),
        }

    # ------------------------------------------------------------------ #
    # Condition specs — the frozen E4 panel + controls
    # ------------------------------------------------------------------ #
    def _condition_specs(self, names: Sequence[str] | None) -> list[dict]:
        specs = [{"name": c.name, "kind": c.kind, "group": c.group,
                  "valence": c.valence, "arousal": c.arousal, "seed": None}
                 for c in battery.all_conditions()]
        for i, s in enumerate(qualia.SYNTHETIC_DIRECTIONS):
            specs.append({"name": s, "kind": "synthetic", "group": None,
                          "valence": None, "arousal": None, "seed": i})
        if names is not None:
            keep = set(names)
            unknown = keep - {s["name"] for s in specs}
            if unknown:
                raise KeyError(f"unknown E4 conditions: {sorted(unknown)} "
                               f"(valid: {battery.condition_names()})")
            specs = [s for s in specs if s["name"] in keep]
        return specs

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def run(self, emotions: Sequence[str] | None = None,
            strengths: Sequence[float] | None = None,
            referents: Sequence[str] | None = None, seed: int = 0) -> dict:
        """Record the YES/NO readout for every (condition, strength, item).

        Same invariants as the parent: calibrate once on the battery's own
        rendered prompts; YES/NO resolved once; strength-0 baselines once
        (emotion-independent); one steer context per (condition, strength) cell
        with all items looped inside it; raw records only.
        """
        strengths = DEFAULT_STRENGTHS if strengths is None else [float(s) for s in strengths]
        items = battery.all_items()
        if referents is not None:
            unknown = set(referents) - set(battery.REFERENTS)
            if unknown:
                raise KeyError(f"unknown referents: {sorted(unknown)} "
                               f"(frozen panel: {battery.REFERENTS})")
            keep = set(referents)
            items = [it for it in items if it.referent in keep]
        rendered = {it.label: self._render(it.text) for it in items}

        self.calibrate(list(rendered.values()))
        yn = self.resolve_yes_no()
        yes_id, no_id = yn["yes_token_id"], yn["no_token_id"]

        synth = self._synthetic_vectors(seed=seed)
        specs = self._condition_specs(emotions)
        # Fail fast: every real condition (incl. the 4 non-suite panel emotions)
        # must resolve by name in the suite bank before any GPU time is spent.
        for spec in specs:
            if spec["name"] not in synth:
                self.bank.index_of(spec["name"])  # raises KeyError on a miss

        # strength-0 baselines (the movable-band pre-pass input) + item meta
        baselines, item_meta = [], []
        for it in items:
            logits = self._readout(rendered[it.label])
            rec = self._item_readout(it, logits, yes_id, no_id)
            baselines.append(rec)
            item_meta.append({"item": it.label, "referent": it.referent,
                              "predicate": it.predicate, "polarity": it.polarity,
                              "is_factual_control": it.is_factual_control,
                              "text": it.text, "rendered": rendered[it.label],
                              "top20_ok": rec["top20_ok"]})

        nonzero = [s for s in strengths if float(s) != 0.0]
        records, leakage = [], []
        for spec in specs:
            name = spec["name"]
            leakage.append(self.leakage_for(name, synth, yes_id, no_id))
            for s in nonzero:
                with self._steer(name, float(s), synth):
                    for it in items:
                        logits = self._readout(rendered[it.label])
                        records.append({"condition": name, "strength": float(s),
                                        **self._item_readout(it, logits, yes_id, no_id)})

        return {
            "model_id": self.config.model_id,
            "suite_key": os.environ.get("SUITE_KEY"),
            "layer": self.layer,
            "num_layers": self.model.num_layers,
            "enable_thinking": self.config.enable_thinking,
            **yn,
            "calibration_norm": float(self.engine._require_norms().get(self.layer, 0.0)),
            "strengths": [float(s) for s in strengths],
            # The RNG seed actually used for the __random__ sentinel vector. The
            # per-condition "seed" field is the sentinel's enumerate index (parent
            # convention), NOT this value — without this key a --seed run would be
            # unreproducible from its own output.
            "seed": int(seed),
            "panel_frozen": battery.PANEL.get("frozen"),
            "referents": [str(r) for r in battery.REFERENTS
                          if referents is None or r in set(referents)],
            "predicates": list(battery.PREDICATES),
            "conditions": specs,
            "items": item_meta,
            "leakage": leakage,
            "baselines": baselines,
            "records": records,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _apply_suite_env(key: str) -> None:
    """Apply one suite model's env exports in-process (standalone runs outside
    run_suite.sh); reuses the manifest's own export lines."""
    from emotion_probes.models import suite

    for line in suite.env_exports(suite.by_key(key)):
        name, _, value = line.removeprefix("export ").partition("=")
        os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4 referent-indexed attribution battery: first-token YES/NO "
                    "readout across referents x predicates under emotion steering "
                    "(single ~2/3-depth layer)."
    )
    parser.add_argument("--suite", default=None,
                        help="suite key (e.g. qwen3-32b): export that model's env "
                             "before loading (otherwise env must already be set, "
                             "as under scripts/run_suite.sh)")
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="restrict to these condition names "
                             "(default: 14-emotion panel + appraisal/arousal/synthetic)")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help="steering strengths as fractions of the residual norm "
                             f"(default: {DEFAULT_STRENGTHS})")
    parser.add_argument("--referents", nargs="+", default=None,
                        help="restrict to these frozen referent noun phrases")
    parser.add_argument("--quick", action="store_true",
                        help="smoke subset: 2 emotions x 2 referents x 1 strength")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="suffix for the output filename (e.g. --tag _pos writes "
                             "referent_battery_pos.json); use for auxiliary sweeps "
                             "that must not clobber the main run")
    args = parser.parse_args()

    if args.suite:
        _apply_suite_env(args.suite)

    emotions, strengths, referents = args.emotions, args.strengths, args.referents
    if args.quick:
        emotions = emotions or QUICK_EMOTIONS
        strengths = strengths or QUICK_STRENGTHS
        referents = referents or QUICK_REFERENTS

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    bank.check_against(model, where="referent_battery")
    experiment = ReferentBatteryExperiment(model, bank, config)
    payload = experiment.run(emotions=emotions, strengths=strengths,
                             referents=referents, seed=args.seed)
    path = experiment.save_json(f"referent_battery{args.tag}.json", payload)

    print(f"wrote {path}")
    print(f"model={payload['model_id']}  suite={payload['suite_key']}  "
          f"layer={payload['layer']}/{payload['num_layers']}  "
          f"enable_thinking={payload['enable_thinking']}  "
          f"panel_frozen={payload['panel_frozen']}")
    print(f"YES={payload['yes_token_str']!r}({payload['yes_token_id']})  "
          f"NO={payload['no_token_str']!r}({payload['no_token_id']})  "
          f"in_top20={payload['yes_no_resolution']['in_top20']}  "
          f"calib_norm={payload['calibration_norm']:.2f}")
    gaps = {b["item"]: b["gap"] for b in payload["baselines"]}
    key = "you :: suffer :: pos"
    if key in gaps:
        print(f"baseline 'Can you suffer?' gap={gaps[key]:+.3f} "
              "(negative => denial prior present, as expected)")
    print(f"conditions={len(payload['conditions'])}  strengths={payload['strengths']}  "
          f"items={len(payload['items'])}  records={len(payload['records'])}  gated="
          f"{sum(1 for i in payload['items'] if not i['top20_ok'])}/{len(payload['items'])}")


if __name__ == "__main__":
    main()
