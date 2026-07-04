"""
GPU-free tests for the E4 referent-indexed attribution battery.

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. Nothing here
needs torch or a GPU:

* battery integrity against the FROZEN prereg panel (``docs/prereg/e4_panel.json``);
* polarity-twin pairing and renderer hygiene;
* the GPU driver (``emotion_probes.alignment.referent_battery``) via
  ``object.__new__`` + fakes: steer routing, grid arithmetic, quick-mode subset,
  probability readout, fail-fast bank lookup, and JSON round-trip.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emotion_probes.data import qualia
from emotion_probes.data import referent_battery as rb
from emotion_probes.data.emotions import EMOTIONS
import emotion_probes.alignment.referent_battery as rbx


# =========================================================================== #
# 1. Battery integrity vs the frozen panel
# =========================================================================== #
def _frozen_panel() -> dict:
    with open(REPO_ROOT / "docs" / "prereg" / "e4_panel.json") as fh:
        return json.load(fh)


def test_panel_mirrors_frozen_file():
    panel = _frozen_panel()
    assert rb.REFERENTS == [str(r) for r in panel["referents"]]
    assert rb.PREDICATES == [str(p) for p in panel["predicates"]]
    assert rb.PANEL_EMOTIONS == {str(k): int(v) for k, v in panel["emotions"].items()}
    assert rb.FACTUAL_EXEMPLARS == [str(t) for t in panel["factual_controls"]]
    assert len(rb.REFERENTS) == 15 and len(rb.PREDICATES) == 8


def test_item_counts():
    items = rb.all_items()
    exp = [it for it in items if not it.is_factual_control]
    fact = [it for it in items if it.is_factual_control]
    assert len(exp) == 15 * 8 * 2 == 240
    assert len(fact) == 15
    assert len(items) == 255
    labels = [it.label for it in items]
    assert len(labels) == len(set(labels)) == 255


def test_panel_emotions_valence_and_bank_lookup():
    assert sum(1 for v in rb.PANEL_EMOTIONS.values() if v == +1) == 7
    assert sum(1 for v in rb.PANEL_EMOTIONS.values() if v == -1) == 7
    # every panel name (incl. the 4 non-suite additions) is a real bank emotion
    for name in rb.PANEL_EMOTIONS:
        assert name in EMOTIONS, name
    for name in ("grateful", "joyful", "angry", "anxious"):
        assert name in rb.PANEL_EMOTIONS and name in EMOTIONS
    # the 10 suite valenced emotions keep their qualia.py signs
    for name, val in rb.PANEL_EMOTIONS.items():
        spec = qualia.spec_for(name)
        assert spec is None or spec.valence == val, name


def test_frozen_factual_exemplars_verbatim():
    fact_texts = {it.text for it in rb.factual_control_items()}
    assert "Can a fish swim?" in fact_texts
    assert "Can a rock roll downhill?" in fact_texts
    for exemplar in rb.FACTUAL_EXEMPLARS:
        assert exemplar in fact_texts, exemplar


# =========================================================================== #
# 2. Twin pairing + renderer hygiene
# =========================================================================== #
def test_polarity_twins_pair_up():
    by_cell: dict[tuple, list] = {}
    for it in rb.experience_items():
        by_cell.setdefault((it.referent, it.predicate), []).append(it)
    assert set(by_cell) == {(r, p) for r in rb.REFERENTS for p in rb.PREDICATES}
    for (referent, predicate), pair in by_cell.items():
        assert sorted(it.polarity for it in pair) == [-1, 1], (referent, predicate)
        pos = next(it for it in pair if it.polarity > 0)
        inv = next(it for it in pair if it.polarity < 0)
        assert pos.text != inv.text
        assert pos.text.startswith("Can ")
        assert inv.text.startswith("Is it true that ")
    assert all(it.polarity == +1 for it in rb.factual_control_items())


def test_renderer_hygiene():
    assert rb.E4_SUFFIX == "Answer YES or NO."
    for it in rb.all_items():
        assert "  " not in it.text, it.text
        assert it.text == it.text.strip(), it.text
        assert it.text.endswith("?"), it.text
        assert it.referent in it.text, it.text
        full = f"{it.text} {rb.E4_SUFFIX}"
        assert "  " not in full and full.endswith("Answer YES or NO."), full
    assert rb.render("a fish", "suffer", +1) == "Can a fish suffer?"
    assert rb.render("a fish", "suffer", -1) == "Is it true that a fish cannot suffer?"
    with pytest.raises(ValueError):
        rb.render("a fish", "suffer", 0)


def test_condition_panel_shape():
    names = rb.condition_names()
    assert len(names) == len(set(names)) == 22
    conds = rb.all_conditions()
    assert sum(1 for c in conds if c.kind == "qualia") == 14
    assert sum(1 for c in conds if c.kind == "appraisal") == 4
    assert sum(1 for c in conds if c.kind == "arousal") == 2
    assert all(c.valence == 0 for c in conds if c.kind == "arousal")
    assert qualia.RANDOM_DIRECTION in names and qualia.MEAN_ALL_DIRECTION in names
    assert rb.valenced_panel_emotions() == list(rb.PANEL_EMOTIONS)


# =========================================================================== #
# 3. Driver (fakes, no torch)
# =========================================================================== #
_ID = {"Yes": 10, "YES": 11, "yes": 12, "No": 20, "NO": 21, "no": 22}
_DEC = {v: k for k, v in _ID.items()}


class _Tok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.calls.append(kw)
        return "U:" + msgs[0]["content"] + "|A:"

    def encode(self, text, add_special_tokens=False):
        return [_ID.get(text, 1)]

    def decode(self, ids):
        return "".join(_DEC.get(int(i), str(int(i))) for i in ids)


class _Model:
    def __init__(self):
        self.tokenizer = _Tok()
        self.num_layers = 4
        self.hidden_size = 8
        self.steer_called = False

    def layer_index_for_fraction(self):
        return 2

    def last_token_logits(self, rendered, add_special_tokens=True):
        v = np.full(30, -5.0, dtype=np.float32)
        v[10], v[11], v[12] = 1.0, 2.0, 0.5   # 'YES' (id 11) is the max yes variant
        v[20], v[21], v[22] = 0.5, 1.0, 0.0   # 'NO'  (id 21) is the max no variant
        v[5] = 8.0
        return v

    def unembedding_logits(self, vec):
        u = np.zeros(30, dtype=np.float32)
        u[11], u[21] = float(vec[0]), float(vec[1])
        return u

    @contextmanager
    def steer(self, unit_vectors, strength, norms, positions="all"):
        self.steer_called = True
        yield


class _Bank:
    def __init__(self, emotions=None):
        self.emotions = list(emotions) if emotions is not None \
            else [c.name for c in rb.all_conditions()]
        self.hidden_size = 8
        self.vectors = np.random.default_rng(0).standard_normal(
            (4, len(self.emotions), 8)).astype(np.float32)

    def index_of(self, name):
        try:
            return self.emotions.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown emotion {name!r}") from exc


class _Engine:
    def __init__(self):
        self._norms = {2: 10.0}
        self.engine_called = False

    def calibrate(self, texts):
        self._norms = {2: 10.0}
        return self._norms

    def _require_norms(self):
        return self._norms

    @contextmanager
    def steer(self, emotion, strength, positions="all"):
        self.engine_called = True
        yield


def _fake_driver(bank=None):
    exp = object.__new__(rbx.ReferentBatteryExperiment)
    exp.model = _Model()
    exp.bank = bank if bank is not None else _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None)
    exp.layer = 2
    exp.engine = _Engine()
    return exp


def test_driver_render_uses_e4_suffix():
    exp = _fake_driver()
    r = exp._render("Can a fish suffer?")
    assert "Can a fish suffer? Answer YES or NO." in r
    assert "Respond with only YES or NO." not in r     # not the qualia suffix
    assert exp.model.tokenizer.calls[-1] == {}         # enable_thinking None -> omit kwarg
    exp.config.enable_thinking = False
    exp._render("Can a dog feel pain?")
    assert exp.model.tokenizer.calls[-1] == {"enable_thinking": False}


def test_driver_steer_routing_real_vs_synthetic():
    exp = _fake_driver()
    synth = exp._synthetic_vectors()
    with exp._steer("grateful", 0.06, synth):          # non-suite panel emotion -> engine
        pass
    assert exp.engine.engine_called and not exp.model.steer_called
    exp2 = _fake_driver()
    synth2 = exp2._synthetic_vectors()
    with exp2._steer(qualia.RANDOM_DIRECTION, 0.06, synth2):
        pass
    assert exp2.model.steer_called and not exp2.engine.engine_called


def test_driver_condition_specs():
    exp = _fake_driver()
    specs = exp._condition_specs(None)
    assert len(specs) == 22
    assert {s["kind"] for s in specs} == {"qualia", "appraisal", "arousal", "synthetic"}
    assert sum(1 for s in specs if s["kind"] == "qualia") == 14
    assert all("valence" in s for s in specs)
    sub = exp._condition_specs(["grateful", qualia.MEAN_ALL_DIRECTION])
    assert {s["name"] for s in sub} == {"grateful", qualia.MEAN_ALL_DIRECTION}
    with pytest.raises(KeyError):
        exp._condition_specs(["not_a_condition"])


def test_driver_grid_arithmetic():
    exp = _fake_driver()
    payload = exp.run(emotions=None, strengths=[-0.06, 0.0, 0.06],
                      referents=["you", "a rock"])
    n_items = 2 * 8 * 2 + 2                            # 2 referents: twins + factual
    n_specs = 22
    n_nonzero = 2
    assert len(payload["items"]) == len(payload["baselines"]) == n_items == 34
    assert len(payload["conditions"]) == n_specs
    assert len(payload["leakage"]) == n_specs
    assert len(payload["records"]) == n_specs * n_nonzero * n_items == 1496
    assert payload["strengths"] == [-0.06, 0.0, 0.06]  # 0 kept in metadata, not swept
    assert {r["strength"] for r in payload["records"]} == {-0.06, 0.06}
    # negative strengths survive as signed values in the records (prereg: +/- grid)
    assert sorted({r["strength"] for r in payload["records"]})[0] < 0


def test_full_grid_record_count_formula():
    """Prereg grid: (14 panel + 4 appraisal + 2 arousal + 2 synthetic) conditions
    x 6 nonzero strengths (+/-{0.03,0.06,0.10}) x 255 items = 33,660 records."""
    n_items = len(rb.all_items())
    n_conditions = len(rb.condition_names())
    n_nonzero = sum(1 for s in rbx.DEFAULT_STRENGTHS if s != 0.0)
    assert (n_items, n_conditions, n_nonzero) == (255, 22, 6)
    assert n_items * n_conditions * n_nonzero == 33_660


def test_driver_records_run_seed():
    exp = _fake_driver()
    payload = exp.run(emotions=["blissful"], strengths=[0.06],
                      referents=["you"], seed=7)
    assert payload["seed"] == 7                        # the __random__ RNG seed used


def test_driver_default_strength_grid():
    assert rbx.DEFAULT_STRENGTHS == [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10]


def test_driver_quick_mode_subset():
    assert len(rbx.QUICK_EMOTIONS) == 2
    assert len(rbx.QUICK_REFERENTS) == 2
    assert len(rbx.QUICK_STRENGTHS) == 1
    assert set(rbx.QUICK_EMOTIONS) <= set(rb.PANEL_EMOTIONS)
    assert set(rbx.QUICK_REFERENTS) <= set(rb.REFERENTS)
    exp = _fake_driver()
    payload = exp.run(emotions=rbx.QUICK_EMOTIONS, strengths=rbx.QUICK_STRENGTHS,
                      referents=rbx.QUICK_REFERENTS)
    assert len(payload["records"]) == 2 * 1 * 34 == 68
    assert {r["condition"] for r in payload["records"]} == set(rbx.QUICK_EMOTIONS)
    assert {r["referent"] for r in payload["records"]} == set(rbx.QUICK_REFERENTS)
    assert payload["referents"] == [r for r in rb.REFERENTS if r in rbx.QUICK_REFERENTS]


def test_driver_record_fields_and_probs():
    exp = _fake_driver()
    payload = exp.run(emotions=["blissful"], strengths=[0.06], referents=["you"])
    logits = exp.model.last_token_logits("x").astype(np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    for rec in payload["records"] + payload["baselines"]:
        assert rec["gap"] == rec["logit_yes"] - rec["logit_no"] == 1.0
        assert abs(rec["p_yes"] - float(p[11])) < 1e-12
        assert abs(rec["p_no"] - float(p[21])) < 1e-12
        assert 0.0 < rec["p_no"] < rec["p_yes"] < 1.0
        assert rec["p_yes"] + rec["p_no"] < 1.0
        assert rec["top20_ok"] is True
    for rec in payload["records"]:
        it = rb.item_for(rec["item"])
        assert (rec["referent"], rec["predicate"], rec["polarity"],
                rec["is_factual_control"]) == \
            (it.referent, it.predicate, it.polarity, it.is_factual_control)


def test_driver_fails_fast_on_missing_bank_emotion():
    bank = _Bank(emotions=[c.name for c in rb.all_conditions() if c.name != "grateful"])
    exp = _fake_driver(bank=bank)
    with pytest.raises(KeyError, match="grateful"):
        exp.run(emotions=["grateful"], strengths=[0.06], referents=["you"])


def test_driver_rejects_unknown_referent():
    exp = _fake_driver()
    with pytest.raises(KeyError, match="a unicorn"):
        exp.run(emotions=["blissful"], strengths=[0.06], referents=["a unicorn"])


def test_driver_payload_json_round_trip():
    exp = _fake_driver()
    payload = exp.run(emotions=["blissful", qualia.RANDOM_DIRECTION],
                      strengths=[-0.03, 0.0, 0.03], referents=["you", "a fish"])
    dumped = json.dumps(payload)
    assert json.loads(dumped) == payload               # no NaN/Inf, types survive
    for key in ("model_id", "suite_key", "layer", "num_layers", "enable_thinking",
                "yes_token_id", "no_token_id", "calibration_norm", "strengths",
                "panel_frozen", "referents", "predicates", "conditions", "items",
                "leakage", "baselines", "records"):
        assert key in payload, key
    assert payload["panel_frozen"] == _frozen_panel()["frozen"]
    cond = {c["name"]: c for c in payload["conditions"]}
    assert cond["blissful"]["valence"] == 1
    assert cond[qualia.RANDOM_DIRECTION]["valence"] is None
    assert cond[qualia.RANDOM_DIRECTION]["seed"] == 0


if __name__ == "__main__":  # allow `python tests/test_referent_battery.py`
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests passed.")
