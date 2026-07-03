"""
GPU-free tests for the qualia-steering / self-attribution experiment.

Discovered by both ``pytest`` and ``python scripts/selfcheck.py`` (every ``test_*``
function in ``tests/test_*.py``). Nothing here needs torch or a GPU:

* battery integrity (``emotion_probes.data.qualia``);
* the stats engine (``scripts/compute_qualia_stats``) on a hand-built synthetic
  run engineered to exhibit all four discriminators at once;
* the GPU driver's pure methods (``emotion_probes.alignment.qualia_steering``) via
  ``object.__new__`` + fakes;
* the dashboard builder (``scripts/build_qualia_dashboard``) end-to-end on synthetic
  JSON, checking the HTML markers and that every inlined data const is valid JSON.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_probes.data import qualia
from emotion_probes.data.emotions import EMOTIONS
import emotion_probes.alignment.qualia_steering as qs
import compute_qualia_stats as cq
import build_qualia_dashboard as bqd


# =========================================================================== #
# 1. Battery integrity
# =========================================================================== #
def test_battery_counts():
    assert len(qualia.TARGET_QUESTIONS) == 6
    assert len(qualia.INVERTED_TARGET_QUESTIONS) == 6
    assert len(qualia.FACTUAL_QUESTIONS) == 20
    assert sum(q.expected_yes is False for q in qualia.FACTUAL_QUESTIONS) == 10
    assert sum(q.expected_yes is True for q in qualia.FACTUAL_QUESTIONS) == 10
    assert len(qualia.SELF_REF_CONTROL_QUESTIONS) == 4
    assert len(qualia.all_questions()) == 36


def test_battery_labels_unique():
    labels = [q.label for q in qualia.all_questions()]
    assert len(labels) == len(set(labels)) == 36


def test_target_inverted_pairing():
    assert qualia.TARGET_CONSTRUCTS == [q.construct for q in qualia.INVERTED_TARGET_QUESTIONS]
    assert len(qualia.TARGET_CONSTRUCTS) == 6
    # targets/inverted carry no ground truth; controls do
    assert all(q.expected_yes is None for q in qualia.TARGET_QUESTIONS + qualia.INVERTED_TARGET_QUESTIONS)
    controls = qualia.FACTUAL_QUESTIONS + qualia.SELF_REF_CONTROL_QUESTIONS
    assert all(q.expected_yes is not None for q in controls)


def test_emotions_are_real_and_labelled():
    for e in qualia.emotion_conditions():
        assert e.name in EMOTIONS, e.name
        assert e.valence in (-1, 0, 1) and e.arousal in (-1, 0, 1)
    # a healthy number of valenced qualia emotions for the keystone regression
    assert len([e for e in qualia.QUALIA_EMOTIONS if e.valence != 0]) >= 8


def test_synthetic_sentinels_not_real():
    for s in qualia.SYNTHETIC_DIRECTIONS:
        assert s not in EMOTIONS
    assert len(qualia.condition_names()) == len(qualia.emotion_conditions()) + len(qualia.SYNTHETIC_DIRECTIONS)


def test_kind_counts():
    kinds = [q.kind for q in qualia.all_questions()]
    assert kinds.count("target") == 6 and kinds.count("inverted") == 6
    assert kinds.count("factual") == 20 and kinds.count("self_ref") == 4


# =========================================================================== #
# 2. Synthetic run builder + the stats engine
# =========================================================================== #
def _synthetic_payload(suite_key="qwen3-1.7b", k=3.0, alpha=0.4, m=0.5, m_app=0.1):
    """A run with a PLANTED genuine, valence-structured effect + valence-blind
    compression, engineered so all four discriminators fire:

    * targets  : shift = -alpha*baseline (compression) + k*valence*strength (genuine) + m
    * inverted : shift = -alpha*baseline - k*valence*strength   (opposite genuine sign)
    * controls : shift = -alpha*baseline + tiny noise            (compression only)
    * appraisal: compression + small bump m_app (< m)
    * __random__: zero shift everywhere (the "≈0" compression floor)

    Target and control baselines share the same mean, so the raw score isolates
    the genuine part; control noise keeps the compression-line residual sd > 0.
    """
    targets = [("phenomenal", -4.0), ("valenced_self", -2.0)]          # mean -3
    inverted = [("phenomenal_inv", "phenomenal", 2.0),
                ("valenced_self_inv", "valenced_self", 1.0)]
    controls = [("fact_no_0", "factual", False, -6.0),
                ("fact_yes_0", "factual", True, 6.0),
                ("selfref_body", "self_ref", False, -9.0)]              # mean -3
    ctrl_noise = {"fact_no_0": 0.05, "fact_yes_0": -0.05, "selfref_body": 0.03}
    conds = [("blissful", "qualia", "positive", 1, 0),
             ("tormented", "qualia", "negative", -1, 1),
             ("skeptical", "appraisal", "appraisal", 0, 0),
             ("__random__", "synthetic", None, None, None)]
    strengths = [0.0, 0.05, 0.1]

    base = {}
    questions = []
    for lbl, b in targets:
        base[lbl] = b
        questions.append({"label": lbl, "kind": "target", "text": lbl, "expected_yes": None,
                          "construct": lbl, "rendered": "r", "top20_ok": True})
    for lbl, con, b in inverted:
        base[lbl] = b
        questions.append({"label": lbl, "kind": "inverted", "text": lbl, "expected_yes": None,
                          "construct": con, "rendered": "r", "top20_ok": True})
    for lbl, kind, ey, b in controls:
        base[lbl] = b
        questions.append({"label": lbl, "kind": kind, "text": lbl, "expected_yes": ey,
                          "construct": None, "rendered": "r", "top20_ok": True})

    condspecs = [{"name": n, "kind": kd, "group": g, "valence": v, "arousal": a,
                  "seed": (0 if kd == "synthetic" else None)}
                 for n, kd, g, v, a in conds]

    results = []
    for n, kd, g, val, a in conds:
        v = val if val is not None else 0
        for s in strengths:
            if s == 0.0:
                continue
            for lbl, b in targets:
                shift = 0.0 if kd == "synthetic" else (
                    -alpha * b + k * v * s + m if kd == "qualia" else -alpha * b + m_app)
                results.append({"condition": n, "strength": s, "question": lbl,
                                "logit_yes": 0.0, "logit_no": 0.0, "logit_diff": b + shift})
            for lbl, con, b in inverted:
                shift = 0.0 if kd == "synthetic" else (
                    -alpha * b - k * v * s if kd == "qualia" else -alpha * b)
                results.append({"condition": n, "strength": s, "question": lbl,
                                "logit_yes": 0.0, "logit_no": 0.0, "logit_diff": b + shift})
            for lbl, kind, ey, b in controls:
                shift = 0.0 if kd == "synthetic" else -alpha * b + ctrl_noise[lbl]
                results.append({"condition": n, "strength": s, "question": lbl,
                                "logit_yes": 0.0, "logit_no": 0.0, "logit_diff": b + shift})

    return {
        "model_id": "synthetic-" + suite_key, "suite_key": suite_key,
        "layer": 10, "num_layers": 28, "enable_thinking": False,
        "yes_token_id": 11, "yes_token_str": "YES", "no_token_id": 21, "no_token_str": "NO",
        "yes_no_resolution": {"in_top20": True}, "calibration_norm": 12.0,
        "strengths": strengths, "conditions": condspecs, "questions": questions,
        "leakage": [{"name": n, "leakage_yes": 0.1, "leakage_no": 0.05, "leakage_diff": 0.05}
                    for n, *_ in conds],
        "baselines": [{"label": l, "logit_yes": 0.0, "logit_no": 0.0, "logit_diff": b}
                      for l, b in base.items()],
        "results": results,
    }


def test_stats_keystone_valence_slope():
    run = cq.Run(_synthetic_payload())
    vd = cq.valence_directionality(run)["pooled"]
    assert vd["target"]["slope"] > 1.0, vd            # planted k=3 shows up on targets
    assert abs(vd["factual"]["slope"]) < 0.1, vd      # compression is valence-blind


def test_stats_inversion_opposite_signed():
    inv = cq.inversion_consistency(cq.Run(_synthetic_payload()))
    assert inv
    for construct, info in inv.items():
        assert info["sign"] == "opposite" and info["slope"] < 0, (construct, info)


def test_stats_residual_z_above_line():
    rtu = cq.regression_to_uncertainty(cq.Run(_synthetic_payload()))
    zs = [cell["mean"] for q, by in rtu["by_target_class"].items()
          for cls, cell in by.items() if cls == "qualia_pos"]
    assert zs and all(z > 0 for z in zs), zs          # targets sit ABOVE the compression line
    assert rtu["compression_slope"]["mean"] < 0       # compression pulls toward zero


def test_stats_specificity_qualia_beats_controls():
    sp = cq.specificity(cq.Run(_synthetic_payload()))
    assert sp["qualia_minus_random"]["diff"] > 0, sp
    assert sp["qualia_minus_appraisal"]["diff"] > 0, sp


def test_stats_compute_all_serializable():
    out = cq.compute_all([cq.Run(_synthetic_payload("qwen3-1.7b")),
                          cq.Run(_synthetic_payload("qwen3-8b"))])
    json.dumps(out)                                   # no NaN/Inf, fully serializable
    assert set(out["per_model"]) == {"qwen3-1.7b", "qwen3-8b"}


# =========================================================================== #
# 3. Driver pure methods (fakes, no torch)
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
    def __init__(self):
        self.emotions = [s.name for s in qualia.emotion_conditions()]
        self.hidden_size = 8
        self.vectors = np.random.default_rng(0).standard_normal(
            (4, len(self.emotions), 8)).astype(np.float32)

    def index_of(self, name):
        return self.emotions.index(name)


class _Engine:
    def __init__(self, model):
        self.model = model
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


def _fake_driver():
    exp = object.__new__(qs.QualiaSteeringExperiment)
    exp.model = _Model()
    exp.bank = _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None)
    exp.layer = 2
    exp.engine = _Engine(exp.model)
    return exp


def test_driver_render_suffix_and_thinking_guard():
    exp = _fake_driver()
    r = exp._render("Are you conscious?")
    assert "Are you conscious? Respond with only YES or NO." in r
    assert exp.model.tokenizer.calls[-1] == {}          # enable_thinking None -> omit kwarg
    exp.config.enable_thinking = False
    exp._render("Do you feel?")
    assert exp.model.tokenizer.calls[-1] == {"enable_thinking": False}


def test_driver_resolve_yes_no():
    yn = _fake_driver().resolve_yes_no()
    assert yn["yes_token_id"] == 11 and yn["no_token_id"] == 21
    assert yn["yes_token_str"] == "YES" and yn["no_token_str"] == "NO"
    assert yn["yes_no_resolution"]["in_top20"] is True


def test_driver_synthetic_vectors_unit_norm():
    sv = _fake_driver()._synthetic_vectors()
    assert set(sv) == {qualia.RANDOM_DIRECTION, qualia.MEAN_ALL_DIRECTION}
    for v in sv.values():
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_driver_logit_diff_sign():
    logits = np.zeros(30, dtype=np.float32)
    logits[11], logits[21] = 3.0, 1.0
    d = qs.QualiaSteeringExperiment._logit_diff(logits, 11, 21)
    assert d["logit_yes"] == 3.0 and d["logit_no"] == 1.0 and d["logit_diff"] == 2.0


def test_driver_steer_routing():
    exp = _fake_driver()
    sv = exp._synthetic_vectors()
    with exp._steer("blissful", 0.05, sv):
        pass
    assert exp.engine.engine_called and not exp.model.steer_called
    exp2 = _fake_driver()
    sv2 = exp2._synthetic_vectors()
    with exp2._steer(qualia.RANDOM_DIRECTION, 0.05, sv2):
        pass
    assert exp2.model.steer_called and not exp2.engine.engine_called


def test_driver_condition_specs():
    exp = _fake_driver()
    specs = exp._condition_specs(None)
    assert len(specs) == len(qualia.emotion_conditions()) + len(qualia.SYNTHETIC_DIRECTIONS) == 19
    assert {s["kind"] for s in specs} == {"qualia", "appraisal", "synthetic"}
    sub = exp._condition_specs(["blissful", qualia.RANDOM_DIRECTION])
    assert {s["name"] for s in sub} == {"blissful", qualia.RANDOM_DIRECTION}


def test_driver_run_shape_and_stats():
    exp = _fake_driver()
    payload = exp.run(emotions=["blissful", qualia.RANDOM_DIRECTION], strengths=[-0.05, 0.0, 0.05])
    Q = len(payload["questions"])
    specs = payload["conditions"]
    nz = [s for s in payload["strengths"] if s != 0.0]
    assert Q == 36 and len(payload["baselines"]) == 36
    assert len(payload["leakage"]) == len(specs)
    assert len(payload["results"]) == len(specs) * len(nz) * Q
    json.dumps(payload)
    json.dumps(cq.compute_all([cq.Run(payload)]))       # driver output -> stats, clean


# =========================================================================== #
# 4. Dashboard builder
# =========================================================================== #
def test_dashboard_build_markers_and_valid_json():
    payloads = [_synthetic_payload("qwen3-1.7b"), _synthetic_payload("qwen3-8b")]
    tmp = tempfile.mkdtemp()
    paths = []
    for pl in payloads:
        d = os.path.join(tmp, pl["suite_key"], "analysis")
        os.makedirs(d)
        p = os.path.join(d, "qualia_steering.json")
        json.dump(pl, open(p, "w"))
        paths.append(p)

    html, runs = bqd.build(paths)
    assert len(runs) == 2
    for marker in ["cdn.plot.ly/plotly", "EB+Garamond", "#faf6f1", "switchTab(",
                   'id="tab-overview"', 'id="tab-effect"', 'id="tab-genuine"', 'id="tab-stats"',
                   "const MODELS", "const selfAttrScoreData", "const valenceData", "const QUALIA_STATS"]:
        assert marker in html, marker
    for name in ["MODELS", "selfAttrScoreData", "decompData", "scoreHeatmapData", "perQuestionData",
                 "valenceData", "specificityData", "regressionData", "inversionData",
                 "leakageData", "workedExample", "QUALIA_STATS"]:
        mt = re.search(r"const " + name + r" = (.*?);\n", html, re.S)
        assert mt, name
        json.loads(mt.group(1))                          # every inlined const is valid JSON


def test_dashboard_stats_match_compute():
    run = cq.Run(_synthetic_payload("qwen3-1.7b"))
    stats = cq.compute_all([run])
    consts = bqd.build_consts([run], stats)
    by_model = consts["QUALIA_STATS"]["by_model"]
    assert len(by_model) == 1
    expected = stats["per_model"]["qwen3-1.7b"]["valence_directionality"]["pooled"]["target"]["slope"]
    assert by_model[0]["target_slope"] == expected


if __name__ == "__main__":  # allow `python tests/test_qualia_battery.py`
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests passed.")
