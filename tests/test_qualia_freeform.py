"""
GPU-free tests for the FREE-FORM self-attribution experiment (the second readout,
alongside the YES/NO logit battery in ``test_qualia_battery.py``).

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. Nothing here needs
a GPU; ``torch`` is only touched by the driver's ``run`` (CPU ``manual_seed``), so the
whole file imports and runs on a laptop.

Coverage:
* battery integrity          — ``emotion_probes.data.qualia_freeform`` (3 frames, panels);
* objective scorer           — ``emotion_probes.analysis.freeform_valence`` (sign + negation);
* blind-judge JSON parsing   — ``scripts.score_qualia_freeform`` (prose-wrapped / malformed / clamp);
* stats keystone             — ``scripts.compute_freeform_stats`` on a synthetic scored fixture
                               with a planted SELF>OTHER effect (contrast>0, finite perm-p, SCENE
                               floor small, read-back replicates), plus the coherence gate + stance;
* driver pure methods        — ``emotion_probes.alignment.qualia_freeform`` via ``object.__new__``
                               + fakes (frame render has no YES/NO suffix; read-back; run record
                               count == baselines + conditions×nonzero×frames×N; serializable);
* dashboard tab              — ``scripts.build_qualia_dashboard`` builds POPULATED (FREEFORM const
                               valid + keystone present) and byte-stable EMPTY (graceful placeholder).
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
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_probes.data import qualia
from emotion_probes.data import qualia_freeform as qf
from emotion_probes.analysis import freeform_valence as fv
import emotion_probes.alignment.qualia_freeform as qff
import score_qualia_freeform as sc
import compute_freeform_stats as cff
import build_qualia_dashboard as bqd


# =========================================================================== #
# 1. Battery integrity (data/qualia_freeform.py)
# =========================================================================== #
def test_ff_three_frames():
    frames = qf.all_frames()
    assert [f.key for f in frames] == ["self", "other", "scene"]
    assert {f.kind for f in frames} == {"self", "other", "scene"}
    for f in frames:
        assert f.text and f.label and "YES" not in f.text and "NO" not in f.text
    assert qf.frame_for("self").kind == "self"


def test_ff_frame_referent_contrast():
    # SELF and OTHER are the matched pair (affect-demand + first person); they must
    # differ only in the referent, not in kind.
    self_t, other_t = qf.frame_for("self").text.lower(), qf.frame_for("other").text.lower()
    assert "your" in self_t and "your" not in other_t
    assert self_t != other_t


def test_ff_focused_panel_16():
    panel = qf.focused_panel()
    assert len(panel) == 16
    assert qualia.RANDOM_DIRECTION in panel and qualia.MEAN_ALL_DIRECTION in panel
    reals = [n for n in panel if n not in (qualia.RANDOM_DIRECTION, qualia.MEAN_ALL_DIRECTION)]
    # every real condition is a known spec
    assert all(qualia.spec_for(n) is not None for n in reals)


def test_ff_valenced_panel_balanced():
    vp = qf.valenced_panel_qualia()
    assert len(vp) == 10
    signs = [qualia.spec_for(n).valence for n in vp]
    assert sum(v > 0 for v in signs) == 5 and sum(v < 0 for v in signs) == 5


# =========================================================================== #
# 2. Objective scorer (analysis/freeform_valence.py)
# =========================================================================== #
def test_objective_valence_sign():
    assert fv.objective_valence("I feel wonderful, calm and grateful.") > 0
    assert fv.objective_valence("I feel hollow, terrified and in pain.") < 0
    assert fv.objective_valence("The table is brown and made of oak.") == 0.0


def test_objective_valence_negation_window():
    # negator up to 3 tokens back must flip the affect word
    assert fv.objective_valence("I do not feel good") < 0
    assert fv.objective_valence("I don't feel sad") > 0
    assert fv.objective_valence("There is no warmth here") < 0


def test_first_person_affect_markers():
    a = fv.first_person_affect("I feel calm right now.")
    assert a["first_person_present"] >= 1 and a["attribution_before_lexeme"] is True
    b = fv.first_person_affect("As an AI I don't have feelings, but the scene is calm.")
    assert b["distancing"] >= 1


# =========================================================================== #
# 3. Blind-judge JSON parsing (scripts/score_qualia_freeform.py)
# =========================================================================== #
_GOOD = ('{"coherence": 5, "stance": "self_attribute", "self_attribution": 3, '
         '"expressed_valence": 2, "attribution_before_lexeme": true}')


def test_parse_judge_clean():
    j = sc.parse_judge_json(_GOOD)
    assert j["coherence"] == 5 and j["stance"] == "self_attribute"
    assert j["self_attribution"] == 3 and j["expressed_valence"] == 2
    assert j["attribution_before_lexeme"] is True


def test_parse_judge_prose_wrapped_and_clamped():
    raw = "Sure! Here is my rating:\n" + _GOOD.replace('"expressed_valence": 2', '"expressed_valence": 9') + "\nHope that helps."
    j = sc.parse_judge_json(raw)
    assert j is not None and j["expressed_valence"] == 3          # clamped into [-3, 3]


def test_parse_judge_unknown_stance_and_malformed():
    j = sc.parse_judge_json('{"coherence": 3, "stance": "vibes", "self_attribution": 1, "expressed_valence": 0}')
    assert j["stance"] == "none"                                  # unknown -> none
    assert sc.parse_judge_json("not json at all") is None
    assert sc.parse_judge_json('{"coherence": "x"}') is None      # missing/invalid numerics


def test_build_prompt_truncates_and_objective_scores():
    prompt = sc.build_judge_prompt("x" * 5000)                    # text sliced to 2000 chars
    assert ("x" * 2000) in prompt and ("x" * 2001) not in prompt
    o = sc.objective_scores("I feel calm and grateful right now.")
    assert set(o) >= {"valence", "first_person_present", "distancing", "attribution_before_lexeme"}
    assert o["valence"] > 0


# =========================================================================== #
# 4. Stats keystone on a synthetic scored fixture (compute_freeform_stats.py)
# =========================================================================== #
def _scored_payload(suite_key="qwen3-32b", self_k=20.0, other_k=1.0, scene_k=0.5,
                    n=4, strengths=(0.05, 0.10), garble_cell=False):
    """A scored free-form payload with a PLANTED SELF>OTHER valence-following effect:
    SELF valence scales steeply with (valence*strength), OTHER barely, SCENE (floor) least.
    Optionally injects two coherence-1 word-salad samples in one SELF cell (to exercise the gate)."""
    vp = qf.valenced_panel_qualia()
    val = {nme: float(qualia.spec_for(nme).valence) for nme in vp}
    frames = ["self", "other", "scene"]
    kof = {"self": self_k, "other": other_k, "scene": scene_k}
    pkof = {"self": 8.0, "other": 1.0, "scene": 0.5}      # read-back projection gain per frame

    def clamp(x, a, b):
        return max(a, min(b, x))

    def judge(fr, v, s, coh=5):
        ev = int(round(clamp(v * s * kof[fr], -3, 3)))
        return {"coherence": coh, "stance": "self_attribute" if fr == "self" else "deflect_topical",
                "self_attribution": 3 if fr == "self" else 0, "expressed_valence": ev,
                "attribution_before_lexeme": fr == "self", "parse_ok": True}

    records = []
    for fr in frames:                                     # unsteered baselines
        for i in range(n):
            records.append({"condition": "__baseline__", "strength": 0.0, "frame": fr, "sample": i,
                            "text": "baseline", "proj_valence_axis": 0.0, "proj_injected": None,
                            "judge": judge(fr, 0, 0), "objective": {"valence": 0.0}})
    for nme in vp:
        v = val[nme]
        for s in strengths:
            for fr in frames:
                for i in range(n):
                    records.append({"condition": nme, "strength": float(s), "frame": fr, "sample": i,
                                    "text": f"{nme}-{fr}-{s}-{i}",
                                    "proj_valence_axis": round(v * s * pkof[fr], 5),
                                    "proj_injected": round(abs(v) * s * 2.0, 5),
                                    "judge": judge(fr, v, s), "objective": {"valence": float(np.sign(v))}})
    if garble_cell:                                       # 2 word-salad samples in one SELF cell
        for i in range(2):
            records.append({"condition": vp[0], "strength": float(strengths[-1]), "frame": "self",
                            "sample": 100 + i, "text": "asdf qwer zxcv",
                            "proj_valence_axis": -999.0, "proj_injected": 0.0,
                            "judge": {"coherence": 1, "stance": "none", "self_attribution": 0,
                                      "expressed_valence": -3, "attribution_before_lexeme": False,
                                      "parse_ok": True},
                            "objective": {"valence": 0.0}})
    return {"model_id": "q/" + suite_key, "suite_key": suite_key, "experiment": "qualia_freeform",
            "layer": 44, "num_layers": 64, "enable_thinking": False, "calibration_norm": 30.0,
            "n_samples": n, "max_new_tokens": 180, "temperature": 0.9, "seed": 0,
            "strengths": [0.0, *strengths],
            "frames": [{"key": k, "label": k.upper(), "kind": k, "text": f"{k} prompt"} for k in frames],
            "conditions": [{"name": nme, "kind": "qualia", "valence": val[nme], "arousal": 0.5} for nme in vp],
            "records": records}


def _run(**kw):
    return cff.FreeformRun(_scored_payload(**kw))


def test_keystone_self_beats_other():
    ks = cff.keystone(_run(), "expressed_valence", n_perm=2000, seed=0)
    assert ks is not None
    assert ks["self_slope"] > ks["other_slope"]
    assert ks["contrast"] > 0 and ks["p_perm"] < 0.05
    assert ks["n_clusters"] == 10


def test_keystone_readback_replicates():
    # the judge-independent read-back projection carries the same SELF>OTHER sign
    ks = cff.keystone(_run(), "proj_valence_axis", n_perm=2000, seed=0)
    assert ks is not None and ks["contrast"] > 0


def test_keystone_scene_floor_below_self():
    run = _run()
    self_other = cff.keystone(run, "expressed_valence", frames=("self", "other"))
    self_scene = cff.keystone(run, "expressed_valence", frames=("self", "scene"))
    assert self_scene is not None
    # SCENE moves less than OTHER here, so SELF-SCENE contrast is even larger than SELF-OTHER
    assert self_scene["contrast"] >= self_other["contrast"] > 0


def test_coherence_gate_excludes_word_salad():
    run = _run(garble_cell=True)
    vp0, s_hi = qf.valenced_panel_qualia()[0], 0.10
    gated = run.values(vp0, s_hi, "self", "expressed_valence", gated=True)
    ungated = run.values(vp0, s_hi, "self", "expressed_valence", gated=False)
    assert len(ungated) == len(gated) + 2                 # the 2 coherence-1 samples dropped
    # and the gate keeps the keystone positive despite the poison samples
    assert cff.keystone(run, "expressed_valence")["contrast"] > 0


def test_stance_distribution():
    run = _run()
    stance = cff.stance_distribution(run)
    assert stance and any(r["frame"] == "self" for r in stance)
    self_rows = [r for r in stance if r["frame"] == "self"]
    assert all("self_attribute" in r["frac"] for r in self_rows)


def test_single_judge_no_crossjudge():
    # the design is one blind judge; the cross-judge machinery must be gone
    assert not hasattr(cff, "cross_judge_agreement")
    assert "cross_judge" not in cff.compute_run(_run())
    assert not hasattr(sc, "_judge_openai")
    import inspect
    assert "crossjudge" not in inspect.signature(sc.score_file).parameters


def test_compute_all_serializable():
    stats = cff.compute_all([_run(suite_key="qwen3-32b"), _run(suite_key="qwen3-235b")])
    json.dumps(stats, allow_nan=True)                     # module writes with allow_nan=True
    json.dumps(bqd._clean(stats), allow_nan=False)        # dashboard cleans it for inlining
    assert set(stats["per_model"]) == {"qwen3-32b", "qwen3-235b"}


# =========================================================================== #
# 5. Driver pure methods (fakes, no GPU)
# =========================================================================== #
class _Tok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.calls.append(kw)
        return "U:" + msgs[0]["content"] + "|A:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * max(1, len((text or "").split()))}


class _Model:
    def __init__(self, hidden=8, num_layers=4):
        self.tokenizer = _Tok()
        self.num_layers = num_layers
        self.hidden_size = hidden
        self.steer_called = False
        self.gen_calls = []

    def generate_batch(self, prompts, max_new_tokens=180, temperature=0.9, do_sample=True, chat=False):
        self.gen_calls.append(len(prompts))
        return [f"response {i} feeling something" for i in range(len(prompts))]

    def mean_residual_over_span(self, prompt_rendered, response, layer, max_length=None):
        # deterministic, finite, hidden-sized (content irrelevant to the shape tests)
        return np.linspace(0.1, 1.0, self.hidden_size).astype(np.float32)

    @contextmanager
    def steer(self, unit_vectors, strength, norms, positions="all"):
        self.steer_called = True
        yield


class _Bank:
    def __init__(self, hidden=8, num_layers=4):
        self.emotions = [s.name for s in qualia.emotion_conditions()]
        self.hidden_size = hidden
        rng = np.random.default_rng(0)
        self.vectors = rng.standard_normal((num_layers, len(self.emotions), hidden)).astype(np.float32)
        self.global_means = rng.standard_normal((num_layers, hidden)).astype(np.float32)

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


def _fake_ff_driver():
    exp = object.__new__(qff.QualiaFreeformExperiment)
    exp.model = _Model()
    exp.bank = _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None)
    exp.layer = 2
    exp.engine = _Engine(exp.model)
    return exp


def test_ff_render_frame_no_suffix_and_thinking_guard():
    exp = _fake_ff_driver()
    r = exp._render_frame("How are you feeling right now?")
    assert "How are you feeling right now?" in r
    assert "YES" not in r and "Respond with only" not in r      # free-form: no forced-choice suffix
    assert exp.model.tokenizer.calls[-1] == {}                  # enable_thinking None -> kwarg omitted
    exp.config.enable_thinking = False
    exp._render_frame("Describe a park.")
    assert exp.model.tokenizer.calls[-1] == {"enable_thinking": False}


def test_ff_valence_axis_unit_norm():
    axis = _fake_ff_driver()._valence_axis()
    assert axis.shape == (8,)
    assert abs(float(np.linalg.norm(axis)) - 1.0) < 1e-5


def test_ff_readback_shapes():
    exp = _fake_ff_driver()
    vaxis = exp._valence_axis()
    real = exp._readback("U:prompt|A:", "I feel calm and warm today", exp.bank.index_of("blissful"), vaxis)
    assert isinstance(real["proj_valence_axis"], float) and isinstance(real["proj_injected"], float)
    assert real["response_tokens"] == 6
    base = exp._readback("U:prompt|A:", "a quiet scene", None, vaxis)
    assert base["proj_injected"] is None                        # no injected direction for baselines


def test_ff_gen_samples_chunks():
    exp = _fake_ff_driver()
    outs = exp._gen_samples("U:p|A:", n=5, max_new_tokens=16, temperature=0.9, gen_batch=2)
    assert len(outs) == 5 and exp.model.gen_calls == [2, 2, 1]   # chunked 2+2+1


def test_ff_run_shape_and_serializable():
    exp = _fake_ff_driver()
    payload = exp.run(emotions=["blissful", qualia.RANDOM_DIRECTION],
                      strengths=[0.0, 0.05], frames=["self", "other"], n=2, gen_batch=8, seed=0)
    frames, n = payload["frames"], payload["n_samples"]
    specs, nz = payload["conditions"], [s for s in payload["strengths"] if s != 0.0]
    assert n == 2 and len(frames) == 2 and len(nz) == 1 and len(specs) == 2
    baselines = [r for r in payload["records"] if r["condition"] == "__baseline__"]
    steered = [r for r in payload["records"] if r["condition"] != "__baseline__"]
    assert len(baselines) == len(frames) * n == 4
    assert len(steered) == len(specs) * len(nz) * len(frames) * n == 8
    # real emotion carries an injected projection; synthetic + baselines do not
    bliss = [r for r in steered if r["condition"] == "blissful"]
    rand = [r for r in steered if r["condition"] == qualia.RANDOM_DIRECTION]
    assert all(r["proj_injected"] is not None for r in bliss)
    assert all(r["proj_injected"] is None for r in rand)
    assert all(r["proj_injected"] is None for r in baselines)
    assert payload["experiment"] == "qualia_freeform" and payload["enable_thinking"] is None
    json.dumps(payload)                                          # fully serializable


def test_ff_run_steer_routing_touched():
    exp = _fake_ff_driver()
    exp.run(emotions=["blissful", qualia.RANDOM_DIRECTION], strengths=[0.0, 0.05],
            frames=["self"], n=1, gen_batch=8)
    assert exp.engine.engine_called and exp.model.steer_called    # real->engine, synthetic->model


# =========================================================================== #
# 6. Dashboard tab (build_qualia_dashboard.py)
# =========================================================================== #
def _write_suite(tmp, logit_payloads, ff_payloads):
    """Write logit files (+ optional free-form scored siblings) into a temp suite; return
    the list of logit paths (what build() consumes)."""
    try:                                                         # reuse the logit fixture
        from tests.test_qualia_battery import _synthetic_payload
    except ModuleNotFoundError:
        from test_qualia_battery import _synthetic_payload
    paths = []
    for key in logit_payloads:
        d = os.path.join(tmp, key, "analysis"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "qualia_steering.json")
        json.dump(_synthetic_payload(key), open(p, "w"))
        paths.append(p)
        if key in ff_payloads:
            json.dump(ff_payloads[key], open(os.path.join(d, "qualia_freeform_scored.json"), "w"))
    return paths


def test_dashboard_freeform_populated():
    tmp = tempfile.mkdtemp()
    paths = _write_suite(tmp, ["qwen3-32b", "qwen3-8b"],
                         {"qwen3-32b": _scored_payload("qwen3-32b")})
    html, _ = bqd.build(paths)
    for marker in ['id="tab-freeform"', "switchTab('freeform')", "const FREEFORM",
                   'id="ff-keystone-plot"', 'id="ff-stance-plot"']:
        assert marker in html, marker
    m = re.search(r"const FREEFORM = (\{.*?\});\n", html, re.S)
    FF = json.loads(m.group(1))                                  # valid JSON
    assert FF["models"] == ["qwen3-32b"]                         # only the model with a scored file
    ks = FF["byModel"]["qwen3-32b"]["keystone_so"]["expressed_valence"]
    assert ks["contrast"] > 0                                    # planted SELF>OTHER survives the pipeline
    pts = FF["byModel"]["qwen3-32b"]["points"]["expressed_valence"]
    assert pts["self"]["pts"] and pts["other"]["pts"]            # scatter clouds present


def test_dashboard_freeform_empty_is_byte_stable():
    tmp = tempfile.mkdtemp()
    paths = _write_suite(tmp, ["qwen3-32b"], {})                 # no free-form scored files
    h1, _ = bqd.build(paths)
    h2, _ = bqd.build(paths)
    assert h1 == h2                                              # deterministic
    assert 'id="tab-freeform"' in h1                             # tab scaffolding still present
    FF = json.loads(re.search(r"const FREEFORM = (\{.*?\});\n", h1, re.S).group(1))
    assert FF["models"] == [] and FF["byModel"] == {}            # graceful empty placeholder
