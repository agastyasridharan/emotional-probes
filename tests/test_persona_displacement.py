"""
GPU-free tests for the E3 persona-displacement re-encode driver
(``emotion_probes.alignment.persona_displacement``).

Fakes follow the repo idiom (``object.__new__`` + hand-set attributes, see
``tests/test_qualia_freeform.py``); torch is CPU-only here. The fake network
produces a hand-computable residual stream — activation at (layer l, position t)
is ``(t+1) * (l+1) * [1, 2, ..., H]`` — so span pooling, washout masking, the
projection math and the direct-add analytic null are all checked against exact
hand-derived values.

Coverage:
* multi-layer capture helper     — hook registration/removal, per-layer tensors;
* record selection               — confirmatory strengths, descriptive_only flag on
                                   0.15, 0.03 dropped, baselines kept, per-cell cap;
* span masking                   — pooled mean covers RESPONSE tokens only (from
                                   recorded prompt_tokens), washout positions;
* projection math                — hand-computed dots, v_injected None handling;
* direct-add analytic null       — α·v_e added at layers >= steer layer only;
* arm bookkeeping + record counts— per-arm row counts, swap arms swap the encode
                                   frame, steer routing (real -> engine, synthetic
                                   -> model.steer), one context per cell, unknown
                                   conditions skipped;
* payload                        — provenance header (Gram, cos_v_e table), JSON-
                                   serializable.
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
import emotion_probes.alignment.persona_displacement as pd


HIDDEN = 6
NUM_LAYERS = 8
CAPTURE = [2, 5]
STEER_LAYER = 3
BASE = np.arange(1, HIDDEN + 1, dtype=np.float32)   # [1..6]


# =========================================================================== #
# Fakes
# =========================================================================== #
class _Tok:
    """Whitespace tokenizer with a persistent id<->word vocab (decode works)."""

    def __init__(self):
        self.vocab: list[str] = []
        self._ids: dict[str, int] = {}
        self.template_calls = []

    def _id(self, w: str) -> int:
        if w not in self._ids:
            self._ids[w] = len(self.vocab)
            self.vocab.append(w)
        return self._ids[w]

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": [self._id(w) for w in str(text).split()]}

    def decode(self, ids, **kw):
        return " ".join(self.vocab[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.template_calls.append(kw)
        return "|".join(f"{m['role']}:{m['content']}" for m in msgs) + "|assistant:"


class _Layer:
    """Fake decoder block exposing register_forward_hook (real-signature handles)."""

    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        layer, hooks = self, self.hooks

        class _Handle:
            def remove(_self):
                hooks.remove(fn)
        return _Handle()


class _Net:
    """Fake HF module: calling it fires every registered hook with the
    deterministic activation ``(t+1) * (layer+1) * BASE`` at each position.
    Each forward logs the owner's currently-active steer context (or None) so
    tests can prove the steered arms teacher-force INSIDE the steer hook."""

    def __init__(self, layers, hidden, owner=None):
        self._layers = layers
        self._hidden = hidden
        self._owner = owner

    def __call__(self, input_ids=None, **kw):
        import torch
        if self._owner is not None:
            self._owner.forward_log.append(self._owner.active_steer)
        B, T = input_ids.shape
        t_fac = torch.arange(1, T + 1, dtype=torch.float32).view(1, T, 1)
        base = torch.arange(1, self._hidden + 1, dtype=torch.float32).view(1, 1, -1)
        for li, layer in enumerate(self._layers):
            out = t_fac * base * float(li + 1)
            out = out.expand(B, T, self._hidden).clone()
            for h in list(layer.hooks):
                h(layer, (input_ids,), (out,))
        return None


class _Model:
    def __init__(self, hidden=HIDDEN, num_layers=NUM_LAYERS):
        self.tokenizer = _Tok()
        self.num_layers = num_layers
        self.hidden_size = hidden
        self._layers = [_Layer() for _ in range(num_layers)]
        self._input_device = "cpu"
        self.model = _Net(self._layers, hidden, owner=self)
        self.steer_calls = []
        self.gen_calls = []
        self.active_steer = None     # set while a steer context is open
        self.forward_log = []        # active_steer at each forward pass

    def generate_batch(self, prompts, max_new_tokens=180, temperature=0.9,
                       do_sample=True, chat=False):
        self.gen_calls.append(len(prompts))
        return [f"resp{i} I feel calm" for i in range(len(prompts))]

    @contextmanager
    def steer(self, unit_vectors, strength, norms, positions="all"):
        self.steer_calls.append(("synthetic", float(strength)))
        self.active_steer = ("synthetic", float(strength))
        try:
            yield
        finally:
            self.active_steer = None


class _Bank:
    def __init__(self, hidden=HIDDEN, num_layers=NUM_LAYERS):
        self.emotions = [s.name for s in qualia.emotion_conditions()]
        self.hidden_size = hidden
        rng = np.random.default_rng(7)
        v = rng.standard_normal((num_layers, len(self.emotions), hidden)).astype(np.float32)
        self.vectors = v / np.linalg.norm(v, axis=-1, keepdims=True)

    def index_of(self, name):
        return self.emotions.index(name)


class _Engine:
    def __init__(self, model=None):
        self._norms = {STEER_LAYER: 10.0}
        self.calls = []
        self._model = model

    def calibrate(self, texts):
        return self._norms

    def _require_norms(self):
        return self._norms

    @contextmanager
    def steer(self, emotion, strength, positions="all"):
        self.calls.append((emotion, float(strength)))
        if self._model is not None:
            self._model.active_steer = (emotion, float(strength))
        try:
            yield
        finally:
            if self._model is not None:
                self._model.active_steer = None


def _fake_exp():
    exp = object.__new__(pd.PersonaDisplacementExperiment)
    exp.model = _Model()
    exp.bank = _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None,
                                 deflection_max_length=512)
    exp.layer = STEER_LAYER
    exp.capture_layers = list(CAPTURE)
    exp.engine = _Engine(exp.model)
    return exp


def _dirs():
    """Frozen-coordinate fixture: axis-aligned unit vectors (hand-computable dots)."""
    e = np.eye(HIDDEN, dtype=np.float32)
    out = {}
    for L in CAPTURE:
        d = {"assistant_axis": e[0], "human_axis": e[1],
             "human_axis_purged": e[2], "experiencer_axis": e[4],
             "valence_axis": e[3]}
        for i in range(1, 9):
            d[f"pc{i}"] = e[(i + 3) % HIDDEN]
        out[L] = d
    return out


def _scored_payload(n=2, strengths=(0.03, 0.06, 0.15), conditions=("blissful", "__random__"),
                    frames=("self", "other"), extra_condition=None):
    """Minimal qualia_freeform_scored-shaped payload. Rendered prompts are 3
    tokens; every text starts with a space so whitespace tokenization gives an
    exact 3-token prompt / 3-token response split."""
    frame_meta = [{"key": k, "label": k.upper(), "kind": k, "text": f"{k} q",
                   "rendered": f"{k}p1 {k}p2 {k}p3", "prompt_tokens": 3}
                  for k in frames]
    records = []
    for fk in frames:
        for i in range(n):
            records.append({"condition": pd.BASELINE, "strength": 0.0, "frame": fk,
                            "sample": i, "text": " I feel warm"})
    conds = list(conditions) + ([extra_condition] if extra_condition else [])
    for c in conds:
        for s in strengths:
            for fk in frames:
                for i in range(n):
                    records.append({"condition": c, "strength": s, "frame": fk,
                                    "sample": i, "text": " I feel warm"})
    return {"model_id": "fake", "seed": 0, "layer": STEER_LAYER,
            "calibration_norm": 10.0, "records": records, "frames": frame_meta}


def _write_scored(tmp_path, payload, name="qualia_freeform_scored.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


# =========================================================================== #
# 1. Multi-layer capture helper
# =========================================================================== #
def test_multi_capture_lifecycle_and_shapes():
    import torch
    model = _Model()
    ids = torch.tensor([[0, 1, 2, 3]])
    with pd.multi_layer_capture(model, CAPTURE) as cap:
        model.model(input_ids=ids)
        assert set(cap) == set(CAPTURE)
        a2 = cap[2].numpy()[0]
        assert a2.shape == (4, HIDDEN)
        # layer 2 -> factor 3; position t -> factor t+1
        assert np.allclose(a2[0], 1 * 3 * BASE) and np.allclose(a2[3], 4 * 3 * BASE)
    # hooks removed on exit; other layers never hooked
    assert all(not l.hooks for l in model._layers)


def test_multi_capture_removes_hooks_on_error():
    model = _Model()
    with pytest.raises(RuntimeError):
        with pd.multi_layer_capture(model, [1]):
            raise RuntimeError("boom")
    assert not model._layers[1].hooks


# =========================================================================== #
# 2. Record selection (strength filter + descriptive flag + cell cap)
# =========================================================================== #
def test_select_records_strengths_and_flags():
    recs = _scored_payload()["records"]
    recs = [{**r, "record_id": f"main#{i}"} for i, r in enumerate(recs)]
    kept = pd.select_records(recs, pd.CONFIRMATORY_STRENGTHS)
    strengths = {r["strength"] for r in kept}
    assert 0.03 not in strengths                       # dropped
    assert {0.0, 0.06, 0.15} == strengths              # confirmatory + descriptive
    assert all(r["descriptive_only"] for r in kept if r["strength"] == 0.15)
    assert not any(r["descriptive_only"] for r in kept if r["strength"] == 0.06)
    # baselines always kept, even under a condition restriction
    kept_b = pd.select_records(recs, pd.CONFIRMATORY_STRENGTHS, emotions=["blissful"])
    assert any(r["condition"] == pd.BASELINE for r in kept_b)
    assert not any(r["condition"] == "__random__" for r in kept_b)


def test_select_records_cell_cap():
    recs = _scored_payload(n=5)["records"]
    recs = [{**r, "record_id": f"main#{i}"} for i, r in enumerate(recs)]
    kept = pd.select_records(recs, pd.CONFIRMATORY_STRENGTHS, max_per_cell=2)
    cells = {}
    for r in kept:
        cells[(r["condition"], r["strength"], r["frame"])] = \
            cells.get((r["condition"], r["strength"], r["frame"]), 0) + 1
    assert cells and all(v == 2 for v in cells.values())


def test_load_scored_record_ids_unique_across_sources(tmp_path):
    p1 = _write_scored(tmp_path, _scored_payload(n=1), "a.json")
    p2 = _write_scored(tmp_path, _scored_payload(n=1), "b.json")
    records, frames, meta = pd.load_scored([("main", p1), ("extra0", p2)])
    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids)) and ids[0] == "main#0"
    assert any(i.startswith("extra0#") for i in ids)
    assert set(frames) == {"self", "other"} and frames["self"]["prompt_tokens"] == 3
    # source provenance surfaced (seed / calibration guard inputs)
    assert [m["tag"] for m in meta] == ["main", "extra0"]
    assert meta[0]["seed"] == 0 and meta[0]["calibration_norm"] == 10.0


def test_select_records_keeps_baselines_without_zero_strength():
    recs = _scored_payload()["records"]
    recs = [{**r, "record_id": f"main#{i}"} for i, r in enumerate(recs)]
    kept = pd.select_records(recs, [0.06])       # no 0.0 in the confirmatory set
    assert any(r["condition"] == pd.BASELINE for r in kept)
    assert not any(r["descriptive_only"] for r in kept if r["condition"] == pd.BASELINE)


# =========================================================================== #
# 3. Span masking (pool covers response tokens only) + washout guard
# =========================================================================== #
def test_reencode_pools_response_span_only():
    exp = _fake_exp()
    # prompt 3 tokens + response " I feel warm" (3 tokens) => positions 3,4,5
    enc = exp._reencode("p1 p2 p3", " I feel warm", n_prompt=3)
    assert enc["response_tokens"] == 3
    # mean over t=3..5 -> factors (4+5+6)/3 = 5; layer 2 -> *3, layer 5 -> *6
    assert np.allclose(enc["pooled"][2], 5 * 3 * BASE)
    assert np.allclose(enc["pooled"][5], 5 * 6 * BASE)
    # washout: "I" (pronoun) at span idx 0 -> t=3, "feel" (experience) idx 1 -> t=4
    assert enc["washout_tokens"] == 2
    assert np.allclose(enc["washout_pooled"][2], (4 + 5) / 2 * 3 * BASE)


def test_reencode_uses_recorded_prompt_tokens_not_tokenizer():
    exp = _fake_exp()
    # recorded prompt_tokens=4 overrides the 3-token tokenization of the prompt
    enc = exp._reencode("p1 p2 p3", " I feel warm", n_prompt=4)
    # span t=4,5 -> mean factor 5.5
    assert np.allclose(enc["pooled"][2], 5.5 * 3 * BASE)
    assert enc["response_tokens"] == 2


def test_reencode_empty_span_falls_back_to_last_token():
    exp = _fake_exp()
    enc = exp._reencode("p1 p2 p3", "", n_prompt=3)   # full=3 tokens, clamp to 2
    # span = positions 2.. -> factor 3 at layer 2 -> 3*3*BASE
    assert np.allclose(enc["pooled"][2], 3 * 3 * BASE)


def test_prompt_acts_final_position():
    exp = _fake_exp()
    acts = exp._prompt_acts("p1 p2 p3")
    assert np.allclose(acts[2], 3 * 3 * BASE)         # t=2 -> factor 3, layer 2 -> *3
    assert np.allclose(acts[5], 3 * 6 * BASE)


# =========================================================================== #
# 4. Projection math
# =========================================================================== #
def test_project_hand_computed():
    dirs = _dirs()[2]
    vec = 15 * BASE                                   # pooled at layer 2 in the fixture
    v_e = pd._unit(np.ones(HIDDEN, dtype=np.float32))
    proj = pd.PersonaDisplacementExperiment._project(vec, dirs, v_e, v_e)
    assert proj["assistant_axis"] == pytest.approx(15.0)     # 15*BASE @ e0 = 15*1
    assert proj["human_axis"] == pytest.approx(30.0)         # 15*2
    assert proj["valence_axis"] == pytest.approx(60.0)       # 15*4
    expected_ve = float(vec @ v_e)
    assert proj["v_injected"] == pytest.approx(expected_ve, abs=1e-4)
    assert proj["v_injected_steer"] == pytest.approx(expected_ve, abs=1e-4)


def test_project_none_injected():
    proj = pd.PersonaDisplacementExperiment._project(BASE, _dirs()[2], None, None)
    assert proj["v_injected"] is None and proj["v_injected_steer"] is None


def test_unit_rejects_zero():
    with pytest.raises(ValueError):
        pd._unit(np.zeros(4))


def test_experiencer_axis_hand_computed():
    """z_x = unit(mean(experiencer_high) − mean(experiencer_low)), orthogonalized
    against a — from the FROZEN subsets, per prereg E3 'Coordinates'."""
    from emotion_probes.persona.assets import AssistantAxis
    H = 4
    e = np.eye(H, dtype=np.float32)
    axis = np.stack([e[0], e[0]])                        # a = e0 at layers 0, 1
    roles = {
        "r_hi1": np.stack([np.zeros(H, np.float32), 3 * e[0] + 2 * e[1]]),
        "r_hi2": np.stack([np.zeros(H, np.float32), 1 * e[0] + 4 * e[1]]),
        "r_lo":  np.stack([np.zeros(H, np.float32), 2 * e[0]]),
    }
    ax = AssistantAxis(axis=axis, default_vector=np.zeros((2, H), np.float32),
                       role_vectors=roles, path="fake")
    subsets = {"subsets": {"experiencer_high": ["r_hi1", "r_hi2"],
                           "experiencer_low": ["r_lo"]}}
    x = pd.experiencer_axis(ax, subsets, 1)
    # mean(high) − mean(low) = (2e0 + 3e1) − 2e0 = 3e1; purge a=e0; unit -> e1
    assert np.allclose(x, e[1], atol=1e-6)
    assert abs(float(x @ e[0])) < 1e-6                   # orthogonal to a
    with pytest.raises(KeyError):                        # frozen sets required
        pd.experiencer_axis(ax, {"subsets": {"human": []}}, 1)
    with pytest.raises(KeyError):                        # unknown role name
        pd.experiencer_axis(ax, {"subsets": {"experiencer_high": ["ghost"],
                                             "experiencer_low": ["r_lo"]}}, 1)


# =========================================================================== #
# 5. Run: arm bookkeeping, counts, steer routing, direct-add analytics
# =========================================================================== #
def _run_payload(tmp_path, **kw):
    exp = _fake_exp()
    payload = _scored_payload(extra_condition=kw.pop("extra_condition", "__random_2__"))
    p = _write_scored(tmp_path, payload)
    defaults = dict(scored_path=p, directions=_dirs(), n_control=2, gen_batch=8, seed=0)
    defaults.update(kw)
    return exp, exp.run(**defaults)


def test_run_arm_row_counts(tmp_path):
    exp, payload = _run_payload(tmp_path)
    rows = payload["records"]
    by_arm = {}
    for r in rows:
        by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
    n_layers = len(CAPTURE)
    # kept (0.03 dropped, 0.15 kept-descriptive): baselines 2x2=4; blissful,
    # __random__, __random_2__ each @{.06,.15} x 2 frames x 2 = 8 => 28 records
    assert payload["n_kept_records"] == 28
    assert by_arm[pd.ARM_REENC_UNSTEERED] == 28 * n_layers
    assert by_arm[pd.ARM_REENC_UNSTEERED_SWAP] == 28 * n_layers
    # steerable nonzero: blissful + __random__ each @{.06,.15} x 4 records = 16
    # (__random_2__ is unknown => excluded from steered/analytic arms)
    assert by_arm[pd.ARM_REENC_STEERED] == 16 * n_layers
    assert by_arm[pd.ARM_REENC_STEERED_SWAP] == 16 * n_layers
    # direct-add: 8 steerable (cond,s,frame) cells + 2 strength-0 frame cells
    assert by_arm[pd.ARM_DIRECT_ADD] == (8 + 2) * n_layers
    # prompt-steered: 4 (cond,s) cells x 2 frames
    assert by_arm[pd.ARM_PROMPT_STEERED] == 4 * 2 * n_layers
    # persona control: 3 roles x 2 frames x n_control=2
    assert by_arm[pd.ARM_PERSONA_CONTROL] == 3 * 2 * 2 * n_layers
    # unknown condition surfaced, not silently dropped
    assert payload["skipped_conditions"] == ["__random_2__"]
    steered_conds = {r["condition"] for r in rows if r["arm"] == pd.ARM_REENC_STEERED}
    assert "__random_2__" not in steered_conds
    # every row carries the flat schema
    for r in rows[:10]:
        assert {"record_id", "arm", "condition", "strength", "frame", "layer",
                "projections"} <= set(r)
        assert r["layer"] in CAPTURE


def test_run_steer_routing_and_one_context_per_cell(tmp_path):
    exp, payload = _run_payload(tmp_path)
    # real emotion -> engine: blissful cells (.06, .15), ONE context per cell
    # per steered arm (reencode_steered, reencode_steered_swap, prompt_steered)
    assert all(c == "blissful" for c, _ in exp.engine.calls)
    assert sorted(s for _, s in exp.engine.calls) == sorted([0.06, 0.15] * 3)
    # synthetic -> model.steer: __random__ cells (.06, .15) once per steered arm
    assert exp.model.steer_calls == [("synthetic", 0.06), ("synthetic", 0.15)] * 3


def test_steered_reencode_forwards_run_inside_steer_context(tmp_path):
    """The steering hook must be LIVE during the teacher-forced forward pass
    (real emotions via the engine AND synthetic via model.steer) — not merely
    opened and closed around nothing."""
    exp = _fake_exp()
    p = _write_scored(tmp_path, _scored_payload())
    out = exp.run(scored_path=p, directions=_dirs(), seed=0,
                  arms=[pd.ARM_REENC_STEERED])
    n_rows = sum(1 for r in out["records"] if r["arm"] == pd.ARM_REENC_STEERED)
    assert len(exp.model.forward_log) == n_rows // len(CAPTURE)   # 1 forward/record
    assert exp.model.forward_log and all(st is not None for st in exp.model.forward_log)
    assert {st[0] for st in exp.model.forward_log} == {"blissful", "synthetic"}
    assert {st[1] for st in exp.model.forward_log} == {0.06, 0.15}
    # and the unsteered arm teacher-forces with NO steer context active
    exp2 = _fake_exp()
    out2 = exp2.run(scored_path=p, directions=_dirs(), seed=0,
                    arms=[pd.ARM_REENC_UNSTEERED])
    assert exp2.model.forward_log and all(st is None for st in exp2.model.forward_log)


def test_random_seed_mismatch_excludes_random_from_steered_arms(tmp_path):
    """__random__ is only reconstructible under the source run's seed; with a
    mismatched seed it must be skipped from steered/analytic arms (else the
    control band would be steered with a DIFFERENT vector but the same label)."""
    exp = _fake_exp()
    payload = _scored_payload()
    payload["seed"] = 3                                   # source ran with seed 3
    p = _write_scored(tmp_path, payload)
    out = exp.run(scored_path=p, directions=_dirs(), seed=0)
    assert "__random__" in out["skipped_conditions"]
    for arm in (pd.ARM_REENC_STEERED, pd.ARM_REENC_STEERED_SWAP,
                pd.ARM_DIRECT_ADD, pd.ARM_PROMPT_STEERED):
        assert "__random__" not in {r["condition"] for r in out["records"]
                                    if r["arm"] == arm}
    # its records still flow through the unsteered re-encode (own text, no hook)
    assert "__random__" in {r["condition"] for r in out["records"]
                            if r["arm"] == pd.ARM_REENC_UNSTEERED}
    # matching seed keeps it steerable (the default fixture path)
    exp2, out2 = _run_payload(tmp_path)
    assert "__random__" in {r["condition"] for r in out2["records"]
                            if r["arm"] == pd.ARM_REENC_STEERED}


def test_run_rejects_directions_missing_experiencer_axis(tmp_path):
    """T1/T3/T4/T6 are built on z_x; a direction set without it must fail fast,
    not burn a GPU run producing projections the confirmatory family can't use."""
    exp = _fake_exp()
    p = _write_scored(tmp_path, _scored_payload())
    dirs = {L: {n: v for n, v in d.items() if n != "experiencer_axis"}
            for L, d in _dirs().items()}
    with pytest.raises(KeyError, match="experiencer_axis"):
        exp.run(scored_path=p, directions=dirs)


def test_run_swap_arms_swap_encode_frame(tmp_path):
    exp, payload = _run_payload(tmp_path)
    for r in payload["records"]:
        if r["arm"] in (pd.ARM_REENC_UNSTEERED_SWAP, pd.ARM_REENC_STEERED_SWAP):
            assert r["encode_frame"] == pd.SWAP_FRAMES[r["frame"]]
        elif r["arm"] in (pd.ARM_REENC_UNSTEERED, pd.ARM_REENC_STEERED):
            assert r["encode_frame"] == r["frame"]


def test_run_descriptive_flag_and_strength_filter(tmp_path):
    exp, payload = _run_payload(tmp_path)
    strengths = {r["strength"] for r in payload["records"]}
    assert 0.03 not in strengths
    for r in payload["records"]:
        if r["condition"] == "blissful":
            assert r["descriptive_only"] == (r["strength"] == 0.15)


def test_direct_add_analytic_null(tmp_path):
    exp, payload = _run_payload(tmp_path)
    da = [r for r in payload["records"] if r["arm"] == pd.ARM_DIRECT_ADD]
    base = {(r["frame"], r["layer"]): r["projections"]
            for r in da if r["condition"] == pd.BASELINE}
    v_e = pd._unit(exp.bank.vectors[STEER_LAYER, exp.bank.index_of("blissful")])
    alpha = 0.06 * 10.0                                # strength x calibration norm
    for r in da:
        if r["condition"] != "blissful" or r["strength"] != 0.06:
            continue
        b = base[(r["frame"], r["layer"])]
        for name, d in _dirs()[r["layer"]].items():
            got, expect_base = r["projections"][name], b[name]
            if r["layer"] >= STEER_LAYER:              # L5: injection propagates
                assert got == pytest.approx(expect_base + alpha * float(v_e @ d), abs=1e-3)
            else:                                      # L2: below injection, unchanged
                assert got == pytest.approx(expect_base, abs=1e-4)
    # span bookkeeping
    assert all(r["span"] == "prompt_final" for r in da)


def test_persona_control_generates_and_projects(tmp_path):
    exp, payload = _run_payload(tmp_path)
    pc = [r for r in payload["records"] if r["arm"] == pd.ARM_PERSONA_CONTROL]
    assert {r["condition"] for r in pc} == {"role:empath", "role:poet", "role:accountant"}
    assert all(r["strength"] == 0.0 and r["span"] == "response" for r in pc)
    assert all(r["projections"]["v_injected"] is None for r in pc)
    # system prompts recorded in the header, loaded from the real repo assets
    assert set(payload["control_roles"]) == {"empath", "poet", "accountant"}
    assert "empath" in payload["control_roles"]["empath"].lower()
    # generation happened (2 per role x frame, chunked under gen_batch=8)
    assert sum(exp.model.gen_calls) == 3 * 2 * 2


def test_run_arm_restriction(tmp_path):
    exp, payload = _run_payload(tmp_path, arms=[pd.ARM_REENC_UNSTEERED])
    assert {r["arm"] for r in payload["records"]} == {pd.ARM_REENC_UNSTEERED}
    assert payload["arms"] == [pd.ARM_REENC_UNSTEERED]
    exp2 = _fake_exp()
    with pytest.raises(ValueError):
        exp2.run(scored_path="/nonexistent", arms=["bogus_arm"], directions=_dirs())


# =========================================================================== #
# 6. Provenance header + serialization
# =========================================================================== #
def test_header_gram_and_cos_table(tmp_path):
    exp, payload = _run_payload(tmp_path)
    assert payload["experiment"] == "persona_displacement"
    assert payload["steer_layer"] == STEER_LAYER
    assert payload["capture_layers"] == CAPTURE
    assert payload["calibration_norm"] == 10.0
    assert payload["strengths_confirmatory"] == [0.0, 0.06, 0.1]
    assert payload["strengths_descriptive"] == [0.15]
    assert set(payload["direction_names"]) == set(pd.DIRECTION_NAMES)
    # Gram: orthonormal fixture -> identity
    g = payload["direction_gram"]["2"]
    assert g["assistant_axis"]["assistant_axis"] == pytest.approx(1.0)
    assert g["assistant_axis"]["human_axis"] == pytest.approx(0.0)
    # cos(v_e, coordinate) table published up front, per layer
    cv = payload["cos_v_e"]["blissful"]["2"]
    v = pd._unit(exp.bank.vectors[2, exp.bank.index_of("blissful")])
    assert cv["assistant_axis"] == pytest.approx(float(v[0]), abs=1e-4)
    # layer roles annotated (non-default layers are exploratory)
    assert payload["layer_roles"] == {"2": "exploratory", "5": "exploratory"}
    assert pd.LAYER_ROLES[50] == "primary" and pd.LAYER_ROLES[32] == "text_mediated_comparator"


def test_payload_json_serializable(tmp_path):
    exp, payload = _run_payload(tmp_path)
    dumped = json.dumps(payload)                       # no numpy leakage
    assert "persona_displacement" in dumped


def test_directions_npz_roundtrip(tmp_path):
    dirs = _dirs()
    arrs = {f"{n}@L{L}": v for L, d in dirs.items() for n, v in d.items()}
    p = tmp_path / "dirs.npz"
    np.savez(p, **arrs)
    loaded = pd.load_directions_npz(p, CAPTURE)
    assert set(loaded) == set(CAPTURE)
    assert np.allclose(loaded[2]["assistant_axis"], dirs[2]["assistant_axis"])
    with pytest.raises(KeyError):
        pd.load_directions_npz(p, [63])
