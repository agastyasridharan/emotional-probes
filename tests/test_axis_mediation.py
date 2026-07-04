"""
GPU-free tests for the E5 axis-mediation port (portfolio.md §E5, docs/prereg/E5.md).

Nothing here needs a GPU; torch runs on CPU for the steering-ops math and the fake
model returns hand-built tensors (the ``object.__new__`` + fakes idiom from
``test_qualia_battery.py`` / ``test_qualia_freeform.py``).

Coverage:
* steering-ops math          — ``emotion_probes.persona.steering_ops`` on hand-built
                               tensors (ablation zeroes the projection; cap clips only
                               above tau; lower-cap only lifts below tau; mean-ablation;
                               shapes/dtypes preserved incl. bf16; project_batch/tokens);
* graded add-back            — λ-floor semantics (λ=0 ≡ cap-to-baseline, monotone in λ);
* hook composition           — ``apply_ops`` registration/removal on fake layers,
                               same-layer ordering (steer-then-cap vs cap-then-steer),
                               tuple-output handling;
* capping config             — experiment-name parsing + missing-file fallback;
* λ grid / cell bookkeeping  — ``intervention_grid``, ``build_cells``,
                               ``expected_record_counts``, tau percentile, panel Δ;
* driver                     — constructor axis-provenance guard, sufficiency-arm
                               −axis routing, full ``run`` on fakes (record counts ==
                               formula, lam bookkeeping, provenance, hooks active
                               simultaneously with the steer context, serializable).
"""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emotion_probes.data import qualia
import emotion_probes.alignment.axis_mediation as am
import emotion_probes.persona.steering_ops as sops
from emotion_probes.persona.assets import ProvenanceError, unit


def _proj(h, v):
    vu = v.to(torch.float32) / v.to(torch.float32).norm()
    return torch.einsum("bld,d->bl", h.to(torch.float32), vu)


def _h_with_projections(projs, hidden=8):
    """(1, T, H) tensor whose projection onto e0 is ``projs``; e1 carries an
    orthogonal component so ops can be checked for collateral damage."""
    h = torch.zeros(1, len(projs), hidden)
    h[0, :, 0] = torch.tensor(projs, dtype=torch.float32)
    h[0, :, 1] = 7.0
    return h


_V0 = torch.zeros(8)
_V0[0] = 2.0  # non-unit on purpose: ops must unit-normalize internally


# =========================================================================== #
# 1. Steering-ops math
# =========================================================================== #
def test_ablation_zeroes_projection_keeps_orthogonal():
    torch.manual_seed(0)
    h, v = torch.randn(2, 3, 8), torch.randn(8)
    out = sops.apply_ablation(h, v, 0.0)
    assert torch.allclose(_proj(out, v), torch.zeros(2, 3), atol=1e-5)
    vu = v / v.norm()
    orth = h - torch.einsum("bl,d->bld", _proj(h, v), vu)
    assert torch.allclose(out, orth, atol=1e-5)


def test_ablation_addback_uses_raw_vector():
    torch.manual_seed(1)
    h, v = torch.randn(2, 3, 8), torch.randn(8)
    out = sops.apply_ablation(h, v, 1.5)
    # projection becomes coeff * ||v|| (verbatim port adds coeff * RAW vector)
    want = torch.full((2, 3), 1.5 * float(v.norm()))
    assert torch.allclose(_proj(out, v), want, atol=1e-4)


def test_cap_clips_only_above_tau():
    h = _h_with_projections([-1.0, 0.5, 3.0])
    out = sops.apply_cap(h, _V0, 1.0)
    assert torch.allclose(_proj(out, _V0)[0], torch.tensor([-1.0, 0.5, 1.0]), atol=1e-5)
    assert torch.allclose(out[0, :, 1], h[0, :, 1])            # orthogonal untouched
    assert torch.allclose(sops.apply_cap(h, _V0, 10.0), h)     # all below tau -> no-op


def test_cap_lower_lifts_only_below_tau():
    h = _h_with_projections([-1.0, 0.5, 3.0])
    out = sops.apply_cap_lower(h, _V0, 1.0)
    assert torch.allclose(_proj(out, _V0)[0], torch.tensor([1.0, 1.0, 3.0]), atol=1e-5)
    assert torch.allclose(out[0, :, 1], h[0, :, 1])
    assert torch.allclose(sops.apply_cap_lower(h, _V0, -5.0), h)  # all above -> no-op


def test_mean_ablation():
    torch.manual_seed(2)
    h, v = torch.randn(2, 3, 8), torch.randn(8)
    mean = torch.randn(8)
    out = sops.apply_mean_ablation(h, v, mean)
    vu = v / v.norm()
    orth = h - torch.einsum("bl,d->bld", _proj(h, v), vu)
    assert torch.allclose(out, orth + mean, atol=1e-5)


def test_ops_shape_and_dtype_preserved():
    for dtype in (torch.float32, torch.bfloat16):
        h = torch.randn(2, 4, 8).to(dtype)
        v32 = torch.randn(8)                                  # fp32 v vs bf16 h
        for out in (sops.apply_ablation(h, v32, 0.5),
                    sops.apply_cap(h, v32, 0.3),
                    sops.apply_cap_lower(h, v32, 0.3),
                    sops.apply_mean_ablation(h, v32, torch.randn(8))):
            assert out.shape == h.shape and out.dtype == dtype


def test_project_batch_verbatim():
    torch.manual_seed(3)
    acts = torch.randn(5, 4, 8)                               # (batch, n_layers, hidden)
    axis = torch.randn(4, 8)
    out = sops.project_batch(acts, axis, layer=2)
    assert out.shape == (5,) and out.dtype == torch.float32
    ax = axis[2] / (axis[2].norm() + 1e-8)
    assert torch.allclose(out, acts[:, 2, :] @ ax, atol=1e-5)
    raw = sops.project_batch(acts, axis, layer=2, normalize=False)
    assert torch.allclose(raw, acts[:, 2, :] @ axis[2], atol=1e-4)


def test_project_tokens_shape():
    h = torch.randn(3, 5, 8).to(torch.bfloat16)
    out = sops.project_tokens(h, torch.randn(8))
    assert out.shape == (3, 5) and out.dtype == torch.float32


def test_addback_op_grades_the_floor():
    h = _h_with_projections([-2.0, 0.0, 3.0])
    tau, delta = 1.0, -2.0                                    # Δ_emo < 0 = human-ward
    cap = sops.cap_lower_op(_V0, tau)(h)
    lam0 = sops.cap_lower_addback_op(_V0, tau, 0.0, delta)(h)
    assert torch.allclose(lam0, cap, atol=1e-6)               # λ=0 ≡ cap-to-baseline
    projs = {}
    for lam in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0):             # −0.5 = 150% overshoot
        out = sops.cap_lower_addback_op(_V0, tau, lam, delta)(h)
        p = _proj(out, _V0)[0]
        assert torch.allclose(p, torch.clamp(p, min=tau + lam * delta))
        assert float(p[0]) == pytest.approx(tau + lam * delta, abs=1e-5)
        projs[lam] = p
    for a, b in zip((-0.5, 0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75, 1.0)):
        assert torch.all(projs[a] >= projs[b] - 1e-6)         # monotone release in λ
    assert torch.allclose(projs[1.0][2], _proj(h, _V0)[0][2]) # untouched above floor
    # overshoot arm: floor tau − 0.5Δ = 2.0 sits ABOVE baseline tau
    assert torch.allclose(projs[-0.5], torch.tensor([2.0, 2.0, 3.0]), atol=1e-5)


# =========================================================================== #
# 2. Hook composition (fakes)
# =========================================================================== #
class _FakeLayer:
    """Mimics an nn.Module's forward-hook registry + torch's hook semantics
    (registration order; a non-None return replaces the output for later hooks)."""

    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        layer = self

        class _H:
            def remove(_self):
                layer.hooks.remove(fn)
        return _H()

    def forward_sim(self, output):
        for fn in self.hooks:
            r = fn(self, (), output)
            if r is not None:
                output = r
        return output


def _steer_hook(vec, scaled):
    """A ProbedModel.steer-style additive hook (tuple-aware), for ordering tests."""
    def hook(_m, _i, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h + scaled * vec
        return (h, *output[1:]) if is_tuple else h
    return hook


def test_apply_ops_register_and_remove():
    model = SimpleNamespace(_layers=[_FakeLayer() for _ in range(6)])
    ops = {4: sops.cap_lower_op(_V0, 1.0), 5: sops.cap_op(_V0, 2.0)}
    with sops.apply_ops(model, ops):
        assert [len(l.hooks) for l in model._layers] == [0, 0, 0, 0, 1, 1]
    assert all(not l.hooks for l in model._layers)            # removed on exit


def test_composition_steer_then_cap_same_layer():
    layer = _FakeLayer()
    model = SimpleNamespace(_layers=[layer])
    vu = _V0 / _V0.norm()
    h = _h_with_projections([0.5, 0.9])                       # both below tau=1.0
    layer.register_forward_hook(_steer_hook(vu, 5.0))         # emotion steer FIRST
    with sops.apply_ops(model, {0: sops.cap_op(_V0, 1.0)}):
        out = layer.forward_sim((h, "kv_cache"))
    assert isinstance(out, tuple) and out[1] == "kv_cache"    # tuple passthrough
    # cap ran after the +5 push, so the excess is clipped back to tau
    assert torch.allclose(_proj(out[0], _V0)[0], torch.tensor([1.0, 1.0]), atol=1e-5)


def test_composition_order_is_registration_order():
    layer = _FakeLayer()
    model = SimpleNamespace(_layers=[layer])
    vu = _V0 / _V0.norm()
    h = _h_with_projections([0.5])
    with sops.apply_ops(model, {0: sops.cap_op(_V0, 1.0)}):   # cap FIRST this time
        layer.register_forward_hook(_steer_hook(vu, 5.0))     # steer second
        out = layer.forward_sim(h)
    # the cap never saw the push -> projection ends at 0.5 + 5.0
    assert float(_proj(out, _V0)[0, 0]) == pytest.approx(5.5, abs=1e-5)


# =========================================================================== #
# 3. Capping config + grid bookkeeping (pure CPU)
# =========================================================================== #
def test_parse_capping_experiment():
    # Python-slice band semantics: the real config's layers_46:54-p0.25 experiment
    # has 8 interventions at layers 46..53 ("Lu's band L46-53", docs/prereg/E5.md).
    layers, pct = am.parse_capping_experiment("layers_46:54-p0.25")
    assert layers == list(range(46, 54)) and len(layers) == 8 and pct == 0.25
    layers, _ = am.parse_capping_experiment("layers_60:64-p0.5")   # hi == N_LAYERS ok
    assert layers == [60, 61, 62, 63]
    for bad in ("layers_46-54", "layers_54:46-p0.25", "layers_46:46-p0.25",
                "layers_1:99-p0.25", "x"):
        with pytest.raises(ValueError):
            am.parse_capping_experiment(bad)


def test_load_capping_config_fallback():
    cfg = am.load_capping_config(tempfile.mkdtemp())          # no capping_config.pt
    assert cfg["source"] == "fallback-default"
    assert cfg["cap_layers"] == list(range(46, 54)) and cfg["percentile"] == 0.25
    assert cfg["experiment"] == am.RECOMMENDED_EXPERIMENT


def _write_real_structure_config(tmpdir, iv_layers, hidden=5120):
    """Mimic the VERIFIED upstream capping_config.pt: vectors keyed
    'layer_<L>/contrast_role_pos3_default1' with {'vector', 'layer'} meta;
    experiments keyed 'id' (NOT 'name') with per-layer 'interventions'."""
    names = {L: f"layer_{L}/contrast_role_pos3_default1" for L in iv_layers}
    cfg = {"vectors": {names[L]: {"vector": torch.zeros(hidden), "layer": L}
                       for L in iv_layers},
           "experiments": [
               {"id": "layers_46:54-p0.25",
                "interventions": [{"vector": names[L], "cap": -30.0 - L}
                                  for L in iv_layers]}]}
    torch.save(cfg, Path(tmpdir) / "capping_config.pt")
    return tmpdir


def test_load_capping_config_real_structure():
    cfg = am.load_capping_config(
        _write_real_structure_config(tempfile.mkdtemp(), list(range(46, 54))))
    assert cfg["cap_layers"] == list(range(46, 54))           # 8 layers, 46..53
    assert cfg["percentile"] == 0.25
    assert cfg["experiment"] == "layers_46:54-p0.25"
    assert cfg["source"] != "fallback-default"
    assert cfg["upstream_caps"] == {str(L): -30.0 - L for L in range(46, 54)}


def test_load_capping_config_band_mismatch_and_wrong_model():
    # id parses to 46..53 but interventions cover 46..52 -> hard ProvenanceError
    with pytest.raises(ProvenanceError):
        am.load_capping_config(
            _write_real_structure_config(tempfile.mkdtemp(), list(range(46, 53))))
    # wrong-model hidden size quarantined
    with pytest.raises(ProvenanceError):
        am.load_capping_config(
            _write_real_structure_config(tempfile.mkdtemp(), list(range(46, 54)),
                                         hidden=3584))


def test_intervention_grid():
    ivs = am.intervention_grid([0.25, 0.5, 0.75, 1.0])
    assert ivs[0] == {"intervention": "none", "lam": None}
    assert ivs[1] == {"intervention": "cap", "lam": 0.0}
    assert [i["lam"] for i in ivs[2:]] == [0.25, 0.5, 0.75, 1.0]
    assert all(i["intervention"] == "addback" for i in ivs[2:])


def test_build_cells_and_record_count_formula():
    ivs = am.intervention_grid([0.25, 0.5, 0.75, 1.0])        # 6 arms
    cells = am.build_cells(["a", "b", "c"], [0.06], ivs)
    exp = am.expected_record_counts(3, 1, len(ivs), n_frames=2, n_samples=10)
    assert len(cells) == exp["cells"] == 6 * (1 + 3 * 1) + 1 == 25
    assert exp["battery"] == 25 * 36
    assert exp["freeform"] == 25 * 2 * 10 and exp["audit"] == 25
    off = [c for c in cells if c[0] == am.EMOTION_OFF]
    assert len(off) == 6 and all(s == 0.0 for _, s, _ in off)
    suff = [c for c in cells if c[0] == am.NEG_AXIS]
    assert len(suff) == 1 and suff[0][1] == 0.06
    assert suff[0][2]["intervention"] == "none"               # −axis ALONE, no cap


def test_tau_percentile_and_panel_delta():
    tau = am.tau_from_projections({46: np.arange(101.0), 47: [2.0, 2.0]}, 0.25)
    assert tau[46] == 25.0 and tau[47] == 2.0
    assert am.panel_mean_delta({}, [4, 5]) == {4: 0.0, 5: 0.0}
    d = am.panel_mean_delta({("a", 0.06): {4: -2.0, 5: -4.0},
                             ("b", 0.06): {4: -4.0, 5: -8.0}}, [4, 5])
    assert d == {4: -3.0, 5: -6.0}


# =========================================================================== #
# 4. Driver (object.__new__ + fakes, no GPU)
# =========================================================================== #
_ID = {"Yes": 10, "YES": 11, "yes": 12, "No": 20, "NO": 21, "no": 22}
_DEC = {v: k for k, v in _ID.items()}


class _Tok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "U:" + msgs[0]["content"] + "|A:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * max(1, len((text or "").split()))}

    def encode(self, text, add_special_tokens=False):
        return [_ID.get(text, 1)]

    def decode(self, ids):
        return "".join(_DEC.get(int(i), str(int(i))) for i in ids)


class _Model:
    def __init__(self, hidden=8, num_layers=6):
        self.tokenizer = _Tok()
        self.num_layers, self.hidden_size = num_layers, hidden
        self._layers = [_FakeLayer() for _ in range(num_layers)]
        self._input_device = "cpu"
        self._open_captures = []
        self.steer_active = False
        self.steer_calls = []
        self.gen_log = []                                     # (steer_active, n_hooks)
        self.model = self._forward                            # driver calls model.model(...)

    def layer_index_for_fraction(self):
        return 2

    def _forward(self, input_ids=None, **kw):
        base = 11.0 if self.steer_active else 1.0             # steer shifts the stream
        T = int(input_ids.shape[1])
        for d, layers in self._open_captures:
            for L in layers:
                d[L] = torch.full((1, T, self.hidden_size), base)

    @contextmanager
    def _capture(self, layers):
        d = {}
        self._open_captures.append((d, list(layers)))
        try:
            yield d
        finally:
            self._open_captures.pop()

    def last_token_logits(self, rendered, add_special_tokens=True):
        v = np.full(30, -5.0, dtype=np.float32)
        v[10], v[11], v[12] = 1.0, 2.0, 0.5
        v[20], v[21], v[22] = 0.5, 1.0, 0.0
        return v

    def generate_batch(self, prompts, max_new_tokens=180, temperature=0.9,
                       do_sample=True, chat=False):
        self.gen_log.append((self.steer_active,
                             sum(len(l.hooks) for l in self._layers)))
        return [f"resp {i} feeling something" for i in range(len(prompts))]

    def mean_residual_over_span(self, prompt_rendered, response, layer, max_length=None):
        return np.linspace(0.1, 1.0, self.hidden_size).astype(np.float32)

    @contextmanager
    def steer(self, unit_vectors, strength, norms, positions="all"):
        self.steer_calls.append({k: np.array(v) for k, v in unit_vectors.items()})
        self.steer_active = True
        try:
            yield
        finally:
            self.steer_active = False


class _Bank:
    def __init__(self, hidden=8, num_layers=6):
        self.emotions = [s.name for s in qualia.emotion_conditions()]
        self.hidden_size = hidden
        rng = np.random.default_rng(0)
        self.vectors = rng.standard_normal(
            (num_layers, len(self.emotions), hidden)).astype(np.float32)
        self.global_means = rng.standard_normal((num_layers, hidden)).astype(np.float32)

    def index_of(self, name):
        return self.emotions.index(name)

    def check_against(self, model, where=""):
        pass


class _Engine:
    def __init__(self, model):
        self.model = model
        self._norms = {2: 10.0}

    def calibrate(self, texts):
        return self._norms

    def _require_norms(self):
        return self._norms

    @contextmanager
    def steer(self, emotion, strength, positions="all"):
        self.model.steer_active = True                        # emotion hook "active"
        try:
            yield
        finally:
            self.model.steer_active = False


def _fake_am(cap_layers=(4, 5)):
    exp = object.__new__(am.AxisMediationExperiment)
    exp.model = _Model()
    exp.bank = _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None)
    exp.layer = 2
    exp.engine = _Engine(exp.model)
    rng = np.random.default_rng(1)
    exp.axis_matrix = (np.abs(rng.standard_normal((6, 8))) + 0.1).astype(np.float32)
    exp.axis_source = "test"
    exp.cap_layers = list(cap_layers)
    exp.cap_percentile = 0.25
    exp.capping_provenance = {"source": "test", "experiment": None}
    exp.axis_units = {L: unit(exp.axis_matrix[L]) for L in exp.cap_layers}
    exp.steer_axis_unit = unit(exp.axis_matrix[2])
    exp._axis_tensors = {}
    return exp


def test_constructor_guards():
    model, bank = _Model(), _Bank()
    cfg = SimpleNamespace(model_id="fake", enable_thinking=None)
    with pytest.raises(ValueError):
        am.AxisMediationExperiment(model, bank, cfg)          # axis is mandatory
    with pytest.raises(ProvenanceError):                      # wrong-geometry axis
        am.AxisMediationExperiment(model, bank, cfg,
                                   axis=np.ones((3, 8), np.float32), cap_layers=[4, 5])
    good = (np.abs(np.random.default_rng(0).standard_normal((6, 8))) + 0.1).astype(np.float32)
    exp = am.AxisMediationExperiment(model, bank, cfg, axis=good, cap_layers=[4, 5])
    assert set(exp.axis_units) == {4, 5}
    for u in exp.axis_units.values():
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-5


def test_neg_axis_sufficiency_routing():
    exp = _fake_am()
    synth = exp._synthetic_vectors(seed=0)
    with exp._steer_cell(am.NEG_AXIS, 0.06, synth):
        assert exp.model.steer_active
    vecs = exp.model.steer_calls[-1]
    assert set(vecs) == {2}                                   # steer layer only
    assert np.allclose(vecs[2], -exp.steer_axis_unit)         # −a at matched norm
    with exp._steer_cell(am.EMOTION_OFF, 0.0, synth):
        assert not exp.model.steer_active                     # off -> no hook at all


def test_intervention_ops_layers_and_none():
    exp = _fake_am()
    tau = {4: 1.0, 5: 2.0}
    d = {4: -0.5, 5: -1.0}
    assert exp._intervention_ops({"intervention": "none", "lam": None}, tau, d) == {}
    ops = exp._intervention_ops({"intervention": "cap", "lam": 0.0}, tau, d)
    assert set(ops) == {4, 5}
    h = torch.zeros(1, 2, 8)                                  # proj 0 < tau at both
    for L in (4, 5):
        out = ops[L](h)
        p = float((out[0, 0].numpy() @ exp.axis_units[L]))
        assert p == pytest.approx(tau[L], abs=1e-4)           # lifted to the floor
    ops_ab = exp._intervention_ops({"intervention": "addback", "lam": 1.0}, tau, d)
    p = float((ops_ab[4](h)[0, 0].numpy() @ exp.axis_units[4]))
    assert p == pytest.approx(tau[4] + 1.0 * d[4], abs=1e-4)  # floor = tau + λΔ


def test_run_counts_bookkeeping_and_provenance():
    exp = _fake_am()
    payload = exp.run(emotions=["blissful", "terrified"], strengths=[0.06],
                      frames=["self"], n=2, lambdas=[0.5, 1.0], gen_batch=8, seed=0)
    ivs = payload["interventions"]
    want = am.expected_record_counts(2, 1, len(ivs), n_frames=1, n_samples=2)
    assert len(ivs) == 4 and want["cells"] == 4 * (1 + 2) + 1 == 13
    assert len(payload["battery_records"]) == want["battery"] == 13 * 36
    assert len(payload["freeform_records"]) == want["freeform"] == 13 * 2
    assert len(payload["audit_records"]) == want["audit"] == 13
    # λ bookkeeping on every record type
    for recs in (payload["battery_records"], payload["freeform_records"],
                 payload["audit_records"]):
        lams = {(r["intervention"], r["lam"]) for r in recs}
        assert ("none", None) in lams and ("cap", 0.0) in lams
        assert ("addback", 0.5) in lams and ("addback", 1.0) in lams
    # provenance: tau, capping band, λ grid, measured Δ per (condition, strength)
    assert set(payload["tau_by_layer"]) == {"4", "5"}
    assert payload["cap_layers"] == [4, 5] and payload["cap_percentile"] == 0.25
    assert payload["lambda_grid"] == [0.5, 1.0]
    assert payload["capping_config"]["source"] == "test"
    assert {(d["condition"], d["strength"]) for d in payload["delta_by_cell"]} \
        == {("blissful", 0.06), ("terrified", 0.06)}
    for d in payload["delta_by_cell"]:                        # steered audit shifted +10
        assert all(v > 0 for v in d["delta"].values())
    assert payload["audit_text"] == am.AUDIT_TEXT
    # position profile reported, not just the mean (E5 audit spec)
    assert set(payload["baseline_profile"]) == {"4", "5"}
    for r in payload["audit_records"]:
        prof = r["achieved_profile"]
        assert set(prof) == {"4", "5"}
        for L in ("4", "5"):
            assert isinstance(prof[L], list) and len(prof[L]) >= 1
            assert np.mean(prof[L]) == pytest.approx(
                r["achieved_projection"][L], abs=1e-3)
    # sentinel conditions declared; injected read-back only for real emotions
    names = [c["name"] for c in payload["conditions"]]
    assert names[0] == am.EMOTION_OFF and names[-1] == am.NEG_AXIS
    for r in payload["freeform_records"]:
        if r["condition"] in (am.EMOTION_OFF, am.NEG_AXIS):
            assert r["proj_injected"] is None
        else:
            assert r["proj_injected"] is not None
    json.dumps(payload)                                       # fully serializable


def test_run_hooks_compose_and_release():
    exp = _fake_am(cap_layers=(4, 5))
    exp.run(emotions=["blissful"], strengths=[0.06], frames=["self"], n=1,
            lambdas=[0.5, 1.0], gen_batch=8, seed=0)
    # cell order: __none__ x {none,cap,0.5,1.0}, blissful x same, __neg_axis__ x none
    log = exp.model.gen_log
    assert len(log) == 9
    assert log[0] == (False, 0)                               # off / none: nothing active
    assert log[1] == (False, 2)                               # off / cap: 2 cap hooks only
    assert log[4] == (True, 0)                                # emotion / none: steer only
    assert log[5] == (True, 2)                                # emotion + cap SIMULTANEOUS
    assert log[6] == (True, 2) and log[7] == (True, 2)        # emotion + graded add-back
    assert log[8] == (True, 0)                                # −axis alone (sufficiency)
    assert all(not l.hooks for l in exp.model._layers)        # every hook released
    assert not exp.model.steer_active
