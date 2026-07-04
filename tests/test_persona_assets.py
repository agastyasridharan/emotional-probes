"""
GPU-free tests for the persona asset loaders (``emotion_probes.persona.assets``).

Everything runs on synthetic fixtures in ``tmp_path`` — no real assets, no GPU.
The assistant-axis loader tests shrink the expected geometry constants via
``monkeypatch`` (writing 275 real-size (64, 5120) tensors per test would be silly);
the shape ASSERTS themselves are still exercised, including the mandatory
:class:`ProvenanceError` on the quarantined Qwen2.5-7B (28, 171, 3584) bank shape.
torch is only needed by the ``.pt`` fixtures (importorskip'd there).

Coverage:
* emotion bank: happy path (planted values, fp32 copy semantics, index), quarantine
  shape, wrong source_model_id, missing keys;
* assistant axis: load + unwrap, wrong shape, wrong role count, unit_axis indexing;
* unit / project math (zero-norm raise, scalar vs batch);
* anchor_check pass (margin stats) + fail (AnchorError);
* role subsets: FileNotFoundError message, all three tolerated schemas;
* human_axis: exact matched-weighted construction, determinism (seed ignored),
  unknown-role and no-overlap failures, leave-one-AI-role-out jackknife.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import emotion_probes.persona.assets as pa
from emotion_probes.persona import (
    AnchorError,
    AssistantAxis,
    ProvenanceError,
    anchor_check,
    human_axis,
    load_assistant_axis,
    load_emotion_bank,
    load_role_subsets,
    project,
    unit,
)


# =========================================================================== #
# 1. Emotion bank
# =========================================================================== #
def _write_bank(path, shape=(64, 171, 5120), source="Qwen/Qwen3-32B", plant=None):
    vectors = np.zeros(shape, dtype=np.float32)
    if plant:
        for (layer, emo_idx), value in plant.items():
            vectors[layer, emo_idx, 0] = value
    emotions = np.array([f"emotion_{i}" for i in range(shape[1])], dtype=object)
    np.savez_compressed(path, emotions=emotions, vectors=vectors,
                        global_means=np.zeros((shape[0], shape[2]), dtype=np.float32),
                        source_model_id=np.array(source, dtype=object))


def test_emotion_bank_happy_path(tmp_path):
    p = tmp_path / "bank.npz"
    _write_bank(p, plant={(42, 3): 7.5})
    bank = load_emotion_bank(p)
    assert bank.path == str(p)
    assert len(bank.emotions) == 171 and bank.vectors.shape == (64, 171, 5120)
    assert bank.vectors.dtype == np.float32
    assert bank.index("emotion_3") == 3
    v = bank.vector("emotion_3", 42)
    assert v.shape == (5120,) and v.dtype == np.float32 and v[0] == 7.5
    v[0] = -1.0                                            # must be a copy
    assert bank.vectors[42, 3, 0] == 7.5


def test_emotion_bank_bad_lookups(tmp_path):
    p = tmp_path / "bank.npz"
    _write_bank(p)
    bank = load_emotion_bank(p)
    with pytest.raises(KeyError, match="not_an_emotion"):
        bank.index("not_an_emotion")
    with pytest.raises(IndexError):
        bank.vector("emotion_0", 64)


def test_emotion_bank_quarantine_shape(tmp_path):
    """The Qwen2.5-7B bank shape MUST hard-fail with an explicit quarantine message."""
    p = tmp_path / "qwen25_7b.npz"
    _write_bank(p, shape=(28, 171, 3584))
    with pytest.raises(ProvenanceError, match=r"Qwen2\.5-7B.*QUARANTINED"):
        load_emotion_bank(p)


def test_emotion_bank_wrong_source_model(tmp_path):
    p = tmp_path / "bank.npz"
    _write_bank(p, source="Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(ProvenanceError, match="source_model_id"):
        load_emotion_bank(p)


def test_emotion_bank_missing_keys(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, stuff=np.zeros(3))
    with pytest.raises(ProvenanceError, match="npz keys"):
        load_emotion_bank(p)


# =========================================================================== #
# 2. Assistant-axis loader (shrunk geometry via monkeypatch; asserts intact)
# =========================================================================== #
def _shrink(monkeypatch, n_layers=4, hidden=8, n_roles=3):
    monkeypatch.setattr(pa, "N_LAYERS", n_layers)
    monkeypatch.setattr(pa, "HIDDEN", hidden)
    monkeypatch.setattr(pa, "N_ROLES", n_roles)
    return n_layers, hidden, n_roles


def _write_axis_root(tmp_path, n_layers=4, hidden=8, role_names=("alpha", "beta", "gamma"),
                     wrap_roles=False):
    torch = pytest.importorskip("torch")
    root = tmp_path / "axis"
    (root / "role_vectors").mkdir(parents=True)
    rng = np.random.default_rng(0)
    roles = {n: rng.standard_normal((n_layers, hidden)).astype(np.float32)
             for n in role_names}
    role_mean = np.mean(list(roles.values()), axis=0)
    default = role_mean + rng.standard_normal((n_layers, hidden)).astype(np.float32)
    axis = default - role_mean
    torch.save(torch.tensor(axis, dtype=torch.bfloat16), root / "assistant_axis.pt")
    torch.save(torch.tensor(default, dtype=torch.bfloat16), root / "default_vector.pt")
    for name, v in roles.items():
        obj = torch.tensor(v, dtype=torch.bfloat16)
        torch.save({"vector": obj, "role": name} if wrap_roles else obj,
                   root / "role_vectors" / f"{name}.pt")
    return root, axis, default, roles


def test_axis_loader_happy_path(tmp_path, monkeypatch):
    n_layers, hidden, _ = _shrink(monkeypatch)
    root, axis, default, roles = _write_axis_root(tmp_path, n_layers, hidden)
    ax = load_assistant_axis(root)
    assert ax.path == str(root)
    assert ax.axis.shape == ax.default_vector.shape == (n_layers, hidden)
    assert ax.axis.dtype == ax.default_vector.dtype == np.float32
    assert set(ax.role_vectors) == set(roles)
    for v in ax.role_vectors.values():
        assert v.shape == (n_layers, hidden) and v.dtype == np.float32
    # bf16 round-trip: close, not exact
    assert np.allclose(ax.axis, axis, atol=0.05)


def test_axis_loader_unwraps_dict_role_files(tmp_path, monkeypatch):
    n_layers, hidden, _ = _shrink(monkeypatch)
    root, *_ = _write_axis_root(tmp_path, n_layers, hidden, wrap_roles=True)
    ax = load_assistant_axis(root)
    assert ax.role_vectors["alpha"].shape == (n_layers, hidden)


def test_axis_loader_rejects_wrong_shape(tmp_path, monkeypatch):
    _shrink(monkeypatch, n_layers=4, hidden=8)
    root, *_ = _write_axis_root(tmp_path, n_layers=5, hidden=8)   # 5 != 4 layers
    with pytest.raises(ProvenanceError, match="assistant_axis shape"):
        load_assistant_axis(root)


def test_axis_loader_rejects_wrong_role_count(tmp_path, monkeypatch):
    _shrink(monkeypatch, n_roles=4)                               # expects 4, fixture has 3
    root, *_ = _write_axis_root(tmp_path)
    with pytest.raises(ProvenanceError, match="3 files != 4"):
        load_assistant_axis(root)


def test_unit_axis_layer_indexing():
    ax = _toy_axis()
    u = ax.unit_axis(1)
    assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-6
    assert np.allclose(u, unit(ax.axis[1]))


# =========================================================================== #
# 3. unit / project math
# =========================================================================== #
def test_unit_normalizes_and_rejects_zero():
    v = unit([3.0, 4.0])
    assert np.allclose(v, [0.6, 0.8]) and v.dtype == np.float32
    with pytest.raises(ValueError, match="norm"):
        unit(np.zeros(5))


def test_project_scalar_and_batch():
    d = np.array([2.0, 0.0])                    # unit -> [1, 0]
    assert project(np.array([3.0, 9.0]), d) == pytest.approx(3.0)
    assert isinstance(project(np.array([3.0, 9.0]), d), float)
    out = project(np.array([[1.0, 5.0], [-2.0, 7.0]]), d)
    assert out.shape == (2,) and np.allclose(out, [1.0, -2.0])


# =========================================================================== #
# 4. anchor_check
# =========================================================================== #
def _toy_axis(n_layers=4, hidden=6, n_roles=5, default_offset=3.0, seed=1):
    """AssistantAxis whose default sits +offset along e0 relative to the roles."""
    rng = np.random.default_rng(seed)
    roles = {f"role_{i}": rng.standard_normal((n_layers, hidden)).astype(np.float32) * 0.1
             for i in range(n_roles)}
    role_mean = np.mean(list(roles.values()), axis=0)
    default = role_mean.copy()
    default[:, 0] += default_offset
    axis = default - role_mean
    return AssistantAxis(axis=axis, default_vector=default, role_vectors=roles, path="<toy>")


def test_anchor_check_passes_and_reports_margins():
    ax = _toy_axis()
    out = anchor_check(ax, layers=(0, 1, 2, 3))
    assert set(out) == {0, 1, 2, 3}
    for stats in out.values():
        assert set(stats) == {"default", "role_mean", "role_std", "margin_sd"}
        assert stats["default"] > stats["role_mean"]
        assert stats["margin_sd"] > 0


def test_anchor_check_raises_on_sign_violation():
    ax = _toy_axis()
    ax.axis = -ax.axis          # flipped sign convention -> default projects LOW
    with pytest.raises(AnchorError, match="off-by-one"):
        anchor_check(ax, layers=(0, 1))


def test_anchor_check_raises_on_single_bad_layer():
    ax = _toy_axis()
    ax.default_vector[2] = ax.default_vector[2] - 100.0 * unit(ax.axis[2])
    with pytest.raises(AnchorError, match=r"layers \[2\]"):
        anchor_check(ax, layers=(0, 1, 2, 3))


# =========================================================================== #
# 5. Role subsets + human_axis
# =========================================================================== #
def test_load_role_subsets_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="adjudicat"):
        load_role_subsets(tmp_path / "role_subsets.json")


def test_load_role_subsets_reads_json(tmp_path):
    p = tmp_path / "role_subsets.json"
    payload = {"roles": {"poet": {"category": "human", "emotionality": 2}}}
    p.write_text(json.dumps(payload))
    assert load_role_subsets(p) == payload


def _matching_axis():
    """Six roles with constant per-role vectors so means are exactly checkable."""
    n_layers, hidden = 3, 4
    values = {"h0": 1.0, "h1": 2.0, "h2": 4.0, "h3": 8.0, "a0": -1.0, "a1": -2.0}
    roles = {n: np.full((n_layers, hidden), v, dtype=np.float32) for n, v in values.items()}
    return AssistantAxis(axis=np.ones((n_layers, hidden), dtype=np.float32),
                         default_vector=np.zeros((n_layers, hidden), dtype=np.float32),
                         role_vectors=roles, path="<toy>")


_TOY_SUBSETS = {
    # emotionality: h0,h1 at 0; h2 at 1; h3 at 2 (unmatched) | a0 at 0; a1 at 1
    "human": {"h0": {"emotionality": 0}, "h1": {"emotionality": 0},
              "h2": {"emotionality": 1}, "h3": {"emotionality": 2}},
    "ai_artificial": {"a0": {"emotionality": 0}, "a1": {"emotionality": 1}},
}


def test_human_axis_matched_weighted_and_shape():
    ax = _matching_axis()
    h = human_axis(ax, _TOY_SUBSETS, seed=0)
    assert h.shape == (3, 4) and h.dtype == np.float32
    # overlapping levels {0, 1} with weights min(2,1)=1 and min(1,1)=1 -> 0.5 each:
    # h = [0.5*mean(1,2) + 0.5*4] - [0.5*(-1) + 0.5*(-2)] = 2.75 + 1.5 = 4.25;
    # h3 (level 2, no AI counterpart) never contributes.
    assert np.allclose(h, 4.25)


def test_human_axis_deterministic_and_seed_ignored():
    ax = _matching_axis()
    assert np.array_equal(human_axis(ax, _TOY_SUBSETS, seed=0),
                          human_axis(ax, _TOY_SUBSETS, seed=0))
    outs = {float(human_axis(ax, _TOY_SUBSETS, seed=s)[0, 0]) for s in range(12)}
    assert outs == {4.25}          # construction is deterministic; seed has no effect


def test_human_axis_jackknife_leave_one_ai_out():
    ax = _matching_axis()
    loo = pa.human_axis_jackknife(ax, _TOY_SUBSETS)
    assert set(loo) == {"a0", "a1"}
    # drop a0 -> only level 1 overlaps: h = h2 - a1 = 4 - (-2) = 6
    assert np.allclose(loo["a0"], 6.0)
    # drop a1 -> only level 0 overlaps: h = mean(1, 2) - (-1) = 2.5
    assert np.allclose(loo["a1"], 2.5)
    for v in loo.values():
        assert v.shape == (3, 4) and v.dtype == np.float32


_RATER_STYLE_SUBSETS = {"roles": {
    "h0": {"category": "human", "emotionality": 0},
    "h2": {"category": "human", "emotionality": 1},
    "a0": {"category": "ai_artificial", "emotionality": 0},
    "a1": {"category": "ai_artificial", "emotionality": 1},
    "x": {"category": "nonhuman_other", "emotionality": 1},   # other categories ignored
}}


def test_human_axis_rater_file_schema():
    """Adjudicated file may keep the rater-file shape: {'roles': {name: {...}}}."""
    h = human_axis(_matching_axis(), _RATER_STYLE_SUBSETS, seed=0)
    # exact: ({h0} vs {a0}) + ({h2} vs {a1}) -> mean(1,4) - mean(-1,-2) = 2.5 + 1.5
    assert np.allclose(h, 4.0)


def test_human_axis_unknown_role_raises():
    subsets = {"roles": {**_RATER_STYLE_SUBSETS["roles"],
                         "ghost": {"category": "human", "emotionality": 0}}}
    with pytest.raises(KeyError, match="no vector"):
        human_axis(_matching_axis(), subsets)


def test_human_axis_list_plus_ratings_schema():
    ax = _matching_axis()
    subsets = {"human": ["h0", "h2"], "ai_artificial": ["a0", "a1"],
               "ratings": {"h0": {"emotionality": 0}, "h2": {"emotionality": 1},
                           "a0": {"emotionality": 0}, "a1": {"emotionality": 1}}}
    assert np.allclose(human_axis(ax, subsets, seed=0), 4.0)


def test_human_axis_no_overlap_raises():
    ax = _matching_axis()
    subsets = {"human": {"h0": {"emotionality": 0}},
               "ai_artificial": {"a1": {"emotionality": 2}}}
    with pytest.raises(ValueError, match="histograms do not overlap"):
        human_axis(ax, subsets)


def test_human_axis_missing_category_raises():
    with pytest.raises(KeyError, match="ai_artificial"):
        human_axis(_matching_axis(), {"human": {"h0": {"emotionality": 0}}})


if __name__ == "__main__":  # allow `python tests/test_persona_assets.py`
    raise SystemExit(pytest.main([__file__, "-q"]))
