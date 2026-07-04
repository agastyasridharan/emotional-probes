"""
GPU-free tests for the E1 geometry atlas (``emotion_probes.persona.geometry`` +
``scripts/run_e1_atlas.py``).

Everything runs on toy fixtures in small dims — no real assets, no torch, no GPU.
The toy world plants known structure: bank emotion directions orthonormal at the
test layer, human-vs-AI roles separated by a component INSIDE the emotion span
plus a component OUTSIDE it, so purging has an exactly checkable effect
(cos(h_purged, v_e) -> 0 while the off-span separation survives).

Coverage:
* suite panel resolution (5 pos / 5 neg, frozen names);
* build_directions: keys, unit norms, purge removes the emotion component,
  h retains it, valence axis matches the hand-built mean difference;
* pca_atlas: orthonormal components, descending EVR, planted PC1-axis alignment,
  both centerings, bad center raises;
* cosine_table: symmetry, unit diagonal, absolute values;
* isotropic_floor: mean tracks the analytic sqrt(2/(pi*dim)), percentile order;
* story_nulls: missing dir -> None, planted-signal p at floor vs orthogonal
  target null, layers.json row mapping, topic-contrast block, incomplete cache
  and shape mismatches hard-fail;
* run_e1_atlas.build_atlas + render_summary end-to-end on the toy world
  (exact 252-arrangement permutation block included).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emotion_probes.persona.geometry as geo
from emotion_probes.persona.assets import AssistantAxis, EmotionBank, ProvenanceError, unit

HIDDEN, N_LAYERS, LAYER = 48, 3, 1


# --------------------------------------------------------------------------- #
# Toy world
# --------------------------------------------------------------------------- #
def _panel_names():
    pos, neg = geo.suite_valence_panel()
    return pos, neg, [*pos, *neg]


def _toy_world(seed=0, plant_scale=2.0, off_scale=1.5):
    """Bank + axis + subsets with planted, exactly checkable geometry.

    At LAYER: the 10 panel emotion directions are orthonormal rows E[0..9];
    ``w`` is a unit direction orthogonal to all of them. Human roles sit at
    +plant*E[0] + off*w, AI roles at the negative — so h ~ E[0] and w mixed,
    and h_purged must be exactly ~w.
    """
    pos, neg, names = _panel_names()
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 11)))
    e_rows, w = q[:, :10].T.astype(np.float32), q[:, 10].astype(np.float32)

    vectors = rng.standard_normal((N_LAYERS, len(names) + 2, HIDDEN)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    vectors[LAYER, :10] = e_rows
    bank = EmotionBank(emotions=[*names, "extra_a", "extra_b"], vectors=vectors,
                       path="<toy bank>")

    def full(vec_at_layer, base_seed):
        m = np.tile(vec_at_layer, (N_LAYERS, 1)).astype(np.float32)
        m += 0.01 * np.random.default_rng(base_seed).standard_normal(m.shape).astype(np.float32)
        return m

    delta = plant_scale * e_rows[0] + off_scale * w
    roles = {}
    for i in range(4):
        roles[f"h{i}"] = full(+delta, 100 + i)
        roles[f"a{i}"] = full(-delta, 200 + i)
    role_mean = np.mean(list(roles.values()), axis=0)
    axis_dir = unit(rng.standard_normal(HIDDEN).astype(np.float32))
    default = role_mean + 5.0 * axis_dir
    ax = AssistantAxis(axis=(default - role_mean), default_vector=default,
                       role_vectors=roles, path="<toy axis>")
    subsets = {"human": {f"h{i}": {"emotionality": 0} for i in range(4)},
               "ai_artificial": {f"a{i}": {"emotionality": 0} for i in range(4)},
               # frozen experiencer sets — build_atlas exports z_x into the sidecar
               "experiencer_high": [f"h{i}" for i in range(4)],
               "experiencer_low": [f"a{i}" for i in range(4)]}
    return bank, ax, subsets, e_rows, w


# --------------------------------------------------------------------------- #
# 1. Suite panel
# --------------------------------------------------------------------------- #
def test_suite_valence_panel_frozen():
    pos, neg = geo.suite_valence_panel()
    assert pos == ["blissful", "ecstatic", "euphoric", "serene", "content"]
    assert neg == ["tormented", "terrified", "panicked", "hurt", "overwhelmed"]
    assert not set(pos) & set(neg)


# --------------------------------------------------------------------------- #
# 2. build_directions
# --------------------------------------------------------------------------- #
def test_build_directions_keys_and_unit_norm():
    bank, ax, subsets, _, _ = _toy_world()
    dirs = geo.build_directions(bank, ax, subsets, LAYER)
    _, _, names = _panel_names()
    assert set(dirs) == {"a", "h", "h_purged", "valence", *(f"v_{n}" for n in names)}
    for v in dirs.values():
        assert v.shape == (HIDDEN,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_purging_removes_emotion_component():
    bank, ax, subsets, e_rows, w = _toy_world()
    dirs = geo.build_directions(bank, ax, subsets, LAYER)
    _, _, names = _panel_names()
    # h retains the planted in-span component; h_purged loses ALL of it ...
    assert abs(float(dirs["h"] @ e_rows[0])) > 0.5
    for n in names:
        assert abs(float(dirs["h_purged"] @ dirs[f"v_{n}"])) < 1e-5
    # ... while the off-span separation survives and dominates
    assert abs(float(dirs["h_purged"] @ w)) > 0.99


def test_valence_axis_is_mean_difference():
    bank, ax, subsets, e_rows, _ = _toy_world()
    dirs = geo.build_directions(bank, ax, subsets, LAYER)
    expected = unit(e_rows[:5].mean(axis=0) - e_rows[5:].mean(axis=0))
    assert np.allclose(dirs["valence"], expected, atol=1e-6)


def test_build_directions_deterministic():
    bank, ax, subsets, _, _ = _toy_world()
    d1 = geo.build_directions(bank, ax, subsets, LAYER, seed=0)
    d2 = geo.build_directions(bank, ax, subsets, LAYER, seed=0)
    for k in d1:
        assert np.array_equal(d1[k], d2[k])


def test_span_basis_and_purge_math():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((4, 12)).astype(np.float32)
    v[3] = v[0] + v[1]                                   # rank-deficient
    basis = geo.span_basis(v)
    assert basis.shape[0] == 3                           # rank detected
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
    x = rng.standard_normal(12).astype(np.float32)
    p = geo.purge(x, basis)
    assert np.allclose(basis @ p, 0.0, atol=1e-4)        # nothing left in span
    assert np.allclose(geo.purge(p, basis), p, atol=1e-5)  # idempotent


# --------------------------------------------------------------------------- #
# 3. pca_atlas
# --------------------------------------------------------------------------- #
def _pca_axis(n_roles=12, seed=5):
    """Roles varying mostly along the axis direction, a little along d2."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))
    d1, d2 = q[:, 0].astype(np.float32), q[:, 1].astype(np.float32)
    base = rng.standard_normal(HIDDEN).astype(np.float32)
    roles = {f"r{i}": np.tile(base + 10.0 * np.sin(i) * d1 + 0.5 * np.cos(i) * d2,
                              (N_LAYERS, 1)).astype(np.float32)
             for i in range(n_roles)}
    ax = AssistantAxis(axis=np.tile(d1, (N_LAYERS, 1)).astype(np.float32),
                       default_vector=np.tile(base, (N_LAYERS, 1)).astype(np.float32),
                       role_vectors=roles, path="<toy>")
    return ax, d1


def test_pca_atlas_orthonormal_descending_and_aligned():
    ax, d1 = _pca_axis()
    out = geo.pca_atlas(ax, LAYER, center="mean")
    comps = out["components"]
    assert comps.shape == (out["n_pcs"], HIDDEN)
    assert np.allclose(comps @ comps.T, np.eye(out["n_pcs"]), atol=1e-4)
    evr = out["explained_variance_ratio"]
    assert all(evr[i] >= evr[i + 1] for i in range(len(evr) - 1))
    assert 0.0 < out["cum_explained_variance"] <= 1.0 + 1e-6
    assert out["cos_axis"]["pc1"] > 0.999                # planted alignment
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in out["cos_axis"].values())


def test_pca_atlas_centerings_and_bad_center():
    ax, _ = _pca_axis()
    d = geo.pca_atlas(ax, LAYER, center="default")
    m = geo.pca_atlas(ax, LAYER, center="mean")
    assert d["center"] == "default" and m["center"] == "mean"
    assert d["n_roles"] == m["n_roles"] == 12
    with pytest.raises(ValueError, match="center"):
        geo.pca_atlas(ax, LAYER, center="median")


# --------------------------------------------------------------------------- #
# 4. cosine_table + isotropic_floor
# --------------------------------------------------------------------------- #
def test_cosine_table_symmetric_abs_unit_diag():
    d = {"x": np.array([1.0, 0.0, 0.0]), "neg_x": np.array([-2.0, 0.0, 0.0]),
         "y": np.array([0.0, 3.0, 0.0])}
    t = geo.cosine_table(d)
    assert set(t) == {"x", "neg_x", "y"}
    for a in t:
        assert t[a][a] == pytest.approx(1.0)
        for b in t:
            assert t[a][b] == pytest.approx(t[b][a])
            assert 0.0 <= t[a][b] <= 1.0 + 1e-9
    assert t["x"]["neg_x"] == pytest.approx(1.0)         # abs applied
    assert t["x"]["y"] == pytest.approx(0.0)


def test_isotropic_floor_tracks_analytic():
    out = geo.isotropic_floor(dim=512, n=4000, seed=0)
    analytic = np.sqrt(2.0 / (np.pi * 512))
    assert out["analytic_mean"] == pytest.approx(analytic)
    assert abs(out["mean_abs_cos"] - analytic) / analytic < 0.1
    assert 0 < out["mean_abs_cos"] < out["p95"] <= out["p99"]
    assert out["dim"] == 512 and out["n"] == 4000


# --------------------------------------------------------------------------- #
# 5. story_nulls
# --------------------------------------------------------------------------- #
def _write_story_cache(tmp_path, t_signal, n_stories=6, cache_layers=None,
                       noise=0.05, topics=False, seed=7):
    """Cache with a planted valence signal along t_signal at model layer LAYER."""
    pos, neg, names = _panel_names()
    rng = np.random.default_rng(seed)
    n_rows = len(cache_layers) if cache_layers else N_LAYERS
    row = cache_layers.index(LAYER) if cache_layers else LAYER
    for e in names:
        sgn = 1.0 if e in pos else -1.0
        arr = noise * rng.standard_normal((n_stories, n_rows, HIDDEN)).astype(np.float32)
        arr[:, row, :] += sgn * t_signal
        np.save(tmp_path / f"{e}.npy", arr)
    if cache_layers:
        (tmp_path / "layers.json").write_text(json.dumps(cache_layers))
    if topics:
        tmap = {e: [["cooking", "sports", "geography"][i % 3] for i in range(n_stories)]
                for e in names}
        (tmp_path / "topics.json").write_text(json.dumps(tmap))
    return names


def test_story_nulls_missing_dir_returns_none(tmp_path):
    assert geo.story_nulls(None, {LAYER: {}}) is None
    assert geo.story_nulls(tmp_path / "nope", {LAYER: {}}) is None


def test_story_nulls_planted_signal_and_null_target(tmp_path):
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))
    t, orth = q[:, 0].astype(np.float32), q[:, 1].astype(np.float32)
    _write_story_cache(tmp_path, t_signal=t)
    dirs = {LAYER: {"a": t, "h": orth, "h_purged": orth}}
    out = geo.story_nulls(tmp_path, dirs, n_perm=200, seed=0)
    block = out["layers"][str(LAYER)]
    assert block["n_stories"] == 60 and block["row_in_cache"] == LAYER
    lp = block["label_permutation"]
    assert set(lp) == {"a", "h", "h_purged"}
    assert lp["a"]["observed_cos"] > 0.95                # planted alignment
    assert lp["a"]["p_perm"] == pytest.approx(lp["a"]["floor"])   # at MC floor
    assert not lp["a"]["exact"]                          # label space intractable
    assert abs(lp["h"]["observed_cos"]) < 0.3            # orthogonal target
    assert lp["h"]["p_perm"] > lp["h"]["floor"]
    assert block["topic_contrast"]["status"].startswith("gated")


def test_story_nulls_layers_json_mapping(tmp_path):
    rng = np.random.default_rng(2)
    t = unit(rng.standard_normal(HIDDEN).astype(np.float32))
    cache_layers = [0, LAYER, 7]                          # model layers cached
    _write_story_cache(tmp_path, t_signal=t, cache_layers=cache_layers)
    out = geo.story_nulls(tmp_path, {LAYER: {"a": t, "h": t, "h_purged": t}},
                          n_perm=50, seed=0)
    assert out["layers"][str(LAYER)]["row_in_cache"] == 1
    out2 = geo.story_nulls(tmp_path, {5: {"a": t, "h": t, "h_purged": t}},
                           n_perm=50, seed=0)
    assert "not in cache" in out2["layers"]["5"]["status"]


def test_story_nulls_topic_contrast_block(tmp_path):
    rng = np.random.default_rng(3)
    t = unit(rng.standard_normal(HIDDEN).astype(np.float32))
    _write_story_cache(tmp_path, t_signal=t, topics=True)
    out = geo.story_nulls(tmp_path, {LAYER: {"a": t, "h": t, "h_purged": t}},
                          n_perm=50, seed=0, min_topic_stories=2)
    tc = out["layers"][str(LAYER)]["topic_contrast"]
    assert tc["n_directions"] == 3 and tc["n_topics"] == 3
    assert not tc["enough_directions"]                    # < 50 flagged
    for v in tc["targets"].values():
        assert 0.0 <= v["control_p95_abs_cos"] <= 1.0
        assert "beats_p95" in v and "observed_abs_cos" in v


def test_story_nulls_incomplete_cache_raises(tmp_path):
    pos, _, _ = _panel_names()
    np.save(tmp_path / f"{pos[0]}.npy", np.zeros((2, N_LAYERS, HIDDEN), np.float32))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        geo.story_nulls(tmp_path, {LAYER: {"a": np.ones(HIDDEN)}})


def test_story_nulls_hidden_mismatch_raises(tmp_path):
    t = unit(np.ones(HIDDEN + 1, np.float32))             # wrong dim direction
    _write_story_cache(tmp_path, t_signal=unit(np.ones(HIDDEN, np.float32)))
    with pytest.raises(ProvenanceError, match="hidden"):
        geo.story_nulls(tmp_path, {LAYER: {"a": t, "h": t, "h_purged": t}}, n_perm=10)


# --------------------------------------------------------------------------- #
# 6. run_e1_atlas end-to-end on the toy world
# --------------------------------------------------------------------------- #
def test_build_atlas_and_summary_toy():
    import run_e1_atlas as e1

    bank, ax, subsets, _, _ = _toy_world()
    payload, arrays = e1.build_atlas(bank, ax, subsets, layers=[LAYER],
                                     floor_n=200, trait_dir=None)
    block = payload["layers"][str(LAYER)]

    # exact 252-arrangement valence-sign permutation, identity counted;
    # floor is the prereg's two-sided attainable 2/252 (mirror arrangement ties)
    for t in ("a", "h"):
        perm = block["valence_sign_perm"][t]
        assert perm["exact"] and perm["n_arrangements"] == 252
        assert perm["floor"] == pytest.approx(2 / 252)
        assert perm["p_perm"] >= perm["floor"]
        assert perm["degenerate"] is False
    # h_purged ⊥ span(v_1..v_10) BY CONSTRUCTION -> degenerate, no fake p-value
    hp = block["valence_sign_perm"]["h_purged"]
    assert hp["degenerate"] is True and hp["p_perm"] is None
    assert abs(hp["observed_cos"]) < 1e-5
    assert "unfalsifiable" in hp["note"]
    # planted world: h carries an in-span valence component (analytic ~0.253),
    # gone after the purge
    assert abs(block["signed"]["cos_valence_h"]) > 0.2
    assert abs(block["signed"]["cos_valence_h_purged"]) < 1e-4
    # AI-role jackknife block (n = 4 toy AI roles, honesty bound on raw-h stats)
    jk = payload["human_axis_jackknife"]
    assert jk["n_ai_roles"] == 4 and len(jk["dropped_roles"]) == 4
    jl = jk["layers"][str(LAYER)]
    assert jl["se_cos_valence_h"] >= 0.0 and jl["se_cos_h_a"] >= 0.0
    assert len(jl["loo_cos_h_a"]) == 4
    assert -1.0 <= jl["min_cos_h_loo_h_full"] <= jl["mean_cos_h_loo_h_full"] <= 1.0
    # gating + sidecar arrays
    assert payload["story_nulls"]["status"] == "gated on story cache"
    assert payload["trait_comparability"]["status"].startswith("skipped")
    assert f"L{LAYER}__a" in arrays and f"L{LAYER}__valence" in arrays
    assert arrays[f"L{LAYER}__pca_mean_components"].shape[1] == HIDDEN
    # E3-consumable duplicate keys ("<name>@L<layer>", long names + default PCs)
    for e3_key in (f"assistant_axis@L{LAYER}", f"human_axis@L{LAYER}",
                   f"human_axis_purged@L{LAYER}", f"valence_axis@L{LAYER}",
                   f"experiencer_axis@L{LAYER}", f"pc1@L{LAYER}"):
        assert e3_key in arrays, e3_key
    assert np.array_equal(arrays[f"assistant_axis@L{LAYER}"], arrays[f"L{LAYER}__a"])
    # z_x matches the canonical prereg-E3 builder (single-sourced) and is ⊥ a
    import emotion_probes.alignment.persona_displacement as pd
    zx = arrays[f"experiencer_axis@L{LAYER}"]
    assert np.allclose(zx, pd.experiencer_axis(ax, subsets, LAYER))
    assert abs(float(zx @ ax.unit_axis(LAYER))) < 1e-5
    assert abs(float(np.linalg.norm(zx)) - 1.0) < 1e-5
    assert "components" not in payload["layers"][str(LAYER)]["pca"]["mean"]

    md = e1.render_summary(payload)
    assert "does NOT adjudicate" in md
    assert f"| L{LAYER} |" in md
    assert "GATED" in md
    assert "degenerate" in md                             # degeneracy surfaced
    assert "jackknife" in md                              # n=7 honesty bound surfaced
    json.dumps(payload)                                   # fully serializable


def test_sidecar_satisfies_e3_required_directions(tmp_path):
    """E1→E3 run-time contract: the exported sidecar, loaded through E3's REAL
    loader, must contain every coordinate in REQUIRED_DIRECTIONS — otherwise
    ``PersonaDisplacementExperiment.run(--directions sidecar)`` refuses at
    startup (the guard that previously made the npz path unusable because the
    sidecar lacked experiencer_axis z_x)."""
    import run_e1_atlas as e1
    import emotion_probes.alignment.persona_displacement as pd

    bank, ax, subsets, _, _ = _toy_world()
    _, arrays = e1.build_atlas(bank, ax, subsets, layers=[LAYER],
                               floor_n=200, trait_dir=None)
    p = tmp_path / "e1_atlas_directions.npz"
    np.savez_compressed(p, **arrays)
    loaded = pd.load_directions_npz(p, [LAYER])
    missing = [n for n in pd.REQUIRED_DIRECTIONS if n not in loaded[LAYER]]
    assert not missing, f"sidecar fails E3's run() guard, missing: {missing}"
    for i in range(1, 9):                                 # full pc family too
        assert f"pc{i}" in loaded[LAYER]


def test_valence_sign_test_planted_extreme():
    import run_e1_atlas as e1

    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 10)))
    rows = q.T.astype(np.float32)
    target = unit(rows[:5].mean(axis=0) - rows[5:].mean(axis=0))
    out = e1.valence_sign_test(rows, [1] * 5 + [-1] * 5, target)
    assert out["observed_cos"] == pytest.approx(1.0, abs=1e-5)
    assert out["exact"] and out["n_arrangements"] == 252
    # identity + its mirror are the only arrangements reaching |cos| = 1;
    # the reported floor equals this two-sided attainable minimum (prereg 2/252)
    assert out["p_perm"] == pytest.approx(2 / 252)
    assert out["floor"] == pytest.approx(2 / 252)
    assert out["p_perm"] >= out["floor"]


if __name__ == "__main__":  # allow `python tests/test_persona_geometry.py`
    raise SystemExit(pytest.main([__file__, "-q"]))
