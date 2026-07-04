"""
GPU-free tests for the E4 statistics layer (``scripts/compute_e4_stats.py``).

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. Everything
runs on synthetic ``referent_battery.json`` payloads that mirror the driver
schema (see ``tests/test_referent_battery.py``, which pins it) with a KNOWN
planted effect, so the pipeline must recover sign, magnitude and p-value:

* twin-folding sign math hand-checked on a minimal payload (attribution up =>
  positive folded shift; polarity-shared yes-bias cancels exactly);
* planted SELF-SPECIFIC effect -> keystone recovered at the exact permutation
  floor 2/3432 for every non-self class, per-referent slopes at the planted
  magnitude, gradient verdict 'self-specific';
* planted UNIFORM experiencer-gradient (you co-moving with the ladder, no
  self excess) -> gradient profile flags 'experiencer-gradient not
  self-specific';
* planted factual-control violation -> ``battery_invalid``;
* a null fixture stays null (no keystone, gate passes, verdict no-verdict);
* control flat check + serialization + CLI on-disk round trip.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_probes.data import qualia
from emotion_probes.data import referent_battery as rb
import compute_e4_stats as ce4

FLOOR = 2 / 3432                     # two-sided exact floor at C(14,7) arrangements


# =========================================================================== #
# Synthetic payloads (driver schema; see tests/test_referent_battery.py)
# =========================================================================== #
def _condition_specs() -> list[dict]:
    specs = [{"name": c.name, "kind": c.kind, "group": c.group,
              "valence": c.valence, "arousal": c.arousal, "seed": None}
             for c in rb.all_conditions()]
    for i, s in enumerate(qualia.SYNTHETIC_DIRECTIONS):
        specs.append({"name": s, "kind": "synthetic", "group": None,
                      "valence": None, "arousal": None, "seed": i})
    return specs


def _rec(it, gap, extra=None):
    p_yes = 0.8 / (1.0 + math.exp(-gap))
    return {**(extra or {}),
            "item": it.label, "referent": it.referent, "predicate": it.predicate,
            "polarity": it.polarity, "is_factual_control": it.is_factual_control,
            "logit_yes": float(gap), "logit_no": 0.0, "gap": float(gap),
            "p_yes": p_yes, "p_no": 0.8 - p_yes, "top20_ok": True}


def _payload(slope_map=None, factual_map=None, noise=0.01, seed=1,
             strengths=(-0.06, 0.06), suite_key="qwen3-32b"):
    """A full-schema referent-battery payload with a PLANTED attribution
    effect: for a qualia condition with valence v at strength s, the folded
    experience shift of referent r is slope_map[r]*v*s (positive item gap up,
    inverted twin gap down); factual items shift by factual_map[r]*v*s
    (unfolded). Controls (appraisal/arousal/synthetic) carry noise only."""
    slope_map = slope_map or {}
    factual_map = factual_map or {}
    rng = np.random.default_rng(seed)
    items = rb.all_items()
    specs = _condition_specs()
    base = {it.label: float(rng.normal(0.0, 1.0)) for it in items}

    baselines = [_rec(it, base[it.label]) for it in items]
    records = []
    for spec in specs:
        v = float(spec["valence"] or 0) if spec["kind"] == "qualia" else 0.0
        for s in strengths:
            for it in items:
                if it.is_factual_control:
                    eff = factual_map.get(it.referent, 0.0) * v * s
                else:
                    eff = it.polarity * slope_map.get(it.referent, 0.0) * v * s
                gap = base[it.label] + eff + float(rng.normal(0.0, noise))
                records.append(_rec(it, gap, {"condition": spec["name"],
                                              "strength": float(s)}))
    return {
        "model_id": f"q/{suite_key}", "suite_key": suite_key,
        "layer": 42, "num_layers": 64, "enable_thinking": False,
        "yes_token_id": 11, "yes_token_str": "YES",
        "no_token_id": 21, "no_token_str": "NO",
        "yes_no_resolution": {"in_top20": True},
        "calibration_norm": 30.0, "strengths": [*strengths, 0.0], "seed": 0,
        "panel_frozen": rb.PANEL.get("frozen"),
        "referents": list(rb.REFERENTS), "predicates": list(rb.PREDICATES),
        "conditions": specs,
        "items": [{"item": it.label, "referent": it.referent,
                   "predicate": it.predicate, "polarity": it.polarity,
                   "is_factual_control": it.is_factual_control, "text": it.text,
                   "rendered": f"U:{it.text}|A:", "top20_ok": True}
                  for it in items],
        "leakage": [{"name": sp["name"], "leakage_yes": 0.0,
                     "leakage_no": 0.0, "leakage_diff": 0.0} for sp in specs],
        "baselines": baselines,
        "records": records,
    }


# planted-effect fixtures (stats cached: each compute_run does ~24 exact
# 3432-arrangement enumerations)
_SELF_K = 6.0
_GRAD_K = {  # uniform plausible-experiencer gradient, NO self excess:
    "a rock": 0.0, "a thermostat": 0.5, "an insect": 1.0, "a shrimp": 1.5,
    "a fish": 2.0, "a dog": 3.0, "a human infant": 5.0, "a human stranger": 5.0,
    "the user you are talking to": 5.0, "you": 5.0,
    "AI language models like you": 5.0, "AI assistants": 5.0, "Qwen": 5.0,
    "GPT-4": 5.0, "Llama": 5.0,
}


@lru_cache(maxsize=None)
def _stats(which: str) -> dict:
    if which == "self":
        raw = _payload(slope_map={"you": _SELF_K})
    elif which == "gradient":
        raw = _payload(slope_map=_GRAD_K)
    elif which == "factual_violation":
        raw = _payload(slope_map={"you": 5.0},
                       factual_map={r: 5.0 for r in rb.REFERENTS})
    elif which == "null":
        # seed pinned to a representative null draw: any alpha=0.05 gate fires
        # on ~5% of pure-noise runs by construction, so an unlucky seed would
        # test the coin, not the pipeline
        raw = _payload(noise=0.05, seed=2)
    else:
        raise KeyError(which)
    return ce4.compute_run(ce4.E4Run(raw))


# =========================================================================== #
# 1. Twin folding — sign math hand-checked
# =========================================================================== #
def test_fold_pair_hand_math():
    # attribution UP: positive twin gap 2.0, inverted twin gap -0.6
    assert ce4.fold_pair(2.0, -0.6) == pytest.approx(1.3)
    # pure yes-bias moves both twins the SAME way and cancels exactly
    assert ce4.fold_pair(1.0 + 0.7, 0.4 + 0.7) - ce4.fold_pair(1.0, 0.4) \
        == pytest.approx(0.0)
    assert ce4.fold_pair(1.0, 0.4) == pytest.approx(0.3)


def _mini_payload():
    """1 emotion x 1 referent x 1 predicate with HAND-SET gaps."""
    cond = {"name": "blissful", "kind": "qualia", "group": "positive",
            "valence": 1, "arousal": 1, "seed": None}

    def rec(label, pol, fact, gap, cond_name=None, s=None):
        d = {"item": label, "referent": "you",
             "predicate": "factual" if fact else "suffer", "polarity": pol,
             "is_factual_control": fact, "logit_yes": gap, "logit_no": 0.0,
             "gap": gap, "p_yes": 0.5, "p_no": 0.2, "top20_ok": True}
        if cond_name is not None:
            d.update(condition=cond_name, strength=s)
        return d

    return {
        "model_id": "mini", "suite_key": None, "layer": 1, "num_layers": 2,
        "strengths": [0.0, -0.06, 0.06], "conditions": [cond],
        "referents": ["you"], "predicates": ["suffer"], "items": [],
        "baselines": [rec("p", 1, False, 1.0), rec("i", -1, False, 0.4),
                      rec("f", 1, True, 2.0)],
        "records": [
            # s=+0.06: attribution UP by 1.0 (pos +1.0, inv -1.0); factual +0.5
            rec("p", 1, False, 2.0, "blissful", 0.06),
            rec("i", -1, False, -0.6, "blissful", 0.06),
            rec("f", 1, True, 2.5, "blissful", 0.06),
            # s=-0.06: pure yes-bias +0.7 on BOTH twins -> folded shift 0
            rec("p", 1, False, 1.7, "blissful", -0.06),
            rec("i", -1, False, 1.1, "blissful", -0.06),
            rec("f", 1, True, 2.0, "blissful", -0.06),
        ],
    }


def test_folded_shift_hand_checked():
    run = ce4.E4Run(_mini_payload())
    # folded baseline = (1.0 - 0.4)/2 = 0.3; steered = (2.0 - (-0.6))/2 = 1.3
    assert run.folded_baseline("you", "suffer") == pytest.approx(0.3)
    assert run.folded_shift("blissful", 0.06, "you", "suffer") == pytest.approx(1.0)
    # the polarity-shared yes-bias cell folds to EXACTLY zero
    assert run.folded_shift("blissful", -0.06, "you", "suffer") == pytest.approx(0.0)
    # factual items are unfolded, plain baseline-corrected shifts
    assert run.factual_shift("blissful", 0.06, "you") == pytest.approx(0.5)
    assert run.factual_shift("blissful", -0.06, "you") == pytest.approx(0.0)
    assert run.folded_shift("blissful", 0.06, "you", "feel joy") is None


# =========================================================================== #
# 2. Planted self-specific effect -> keystone at the exact floor
# =========================================================================== #
def test_self_specific_keystone_at_exact_floor():
    ks = _stats("self")["keystone_self_vs_class"]
    assert set(ks) == {"humans", "animals", "other_ai", "inanimate", "all_nonself"}
    for cls, k in ks.items():
        assert k["n_clusters"] == 14
        assert k["n_arrangements"] == 3432 and k["p_perm_exact"] is True
        assert k["p_perm_floor"] == pytest.approx(1 / 3432)
        assert k["p_perm"] == pytest.approx(FLOOR), cls          # AT the floor
        assert k["contrast"] == pytest.approx(_SELF_K, abs=0.15), cls
        assert k["self_slope"] == pytest.approx(_SELF_K, abs=0.15)
        assert abs(k["class_slope"]) < 0.15


def test_self_specific_per_referent_slopes():
    per = _stats("self")["per_referent"]
    you = per["you"]
    assert you["b_val"] == pytest.approx(_SELF_K, abs=0.1)
    assert you["p_perm"] == pytest.approx(FLOOR) and you["p_perm_exact"]
    assert you["b_val_ci_lo"] < _SELF_K < you["b_val_ci_hi"]
    assert you["b_val_prob_space"] > 0                # sign replicates in P space
    assert abs(you["b_dose"]) < 0.2                   # folding cancels dose drift
    for ref in ce4.EXPERIENCER_GRADIENT:              # everything non-self is flat
        assert abs(per[ref]["b_val"]) < 0.2, ref
    assert {d["class"] for d in per.values()} == set(ce4.REFERENT_CLASSES)


def test_self_specific_gradient_verdict():
    gp = _stats("self")["gradient_profile"]
    assert gp["ladder"] == ce4.EXPERIENCER_GRADIENT
    assert gp["verdict"] == "self-specific"
    assert gp["self_excess_significant"] is True
    assert gp["gradient_significant"] is False
    assert gp["slopes_by_magnitude"][0]["referent"] == "you"


def test_self_specific_gates_and_decision():
    st = _stats("self")
    assert st["factual_gate"]["battery_invalid"] is False
    assert st["factual_gate"]["within_margin"] is True
    assert st["control_flat"]["off_manifold_artifact"] is False
    dec = st["decision"]
    assert dec["g2_existence"] is True and dec["battery_invalid"] is False
    # you-alone movement gives D = S_you > 0 at the floor -> B-leaning per the
    # verbatim G3 rule (the prereg's asymmetric claim-scope caveat is prose)
    assert st["primary_D"]["D"] == pytest.approx(_SELF_K, abs=0.15)
    assert st["primary_D"]["p_D"] == pytest.approx(FLOOR)
    assert dec["verdict"] == "B-leaning"


# =========================================================================== #
# 3. Planted uniform gradient -> flagged as NOT self-specific
# =========================================================================== #
def test_uniform_gradient_not_self_specific():
    gp = _stats("gradient")["gradient_profile"]
    assert gp["verdict"] == "experiencer-gradient not self-specific"
    assert gp["gradient_significant"] is True
    assert gp["ladder_trend"]["slope"] > 0 and gp["ladder_trend"]["p"] < 0.05
    assert gp["self_excess_significant"] is False


def test_uniform_gradient_slopes_track_plants():
    st = _stats("gradient")
    per = st["per_referent"]
    for ref, k in _GRAD_K.items():
        assert per[ref]["b_val"] == pytest.approx(k, abs=0.1), ref
    # you co-moves with the humans ceiling -> no self-excess contrast
    hum = st["keystone_self_vs_class"]["humans"]
    assert abs(hum["contrast"]) < 0.1
    # ...but you DOES exceed the inanimate floor (gradient, not boundary)
    assert st["keystone_self_vs_class"]["inanimate"]["contrast"] > 3.0


# =========================================================================== #
# 4. Factual-control violation -> battery_invalid
# =========================================================================== #
def test_factual_violation_flags_battery_invalid():
    st = _stats("factual_violation")
    fg = st["factual_gate"]
    assert fg["b_val"] == pytest.approx(5.0, abs=0.15)
    assert fg["p_perm"] == pytest.approx(FLOOR) and fg["p_perm_exact"]
    assert fg["equivalence_margin"] == pytest.approx(0.5 * abs(fg["you_b_val"]), abs=1e-9)
    assert fg["within_margin"] is False
    assert fg["battery_invalid"] is True
    assert st["decision"]["verdict"] == "battery_invalid"


# =========================================================================== #
# 5. Null fixture stays null
# =========================================================================== #
def test_null_stays_null():
    st = _stats("null")
    assert st["per_referent"]["you"]["p_perm"] > 0.05
    for k in st["keystone_self_vs_class"].values():
        assert k["p_perm"] > 0.05
    assert st["factual_gate"]["battery_invalid"] is False
    assert st["control_flat"]["off_manifold_artifact"] is False
    gp = st["gradient_profile"]
    assert gp["verdict"] == "no self-specific excess, no experiencer-gradient"
    assert st["decision"]["g2_existence"] is False
    assert st["decision"]["verdict"] == "no-verdict"


# =========================================================================== #
# 6. Control flat check + coherence + compression plumbing
# =========================================================================== #
def test_control_classes_covered_and_flat():
    cf = _stats("self")["control_flat"]
    assert set(cf["by_class"]) == {"appraisal", "arousal",
                                   qualia.RANDOM_DIRECTION, qualia.MEAN_ALL_DIRECTION}
    for cls, row in cf["by_class"].items():
        assert row["ratio_vs_qualia"] < 0.5, cls
        assert row["comparable_to_qualia"] is False, cls
    assert cf["qualia_mean_abs_folded_shift"] > 0


def test_compression_fits_and_coherence():
    st = _stats("self")
    comp = st["compression"]
    assert set(comp) == {"experience", "factual"}
    for fit in comp["experience"].values():          # flat sentinels -> ~0 line
        assert abs(fit["slope"]) < 0.05
    coh = st["coherence"]
    assert coh["n_items"] == 255 and coh["n_gated"] == 0
    assert coh["frac_records_gated"] == 0.0


# =========================================================================== #
# 7. Aggregation, serialization, CLI round trip
# =========================================================================== #
def test_compute_all_serializable():
    run = ce4.E4Run(_payload(slope_map={"you": _SELF_K}))
    stats = ce4.compute_all([run])
    assert stats["models"] == ["qwen3-32b"]
    assert "per_model" in stats and "unit_note" in stats
    json.dumps(stats, allow_nan=True)
    assert stats["per_model"]["qwen3-32b"]["size"] == 32.0


def test_cli_round_trip_and_subset_degrades_gracefully(tmp_path, monkeypatch, capsys):
    # a minimal payload (1 emotion / 1 referent) must not crash the CLI: every
    # under-powered analysis degrades to None / insufficient-data
    d = tmp_path / "qwen3-32b" / "analysis"
    d.mkdir(parents=True)
    path = d / "referent_battery.json"
    path.write_text(json.dumps(_mini_payload()))
    assert ce4.discover_paths(str(tmp_path)) == [str(path)]
    out = tmp_path / "e4_stats_generated.json"
    monkeypatch.setattr(sys, "argv",
                        ["compute_e4_stats.py", str(path), "--out", str(out)])
    assert ce4.main() == 0
    stats = json.loads(out.read_text())
    pm = stats["per_model"]["mini"]
    assert pm["per_referent"] == {} and pm["keystone_self_vs_class"] == {}
    assert pm["factual_gate"] is None and pm["primary_D"] is None
    assert pm["decision"]["verdict"] == "insufficient-data"
    assert "wrote" in capsys.readouterr().out


if __name__ == "__main__":  # allow `python tests/test_e4_stats.py`
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            continue
        fn()
        print("ok  ", fn.__name__)
    print(f"\n{len(fns)} tests defined.")
