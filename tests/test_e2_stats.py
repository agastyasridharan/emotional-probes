"""
GPU-free tests for the E2 lexical-knockout ANALYSIS layer
(``scripts/compute_e2_stats.py`` — the prereg docs/prereg/E2.md decision quantities).

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. Coverage, all on
synthetic scored fixtures that PLANT a known effect:

* per-arm keystone b3    — planted SELF>OTHER valence-following recovered in every
                           decode arm, exact p at the 2/252 prereg floor, 252
                           arrangements, 10 clusters;
* kill test SURVIVES     — text-independent fixture (b3 equal across arms):
                           R_ban == 1, jackknife CI-lower >= 0.6, survival verdict +
                           corroboration flag; jackknife CI has real width on the
                           jittered read-back measure;
* kill test KILLED       — lexical-leakage fixture (b3 collapses under ban_emotion
                           ONLY): R_ban == 0, CI-upper <= 0.2, kill verdict, boring-
                           wins ban clause (with G3 passed), ban-arm p off the floor;
* null fixture           — stays null: G1 fails, verdicts void, p_perm == 1;
* G3 residual density    — per-arm rates + steered/baseline splits; the escalation
                           trigger fires exactly at the prereg 70%-reduction line;
* coherence gate         — coherence-1 word salad excluded (per-arm exclusion rates),
                           intent-to-treat companion keeps it;
* raw (unscored) runs    — judge-free quantities still computed, G1 cannot pass;
* plumbing               — prereg constants, driver arm-name parity, discovery
                           preference for scored files, JSON-serializable output.
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

from emotion_probes.data import qualia
from emotion_probes.data import qualia_freeform as qf
import compute_e2_stats as ce

ARMS = list(ce.ARMS)
FLOOR = 2.0 / 252.0                       # prereg two-sided exact floor at 10 clusters


# =========================================================================== #
# Synthetic scored lexical-knockout payloads with PLANTED per-arm effects
# =========================================================================== #
def _clamp(x, a, b):
    return max(a, min(b, x))


def _e2_payload(arm_gain=None, hits=None, n=3, strengths=(0.06, 0.10),
                suite_key="qwen3-32b", null=False, garble_no_ban_cell=False,
                judged=True, seed=0):
    """A scored lexical-knockout payload (driver-schema records + ``judge`` blocks).

    ``arm_gain`` scales the planted SELF-frame effect per arm (1.0 everywhere =
    text-independent effect; ban_emotion 0.0 = lexical leakage). Judge scores are
    deterministic per arm (integer rubric); the read-back projection additionally
    carries a small seeded per-(arm, emotion) jitter so the jackknife CI has real
    width. ``hits`` plants the banned_word_hits rate per arm (G3's raw material).
    """
    vp = qf.valenced_panel_qualia()
    val = {nme: float(qualia.spec_for(nme).valence) for nme in vp}
    arm_gain = arm_gain or {a: 1.0 for a in ARMS}
    hits = hits or {ce.ARM_NO_BAN: 6, ce.ARM_BAN_EMOTION: 1, ce.ARM_BAN_RANDOM: 6}
    rng = np.random.default_rng(seed)
    jitter = {(a, nme): 1.0 + 0.05 * float(rng.standard_normal())
              for a in ARMS for nme in vp}
    KJ, KP = 20.0, 8.0                    # SELF-frame judge / read-back gains

    def judge_block(fr, v, s, gain, coh=5):
        if null:
            ev, sa = 0, 0
        else:
            ev = int(round(_clamp(v * s * (KJ if fr == "self" else 1.0) * gain, -3, 3)))
            sa = 3 if (fr == "self" and s > 0 and gain > 0) else 0
        return {"coherence": coh, "stance": "self_attribute" if sa >= 2 else "deflect_topical",
                "self_attribution": sa, "expressed_valence": ev,
                "attribution_before_lexeme": fr == "self", "parse_ok": True,
                "judge_model": "fake-judge"}

    def record(cond, s, fr, arm, i, v, gain):
        pk = KP if fr == "self" else 1.0
        pj = jitter[(arm, cond)] if (fr == "self" and cond in val) else 1.0
        proj = 0.0 if null else float(v) * float(s) * pk * gain * pj
        r = {"condition": cond, "strength": float(s), "frame": fr, "arm": arm,
             "sample": i, "text": f"{cond}-{fr}-{arm}-{s}-{i}",
             "banned_word_hits": int(hits[arm]),
             "proj_valence_axis": round(float(proj), 5),
             "proj_injected": round(abs(float(v)) * float(s) * 2.0, 5) if s else None,
             "response_tokens": 40}
        if judged:
            r["judge"] = judge_block(fr, v, s, gain)
        return r

    records = []
    for arm in ARMS:                      # per-arm baselines (the unsteered+ban cell)
        for fr in ("self", "other"):
            records += [record("__baseline__", 0.0, fr, arm, i, 0.0, arm_gain[arm])
                        for i in range(n)]
        for nme in vp:
            for s in strengths:
                for fr in ("self", "other"):
                    records += [record(nme, s, fr, arm, i, val[nme], arm_gain[arm])
                                for i in range(n)]
    if garble_no_ban_cell:                # 2 coherence-1 poison samples, one SELF cell
        for i in range(2):
            r = record(vp[0], strengths[-1], "self", ce.ARM_NO_BAN, 100 + i,
                       val[vp[0]], arm_gain[ce.ARM_NO_BAN])
            r["text"] = "asdf qwer zxcv"
            r["proj_valence_axis"] = -999.0
            r["judge"] = {"coherence": 1, "stance": "none", "self_attribution": 0,
                          "expressed_valence": -3, "attribution_before_lexeme": False,
                          "parse_ok": True, "judge_model": "fake-judge"}
            records.append(r)
    return {"model_id": "q/" + suite_key, "suite_key": suite_key,
            "experiment": "lexical_knockout", "layer": 44, "num_layers": 64,
            "enable_thinking": False, "calibration_norm": 30.0, "n_samples": n,
            "max_new_tokens": 180, "temperature": 0.9, "seed": 0,
            "strengths": [0.0, *map(float, strengths)],
            "frames": [{"key": k, "label": k.upper(), "kind": k, "text": f"{k} prompt"}
                       for k in ("self", "other")],
            "conditions": [{"name": nme, "kind": "qualia", "valence": val[nme],
                            "arousal": 0.5} for nme in vp],
            "arms": list(ARMS),
            "banned_lexicon": {"kind": "emotion_lexicon", "n_words": 3, "n_token_ids": 6,
                               "words": ["joy", "sad", "terrified"], "token_ids": [1, 2, 3]},
            "control_lexicon": {"kind": "random_matched", "n_words": 3, "n_token_ids": 6,
                                "words": ["table", "walk", "green"], "token_ids": [4, 5, 6]},
            "records": records}


def _run(**kw):
    return ce.E2Run(_e2_payload(**kw))


_LEAK = dict(arm_gain={ce.ARM_NO_BAN: 1.0, ce.ARM_BAN_EMOTION: 0.0,
                       ce.ARM_BAN_RANDOM: 1.0},
             hits={ce.ARM_NO_BAN: 10, ce.ARM_BAN_EMOTION: 0, ce.ARM_BAN_RANDOM: 10})


# =========================================================================== #
# 1. Per-arm keystone b3
# =========================================================================== #
def test_keystone_per_arm_planted_effect_at_exact_floor():
    run = _run()
    for arm in ARMS:
        for m in ("expressed_valence", "proj_valence_axis"):
            ks = ce.keystone_arm(run, arm, m)
            assert ks is not None, (arm, m)
            assert ks["self_slope"] > ks["other_slope"]
            assert ks["contrast"] > 0 and ks["b3"] > 0
            # prereg inference: exact enumeration over C(10,5)=252 arrangements,
            # two-sided floor 2/252 — the planted effect sits exactly on it
            assert ks["p_perm_exact"] is True and ks["n_arrangements"] == 252
            assert ks["n_clusters"] == 10
            assert ks["p_perm"] == pytest.approx(FLOOR)
            # the balanced 5+/5- panel is negation-symmetric, so the attainable
            # two-sided floor is 2/252 (see _perm_floor), not exact_perm's raw 1/252
            assert ks["p_perm_floor"] == pytest.approx(FLOOR)


def test_keystone_missing_arm_or_frame_returns_none():
    run = _run()
    assert ce.keystone_arm(run, "no_such_arm", "expressed_valence") is None
    payload = _e2_payload()
    payload["frames"] = [f for f in payload["frames"] if f["key"] == "self"]
    assert ce.keystone_arm(ce.E2Run(payload), ce.ARM_NO_BAN, "expressed_valence") is None


# =========================================================================== #
# 2. Kill test — text-independent effect SURVIVES
# =========================================================================== #
def test_kill_test_text_independent_survives():
    pm = ce.compute_run(_run())
    kt = pm["kill_test"]["expressed_valence"]
    rb = kt["R_ban"]
    # judge scores are deterministic and equal across arms -> R_ban exactly 1
    assert rb["R"] == pytest.approx(1.0)
    assert rb["ci_lo"] >= ce.R_SURVIVES_CI_LO
    assert kt["verdict"] == "survives"
    assert kt["ban_survival_corroboration"] is True     # CI-lower >= 0.4
    assert kt["boring_wins_ban_clause"] is False
    assert kt["placebo_collapse"] is False
    assert pm["verdict"] == "survives" and pm["verdict_measure"] == "expressed_valence"
    # equal arms -> paired delta-b3 is ~0, p off the floor
    assert rb["delta_b3_vs_no_ban"] == pytest.approx(0.0, abs=1e-9)
    assert rb["p_delta_perm"] > 0.5


def test_kill_test_jackknife_ci_has_width_on_readback():
    # the jittered objective read-back: R_ban near 1 with a real (nonzero-width)
    # by-emotion jackknife CI that still clears the survival margin
    kt = ce.compute_run(_run())["kill_test"]["proj_valence_axis"]
    rb = kt["R_ban"]
    assert rb["n_jackknife"] == 10
    assert rb["se_jackknife"] > 0
    assert rb["ci_lo"] < rb["R"] < rb["ci_hi"]
    assert 0.8 < rb["R"] < 1.2
    assert rb["ci_lo"] >= ce.R_SURVIVES_CI_LO and kt["verdict"] == "survives"


# =========================================================================== #
# 3. Kill test — lexical leakage KILLED under ban_emotion only
# =========================================================================== #
def test_kill_test_lexical_leakage_killed():
    pm = ce.compute_run(_run(**_LEAK))
    assert pm["gates"]["G1"]["passed"] is True          # no_ban positive control intact
    assert pm["gates"]["G3"]["passed"] is True          # 10 -> 0 hits: fully scrubbed
    for m in ("expressed_valence", "proj_valence_axis"):
        kt = pm["kill_test"][m]
        rb = kt["R_ban"]
        assert rb["R"] == pytest.approx(0.0, abs=1e-9)
        assert rb["ci_hi"] <= ce.R_BORING_WINS_CI_HI
        assert kt["verdict"] == "killed"
        assert kt["boring_wins_ban_clause"] is True     # CI-upper <= 0.2 AND G3 passed
        assert kt["ban_survival_corroboration"] is False
        # the collapse is ban_emotion-SPECIFIC: the matched-random placebo survives
        assert kt["placebo_collapse"] is False
        assert kt["R_random"]["margin_verdict"] == "survives"
        # per-arm exact p: positive control on the floor, ban arm off it
        assert kt["per_arm"][ce.ARM_NO_BAN]["p_perm"] == pytest.approx(FLOOR)
        assert kt["per_arm"][ce.ARM_BAN_EMOTION]["p_perm"] == pytest.approx(1.0)
        # paired across-arm delta is real and lands on the exact floor
        assert rb["delta_b3_vs_no_ban"] > 0
        assert rb["p_delta_perm"] == pytest.approx(FLOOR)
    assert pm["verdict"] == "killed"


# =========================================================================== #
# 4. Null fixture stays null (and G1 voids everything)
# =========================================================================== #
def test_null_fixture_void_and_off_floor():
    pm = ce.compute_run(_run(null=True))
    g1 = pm["gates"]["G1"]
    assert g1["passed"] is False and g1["D_full_pts"] == pytest.approx(0.0)
    for m, kt in pm["kill_test"].items():
        assert kt["verdict"] == "void", m               # G1 fail voids the experiment
        assert kt["R_ban"] is None                      # no positive control -> R undefined
    for arm in ARMS:
        ks = pm["keystone_by_arm"][arm]["expressed_valence"]
        assert ks["contrast"] == pytest.approx(0.0)
        assert ks["p_perm"] == pytest.approx(1.0)
    assert pm["verdict"] == "void"


# =========================================================================== #
# 5. Gate G1 (positive control)
# =========================================================================== #
def test_g1_positive_control_fields_and_pass():
    g1 = ce.g1_positive_control(_run())
    # planted: SELF self_attribution 3, OTHER 0 on steered no_ban records
    assert g1["D_full_pts"] == pytest.approx(3.0)
    assert g1["min_pts"] == 0.5 and g1["max_p"] == 0.05
    assert g1["p_exact"] == pytest.approx(FLOOR)
    assert g1["passed"] is True
    assert "void" in g1["on_fail"]


# =========================================================================== #
# 6. Gate G3 (residual density) — rates and the exact trigger threshold
# =========================================================================== #
def test_g3_rates_by_arm_and_splits():
    g3 = ce.g3_residual_density(_run())
    by = g3["by_arm"]
    assert by[ce.ARM_NO_BAN]["hits_per_record"] == pytest.approx(6.0)
    assert by[ce.ARM_BAN_EMOTION]["hits_per_record"] == pytest.approx(1.0)
    assert by[ce.ARM_BAN_RANDOM]["hits_per_record"] == pytest.approx(6.0)
    for arm in ARMS:                                    # constant-rate fixture: splits agree
        assert by[arm]["hits_per_record_steered"] == pytest.approx(by[arm]["hits_per_record"])
        assert by[arm]["hits_per_record_baseline"] == pytest.approx(by[arm]["hits_per_record"])
        assert by[arm]["n_records"] > 0
    assert g3["reduction_vs_no_ban"] == pytest.approx(1 - 1 / 6)
    assert g3["passed"] is True and g3["escalation_trigger"] is False


def test_g3_trigger_fires_exactly_at_prereg_threshold():
    at = ce.g3_residual_density(_run(hits={ce.ARM_NO_BAN: 20, ce.ARM_BAN_EMOTION: 6,
                                           ce.ARM_BAN_RANDOM: 20}))
    below = ce.g3_residual_density(_run(hits={ce.ARM_NO_BAN: 20, ce.ARM_BAN_EMOTION: 7,
                                              ce.ARM_BAN_RANDOM: 20}))
    # exactly 70% reduced: gate passes, NO escalation
    assert at["reduction_vs_no_ban"] == pytest.approx(0.70)
    assert at["passed"] is True and at["escalation_trigger"] is False
    # 65% reduced: gate fails, escalation trigger fires
    assert below["reduction_vs_no_ban"] == pytest.approx(0.65)
    assert below["passed"] is False and below["escalation_trigger"] is True
    assert "extra-ban-words" in below["on_trigger"] and "dose-response" in below["on_trigger"]


# =========================================================================== #
# 7. Coherence gate + intent-to-treat companion
# =========================================================================== #
def test_coherence_gate_excludes_word_salad_itt_keeps_it():
    clean, garbled = _run(), _run(garble_no_ban_cell=True)
    # gated keystone is untouched by the poison samples...
    ks_c = ce.keystone_arm(clean, ce.ARM_NO_BAN, "expressed_valence")
    ks_g = ce.keystone_arm(garbled, ce.ARM_NO_BAN, "expressed_valence")
    assert ks_g["contrast"] == ks_c["contrast"] and ks_g["n_obs"] == ks_c["n_obs"]
    # ...the intent-to-treat companion includes them
    itt = ce.keystone_arm(garbled, ce.ARM_NO_BAN, "expressed_valence", gated=False)
    assert itt["n_obs"] == ks_g["n_obs"] + 2 and itt["gated"] is False
    # per-arm exclusion rates report exactly the 2 poison records, no_ban only
    coh = ce.coherence_by_arm(garbled)
    assert coh[ce.ARM_NO_BAN]["excluded_by_gate"] == 2
    assert coh[ce.ARM_NO_BAN]["exclusion_rate"] > 0
    assert coh[ce.ARM_BAN_EMOTION]["excluded_by_gate"] == 0
    assert coh[ce.ARM_BAN_RANDOM]["excluded_by_gate"] == 0
    # the companion also lands in compute_run
    pm = ce.compute_run(garbled)
    assert pm["keystone_by_arm_itt"][ce.ARM_NO_BAN]["expressed_valence"]["n_obs"] == itt["n_obs"]


# =========================================================================== #
# 8. Raw (unscored) runs: judge-free quantities only, G1 cannot pass
# =========================================================================== #
def test_unscored_run_judge_free_quantities_only():
    pm = ce.compute_run(_run(judged=False))
    # objective read-back keystone works without a judge
    ks = pm["keystone_by_arm"][ce.ARM_NO_BAN]["proj_valence_axis"]
    assert ks["contrast"] > 0 and ks["p_perm"] == pytest.approx(FLOOR)
    # judge measures are unavailable
    assert pm["keystone_by_arm"][ce.ARM_NO_BAN]["expressed_valence"] is None
    # G3 is objective and still computed
    assert pm["gates"]["G3"]["reduction_vs_no_ban"] == pytest.approx(1 - 1 / 6)
    # but the prereg affirms nothing without the judged positive control
    g1 = pm["gates"]["G1"]
    assert g1["passed"] is False and g1["D_full_pts"] is None
    assert pm["verdict"] == "void"
    # no record is gate-excluded (there is no judge gate to fail)
    assert all(v["excluded_by_gate"] == 0 for v in pm["coherence_by_arm"].values())


# =========================================================================== #
# 9. Plumbing: prereg constants, driver parity, discovery, serializability
# =========================================================================== #
def test_prereg_constants_are_frozen():
    assert ce.GATE == 3                                  # coherence gate >= 3
    assert ce.G1_MIN_D_PTS == 0.5 and ce.G1_MAX_P == 0.05
    assert ce.G3_MIN_REDUCTION == 0.70
    assert (ce.R_SURVIVES_CI_LO, ce.R_KILLED_CI_HI) == (0.6, 0.3)
    assert (ce.R_BAN_SURVIVAL_CI_LO, ce.R_BORING_WINS_CI_HI) == (0.4, 0.2)
    assert ce.CI_LEVEL == 0.90
    assert ce.VERDICT_MEASURE in ce.PRIMARY_MEASURES
    assert set(ce.PRIMARY_MEASURES) == {"expressed_valence", "self_attribution",
                                        "proj_valence_axis"}


def test_arm_names_match_gpu_driver():
    lk = pytest.importorskip("emotion_probes.alignment.lexical_knockout")
    assert ce.ARMS == lk.ARMS
    assert (ce.ARM_NO_BAN, ce.ARM_BAN_EMOTION, ce.ARM_BAN_RANDOM) == \
        (lk.ARM_NO_BAN, lk.ARM_BAN_EMOTION, lk.ARM_BAN_RANDOM)


def test_discover_prefers_scored(tmp_path):
    for key, files in (("m1", ["lexical_knockout.json", "lexical_knockout_scored.json"]),
                       ("m2", ["lexical_knockout.json"]),
                       ("m3", ["other.json"])):
        d = tmp_path / key / "analysis"
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_text("{}")
    paths = ce.discover_paths(str(tmp_path))
    assert paths == [str(tmp_path / "m1" / "analysis" / "lexical_knockout_scored.json"),
                     str(tmp_path / "m2" / "analysis" / "lexical_knockout.json")]


def test_compute_all_serializable_and_keys():
    stats = ce.compute_all([_run(suite_key="qwen3-32b"),
                            ce.E2Run(_e2_payload(suite_key="qwen3-235b", **_LEAK))])
    json.dumps(stats, allow_nan=True)                    # module writes allow_nan=True
    assert set(stats["per_model"]) == {"qwen3-32b", "qwen3-235b"}
    assert stats["prereg"]["doc"] == "docs/prereg/E2.md"
    assert stats["gate"] == ce.GATE
    pm = stats["per_model"]["qwen3-32b"]
    assert set(pm["kill_test"]) == set(ce.PRIMARY_MEASURES)
    assert pm["arms"] == ARMS
    fb = pm["dose_response_fallback"]                    # exploratory INDETERMINATE branch
    assert fb and set(fb["arm_intercepts_zero_dose"]) == set(ARMS)
