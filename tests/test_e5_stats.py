"""
GPU-free tests for the E5 analysis layer (``scripts/compute_e5_stats.py``;
docs/prereg/E5.md binding). Synthetic ``axis_mediation.json`` fixtures PLANT a
known mediation structure and the pipeline must recover it:

* planted FULL mediation   -> verdict "full", monotone kill, elasticity < 0,
                              keystone p at the exact 2/252 valence floor;
* planted NO mediation     -> verdict "none", flat curve, p off the floor;
* planted PARTIAL          -> verdict "partial" (graded intermediate);
* NULL fixture             -> stays null ("no_effect", positive-control gate);
* leaky-clamp audit        -> "audit_failed" STOPS all verdicts;
* overshoot (λ=−0.5)       -> reversal under full mediation, flat under none;
* sufficiency (−axis arm)  -> effect detected/absent by the prereg margins;
* disclaimer trajectory    -> gate-release rise recovered with exact perm p;
* T3 concordance hook      -> concordant / discordant / e3-missing branches;
* audit fraction math, scored-file discovery, JSON-serializable output.

Fixture schema mirrors the driver's ``run()`` payload
(``emotion_probes.alignment.axis_mediation``; pinned by
``tests/test_axis_mediation.py::test_run_counts_bookkeeping_and_provenance``).
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

from emotion_probes.data import qualia, qualia_freeform as qf
import compute_e5_stats as ce5

EMOTION_OFF = ce5.EMOTION_OFF
NEG_AXIS = ce5.NEG_AXIS
LAMBDAS = [-0.5, 0.25, 0.5, 0.75, 1.0]
CAP_LAYERS = [46, 47]
S = 0.06
E = 6.0                                   # planted per-unit emotion effect (logits)

# baseline YES−NO gap per question kind
_BASE = {"target": -2.0, "inverted": 1.0, "factual": 4.0, "self_ref": 3.0}


def _ivs(lambdas=LAMBDAS):
    return [{"intervention": "none", "lam": None},
            {"intervention": "cap", "lam": 0.0},
            *({"intervention": "addback", "lam": float(l)} for l in lambdas)]


def _frac(iv, lam):
    return 0.0 if iv == "none" else 1.0 if iv == "cap" else 1.0 - lam


def _remaining(mediation, f):
    """Planted surviving fraction of the emotion effect at cancelled fraction f."""
    if mediation == "full":
        return 1.0 - f                     # f=1.5 overshoots to −0.5 (reversal)
    if mediation == "none":
        return 1.0                         # cancellation is causally inert
    if mediation == "partial":
        return 1.0 - 0.55 * min(f, 1.0)    # R_cap = 0.45: graded intermediate
    return 0.0                             # "null": no emotion effect at all


def _payload(mediation="full", suff_ratio=1.0, leaky_audit=False, scored=True,
             disclaimer_rise=True, n=4, lambdas=LAMBDAS):
    """A synthetic axis_mediation payload with planted mediation structure."""
    emotions = qf.valenced_panel_qualia()
    val = {e: float(qualia.spec_for(e).valence) for e in emotions}
    amp = {e: E * (1.0 + 0.05 * i) for i, e in enumerate(emotions)}  # break ties
    questions = qualia.all_questions()
    ivs = _ivs(lambdas)
    delta = {L: -3.0 - 0.5 * (L - CAP_LAYERS[0]) for L in CAP_LAYERS}  # human-ward
    baseline_proj = {L: 5.0 + 0.5 * (L - CAP_LAYERS[0]) for L in CAP_LAYERS}

    cells = [(EMOTION_OFF, 0.0, iv) for iv in ivs]
    for e in emotions:
        cells.extend((e, S, iv) for iv in ivs)
    cells.append((NEG_AXIS, S, ivs[0]))

    def gap_effect(cond, f):
        if cond == EMOTION_OFF:
            return 0.0
        if cond == NEG_AXIS:
            return suff_ratio * E
        return val[cond] * amp[cond] * _remaining(mediation, f)

    battery, freeform, audits = [], [], []
    for cond, s, iv in cells:
        f = _frac(iv["intervention"], iv["lam"])
        key = {"condition": cond, "strength": float(s),
               "intervention": iv["intervention"], "lam": iv["lam"]}
        eff = gap_effect(cond, f)
        for q in questions:
            d = _BASE[q.kind] + (eff if q.kind == "target" else 0.0)
            battery.append({**key, "question": q.label, "logit_yes": d, "logit_no": 0.0,
                            "logit_diff": d})
        # audits: achieved − baseline = (1 − achieved_frac)·Δ per layer
        af = 0.0 if iv["intervention"] == "none" else (0.0 if leaky_audit else f)
        if cond in (EMOTION_OFF, NEG_AXIS):
            amb = {L: 0.0 for L in CAP_LAYERS}
        else:
            amb = {L: (1.0 - af) * delta[L] for L in CAP_LAYERS}
        audits.append({**key,
                       "achieved_projection": {str(L): baseline_proj[L] + amb[L]
                                               for L in CAP_LAYERS},
                       "achieved_minus_baseline": {str(L): amb[L] for L in CAP_LAYERS},
                       "achieved_profile": {str(L): [baseline_proj[L] + amb[L]] * 3
                                            for L in CAP_LAYERS}})
        if not scored:
            continue
        # scored free-form: judge self_attribution tracks the surviving effect on
        # SELF only; disclaimer rises with cancelled fraction (gate release)
        is_emo = cond not in (EMOTION_OFF, NEG_AXIS)
        sa_self = 0.2 + (2.0 * max(0.0, _remaining(mediation, f)) if is_emo else
                         (2.0 * suff_ratio if cond == NEG_AXIS else 0.0))
        n_disc = int(round(n * min(1.0, 0.5 * f))) if (disclaimer_rise and is_emo) else 0
        for fr in ("self", "other"):
            for i in range(n):
                disc = fr == "self" and i < n_disc
                judge = {"coherence": 5,
                         "stance": "deny" if disc else
                                   ("self_attribute" if fr == "self" else "deflect_topical"),
                         "self_attribution": round(min(3.0, max(0.0, sa_self)), 3)
                                             if fr == "self" else 0.0,
                         "expressed_valence": 0, "attribution_before_lexeme": False,
                         "disclaimer": bool(disc), "parse_ok": True}
                freeform.append({**key, "frame": fr, "sample": i,
                                 "text": f"{cond}-{fr}-{i}", "proj_valence_axis": 0.0,
                                 "proj_injected": 0.1 if is_emo else None,
                                 "response_tokens": 12, "judge": judge})

    return {
        "model_id": "qwen/qwen3-32b", "suite_key": "qwen3-32b",
        "experiment": "axis_mediation", "layer": 42, "num_layers": 64,
        "enable_thinking": False, "yes_token_id": 10, "yes_token_str": "Yes",
        "no_token_id": 20, "no_token_str": "No", "calibration_norm": 30.0,
        "strengths": [S], "lambda_grid": list(lambdas), "interventions": ivs,
        "cap_layers": CAP_LAYERS, "cap_percentile": 0.25,
        "capping_config": {"source": "test", "experiment": "layers_46:54-p0.25"},
        "axis_source": "test", "audit_text": " audit", "audit_frame": "self",
        "tau_by_layer": {str(L): baseline_proj[L] - 1.0 for L in CAP_LAYERS},
        "baseline_projection": {str(L): baseline_proj[L] for L in CAP_LAYERS},
        "baseline_profile": {str(L): [baseline_proj[L]] * 3 for L in CAP_LAYERS},
        "delta_by_cell": [{"condition": e, "strength": S,
                           "delta": {str(L): delta[L] for L in CAP_LAYERS}}
                          for e in emotions],
        "panel_mean_delta": {str(L): delta[L] for L in CAP_LAYERS},
        "n_samples": n, "max_new_tokens": 180, "temperature": 0.9, "seed": 0,
        "frames": [{"key": k, "label": k.upper(), "kind": k, "text": f"{k} prompt",
                    "rendered": f"U:{k}|A:", "prompt_tokens": 5}
                   for k in ("self", "other")],
        "conditions": ([{"name": EMOTION_OFF, "kind": "baseline", "group": None,
                         "valence": None, "arousal": None, "seed": None}]
                       + [{"name": e, "kind": "qualia", "group": qualia.spec_for(e).group,
                           "valence": val[e], "arousal": None, "seed": None}
                          for e in emotions]
                       + [{"name": NEG_AXIS, "kind": "axis", "group": None,
                           "valence": None, "arousal": None, "seed": None}]),
        "questions": [{"label": q.label, "kind": q.kind, "text": q.text,
                       "expected_yes": q.expected_yes, "construct": q.construct,
                       "rendered": f"U:{q.label}|A:", "top20_ok": True}
                      for q in questions],
        "battery_records": battery, "freeform_records": freeform,
        "audit_records": audits,
    }


def _run(**kw):
    return ce5.E5Run(_payload(**kw))


# =========================================================================== #
# 1. Accessors + audit fraction math
# =========================================================================== #
def test_run_parsing_and_outcomes():
    run = _run()
    assert len(run.qualia_valenced()) == 10
    assert run.cancellation_arms()[0] == ("none", None, 0.0)
    assert ("cap", 0.0, 1.0) in run.cancellation_arms()
    assert ("addback", -0.5, 1.5) in run.cancellation_arms()
    e = run.qualia_valenced()[0]
    v = run.valence(e)
    # emotion_only target gap = planted v·amp; controls unshifted
    assert run.target_gap(e, S, "none", None) == pytest.approx(v * E, rel=1e-6)
    assert run.gap_shift(e, S, "none", None, run.controls()) == pytest.approx(0.0)
    assert run.attribution_score(e, S, "none", None) == pytest.approx(v * E, rel=1e-6)
    # baseline cell is its own reference
    assert run.target_gap(EMOTION_OFF, 0.0, "none", None) == pytest.approx(0.0)


def test_achieved_fraction_math():
    run = _run()
    e = run.qualia_valenced()[0]
    assert run.achieved_fraction(e, S, "none", None) == pytest.approx(0.0)
    assert run.achieved_fraction(e, S, "cap", 0.0) == pytest.approx(1.0)
    assert run.achieved_fraction(e, S, "addback", 0.25) == pytest.approx(0.75)
    assert run.achieved_fraction(e, S, "addback", -0.5) == pytest.approx(1.5)
    # sentinel cells have no per-condition Δ -> unauditable, excluded from gate
    assert run.achieved_fraction(EMOTION_OFF, 0.0, "cap", 0.0) is None


def test_audit_gate_clean_and_leaky():
    clean = ce5.audit_gate(_run())
    assert not clean["audit_failed"]
    assert clean["n_cells"] == 10 * len(_ivs())          # every emotion x arm audited
    assert clean["max_abs_deviation"] == pytest.approx(0.0)
    leaky = ce5.audit_gate(_run(leaky_audit=True))
    assert leaky["audit_failed"] and leaky["failures"]
    assert leaky["max_abs_deviation"] > ce5.AUDIT_TOL    # cap achieved 0 vs nominal 1


def test_vanishing_delta_family_excluded_not_failed():
    """AMENDMENT 2026-07-04: an emotion whose axis displacement is tiny (like
    'content' on real data, Δ=+4.96 vs panel median ≈95) yields junk achieved
    fractions — the machinery's own shift on the audit text dominates the
    denominator. The gate must EXCLUDE such families (disclosed via
    ``range_note``), not fail, and the elasticity must drop them from its
    x-axis rather than nominal-fallback them (which would bias toward flat)."""
    payload = _payload(mediation="full")
    tiny = qf.valenced_panel_qualia()[0]
    for row in payload["delta_by_cell"]:
        if row["condition"] == tiny:       # Δ shrunk; audit records untouched,
            row["delta"] = {L: 0.05 * d    # so fractions go junk (≈ −4 at λ=.25)
                            for L, d in row["delta"].items()}
    run = ce5.E5Run(payload)
    assert set(run.vanishing_families()) == {(tiny, ce5.cq._snap(S))}

    gate = ce5.audit_gate(run)
    assert not gate["audit_failed"]                      # excluded, not failed
    assert gate["n_cells"] == 9 * len(_ivs())            # tiny family dropped
    assert [x["condition"] for x in gate["excluded_vanishing"]] == [tiny]
    assert tiny in gate["range_note"] and "range-limited" in gate["range_note"]

    stats = ce5.compute_run(run)
    assert stats["mediation_verdict"] == "full"          # verdict unpoisoned
    dr = stats["dose_response_battery"]
    assert dr["excluded_vanishing_emotions"] == [tiny]
    assert tiny not in dr["per_emotion_slopes"]          # off the elasticity axis
    assert dr["elasticity"] < 0                          # kill still recovered


# =========================================================================== #
# 2. Planted mediation structures -> verdicts
# =========================================================================== #
def test_full_mediation_verdict_and_monotone_kill():
    stats = ce5.compute_run(_run(mediation="full"))
    assert stats["mediation_verdict"] == "full"
    eq = stats["equivalence_battery"]
    assert eq["R_cap"] == pytest.approx(0.0, abs=1e-6)          # cap ≈ baseline
    assert eq["R_addback1"] == pytest.approx(1.0, abs=1e-6)     # λ=1 ≈ emotion_only
    dr = stats["dose_response_battery"]
    assert dr["monotone_decreasing"]
    assert dr["elasticity"] < 0                                  # the kill slope
    # exact valence enumeration at the preregistered two-sided floor 2/252
    assert dr["perm"]["exact"] and dr["perm"]["n_arrangements"] == 252
    assert dr["perm"]["p_perm"] == pytest.approx(2 / 252)
    assert stats["emotion_effect"]["exists"]
    assert stats["emotion_effect"]["p_perm"] == pytest.approx(2 / 252)


def test_no_mediation_verdict_and_flat_curve():
    stats = ce5.compute_run(_run(mediation="none"))
    assert stats["mediation_verdict"] == "none"
    eq = stats["equivalence_battery"]
    assert eq["R_cap"] == pytest.approx(1.0, abs=1e-6)          # cap ≈ emotion_only
    dr = stats["dose_response_battery"]
    assert dr["elasticity"] == pytest.approx(0.0, abs=1e-9)
    assert dr["perm"]["p_perm"] == pytest.approx(1.0)           # flat: p off floor
    assert stats["emotion_effect"]["exists"]                    # effect present, unkilled


def test_partial_mediation_graded_intermediate():
    stats = ce5.compute_run(_run(mediation="partial"))
    assert stats["mediation_verdict"] == "partial"
    r_cap = stats["equivalence_battery"]["R_cap"]
    assert ce5.MARGIN_KILLED < r_cap < ce5.MARGIN_SURVIVES
    assert stats["dose_response_battery"]["elasticity"] < 0


def test_null_fixture_stays_null():
    stats = ce5.compute_run(_run(mediation="null", suff_ratio=0.0,
                                 disclaimer_rise=False))
    assert stats["mediation_verdict"] == "no_effect"            # positive-control gate
    assert stats["emotion_effect"]["p_perm"] == pytest.approx(1.0)
    assert not stats["emotion_effect"]["exists"]


def test_audit_failure_stops_verdicts():
    stats = ce5.compute_run(_run(mediation="full", leaky_audit=True),
                            e3_stats={"per_model": {"qwen3-32b": {"T3": -0.4}}})
    assert stats["audit_gate"]["audit_failed"]
    assert stats["mediation_verdict"] == "audit_failed"         # despite planted kill
    assert stats["dose_response_battery"] is None
    assert stats["disclaimer"] is None
    assert stats["overshoot"]["classification"] == "audit_failed"
    assert stats["sufficiency"]["classification"] == "audit_failed"
    assert stats["e3_concordance"]["status"] == "not_evaluable"


# =========================================================================== #
# 3. Overshoot + sufficiency
# =========================================================================== #
def test_overshoot_reversal_under_full_mediation():
    stats = ce5.compute_run(_run(mediation="full"))
    ov = stats["overshoot"]
    assert ov["present"] and ov["R_overshoot"] == pytest.approx(-0.5, abs=1e-4)
    assert ov["classification"] == "reversal"
    assert ov["consistent_with_verdict"] is True


def test_overshoot_flat_under_no_mediation():
    ov = ce5.compute_run(_run(mediation="none"))["overshoot"]
    assert ov["classification"] == "flat_full_effect"
    assert ov["R_overshoot"] == pytest.approx(1.0, abs=1e-6)
    assert ov["consistent_with_verdict"] is True


def test_overshoot_absent_without_lambda_minus_half():
    ov = ce5.compute_run(_run(lambdas=[0.25, 0.5, 0.75, 1.0]))["overshoot"]
    assert not ov["present"] and ov["classification"] == "absent"


def test_sufficiency_effect_detected_and_absent():
    suff = ce5.compute_run(_run(suff_ratio=1.0))["sufficiency"]
    assert suff["present"]
    assert suff["target_gap"] == pytest.approx(E, rel=1e-6)     # −axis alone moves it
    assert abs(suff["ratio_vs_emotion_only"]) >= ce5.MARGIN_SURVIVES
    assert suff["classification"] == "reproduces"
    assert suff["per_target_shift"]["n"] == 6                   # 6 target questions
    assert suff["judge_sa_shift"] == pytest.approx(2.0, abs=1e-3)
    none = ce5.compute_run(_run(suff_ratio=0.0))["sufficiency"]
    assert none["classification"] == "absent"
    assert none["target_gap"] == pytest.approx(0.0, abs=1e-9)


# =========================================================================== #
# 4. Judge outcome + disclaimer trajectory (scored records)
# =========================================================================== #
def test_judge_dose_response_mirrors_battery():
    stats = ce5.compute_run(_run(mediation="full"))
    jeq = stats["equivalence_judge"]
    assert jeq is not None
    assert jeq["R_cap"] == pytest.approx(0.0, abs=1e-6)
    assert jeq["R_addback1"] == pytest.approx(1.0, abs=1e-6)
    assert stats["dose_response_judge"]["elasticity"] < 0
    unscored = ce5.compute_run(_run(scored=False))
    assert unscored["equivalence_judge"] is None
    assert unscored["dose_response_judge"] is None
    assert unscored["disclaimer"] is None
    assert not unscored["scored"]


def test_disclaimer_gate_release_trajectory():
    disc = ce5.compute_run(_run(mediation="full", disclaimer_rise=True))["disclaimer"]
    assert disc is not None
    assert disc["baseline_rate"] == pytest.approx(0.0)
    assert disc["slope_vs_cancelled_fraction"] > 0              # rises with cancellation
    assert disc["perm"]["exact"] and disc["perm"]["p_perm"] <= 0.05
    assert disc["gate_release_signature"]
    flat = ce5.compute_run(_run(mediation="full", disclaimer_rise=False))["disclaimer"]
    assert flat["slope_vs_cancelled_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert not flat["gate_release_signature"]


def test_coherence_gate_and_disclaimer_normalization():
    run = _run()
    # inject an incoherent record: must be gated out of judge values
    rec = dict(run._freeform[(run.qualia_valenced()[0], S, "none", None, "self")][0])
    rec["judge"] = {**rec["judge"], "coherence": 1, "self_attribution": 3.0}
    assert not run._passes(rec)
    # disclaimer normalization: bool, 0-1 float, 0-10 int, stance fallback
    for j, want in (({"disclaimer": True}, 1.0), ({"disclaimer": 0.4}, 0.4),
                    ({"disclaimer": 7}, 0.7), ({"stance": "refuse"}, 1.0),
                    ({"stance": "self_attribute"}, 0.0)):
        r = {"judge": {"parse_ok": True, "coherence": 5, **j}}
        run2 = ce5.E5Run({"conditions": [], "questions": [], "frames": []})
        run2._freeform[("x", 0.06, "none", None, "self")] = [r]
        assert run2.disclaimer_rate("x", 0.06, "none", None) == pytest.approx(want)


# =========================================================================== #
# 5. T3 concordance hook — all three branches
# =========================================================================== #
def test_concordance_e3_missing():
    cc = ce5.compute_run(_run(mediation="full"), e3_stats=None)["e3_concordance"]
    assert cc["status"] == "e3_missing" and cc["e3_prediction"] is None


def test_concordance_concordant():
    # E3 B-pattern (T3 < 0 / B-supported) predicted the monotone kill we found
    e3 = {"per_model": {"qwen3-32b": {"verdict": "B-supported",
                                      "T3": {"observed": -0.42, "p_perm": 0.004}}}}
    cc = ce5.compute_run(_run(mediation="full"), e3_stats=e3,
                         e3_path="scripts/e3_stats_generated.json")["e3_concordance"]
    assert cc["status"] == "concordant"
    assert cc["e3_prediction"] == "kill" and cc["e3_t3"] == pytest.approx(-0.42)
    # and A-pattern (T3 > 0) predicted the flat curve we found under no mediation
    e3a = {"per_model": {"qwen3-32b": {"T3": 0.3}}}          # sign-only, no verdict str
    cc2 = ce5.compute_run(_run(mediation="none"), e3_stats=e3a)["e3_concordance"]
    assert cc2["status"] == "concordant" and cc2["e3_prediction"] == "flat"


def test_concordance_discordant():
    # E3 A-pattern predicted flat, but cancellation killed the effect
    e3 = {"per_model": {"qwen3-32b": {"verdict": "A-supported", "T3": 0.5}}}
    cc = ce5.compute_run(_run(mediation="full"), e3_stats=e3)["e3_concordance"]
    assert cc["status"] == "discordant"
    assert cc["e3_prediction"] == "flat" and cc["e5_verdict"] == "full"


# =========================================================================== #
# 6. Orchestration: compute_all, discovery, serialization
# =========================================================================== #
def test_compute_all_serializable(tmp_path):
    stats = ce5.compute_all([_run(mediation="full"), _run(mediation="none")])
    json.dumps(stats, allow_nan=True)
    assert stats["models"] == ["qwen3-32b"]                    # same key: last wins
    assert stats["margins"] == {"survives": 0.6, "killed": 0.3}


def test_discovery_prefers_scored(tmp_path):
    d = tmp_path / "qwen3-32b" / "analysis"
    d.mkdir(parents=True)
    (d / "axis_mediation.json").write_text(json.dumps(_payload(scored=False)))
    assert ce5.discover_paths(str(tmp_path)) == [str(d / "axis_mediation.json")]
    (d / "axis_mediation_scored.json").write_text(json.dumps(_payload()))
    assert ce5.discover_paths(str(tmp_path)) == [str(d / "axis_mediation_scored.json")]
    run = ce5.load_e5_run(ce5.discover_paths(str(tmp_path))[0])
    assert run.scored() and ce5.compute_run(run)["mediation_verdict"] == "full"


def test_load_e3_missing_and_malformed(tmp_path):
    assert ce5.load_e3(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ce5.load_e3(bad) is None
