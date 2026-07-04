"""
CPU statistics for E2 — the bidirectional lexical-priming kill test, hard-ban slice
(docs/prereg/E2.md is binding; portfolio.md §E2 authoritative on any conflict).

Reads the scored per-model file ``lexical_knockout_scored.json`` (the raw
``emotion_probes.alignment.lexical_knockout`` output run through the blind judge of
``scripts/score_qualia_freeform.py``; falls back to the raw ``lexical_knockout.json``,
in which case only the judge-free quantities exist) and computes the prereg E2
decision quantities:

* PER-ARM KEYSTONE b3 — the SELF-vs-OTHER valence-following interaction of
  :func:`compute_freeform_stats.keystone`, computed separately inside each decode
  arm {no_ban, ban_emotion_lexicon, ban_random_matched}, on both judge outcomes
  (``expressed_valence``, ``self_attribution``) and the judge-independent read-back
  projection (``proj_valence_axis``). Within-arm inference is the prereg's exact
  enumeration over C(10,5)=252 valence arrangements (two-sided floor 2/252),
  through :func:`exact_perm.sign_permutation_pvalue` on the UNROUNDED statistic.
* PRIMARY KILL TEST (prereg verbatim) — "Judge self-attribution effect D_X per
  arm; equivalence ratios R_X = D_X / D_full": D_X = mean_e[mean SELF − mean
  OTHER] judge self-attribution over steered records inside each decode arm,
  R_ban / R_random with 90% leave-one-emotion-out jackknife CIs and the prereg
  margins (CI-lower >= 0.6 survives / CI-upper <= 0.3 killed / else
  INDETERMINATE). D_X is a raw point difference — valence-SYMMETRIC, hence
  untestable by valence permutation — so the equivalence decision is purely
  CI-vs-margin; the exact-permutation machinery lives in the b3 companions.
* b3 KILL TEST companions — equivalence ratios ``R_ban`` = b3(ban_emotion)/b3(no_ban)
  and ``R_random`` = b3(ban_random)/b3(no_ban) per measure, with the same
  jackknife CIs, margins and decision-rule flags: the boring-wins
  ban clause (R_ban CI-upper <= 0.2 AND G3 passed — the full BORING-WINS verdict
  additionally needs the judge-validity arm, which is a different driver) and
  ban-survival corroboration (R_ban CI-lower >= 0.4). The paired across-arm Δb3
  gets the same exact enumeration (one shared valence arrangement, arms differenced).
* GATE G1 (in-run positive control) — D_full >= 0.5 pts on the judge 0-3
  self-attribution scale (SELF − OTHER over steered no_ban records), exact
  p <= 0.05 from the no_ban keystone enumeration; failure VOIDS the experiment
  (no null is affirmed without the positive control).
* GATE G3 (residual density) — mean ``banned_word_hits`` per record by arm
  (steered/baseline splits reported); the escalation trigger fires iff the density
  under ban_emotion_lexicon is NOT reduced >= 70% vs no_ban (prereg: escalate
  ``--extra-ban-words``, else demote the arm to dose-response).
* COHERENCE GATE — judge coherence >= 3 (the same gate as compute_freeform_stats),
  with the prereg's intent-to-treat companion (ungated keystones, reported in
  parallel) and per-arm exclusion rates.
* INDETERMINATE fallback (exploratory) — the prereg dose-response regression of
  the per-(arm, emotion, strength) signed judge effect on ACHIEVED dose (signed
  read-back valence + measured affect-word density) with per-arm intercepts; the
  arm intercept at zero lexical dose is the anti-boring quantity.

Documented deviations from the prereg text (none from its decision thresholds):
* the prereg CI is a 90% emotion-level BCa bootstrap; this implements the
  by-emotion jackknife instead (the acceleration ingredient of BCa) — same
  cluster level, deterministic, no resampling;
* the prereg's paired Δ_e test enumerates 2^10 sign-flips; all inference here is
  routed through :func:`exact_perm.sign_permutation_pvalue` (mandated), so the
  Δb3 test enumerates the 252 valence arrangements shared across arms instead —
  still exact, a different (valence-exchangeability) randomization group.

Everything reuses :mod:`compute_qualia_stats` (``cluster_robust_interaction``,
``_snap``, ``SUITE_SIZES``) and mirrors :mod:`compute_freeform_stats` (Run wrapper,
``compute_all``, CLI). Pure numpy/scipy, CPU, importable/testable without a GPU.

  python3 scripts/compute_e2_stats.py --suite-root suite \
      --out scripts/e2_stats_generated.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import scipy.stats as sp

import compute_qualia_stats as cq    # sibling in scripts/ — reuse its stat primitives
import compute_freeform_stats as cff  # Run wrapper + keystone semantics to mirror
import exact_perm as ep               # exact by-cluster permutation enumeration

# Decode arms — must match emotion_probes.alignment.lexical_knockout.ARMS
# (re-declared so the stats layer stays importable without torch; the tests
# cross-check the strings against the driver when it is importable).
ARM_NO_BAN = "no_ban"
ARM_BAN_EMOTION = "ban_emotion_lexicon"
ARM_BAN_RANDOM = "ban_random_matched"
ARMS = [ARM_NO_BAN, ARM_BAN_EMOTION, ARM_BAN_RANDOM]

GATE = cff.GATE                                  # coherence gate >= 3 (prereg)
PRIMARY_MEASURES = list(cff.PRIMARY_MEASURES)    # judge outcomes + objective read-back
# The judged SELF-vs-OTHER valence slope: the b3-companion measure whose exact
# valence-arrangement p backs gate G1 (the raw self-attribution point difference
# is valence-symmetric, hence untestable by valence permutation). The model
# VERDICT itself comes from the prereg primary, ``attribution_kill_test``.
VERDICT_MEASURE = "expressed_valence"
N_PERM = cff.N_PERM

# Prereg constants (docs/prereg/E2.md — binding; do not tune).
G1_MIN_D_PTS = 0.5       # G1: D_full >= 0.5 pts on the judge 0-3 self-attribution scale
G1_MAX_P = 0.05          # ... at exact p <= 0.05, else the whole experiment is void
G3_MIN_REDUCTION = 0.70  # G3: residual affect-lexicon density reduced >= 70% vs no_ban
R_SURVIVES_CI_LO = 0.6   # equivalence margins: CI-lower >= 0.6 survives
R_KILLED_CI_HI = 0.3     # CI-upper <= 0.3 killed; else INDETERMINATE
R_BAN_SURVIVAL_CI_LO = 0.4   # "ban survival (R_ban CI-lower >= 0.4) as corroboration"
R_BORING_WINS_CI_HI = 0.2    # "R_ban CI-upper <= 0.2 with G3 passed" (boring-wins ban clause)
CI_LEVEL = 0.90              # 90% emotion-level CIs


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
class E2Run(cff.FreeformRun):
    """Parsed one-model lexical-knockout results: a FreeformRun re-indexed by arm.

    Unscored (judge-less) runs are accepted: records without a ``judge`` block pass
    the gate untouched — only judge-free measures yield values — so gate G3 and the
    ``proj_valence_axis`` keystones are computable before judging, while every
    judge-based quantity is None and gate G1 cannot pass (the prereg affirms no
    null without the judged positive control)."""

    def __init__(self, raw: dict, gate: int = GATE):
        super().__init__(raw, gate=gate)
        self.arms = list(raw.get("arms", ARMS))
        self._rec = {}
        self._by_arm: dict = {a: [] for a in self.arms}
        for r in raw.get("records", []):
            self._rec.setdefault(
                (r["condition"], cq._snap(r["strength"]), r["frame"], r["arm"]), []).append(r)
            self._by_arm.setdefault(r["arm"], []).append(r)

    def records(self, arm: str) -> list:
        return list(self._by_arm.get(arm, []))

    def _passes(self, r: dict) -> bool:
        if "judge" not in r:      # unscored run: no judge gate is possible
            return True
        return super()._passes(r)

    def values(self, cond: str, strength, frame: str, arm: str, measure: str,
               gated: bool = True) -> list:
        out = []
        for r in self._rec.get((cond, cq._snap(strength), frame, arm), []):
            if gated and not self._passes(r):
                continue
            v = self._measure(r, measure)
            if v is not None:
                out.append(v)
        return out


def load_e2_run(path) -> E2Run:
    with open(path) as f:
        return E2Run(json.load(f))


# --------------------------------------------------------------------------- #
# Keystone core — per-arm b3 with exact valence-arrangement enumeration
# --------------------------------------------------------------------------- #
def _perm_floor(perm: dict, signs) -> float:
    """Smallest attainable p of this permutation test (reported, never decided
    against). exact_perm reports 1/n_arrangements (the identity arrangement
    always counts), but on a negation-symmetric label multiset — the balanced
    5+/5− valence panel — the enumeration also contains the fully NEGATED
    arrangement, whose |statistic| ties |observed| exactly, so the attainable
    two-sided floor is 2/n_arrangements: the prereg's 2/252."""
    fl = float(perm["floor"])
    if not perm.get("exact"):
        return fl
    lab = sorted(round(float(x), 9) for x in np.asarray(signs, float))
    if lab == sorted(round(-float(x), 9) for x in np.asarray(signs, float)):
        return min(1.0, 2.0 * fl)
    return fl


def _panel(run: E2Run):
    """The valenced emotion clusters and their sign vector, in a stable order."""
    emos = run.qualia_valenced()
    signs = np.array([float(run.conditions[n]["valence"]) for n in emos], float)
    return emos, signs


def _arm_design(run: E2Run, arm: str, emos: list, measure: str,
                frames: tuple = ("self", "other"), gated: bool = True):
    """Long-format design for the SELF-vs-OTHER keystone inside one arm:
    (emo_ix, strength, is_hi_frame, y) arrays, or None if under-powered."""
    hi, lo = frames
    if hi not in run.frames() or lo not in run.frames():
        return None
    emo_ix, s_arr, g_arr, y = [], [], [], []
    for j, name in enumerate(emos):
        for s in run.nonzero_strengths():
            for fr, g in ((hi, 1.0), (lo, 0.0)):
                for yv in run.values(name, s, fr, arm, measure, gated=gated):
                    emo_ix.append(j); s_arr.append(float(s))
                    g_arr.append(g); y.append(float(yv))
    if len(y) < 6 or len(set(emo_ix)) < 3:
        return None
    return (np.asarray(emo_ix), np.asarray(s_arr, float),
            np.asarray(g_arr, float), np.asarray(y, float))


def _b3(design, signs: np.ndarray, mask=None) -> float:
    """UNROUNDED interaction coefficient b3 of ``y ~ x + g + x*g`` with
    x = sign[emotion]*strength, g = is_hi_frame. The SAME estimator computes the
    observed and every permuted statistic (see exact_perm's docstring for the
    rounding bug this prevents)."""
    emo_ix, s_arr, g_arr, y = design
    if mask is not None:
        emo_ix, s_arr, g_arr, y = emo_ix[mask], s_arr[mask], g_arr[mask], y[mask]
    x = signs[emo_ix] * s_arr
    A = np.column_stack([np.ones(len(y)), x, g_arr, x * g_arr])
    beta = np.linalg.pinv(A.T @ A) @ A.T @ y
    return float(beta[3])


def keystone_arm(run: E2Run, arm: str, measure: str, frames: tuple = ("self", "other"),
                 n_perm: int = N_PERM, seed: int = 0, gated: bool = True) -> dict | None:
    """Within-arm keystone: b3 = (SELF slope − OTHER slope) of `measure` on
    valence*strength, clustered by emotion, exact valence-arrangement permutation p.
    Mirrors :func:`compute_freeform_stats.keystone` restricted to one decode arm."""
    emos, signs = _panel(run)
    if not emos:
        return None
    design = _arm_design(run, arm, emos, measure, frames, gated=gated)
    if design is None:
        return None
    emo_ix, s_arr, g_arr, yv = design
    x = signs[emo_ix] * s_arr
    obs = cq.cluster_robust_interaction(
        np.column_stack([x, g_arr, x * g_arr]).tolist(), yv, [emos[i] for i in emo_ix])
    if obs is None:
        return None
    perm = ep.sign_permutation_pvalue(signs, lambda sgn: _b3(design, sgn),
                                      n_perm=n_perm, seed=seed)
    hi, lo = frames
    return {
        "arm": arm, "measure": measure, "frames": [hi, lo], "gated": bool(gated),
        f"{hi}_slope": obs["interacted_slope"], f"{lo}_slope": obs["base_slope"],
        "contrast": obs["contrast"], "b3": perm["observed"],
        "n_obs": obs["n_obs"], "n_clusters": obs["n_clusters"],
        "t_cluster": obs["t_cluster"], "p_cluster": obs["p_cluster"],
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"],
        "p_perm_floor": _perm_floor(perm, signs),
    }


# --------------------------------------------------------------------------- #
# Kill test — equivalence ratios with by-emotion jackknife CIs
# --------------------------------------------------------------------------- #
def _jk_ci(R: float, reps: list, level: float = CI_LEVEL) -> dict:
    """Jackknife CI from leave-one-emotion-out replicates. UNROUNDED on purpose:
    the margin verdicts compare these numbers against the prereg thresholds,
    and deciding on a rounded statistic is the same bug class exact_perm.py's
    docstring documents for p-values."""
    n = len(reps)
    if n < 2:
        return {"R": R, "ci_lo": None, "ci_hi": None,
                "se_jackknife": None, "n_jackknife": n, "ci_level": level}
    arr = np.asarray(reps, float)
    se = math.sqrt((n - 1) / n * float(((arr - arr.mean()) ** 2).sum()))
    tcrit = float(sp.t.ppf(0.5 + level / 2, n - 1))
    return {"R": R, "ci_lo": R - tcrit * se, "ci_hi": R + tcrit * se,
            "se_jackknife": se, "n_jackknife": n, "ci_level": level}


def _jackknife_ratio(design_full, design_arm, emos: list, signs: np.ndarray,
                     level: float = CI_LEVEL) -> dict | None:
    """R = b3_arm / b3_full with a leave-one-emotion-out jackknife CI (both arms
    lose the emotion together — the emotion is the cluster unit). Degenerate
    replicates (positive-control b3 at machine zero) are dropped and counted."""
    b3f = _b3(design_full, signs)
    if abs(b3f) < 1e-12:
        return None                     # no positive-control effect: R undefined
    R = _b3(design_arm, signs) / b3f
    reps = []
    for j in range(len(emos)):
        mf, ma = design_full[0] != j, design_arm[0] != j
        if mf.all() and ma.all():
            continue                    # emotion contributes no rows anywhere
        if len(set(design_full[0][mf])) < 3 or len(set(design_arm[0][ma])) < 3:
            continue
        b3f_j = _b3(design_full, signs, mask=mf)
        if abs(b3f_j) < 1e-12:
            continue
        reps.append(_b3(design_arm, signs, mask=ma) / b3f_j)
    return _jk_ci(R, reps, level)


def _margin_verdict(ci: dict | None) -> str:
    """Prereg equivalence margins on the emotion-level CI:
    CI-lower >= 0.6 survives / CI-upper <= 0.3 killed / else INDETERMINATE."""
    if not ci or ci.get("ci_lo") is None:
        return "indeterminate"
    if ci["ci_lo"] >= R_SURVIVES_CI_LO:
        return "survives"
    if ci["ci_hi"] <= R_KILLED_CI_HI:
        return "killed"
    return "indeterminate"


def kill_test(run: E2Run, measure: str, g3_passed: bool,
              frames: tuple = ("self", "other"), n_perm: int = N_PERM,
              seed: int = 0) -> dict | None:
    """The hard-ban kill test on one outcome: per-arm b3 + exact p, equivalence
    ratios R_ban / R_random with jackknife CIs, prereg margins + decision-rule
    flags, and the paired across-arm Δb3 exact p. The ``verdict`` here is the
    R_ban margin verdict; :func:`compute_run` overrides it to "void" when G1 fails."""
    emos, signs = _panel(run)
    if not emos:
        return None
    designs = {a: _arm_design(run, a, emos, measure, frames) for a in run.arms}
    per_arm = {}
    for a in run.arms:
        ks = keystone_arm(run, a, measure, frames, n_perm=n_perm, seed=seed)
        if ks:
            per_arm[a] = {"b3": ks["b3"], "contrast": ks["contrast"],
                          "p_perm": ks["p_perm"], "p_perm_floor": ks["p_perm_floor"],
                          "n_obs": ks["n_obs"]}
    out = {"measure": measure, "frames": list(frames), "per_arm": per_arm,
           "D_full_b3": per_arm.get(ARM_NO_BAN, {}).get("b3")}

    full = designs.get(ARM_NO_BAN)
    for key, arm in (("R_ban", ARM_BAN_EMOTION), ("R_random", ARM_BAN_RANDOM)):
        block = None
        da = designs.get(arm)
        if full is not None and da is not None:
            ci = _jackknife_ratio(full, da, emos, signs)
            if ci is not None:
                delta = ep.sign_permutation_pvalue(
                    signs, lambda sgn, _da=da: _b3(full, sgn) - _b3(_da, sgn),
                    n_perm=n_perm, seed=seed)
                block = {"arm": arm, **ci, "margin_verdict": _margin_verdict(ci),
                         "delta_b3_vs_no_ban": delta["observed"],
                         "p_delta_perm": delta["p_perm"], "p_delta_exact": delta["exact"]}
        out[key] = block

    rb = out["R_ban"]
    out["ban_survival_corroboration"] = bool(
        rb and rb["ci_lo"] is not None and rb["ci_lo"] >= R_BAN_SURVIVAL_CI_LO)
    out["boring_wins_ban_clause"] = bool(
        rb and rb["ci_hi"] is not None and rb["ci_hi"] <= R_BORING_WINS_CI_HI and g3_passed)
    out["boring_wins_note"] = ("full BORING-WINS additionally requires the judge-validity "
                               "arm (prereg decision rule); this file carries the ban arms only")
    out["placebo_collapse"] = bool(
        out["R_random"] and out["R_random"]["margin_verdict"] == "killed")
    out["verdict"] = _margin_verdict(rb)
    return out


# --------------------------------------------------------------------------- #
# PRIMARY kill test — prereg D_X = judge self-attribution effect per arm
# --------------------------------------------------------------------------- #
def attribution_D_by_emotion(run: E2Run, arm: str) -> dict:
    """Per-emotion judge self-attribution effect D_e = mean(SELF) − mean(OTHER)
    over the steered, coherence-gated records of one decode arm (0-3 scale)."""
    out = {}
    for name in run.qualia_valenced():
        hi, lo = [], []
        for s in run.nonzero_strengths():
            hi += run.values(name, s, "self", arm, "self_attribution")
            lo += run.values(name, s, "other", arm, "self_attribution")
        if hi and lo:
            out[name] = float(np.mean(hi)) - float(np.mean(lo))
    return out


def attribution_kill_test(run: E2Run, g3_passed: bool) -> dict | None:
    """The prereg E2 PRIMARY outcome, verbatim: "Judge self-attribution effect
    D_X per arm; equivalence ratios R_X = D_X / D_full", with 90% emotion-level
    leave-one-emotion-out jackknife CIs (the documented CI deviation) and the
    prereg margins. D_X is a raw SELF−OTHER point difference — valence-
    SYMMETRIC, so no valence-arrangement permutation p exists for it; the
    equivalence decision is purely CI-vs-margin, on UNROUNDED numbers. The
    per-measure b3 ``kill_test`` blocks carry the exact-permutation machinery
    as companions."""
    per_arm = {a: attribution_D_by_emotion(run, a) for a in run.arms}
    d_full = per_arm.get(ARM_NO_BAN) or {}
    emos = sorted(d_full)
    if not emos:
        return None
    out = {
        "measure": "self_attribution",
        "definition": "D_X = mean_e[mean SELF − mean OTHER] judge "
                      "self-attribution (0-3) over steered records per arm; "
                      "R_X = D_X / D_full (prereg E2 primary outcome)",
        "D_full_pts": float(np.mean([d_full[e] for e in emos])),
        "per_arm_D_pts": {a: (float(np.mean(list(d.values()))) if d else None)
                          for a, d in per_arm.items()},
        "per_emotion_D_full": {e: round(d_full[e], 5) for e in emos},
    }
    for key, arm in (("R_ban", ARM_BAN_EMOTION), ("R_random", ARM_BAN_RANDOM)):
        d_arm = per_arm.get(arm) or {}
        shared = [e for e in emos if e in d_arm]
        block = None
        if len(shared) >= 2:
            fullv = np.asarray([d_full[e] for e in shared], float)
            armv = np.asarray([d_arm[e] for e in shared], float)
            if abs(float(fullv.mean())) > 1e-12:
                R = float(armv.mean() / fullv.mean())
                reps = []
                for j in range(len(shared)):
                    m = np.arange(len(shared)) != j
                    den = float(fullv[m].mean())
                    if abs(den) < 1e-12:
                        continue
                    reps.append(float(armv[m].mean()) / den)
                ci = _jk_ci(R, reps)
                block = {"arm": arm, "n_emotions": len(shared), **ci,
                         "margin_verdict": _margin_verdict(ci)}
        out[key] = block
    rb = out["R_ban"]
    out["ban_survival_corroboration"] = bool(
        rb and rb["ci_lo"] is not None and rb["ci_lo"] >= R_BAN_SURVIVAL_CI_LO)
    out["boring_wins_ban_clause"] = bool(
        rb and rb["ci_hi"] is not None and rb["ci_hi"] <= R_BORING_WINS_CI_HI and g3_passed)
    out["placebo_collapse"] = bool(
        out["R_random"] and out["R_random"]["margin_verdict"] == "killed")
    out["verdict"] = _margin_verdict(rb)
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def g1_positive_control(run: E2Run, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Gate G1 — the in-run positive control: D_full >= 0.5 pts on the 0-3 judge
    self-attribution scale (SELF − OTHER over steered, gated no_ban records) with
    exact p <= 0.05 from the no_ban keystone's valence-arrangement enumeration
    (the prereg's within-arm b3 test — the raw point difference itself is
    valence-symmetric, hence untestable by valence permutation)."""
    hi_vals, lo_vals = [], []
    for name in run.qualia_valenced():
        for s in run.nonzero_strengths():
            hi_vals += run.values(name, s, "self", ARM_NO_BAN, "self_attribution")
            lo_vals += run.values(name, s, "other", ARM_NO_BAN, "self_attribution")
    d = float(np.mean(hi_vals)) - float(np.mean(lo_vals)) if hi_vals and lo_vals else None
    ks = keystone_arm(run, ARM_NO_BAN, VERDICT_MEASURE, n_perm=n_perm, seed=seed)
    p = ks["p_perm"] if ks else None
    passed = (d is not None and d >= G1_MIN_D_PTS and p is not None and p <= G1_MAX_P)
    return {"D_full_pts": round(d, 4) if d is not None else None,
            "scale": "judge self_attribution (0-3), SELF - OTHER over steered no_ban records",
            "min_pts": G1_MIN_D_PTS, "p_exact": p, "max_p": G1_MAX_P,
            "p_source": f"no_ban keystone b3 ({VERDICT_MEASURE}), "
                        "exact valence-arrangement enumeration",
            "passed": bool(passed),
            "on_fail": "the whole experiment is void (prereg G1)"}


def g3_residual_density(run: E2Run) -> dict:
    """Gate G3 — residual affect-lexicon density: mean ``banned_word_hits`` per
    record by arm (all records, intent-to-treat: the measure is objective, not
    judge-dependent). The escalation trigger fires iff the ban_emotion_lexicon
    density is NOT reduced >= 70% vs no_ban; the trigger compares UNROUNDED rates."""
    def _rate(recs):
        vals = [r["banned_word_hits"] for r in recs if r.get("banned_word_hits") is not None]
        return float(np.mean(vals)) if vals else None

    rates = {}
    for arm in run.arms:
        recs = run.records(arm)
        rates[arm] = {
            "n_records": len(recs),
            "hits_per_record": _rate(recs),
            "hits_per_record_steered": _rate([r for r in recs
                                              if cq._snap(r["strength"]) != 0.0]),
            "hits_per_record_baseline": _rate([r for r in recs
                                               if cq._snap(r["strength"]) == 0.0]),
        }

    def _reduction(which):
        a = (rates.get(ARM_NO_BAN) or {}).get(which)
        b = (rates.get(ARM_BAN_EMOTION) or {}).get(which)
        return 1.0 - b / a if a and b is not None else None

    red = _reduction("hits_per_record")
    passed = red is not None and red >= G3_MIN_REDUCTION
    return {"by_arm": rates,
            "reduction_vs_no_ban": red,
            "reduction_vs_no_ban_steered": _reduction("hits_per_record_steered"),
            "min_reduction": G3_MIN_REDUCTION,
            "passed": bool(passed),
            "escalation_trigger": None if red is None else bool(red < G3_MIN_REDUCTION),
            "on_trigger": "escalate the ban list (--extra-ban-words, embedding-NN synonyms) "
                          "and re-run; if escalation is exhausted, demote the ban arm to "
                          "dose-response (prereg G3)"}


def coherence_by_arm(run: E2Run) -> dict:
    """Per-arm coherence and gate exclusion rates (prereg: reported per arm; the
    intent-to-treat companion analysis is ``keystone_by_arm_itt``)."""
    out = {}
    for arm in run.arms:
        recs = run.records(arm)
        cohs = []
        for r in recs:
            j = r.get("judge") or {}
            if j.get("parse_ok") and isinstance(j.get("coherence"), int):
                cohs.append(j["coherence"])
        n_excl = sum(1 for r in recs if not run._passes(r))
        out[arm] = {"n_records": len(recs), "n_judged": len(cohs),
                    "mean_coherence": round(float(np.mean(cohs)), 3) if cohs else None,
                    "excluded_by_gate": n_excl,
                    "exclusion_rate": round(n_excl / len(recs), 4) if recs else None}
    return out


# --------------------------------------------------------------------------- #
# INDETERMINATE fallback — dose-response on achieved dose (exploratory)
# --------------------------------------------------------------------------- #
def dose_response_fallback(run: E2Run, measure: str = VERDICT_MEASURE,
                           frames: tuple = ("self", "other")) -> dict | None:
    """Prereg INDETERMINATE branch: regress the signed judge effect
    D = sign_e*(mean SELF − mean OTHER) per (arm, emotion, strength) cell on
    ACHIEVED dose — signed read-back valence + measured affect-word density —
    with per-arm intercepts (no global intercept). The arm intercept at zero
    lexical dose is the anti-boring quantity. Exploratory, never gates."""
    emos, signs = _panel(run)
    hi, lo = frames
    arm_list = list(run.arms)
    rows, y = [], []
    for a in arm_list:
        for j, name in enumerate(emos):
            for s in run.nonzero_strengths():
                yh = run.values(name, s, hi, a, measure)
                yl = run.values(name, s, lo, a, measure)
                if not yh or not yl:
                    continue
                rb = run.values(name, s, hi, a, "proj_valence_axis", gated=False)
                dens = [r["banned_word_hits"]
                        for r in run._rec.get((name, cq._snap(s), hi, a), [])
                        if r.get("banned_word_hits") is not None]
                rows.append([signs[j] * float(np.mean(rb)) if rb else 0.0,
                             float(np.mean(dens)) if dens else 0.0]
                            + [1.0 if a == b else 0.0 for b in arm_list])
                y.append(signs[j] * (float(np.mean(yh)) - float(np.mean(yl))))
    if len(y) < len(arm_list) + 3:
        return None
    beta, *_ = np.linalg.lstsq(np.asarray(rows, float), np.asarray(y, float), rcond=None)
    return {"measure": measure,
            "note": "exploratory prereg INDETERMINATE fallback: per-cell signed judge "
                    "effect ~ achieved dose (signed read-back valence + affect-word "
                    "density) with per-arm intercepts; the arm intercept at zero "
                    "lexical dose is the anti-boring quantity",
            "coef_readback": round(float(beta[0]), 5),
            "coef_density": round(float(beta[1]), 5),
            "arm_intercepts_zero_dose": {a: round(float(beta[2 + i]), 5)
                                         for i, a in enumerate(arm_list)},
            "n_cells": len(y)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_run(run: E2Run, n_perm: int = N_PERM, seed: int = 0) -> dict:
    g3 = g3_residual_density(run)
    g1 = g1_positive_control(run, n_perm=n_perm, seed=seed)
    ks = {a: {m: keystone_arm(run, a, m, n_perm=n_perm, seed=seed)
              for m in PRIMARY_MEASURES} for a in run.arms}
    ks_itt = {a: {m: keystone_arm(run, a, m, n_perm=n_perm, seed=seed, gated=False)
                  for m in PRIMARY_MEASURES} for a in run.arms}
    kill = {m: kill_test(run, m, g3_passed=g3["passed"], n_perm=n_perm, seed=seed)
            for m in PRIMARY_MEASURES}
    if not g1["passed"]:                        # G1 fail voids every kill verdict
        for kt in kill.values():
            if kt is not None:
                kt["verdict"] = "void"
    primary = kill.get(VERDICT_MEASURE) or {}
    return {
        "model_id": run.model_id, "suite_key": run.suite_key, "size": run.size(),
        "layer": run.layer, "num_layers": run.num_layers,
        "n_records": sum(len(v) for v in run._rec.values()),
        "arms": list(run.arms),
        "gates": {"G1": g1, "G3": g3},
        "keystone_by_arm": ks,
        "keystone_by_arm_itt": ks_itt,
        "kill_test": kill,
        "coherence_by_arm": coherence_by_arm(run),
        "dose_response_fallback": dose_response_fallback(run),
        "verdict": primary.get("verdict", "indeterminate"),
        "verdict_measure": VERDICT_MEASURE,
    }


def compute_all(runs) -> dict:
    per_model = {}
    for r in runs:
        per_model[r.suite_key or r.model_id or "?"] = compute_run(r)
    return {
        "unit_note": "per-arm keystone b3 = SELF valence-slope − OTHER valence-slope inside "
                     "one decode arm; R_X = b3_arm / b3_no_ban with a 90% by-emotion "
                     "jackknife CI; exact perm = C(10,5)=252 valence-arrangement "
                     "enumeration (two-sided floor 2/252).",
        "prereg": {"doc": "docs/prereg/E2.md",
                   "G1_min_pts": G1_MIN_D_PTS, "G1_max_p": G1_MAX_P,
                   "G3_min_reduction": G3_MIN_REDUCTION,
                   "R_survives_ci_lo": R_SURVIVES_CI_LO, "R_killed_ci_hi": R_KILLED_CI_HI,
                   "R_ban_survival_ci_lo": R_BAN_SURVIVAL_CI_LO,
                   "R_boring_wins_ci_hi": R_BORING_WINS_CI_HI, "ci_level": CI_LEVEL},
        "gate": GATE, "n_perm": N_PERM,
        "models": list(per_model), "per_model": per_model,
    }


def discover_paths(suite_root=None) -> list[str]:
    """One path per suite model: the scored file when present, else the raw one
    (raw runs still yield G3 + the judge-free proj_valence_axis keystones)."""
    root = suite_root or os.environ.get("EMOTION_PROBES_SUITE_ROOT") or "suite"
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "analysis"))):
        scored = os.path.join(d, "lexical_knockout_scored.json")
        raw = os.path.join(d, "lexical_knockout.json")
        if os.path.exists(scored):
            out.append(scored)
        elif os.path.exists(raw):
            out.append(raw)
    return out


def _fmt(x, spec="+.3f"):
    return "n/a" if x is None else format(x, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*",
                    help="explicit lexical_knockout[_scored].json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--out", default="scripts/e2_stats_generated.json")
    args = ap.parse_args()
    paths = args.paths or discover_paths(args.suite_root)
    if not paths:
        raise SystemExit("no lexical_knockout[_scored].json found (pass paths or --suite-root).")
    runs = [load_e2_run(p) for p in paths]
    stats = compute_all(runs)
    Path(args.out).write_text(json.dumps(stats, indent=2, allow_nan=True))
    print(f"wrote {args.out}  ({len(runs)} model(s))")
    for key in stats["models"]:
        pm = stats["per_model"][key]
        g1, g3 = pm["gates"]["G1"], pm["gates"]["G3"]
        print(f"\n=== {key}  (layer {pm['layer']}/{pm['num_layers']}, "
              f"n={pm['n_records']}) ===")
        print(f"  G1 positive control: D_full={_fmt(g1['D_full_pts'])} pts "
              f"p={_fmt(g1['p_exact'], '.4f')} -> "
              f"{'PASS' if g1['passed'] else 'FAIL (experiment VOID)'}")
        print(f"  G3 residual density: reduction={_fmt(g3['reduction_vs_no_ban'], '.3f')} "
              f"(min {G3_MIN_REDUCTION}) -> "
              f"{'PASS' if g3['passed'] else 'ESCALATION TRIGGER'}")
        for m in PRIMARY_MEASURES:
            kt = pm["kill_test"].get(m)
            rb = (kt or {}).get("R_ban")
            if not rb:
                continue
            print(f"  kill[{m:18}] R_ban={_fmt(rb['R'])} "
                  f"CI90=[{_fmt(rb['ci_lo'])}, {_fmt(rb['ci_hi'])}]  "
                  f"survival>={R_BAN_SURVIVAL_CI_LO}:{kt['ban_survival_corroboration']}  "
                  f"verdict={kt['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
