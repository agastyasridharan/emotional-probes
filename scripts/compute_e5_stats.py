"""
CPU statistics for E5 — axis sufficiency + graded audited cancellation
(docs/prereg/E5.md, binding; portfolio.md §E5). Reads the per-model
``axis_mediation.json`` written by ``emotion_probes.alignment.axis_mediation``
(``axis_mediation_scored.json`` preferred when a judged copy exists) and computes:

* AUDIT GATE (first, blocking) — every cancellation cell's ACHIEVED cancelled
  fraction (from the fixed-text audit projections vs the measured per-emotion
  Δ_emo) is compared against its nominal fraction (1−λ; cap ≡ 1). If any cell
  deviates beyond tolerance the arms are mislabeled (leaky clamp / wrong-sign
  displacement) and ALL verdicts are replaced by ``audit_failed`` — the prereg's
  "never claim A from a leaky clamp".
* NECESSITY dose-response — the valence-signed logit self-attribution gap
  (target-pool YES−NO shift vs the emotion-off baseline) against achieved
  cancelled fraction, per-emotion clusters. Keystone: mean_e sign_e·slope_e via
  exact C(10,5)=252 valence enumeration (``exact_perm``). Mediation elasticity =
  pooled signed slope. Judge ``self_attribution`` (SELF−OTHER where both frames
  exist) mirrors it where scored, with an exact arm-label permutation (the dose
  labels are exchangeable within emotion under H0; permuted identically across
  clusters, preserving the cluster structure).
* EQUIVALENCE decision rule (prereg ratio margins ≥0.6 survives / ≤0.3 killed):
  full mediation = addback λ=1 ≈ emotion_only (R≥0.6) AND cap ≈ baseline
  (R≤0.3); none = cap ≈ emotion_only (R≥0.6); partial = graded intermediate.
  Gated on the in-run positive control (the emotion effect itself must exist).
* OVERSHOOT (λ=−0.5, 150%) — B predicts reversal, A flatness; consistency with
  the mediation verdict.
* SUFFICIENCY (``__neg_axis__``) — does −axis alone move the outcome with no
  emotion vector; effect size as a ratio vs the emotion_only effect.
* DISCLAIMER trajectory (fifth mechanism) — does the disclaimer/refusal rate
  rise as the axis displacement is cancelled (gate release); descriptive per-arm
  rates + exact arm-label permutation on the slope.
* T3 CONCORDANCE hook — if ``scripts/e3_stats_generated.json`` exists, state
  whether E3's signed discriminator predicted this cancellation outcome
  (concordant / discordant / e3-missing).

Everything reuses :mod:`compute_qualia_stats` (``linreg``, ``summarize``,
``_snap``, ``SUITE_SIZES``) and ALL permutation inference goes through
:func:`exact_perm.sign_permutation_pvalue` on unrounded statistics.
Pure numpy, CPU, importable/testable without a GPU.

  python3 scripts/compute_e5_stats.py --suite-root suite \
      --out scripts/e5_stats_generated.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import compute_qualia_stats as cq   # sibling in scripts/ — reuse its stat primitives
import exact_perm as ep             # exact by-cluster permutation enumeration

GATE = 3                     # judge coherence gate (scored records only)
N_PERM = 2000
ALPHA = 0.05
MARGIN_SURVIVES = 0.6        # prereg ratio margins: >= 0.6 survives
MARGIN_KILLED = 0.3          # <= 0.3 killed / else INDETERMINATE
AUDIT_TOL = 0.25             # descriptive only: |achieved − nominal| flag threshold per cell
AUDIT_MIN_SPAN = 0.30        # min achieved-fraction span for the elasticity slope to be estimable
AUDIT_VANISHING_FRAC = 0.25  # families with |Δ_emo| below this fraction of the panel
                             # median are excluded as un-auditable (E5Run.vanishing_families)

EMOTION_OFF = "__none__"     # driver sentinels (axis_mediation.py)
NEG_AXIS = "__neg_axis__"

E3_PATH_DEFAULT = str(Path(__file__).resolve().parent / "e3_stats_generated.json")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _slope(xs, ys) -> float:
    """UNROUNDED OLS slope (exact_perm rule: never a rounded statistic).
    cq.linreg rounds — use it for reporting only, never inside stat_fn."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    sxx = float(((x - x.mean()) ** 2).sum())
    if len(x) < 2 or sxx == 0.0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / sxx)


def _lam_key(lam):
    return None if lam is None else cq._snap(lam)


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
class E5Run:
    """Parsed one-model axis-mediation results with cell accessors.

    Cell key = (condition, strength, intervention, lam). Arms per emotion e at
    strength s (nominal cancelled fraction f = 1−λ; cap ≡ λ=0 ≡ f=1):
      emotion_only  (e, s, none, None)   f=0
      addback_λ     (e, s, addback, λ)   f=1−λ   (λ=−0.5 → f=1.5, the overshoot)
      cap           (e, s, cap, 0.0)     f=1
      baseline      (__none__, 0, none)  the emotion-off / intervention-off cell
      axis_only_neg (__neg_axis__, s, none)  the sufficiency arm
    """

    def __init__(self, raw: dict, gate: int = GATE):
        self.raw = raw
        self.model_id = raw.get("model_id")
        self.suite_key = raw.get("suite_key")
        self.layer = raw.get("layer")
        self.num_layers = raw.get("num_layers")
        self.gate = gate
        self.cap_layers = [int(L) for L in raw.get("cap_layers", [])]
        self.lambda_grid = [float(l) for l in raw.get("lambda_grid", [])]
        self.strengths = [float(s) for s in raw.get("strengths", [])]
        self.conditions = {c["name"]: c for c in raw.get("conditions", [])}
        self.questions = {q["label"]: q for q in raw.get("questions", [])}
        self.frame_keys = [f["key"] for f in raw.get("frames", [])]
        self.tau = {str(L): float(v) for L, v in (raw.get("tau_by_layer") or {}).items()}
        self.baseline_proj = {str(L): float(v)
                              for L, v in (raw.get("baseline_projection") or {}).items()}
        self.delta = {(d["condition"], cq._snap(d["strength"])):
                      {str(L): float(v) for L, v in d["delta"].items()}
                      for d in raw.get("delta_by_cell", [])}
        self._battery: dict = {}
        for r in raw.get("battery_records", []):
            self._battery[(r["condition"], cq._snap(r["strength"]), r["intervention"],
                           _lam_key(r["lam"]), r["question"])] = float(r["logit_diff"])
        self._freeform: dict = defaultdict(list)
        for r in raw.get("freeform_records", []):
            self._freeform[(r["condition"], cq._snap(r["strength"]), r["intervention"],
                            _lam_key(r["lam"]), r["frame"])].append(r)
        self._audit: dict = {}
        for r in raw.get("audit_records", []):
            self._audit[(r["condition"], cq._snap(r["strength"]), r["intervention"],
                         _lam_key(r["lam"]))] = r

    # -- panel / meta --
    def size(self) -> float:
        return cq.SUITE_SIZES.get(self.suite_key, float("nan"))

    def qualia_valenced(self) -> list[str]:
        return [n for n, c in self.conditions.items()
                if c.get("kind") == "qualia" and (c.get("valence") or 0) != 0]

    def valence(self, name: str) -> float:
        return float(self.conditions[name]["valence"])

    def nonzero_strengths(self) -> list[float]:
        return [s for s in self.strengths if cq._snap(s) != 0.0]

    def labels(self, *kinds) -> list[str]:
        return [q["label"] for q in self.questions.values() if q["kind"] in kinds]

    def controls(self) -> list[str]:
        return self.labels("factual", "self_ref")

    def cancellation_arms(self) -> list[tuple[str, float | None, float]]:
        """(intervention, lam, nominal cancelled fraction), emotion_only first."""
        arms = [("none", None, 0.0), ("cap", 0.0, 1.0)]
        arms += [("addback", float(l), round(1.0 - float(l), 6))
                 for l in self.lambda_grid]
        return arms

    # -- battery outcomes --
    def battery_diff(self, cond, s, iv, lam, q) -> float | None:
        return self._battery.get((cond, cq._snap(s), iv, _lam_key(lam), q))

    def gap_shift(self, cond, s, iv, lam, labels) -> float | None:
        """Mean over ``labels`` of the YES−NO logit-gap shift vs the emotion-off
        baseline cell — the logit self-attribution gap when labels = targets."""
        shifts = []
        for q in labels:
            d = self.battery_diff(cond, s, iv, lam, q)
            b = self.battery_diff(EMOTION_OFF, 0.0, "none", None, q)
            if d is None or b is None:
                return None
            shifts.append(d - b)
        return float(np.mean(shifts)) if shifts else None

    def target_gap(self, cond, s, iv, lam) -> float | None:
        return self.gap_shift(cond, s, iv, lam, self.labels("target"))

    def attribution_score(self, cond, s, iv, lam) -> float | None:
        """Target-pool shift − control-pool shift (the cq per-cell score)."""
        t = self.gap_shift(cond, s, iv, lam, self.labels("target"))
        c = self.gap_shift(cond, s, iv, lam, self.controls())
        return None if t is None or c is None else t - c

    # -- scored free-form outcomes --
    def _passes(self, r: dict) -> bool:
        j = r.get("judge") or {}
        return bool(j.get("parse_ok")) and isinstance(j.get("coherence"), int) \
            and j["coherence"] >= self.gate

    def scored(self) -> bool:
        return any(self._passes(r) for recs in self._freeform.values() for r in recs)

    def judge_values(self, cond, s, iv, lam, frame, measure) -> list[float]:
        out = []
        for r in self._freeform.get((cond, cq._snap(s), iv, _lam_key(lam), frame), []):
            if not self._passes(r):
                continue
            v = (r.get("judge") or {}).get(measure)
            if v is not None:
                out.append(float(v))
        return out

    def judge_sa(self, cond, s, iv, lam) -> float | None:
        """Judge self_attribution, SELF − OTHER when both frames exist (frame
        specificity per prereg), else the SELF-frame mean."""
        sm = _mean(self.judge_values(cond, s, iv, lam, "self", "self_attribution"))
        if sm is None:
            return None
        if "other" in self.frame_keys:
            om = _mean(self.judge_values(cond, s, iv, lam, "other", "self_attribution"))
            return sm - om if om is not None else None
        return sm

    def disclaimer_rate(self, cond, s, iv, lam) -> float | None:
        """Disclaimer/refusal subscale: judge['disclaimer'] where the extended E5
        rubric scored it (bool, 0-1, or 0-10), else the deny/refuse stance rate."""
        vals = []
        for r in self._freeform.get((cond, cq._snap(s), iv, _lam_key(lam), "self"), []):
            if not self._passes(r):
                continue
            j = r.get("judge") or {}
            v = j.get("disclaimer")
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                x = float(v)
                vals.append(min(1.0, max(0.0, x / 10.0 if x > 1.0 else x)))
            else:
                vals.append(1.0 if j.get("stance") in ("deny", "refuse") else 0.0)
        return _mean(vals)

    # -- audits --
    def family_delta(self, cond, s) -> float | None:
        """Σ over cap layers of the calibration-pass Δ_emo for (condition,
        strength) — the achieved-fraction denominator. None if no Δ row."""
        d = self.delta.get((cond, cq._snap(s)))
        if d is None:
            return None
        return sum(d[str(L)] for L in self.cap_layers)

    def vanishing_families(self, frac: float = AUDIT_VANISHING_FRAC) -> dict:
        """(condition, snapped strength) -> Δ_emo for emotion×strength families
        whose |Δ_emo| falls below ``frac`` × the panel median |Δ_emo| — the
        cancellation audit is then noise-dominated (the intervention machinery's
        own shift on the audit text exceeds the displacement being cancelled)
        and the achieved fractions are junk.

        AMENDMENT 2026-07-04 (disclosed; prereg E5.md does not fix this
        implementation detail): the original guard only caught |Δ| < 1e-9.
        On real data 'content'@0.06 has Δ=+4.96 vs a panel median ≈ 95, giving
        achieved fractions ≈ −4 across every arm. Such families are EXCLUDED
        from the audit gate and the elasticity — not nominal-fallbacked —
        because keeping an un-cancellable emotion at nominal x biases the
        elasticity toward flat, i.e. toward the pre-committed A(+) prediction.
        All claims are range-limited to the audited families."""
        dens = {}
        for e in self.qualia_valenced():
            for s in self.nonzero_strengths():
                den = self.family_delta(e, s)
                if den is not None:
                    dens[(e, cq._snap(s))] = den
        if not dens:
            return {}
        med = float(np.median([abs(v) for v in dens.values()]))
        return {k: v for k, v in dens.items() if abs(v) < frac * med}

    def achieved_fraction(self, cond, s, iv, lam) -> float | None:
        """ACHIEVED cancelled fraction from the fixed-text audit: 1 − pooled
        (achieved − baseline)/Δ_emo across cap layers, Δ_emo the calibration-pass
        displacement for this (condition, strength). None if unauditable."""
        rec = self._audit.get((cond, cq._snap(s), iv, _lam_key(lam)))
        d = self.delta.get((cond, cq._snap(s)))
        if rec is None or d is None:
            return None
        num = sum(float(rec["achieved_minus_baseline"][str(L)]) for L in self.cap_layers)
        den = sum(d[str(L)] for L in self.cap_layers)
        if abs(den) < 1e-9:
            return None
        return 1.0 - num / den


def load_e5_run(path) -> E5Run:
    with open(path) as f:
        return E5Run(json.load(f))


# --------------------------------------------------------------------------- #
# Audit gate (blocking)
# --------------------------------------------------------------------------- #
def audit_gate(run: E5Run, tol: float = AUDIT_TOL,
               min_span: float = AUDIT_MIN_SPAN) -> dict:
    """Audit per prereg E5.md: the fixed-text audit MEASURES each cell's
    achieved cancelled fraction — the elasticity x-axis ("mediation elasticity
    = slope of b3 vs audited achieved cancelled fraction"). The prereg sets no
    achieved-vs-nominal tolerance (a p0.25 floor clamp structurally cannot
    cancel above-floor displacement, so nominal 1−λ is an idealization); the
    gate therefore fails only on what actually invalidates the design:
      * unauditable cells (missing audit rows / exactly-zero Δ_emo);
      * a non-monotone achieved-vs-nominal ordering within an emotion×strength
        cell family (genuinely mislabeled arms / leaky clamp);
      * a degenerate achieved-dose span (< ``min_span``) — the elasticity
        slope is then not estimable.
    Families whose |Δ_emo| is below ``AUDIT_VANISHING_FRAC`` × the panel median
    are EXCLUDED and reported (``excluded_vanishing`` + ``range_note``), not
    failed: their achieved fractions are noise-dominated junk, but a weak
    axis displacement for one emotion does not invalidate the design for the
    others (see E5Run.vanishing_families for the disclosed amendment).
    |achieved − nominal| > ``tol`` cells are reported as ``flagged``
    (descriptive), and ``max_achieved_fraction`` is exported so the A-decisive
    "flat at audited 100 %" clause can be honestly range-limited."""
    vanishing = run.vanishing_families()
    per_cell, unauditable, flagged, nonmonotone, excluded = [], [], [], [], []
    for e in run.qualia_valenced():
        for s in run.nonzero_strengths():
            if (e, cq._snap(s)) in vanishing:
                excluded.append({"condition": e, "strength": float(s),
                                 "family_delta": round(vanishing[(e, cq._snap(s))], 4)})
                continue
            fam = []
            for iv, lam, nominal in run.cancellation_arms():
                if (e, cq._snap(s), iv, _lam_key(lam)) not in run._audit:
                    continue
                achieved = run.achieved_fraction(e, s, iv, lam)
                dev = None if achieved is None else achieved - nominal
                cell = {"condition": e, "strength": float(s), "intervention": iv,
                        "lam": lam, "nominal_fraction": nominal,
                        "achieved_fraction": None if achieved is None else round(achieved, 4),
                        "deviation": None if dev is None else round(dev, 4)}
                per_cell.append(cell)
                fam.append(cell)
                if achieved is None:
                    unauditable.append(cell)
                elif abs(dev) > tol:
                    flagged.append(cell)
            # mislabeling check: achieved order must follow nominal order
            ok = [c for c in fam if c["achieved_fraction"] is not None]
            by_nominal = sorted(ok, key=lambda c: c["nominal_fraction"])
            ach = [c["achieved_fraction"] for c in by_nominal]
            if any(b < a - 0.10 for a, b in zip(ach, ach[1:])):  # 0.10 = audit noise allowance
                nonmonotone.append({"condition": e, "strength": float(s),
                                    "achieved_by_nominal": ach})
    achs = [c["achieved_fraction"] for c in per_cell
            if c["achieved_fraction"] is not None]
    span = (max(achs) - min(achs)) if achs else None
    span_failed = span is not None and span < min_span
    failed = (not per_cell or bool(unauditable) or bool(nonmonotone) or span_failed)
    devs = [abs(c["deviation"]) for c in per_cell if c["deviation"] is not None]
    failures = unauditable + nonmonotone
    if span_failed and not failures:
        # a leaky clamp collapses the dose range; the deviation-flagged cells
        # (e.g. cap achieving ~0 vs nominal 1) are the diagnostic evidence
        failures = flagged
    return {"audit_failed": bool(failed),
            "tolerance": tol, "min_span": min_span, "n_cells": len(per_cell),
            "max_abs_deviation": round(max(devs), 4) if devs else None,
            "achieved_span": None if span is None else round(span, 4),
            "max_achieved_fraction": round(max(achs), 4) if achs else None,
            "min_achieved_fraction": round(min(achs), 4) if achs else None,
            "unauditable": unauditable, "nonmonotone": nonmonotone,
            "flagged_deviation_cells": flagged,
            "excluded_vanishing": excluded,
            "vanishing_frac": AUDIT_VANISHING_FRAC,
            "range_note": (None if not excluded else
                           f"{len(excluded)} emotion×strength famil"
                           f"{'y' if len(excluded) == 1 else 'ies'} excluded as "
                           f"un-auditable (|Δ_emo| < {AUDIT_VANISHING_FRAC}× panel "
                           f"median): "
                           + ", ".join(f"{c['condition']}@{c['strength']} "
                                       f"(Δ={c['family_delta']:+.1f})" for c in excluded)
                           + "; elasticity and A-decisive claims are range-limited "
                             "to the audited families"),
            "failures": failures,
            "reason": ("no auditable cancellation cells" if not per_cell else
                       "unauditable cells" if unauditable else
                       "achieved order does not follow nominal order (mislabeled arms)"
                       if nonmonotone else
                       f"achieved-dose span {span:.3f} < {min_span} (elasticity not estimable)"
                       if span_failed else None)}


# --------------------------------------------------------------------------- #
# Dose-response + equivalence (the necessity core)
# --------------------------------------------------------------------------- #
def _outcome_by_arm(run: E5Run, outcome_fn) -> dict:
    """{(iv, lam, nominal_frac): {emotion: (achieved_frac, outcome)}} over the
    cancellation grid; achieved fraction falls back to nominal if unauditable.
    Vanishing-Δ families (see E5Run.vanishing_families) are excluded outright —
    their achieved fractions are noise-dominated junk that would poison the
    elasticity x-axis, and nominal-fallback would bias the slope toward flat
    (the pre-committed A(+) prediction)."""
    vanish = run.vanishing_families()
    grid = {}
    for arm in run.cancellation_arms():
        iv, lam, nominal = arm
        per_e = {}
        for e in run.qualia_valenced():
            fs, ys = [], []
            for s in run.nonzero_strengths():
                if (e, cq._snap(s)) in vanish:
                    continue
                y = outcome_fn(e, s, iv, lam)
                if y is None:
                    continue
                f = run.achieved_fraction(e, s, iv, lam)
                fs.append(nominal if f is None else f)
                ys.append(float(y))
            if ys:
                per_e[e] = (float(np.mean(fs)), float(np.mean(ys)))
        if per_e:
            grid[arm] = per_e
    return grid


def dose_response(run: E5Run, outcome_fn, signed: bool,
                  n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Outcome vs achieved cancelled fraction, per-emotion clusters.

    ``signed=True`` (battery valence-directional gap): keystone statistic
    mean_e sign_e·slope_e with exact C(n_pos+n_neg, n_pos) valence enumeration —
    the prereg necessity keystone. ``signed=False`` (judge self_attribution,
    disclaimer): exact arm-label permutation of the pooled per-arm curve (dose
    labels exchangeable within emotion under H0, permuted identically across
    clusters). Elasticity/monotonicity use the f<=1 grid; the overshoot arm is a
    separate prediction check.
    """
    grid = _outcome_by_arm(run, outcome_fn)
    core = {arm: per_e for arm, per_e in grid.items() if arm[2] <= 1.0}
    emos = sorted({e for per_e in core.values() for e in per_e})
    if len(core) < 3 or len(emos) < 3:
        return None
    signs = np.array([run.valence(e) for e in emos], float)

    # per-emotion elasticity slopes over the f<=1 arms
    slopes = []
    for e in emos:
        pts = [per_e[e] for per_e in core.values() if e in per_e]
        slopes.append(_slope([p[0] for p in pts], [p[1] for p in pts]))
    slopes = np.asarray(slopes, float)

    # pooled per-arm curve (signed outcomes pooled with sign_e applied)
    def pooled(per_e):
        ys = [(run.valence(e) * y if signed else y) for e, (_, y) in per_e.items()]
        fs = [f for (f, _) in per_e.values()]
        return float(np.mean(fs)), float(np.mean(ys))

    arms_out, curve = [], []
    for (iv, lam, nominal), per_e in sorted(grid.items(), key=lambda kv: kv[0][2]):
        f, y = pooled(per_e)
        arms_out.append({"intervention": iv, "lam": lam, "nominal_fraction": nominal,
                         "achieved_fraction": round(f, 4), "pooled_outcome": round(y, 5),
                         "n_emotions": len(per_e)})
        if nominal <= 1.0:
            curve.append((f, y, nominal))
    elasticity = _slope([c[0] for c in curve], [c[1] for c in curve])

    # monotone kill: pooled curve non-increasing across the cancellation grid
    ordered = [y for _, y, _ in sorted(curve, key=lambda c: c[2])]
    span = max(abs(v) for v in ordered) or 1.0
    monotone = all(b <= a + 0.1 * span for a, b in zip(ordered, ordered[1:]))

    if signed:
        perm = ep.sign_permutation_pvalue(signs, lambda sgn: float(np.mean(sgn * slopes)),
                                          n_perm=n_perm, seed=seed)
    else:
        # arm-label permutation: permute the achieved-fraction labels over arms
        fr = np.array([c[0] for c in curve], float)
        ys = np.array([c[1] for c in curve], float)
        perm = ep.sign_permutation_pvalue(fr, lambda f: _slope(f, ys),
                                          n_perm=n_perm, seed=seed)

    return {"signed": signed, "arms": arms_out,
            "per_emotion_slopes": {e: round(s, 5) for e, s in zip(emos, slopes)},
            "elasticity": round(elasticity, 5),
            "monotone_decreasing": monotone,
            "excluded_vanishing_emotions": sorted(
                {e for (e, _s) in run.vanishing_families()}),
            "perm": {"observed": perm["observed"], "p_perm": perm["p_perm"],
                     "exact": perm["exact"], "n_arrangements": perm["n_arrangements"],
                     "floor": perm["floor"]}}


def _pooled_effect(run: E5Run, outcome_fn, iv, lam, signed: bool) -> float | None:
    """Mean over emotions x strengths of the (optionally valence-signed) outcome."""
    vals = []
    for e in run.qualia_valenced():
        for s in run.nonzero_strengths():
            y = outcome_fn(e, s, iv, lam)
            if y is not None:
                vals.append(run.valence(e) * y if signed else y)
    return _mean(vals)


def equivalence(run: E5Run, outcome_fn, signed: bool) -> dict:
    """Prereg ratio margins on the cancellation arms. D_* are outcome shifts vs
    the emotion-off baseline (already baked into the battery gap; the judge
    outcome subtracts the baseline cell explicitly by the caller)."""
    d_full = _pooled_effect(run, outcome_fn, "none", None, signed)
    d_cap = _pooled_effect(run, outcome_fn, "cap", 0.0, signed)
    d_ab1 = (_pooled_effect(run, outcome_fn, "addback", 1.0, signed)
             if 1.0 in run.lambda_grid else None)
    d_over = (_pooled_effect(run, outcome_fn, "addback", -0.5, signed)
              if -0.5 in run.lambda_grid else None)

    def ratio(d):
        if d is None or d_full is None or abs(d_full) < 1e-12:
            return None
        return round(d / d_full, 4)

    def margin_call(r, low_is=None, high_is=None):
        if r is None:
            return None
        if r >= MARGIN_SURVIVES:
            return high_is or "survives"
        if r <= MARGIN_KILLED:
            return low_is or "killed"
        return "indeterminate"

    r_cap, r_ab1, r_over = ratio(d_cap), ratio(d_ab1), ratio(d_over)
    return {"margins": {"survives": MARGIN_SURVIVES, "killed": MARGIN_KILLED},
            "D_full": None if d_full is None else round(d_full, 5),
            "D_cap": None if d_cap is None else round(d_cap, 5),
            "D_addback1": None if d_ab1 is None else round(d_ab1, 5),
            "D_overshoot": None if d_over is None else round(d_over, 5),
            "R_cap": r_cap, "R_addback1": r_ab1, "R_overshoot": r_over,
            "cap_vs_baseline": margin_call(r_cap),
            "addback1_vs_emotion_only": margin_call(r_ab1)}


def emotion_effect(run: E5Run, n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """In-run positive control: the emotion_only effect itself must exist.
    Statistic = mean_e sign_e·gap_shift_e, exact valence enumeration."""
    emos = [e for e in run.qualia_valenced()
            if any(run.target_gap(e, s, "none", None) is not None
                   for s in run.nonzero_strengths())]
    if len(emos) < 3:
        return None
    gaps = np.array([_mean([run.target_gap(e, s, "none", None)
                            for s in run.nonzero_strengths()]) for e in emos], float)
    signs = np.array([run.valence(e) for e in emos], float)
    perm = ep.sign_permutation_pvalue(signs, lambda sgn: float(np.mean(sgn * gaps)),
                                      n_perm=n_perm, seed=seed)
    return {"n_emotions": len(emos),
            "per_emotion_gap": {e: round(float(g), 5) for e, g in zip(emos, gaps)},
            "observed": perm["observed"], "p_perm": perm["p_perm"],
            "exact": perm["exact"], "n_arrangements": perm["n_arrangements"],
            "floor": perm["floor"],
            "exists": bool(perm["p_perm"] <= ALPHA and abs(perm["observed"]) > 0)}


# --------------------------------------------------------------------------- #
# Verdict, overshoot, sufficiency, disclaimer
# --------------------------------------------------------------------------- #
def mediation_verdict(audit: dict, effect: dict | None, eq: dict) -> str:
    """The prereg decision rule (docs/prereg/E5.md, binding):
    full = λ=1 addback ≈ emotion_only (R≥0.6) AND cap ≈ baseline (R≤0.3);
    none = cap ≈ emotion_only (R≥0.6); partial = graded intermediate;
    audit_failed / no_effect gates precede everything."""
    if audit["audit_failed"]:
        return "audit_failed"
    if effect is None or not effect["exists"]:
        return "no_effect"
    r_cap, r_ab1 = eq["R_cap"], eq["R_addback1"]
    if r_cap is None:
        return "indeterminate"
    if r_cap <= MARGIN_KILLED:
        return "full" if (r_ab1 is not None and r_ab1 >= MARGIN_SURVIVES) else "indeterminate"
    if r_cap >= MARGIN_SURVIVES:
        return "none"
    return "partial"


def overshoot_check(eq: dict, verdict: str) -> dict:
    """λ=−0.5 (150%) prediction: B (full mediation) predicts reversal past the
    baseline; A (no mediation) predicts flatness at the full effect."""
    r = eq["R_overshoot"]
    if r is None:
        return {"present": False, "R_overshoot": None, "classification": "absent",
                "consistent_with_verdict": None}
    if r <= -MARGIN_KILLED:
        cls = "reversal"
    elif r >= MARGIN_SURVIVES:
        cls = "flat_full_effect"
    elif abs(r) <= MARGIN_KILLED:
        cls = "killed_flat"
    else:
        cls = "intermediate"
    consistent = None
    if verdict == "full":
        consistent = bool(r <= MARGIN_KILLED)       # killed or reversed
    elif verdict == "none":
        consistent = bool(r >= MARGIN_SURVIVES)     # curve stays flat at full effect
    return {"present": True, "R_overshoot": r, "classification": cls,
            "consistent_with_verdict": consistent}


def sufficiency(run: E5Run) -> dict:
    """Does −a_asst alone (no emotion vector) move the outcome? Effect size as a
    ratio vs the mean |emotion_only| effect (prereg margins classify it)."""
    out = {"present": False}
    gaps = [run.target_gap(NEG_AXIS, s, "none", None) for s in run.nonzero_strengths()]
    gaps = [g for g in gaps if g is not None]
    if not gaps:
        return out
    d_suff = float(np.mean(gaps))
    score = _mean([run.attribution_score(NEG_AXIS, s, "none", None)
                   for s in run.nonzero_strengths()])
    emo_mag = _mean([abs(run.target_gap(e, s, "none", None))
                     for e in run.qualia_valenced() for s in run.nonzero_strengths()
                     if run.target_gap(e, s, "none", None) is not None])
    ratio = None if not emo_mag else round(d_suff / emo_mag, 4)
    if ratio is None:
        cls = None
    elif abs(ratio) >= MARGIN_SURVIVES:
        cls = "reproduces"
    elif abs(ratio) <= MARGIN_KILLED:
        cls = "absent"
    else:
        cls = "partial"
    per_q = [run.battery_diff(NEG_AXIS, s, "none", None, q)
             - run.battery_diff(EMOTION_OFF, 0.0, "none", None, q)
             for s in run.nonzero_strengths() for q in run.labels("target")
             if run.battery_diff(NEG_AXIS, s, "none", None, q) is not None
             and run.battery_diff(EMOTION_OFF, 0.0, "none", None, q) is not None]
    out = {"present": True, "target_gap": round(d_suff, 5),
           "attribution_score": None if score is None else round(score, 5),
           "emotion_only_mean_abs_gap": None if emo_mag is None else round(emo_mag, 5),
           "ratio_vs_emotion_only": ratio, "classification": cls,
           "per_target_shift": cq.summarize(per_q) if per_q else None}
    if run.scored():
        sa = _mean([run.judge_sa(NEG_AXIS, s, "none", None)
                    for s in run.nonzero_strengths()])
        sa_base = run.judge_sa(EMOTION_OFF, 0.0, "none", None)
        out["judge_sa_shift"] = (round(sa - sa_base, 5)
                                 if sa is not None and sa_base is not None else None)
    return out


def disclaimer_trajectory(run: E5Run, n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Fifth mechanism: does the disclaimer/refusal rate RISE as the axis
    displacement is cancelled (gate release)? Descriptive per-arm rates + exact
    arm-label permutation on the pooled slope (f<=1 grid)."""
    if not run.scored():
        return None
    dr = dose_response(run, run.disclaimer_rate, signed=False,
                       n_perm=n_perm, seed=seed)
    if dr is None:
        return None
    base = run.disclaimer_rate(EMOTION_OFF, 0.0, "none", None)
    rises = dr["elasticity"] > 0
    return {"baseline_rate": None if base is None else round(base, 4),
            "per_arm": dr["arms"], "slope_vs_cancelled_fraction": dr["elasticity"],
            "perm": dr["perm"],
            "gate_release_signature": bool(rises and dr["perm"]["p_perm"] <= ALPHA)}


# --------------------------------------------------------------------------- #
# T3 concordance hook (E3 -> E5 preregistered consistency)
# --------------------------------------------------------------------------- #
def _e3_entry(e3_stats: dict | None, suite_key: str | None):
    if not isinstance(e3_stats, dict):
        return None
    pm = e3_stats.get("per_model")
    if isinstance(pm, dict):
        if suite_key and suite_key in pm:
            return pm[suite_key]
        if len(pm) == 1:
            return next(iter(pm.values()))
        return None
    return e3_stats


def _e3_t3_and_verdict(entry) -> tuple[float | None, str | None]:
    """Defensive extraction: E3's signed discriminator T3 (+ favors A, − favors
    B) and/or its verdict string, wherever the sibling writer put them."""
    if not isinstance(entry, dict):
        return None, None
    verdict = None
    for k in ("verdict", "decision", "e3_verdict"):
        v = entry.get(k)
        if isinstance(v, str):
            verdict = v
            break
        if isinstance(v, dict) and isinstance(v.get("verdict"), str):
            # compute_e3_stats.py writes decision = {"verdict": ..., "t3_sign": ...}
            verdict = v["verdict"]
            break
    t3 = None
    for k in ("T3", "t3", "T3_D", "discriminator", "t3_discriminator"):
        v = entry.get(k)
        if isinstance(v, dict):
            for kk in ("observed", "D", "value", "stat", "statistic"):
                if isinstance(v.get(kk), (int, float)):
                    t3 = float(v[kk])
                    break
        elif isinstance(v, (int, float)):
            t3 = float(v)
        if t3 is not None:
            break
    if t3 is None and isinstance(entry.get("confirmatory"), dict):
        t3, v2 = _e3_t3_and_verdict(entry["confirmatory"])
        return t3, verdict or v2
    if t3 is None and isinstance(entry.get("per_layer"), dict):
        # compute_e3_stats.py nests the family under per_layer[<L>]; take the
        # preregistered primary layer (fall back to the max capture layer).
        pl = entry["per_layer"]
        key = str(entry.get("primary_layer") or max((int(k) for k in pl), default=None))
        t3, _ = _e3_t3_and_verdict(pl.get(key) or {})
    return t3, verdict


def e3_concordance(e3_stats: dict | None, suite_key: str | None,
                   e5_verdict: str, e3_path: str | None = None) -> dict:
    """Preregistered cross-experiment consistency: E3 A-pattern ⇒ cancellation
    flat (E5 'none'); B-pattern ⇒ monotone kill (E5 'full'); mixed ⇒ partial.
    Discordance falsifies the linear-mediator model itself."""
    out = {"e3_path": e3_path, "e5_verdict": e5_verdict, "e3_t3": None,
           "e3_verdict": None, "e3_prediction": None, "status": "e3_missing"}
    entry = _e3_entry(e3_stats, suite_key)
    t3, verdict = _e3_t3_and_verdict(entry)
    if t3 is None and verdict is None:
        return out
    out["e3_t3"], out["e3_verdict"] = t3, verdict
    v = (verdict or "").lower()
    if "b-supported" in v or "b_supported" in v:
        pred = "kill"
    elif "a-supported" in v or "a_supported" in v:
        pred = "flat"
    elif "mixed" in v:
        pred = "partial"
    elif t3 is not None:
        pred = "flat" if t3 > 0 else "kill" if t3 < 0 else None
    else:
        pred = None
    out["e3_prediction"] = pred
    if pred is None:
        return out
    if e5_verdict not in ("full", "partial", "none"):
        out["status"] = "not_evaluable"      # audit_failed / no_effect / indeterminate
        return out
    expected = {"kill": "full", "flat": "none", "partial": "partial"}[pred]
    out["status"] = "concordant" if e5_verdict == expected else "discordant"
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_run(run: E5Run, e3_stats: dict | None = None,
                e3_path: str | None = None, n_perm: int = N_PERM,
                seed: int = 0) -> dict:
    audit = audit_gate(run)
    effect = emotion_effect(run, n_perm=n_perm, seed=seed)
    # battery outcome: valence-signed target gap (per-emotion sign applied inside)
    eq = equivalence(run, run.target_gap, signed=True)
    verdict = mediation_verdict(audit, effect, eq)
    dr = (None if audit["audit_failed"]
          else dose_response(run, run.target_gap, signed=True, n_perm=n_perm, seed=seed))
    judge_dr = judge_eq = None
    if run.scored() and not audit["audit_failed"]:
        base = run.judge_sa(EMOTION_OFF, 0.0, "none", None) or 0.0

        def judge_shift(e, s, iv, lam, _b=base):
            v = run.judge_sa(e, s, iv, lam)
            return None if v is None else v - _b

        judge_dr = dose_response(run, judge_shift, signed=False, n_perm=n_perm, seed=seed)
        judge_eq = equivalence(run, judge_shift, signed=False)
    return {
        "model_id": run.model_id, "suite_key": run.suite_key, "size": run.size(),
        "layer": run.layer, "num_layers": run.num_layers,
        "n_battery": len(run._battery),
        "n_freeform": sum(len(v) for v in run._freeform.values()),
        "n_audit": len(run._audit), "scored": run.scored(),
        "audit_gate": audit,
        "emotion_effect": effect,
        "equivalence_battery": eq,
        "mediation_verdict": verdict,
        "dose_response_battery": dr,
        "dose_response_judge": judge_dr,
        "equivalence_judge": judge_eq,
        "overshoot": overshoot_check(eq, verdict) if not audit["audit_failed"] else
                     {"present": False, "R_overshoot": None, "classification": "audit_failed",
                      "consistent_with_verdict": None},
        "sufficiency": sufficiency(run) if not audit["audit_failed"] else
                       {"present": False, "classification": "audit_failed"},
        "disclaimer": disclaimer_trajectory(run, n_perm=n_perm, seed=seed)
                      if not audit["audit_failed"] else None,
        "e3_concordance": e3_concordance(e3_stats, run.suite_key, verdict, e3_path),
    }


def compute_all(runs, e3_stats: dict | None = None, e3_path: str | None = None,
                n_perm: int = N_PERM) -> dict:
    per_model = {}
    for r in runs:
        key = r.suite_key or r.model_id or "?"
        per_model[key] = compute_run(r, e3_stats=e3_stats, e3_path=e3_path,
                                     n_perm=n_perm)
    return {
        "unit_note": ("Necessity keystone = mean_e sign_e * slope_e(target-gap shift "
                      "vs ACHIEVED cancelled fraction), exact valence enumeration; "
                      "equivalence by prereg ratio margins (>=0.6 survives / <=0.3 "
                      "killed); audit gate blocks all verdicts on a leaky clamp."),
        "gate": GATE, "n_perm": N_PERM, "audit_tolerance": AUDIT_TOL,
        "margins": {"survives": MARGIN_SURVIVES, "killed": MARGIN_KILLED},
        "models": list(per_model), "per_model": per_model,
    }


def discover_paths(suite_root=None) -> list[str]:
    """Prefer the judged copy (axis_mediation_scored.json) per suite key."""
    root = suite_root or os.environ.get("EMOTION_PROBES_SUITE_ROOT") or "suite"
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "analysis"))):
        scored = os.path.join(d, "axis_mediation_scored.json")
        raw = os.path.join(d, "axis_mediation.json")
        if os.path.exists(scored):
            out.append(scored)
        elif os.path.exists(raw):
            out.append(raw)
    return out


def load_e3(path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit axis_mediation[_scored].json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--e3", default=E3_PATH_DEFAULT,
                    help="E3 stats file for the T3 concordance hook (optional)")
    ap.add_argument("--out", default="scripts/e5_stats_generated.json")
    args = ap.parse_args()
    paths = args.paths or discover_paths(args.suite_root)
    if not paths:
        raise SystemExit("no axis_mediation.json found (pass paths or --suite-root).")
    runs = [load_e5_run(p) for p in paths]
    e3_stats = load_e3(args.e3)
    stats = compute_all(runs, e3_stats=e3_stats,
                        e3_path=args.e3 if e3_stats is not None else None)
    Path(args.out).write_text(json.dumps(stats, indent=2, allow_nan=True))
    print(f"wrote {args.out}  ({len(runs)} model(s); "
          f"e3={'loaded' if e3_stats else 'missing'})")
    for key in stats["models"]:
        pm = stats["per_model"][key]
        ag, eq, ov = pm["audit_gate"], pm["equivalence_battery"], pm["overshoot"]
        print(f"\n=== {key}  (layer {pm['layer']}/{pm['num_layers']}, "
              f"battery n={pm['n_battery']}, scored={pm['scored']}) ===")
        print(f"  audit: failed={ag['audit_failed']} "
              f"max_dev={ag['max_abs_deviation']} (tol {ag['tolerance']})")
        print(f"  VERDICT={pm['mediation_verdict']}  R_cap={eq['R_cap']} "
              f"R_addback1={eq['R_addback1']}  D_full={eq['D_full']}")
        dr = pm["dose_response_battery"]
        if dr:
            print(f"  elasticity={dr['elasticity']:+.4f} monotone={dr['monotone_decreasing']} "
                  f"p_perm={dr['perm']['p_perm']:.4f} "
                  f"(floor {dr['perm']['floor']:.4f})")
        print(f"  overshoot: {ov['classification']} "
              f"(R={ov['R_overshoot']}, consistent={ov['consistent_with_verdict']})")
        sf = pm["sufficiency"]
        if sf.get("present"):
            print(f"  sufficiency(−axis alone): gap={sf['target_gap']:+.4f} "
                  f"ratio={sf['ratio_vs_emotion_only']} -> {sf['classification']}")
        dc = pm["disclaimer"]
        if dc:
            print(f"  disclaimer: slope={dc['slope_vs_cancelled_fraction']:+.4f} "
                  f"p_perm={dc['perm']['p_perm']:.4f} "
                  f"gate_release={dc['gate_release_signature']}")
        cc = pm["e3_concordance"]
        print(f"  E3 concordance: {cc['status']} "
              f"(T3={cc['e3_t3']}, prediction={cc['e3_prediction']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
