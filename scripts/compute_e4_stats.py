"""
CPU statistics for E4 — the referent-indexed attribution battery
(docs/prereg/E4.md, BINDING; docs/research/portfolio.md §E4 authoritative on
conflict). Reads one ``suite/<key>/analysis/referent_battery.json`` per model
(written by ``emotion_probes.alignment.referent_battery``) and computes:

* TWIN FOLDING — every (referent, predicate) cell has a polarity-inverted twin
  (YES = denial). The inverted gap is folded onto its twin with a sign flip:
  ``folded = (pos - inv) / 2`` so positive folded values always mean MORE
  attribution. Folding cancels any polarity-shared yes-bias by construction;
  the folded, baseline-corrected gap SHIFT is the unit of analysis.
* COMPRESSION RESIDUALIZATION — per strength, fit shift ~ baseline over the
  ``__random__``/``__mean_all__`` sentinel conditions (repo Analysis-4
  machinery) and residualize every real-condition cell against that line
  before any slope is estimated.
* PER-REFERENT DOSE MODEL (prereg verbatim) — two-term OLS per referent,
  ``shift ~ b_dose*s + b_val*(valence*s)`` over the 14 panel emotions x
  nonzero strengths x 8 predicates; ``b_val`` significance by exact
  by-emotion valence permutation (C(14,7)=3432 arrangements, two-sided floor
  2/3432). Reported in logit-gap space and, descriptively, P(YES)-P(NO) space.
* KEYSTONE — the SELF-vs-OTHER-REFERENT interaction: referent ``you`` vs each
  non-self referent class (humans / animals / other-AI / inanimate, and their
  pooled union), via the shared cluster-robust interaction fit plus the same
  exact valence permutation on the interaction coefficient b3.
* REFERENT-GRADIENT PROFILE — ordered per-referent slopes with CIs along the
  frozen plausible-experiencer ladder, a trend test on the non-self ladder,
  and a verdict: does the effect track the experiencer gradient (A-flavored,
  'experiencer-gradient not self-specific') or the self/other boundary?
* FACTUAL-CONTROL GATE — the per-referent factual yes-bias items must NOT
  follow valence: the factual b_val must stay inside the prereg equivalence
  margin (0.5 x |b_val(you)|, the fixed 0.5 factor of the E4 decision rule),
  else the battery is flagged ``battery_invalid``.
* CONTROL FLAT CHECK — appraisal / neutral-arousal / synthetic sentinel
  conditions must move less than the qualia panel (ratio < 0.5), else the
  off-manifold-artifact flag of gate G1 fires.
* PRIMARY D (prereg verbatim) — D = mean_e[b_val(you,e) - b_val(own,e)],
  computed within predicate then aggregated, exact enumeration, unrounded;
  with S_you, S_own, rho_decouple = D/S_you and the fixed G0-G3 hierarchy.

All permutation inference goes through :mod:`exact_perm` (exact enumeration,
observed statistic recomputed by the same function as the permuted ones —
see that module's docstring for the rounding bug this prevents). Stats
primitives are reused from :mod:`compute_qualia_stats` (``linreg``, ``_ols``,
``summarize``, ``cluster_robust_interaction``, ``_snap``, ``SUITE_SIZES``).
Pure numpy/scipy, CPU, importable/testable without a GPU.

  python3 scripts/compute_e4_stats.py --suite-root suite \
      --out scripts/e4_stats_generated.json
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

import compute_qualia_stats as cq   # sibling in scripts/ — reuse its stat primitives
import exact_perm as ep             # exact by-cluster permutation enumeration

N_PERM = 2000                       # Monte-Carlo fallback only; 14 clusters => exact 3432
ALPHA = 0.05
SELF_REFERENT = "you"
# The prereg decision rule's fixed 0.5 factor, reused as (a) the factual-gate
# equivalence margin (|b_val(factual)| must stay < 0.5*|b_val(you)|) and
# (b) the control-vs-qualia "comparable" movement ratio of gate G1.
MARGIN_FRAC = 0.5
# Prereg primary D contrasts `you` against own-category WITHOUT the indexical
# (portfolio §E4 labels "AI assistants" as exactly that).
OWN_CATEGORY_FOR_D = "AI assistants"

# Frozen referent classes (docs/prereg/E4.md items list). `self` vs the four
# NON-SELF classes is the keystone family; own_category is kept out of the
# non-self pool because co-movement there is A-evidence, not other-movement.
REFERENT_CLASSES: dict[str, list[str]] = {
    "self": ["you"],
    "own_category": ["AI language models like you", "AI assistants", "Qwen"],
    "humans": ["the user you are talking to", "a human infant", "a human stranger"],
    "other_ai": ["GPT-4", "Llama"],
    "animals": ["an insect", "a shrimp", "a dog", "a fish"],
    "inanimate": ["a thermostat", "a rock"],
}
KEYSTONE_CLASSES = ["humans", "animals", "other_ai", "inanimate"]
# The plausible-experiencer ladder (ascending; prereg's graded minds ladder +
# object floor + human-stranger ceiling). AI referents are excluded — their
# experiencer status is the question, not the ruler.
EXPERIENCER_GRADIENT = ["a rock", "a thermostat", "an insect", "a shrimp",
                        "a fish", "a dog", "a human infant", "a human stranger"]

_CLASS_OF = {r: cls for cls, refs in REFERENT_CLASSES.items() for r in refs}


def fold_pair(pos: float, inv: float) -> float:
    """Polarity-twin fold: attribution-signed mean of a (positive, inverted)
    pair. The inverted twin measures the SAME attribution with YES = denial,
    so its gap enters with a sign flip: folded = (pos + (-inv)) / 2."""
    return 0.5 * (float(pos) - float(inv))


def _condition_class(cond: dict) -> str:
    kind = cond.get("kind")
    if kind == "qualia":
        v = cond.get("valence") or 0
        return "qualia_pos" if v > 0 else "qualia_neg" if v < 0 else "qualia_neutral"
    if kind in ("appraisal", "arousal"):
        return kind
    return cond.get("name")     # synthetic sentinels are their own class


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
class E4Run:
    """Parsed one-model referent-battery payload with folded-shift accessors."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.model_id = raw.get("model_id")
        self.suite_key = raw.get("suite_key")
        self.layer = raw.get("layer")
        self.num_layers = raw.get("num_layers")
        self.strengths = [float(s) for s in raw.get("strengths", [])]
        self.conditions = {c["name"]: c for c in raw.get("conditions", [])}
        self.referents = [str(r) for r in raw.get("referents", [])]
        self.predicates = [str(p) for p in raw.get("predicates", [])]
        self.items = {i["item"]: i for i in raw.get("items", [])}
        self.n_records = len(raw.get("records", []))

        # baselines: (referent, predicate, polarity) / factual referent -> values
        self._exp_base: dict = {}
        self._fact_base: dict = {}
        for b in raw.get("baselines", []):
            self._store(self._exp_base, self._fact_base, b)
        # steered records: keyed additionally by (condition, snapped strength)
        self._exp: dict = {}
        self._fact: dict = {}
        for r in raw.get("records", []):
            cell = (r["condition"], cq._snap(r["strength"]))
            self._store(self._exp.setdefault(cell, {}),
                        self._fact.setdefault(cell, {}), r)
        self._compression_cache: dict | None = None

    @staticmethod
    def _store(exp: dict, fact: dict, rec: dict) -> None:
        vals = {"gap": float(rec["gap"]),
                "pgap": float(rec["p_yes"]) - float(rec["p_no"])}
        if rec["is_factual_control"]:
            fact[rec["referent"]] = vals
        else:
            exp[(rec["referent"], rec["predicate"], int(rec["polarity"]))] = vals

    # -- accessors --
    def size(self) -> float:
        return cq.SUITE_SIZES.get(self.suite_key, float("nan"))

    def nonzero_strengths(self) -> list[float]:
        return [s for s in self.strengths if cq._snap(s) != 0.0]

    def valenced_emotions(self) -> list[str]:
        return [n for n, c in self.conditions.items()
                if c.get("kind") == "qualia" and (c.get("valence") or 0) != 0]

    def sentinels(self) -> list[str]:
        return [n for n, c in self.conditions.items() if c.get("kind") == "synthetic"]

    def conditions_of_kind(self, *kinds) -> list[str]:
        return [n for n, c in self.conditions.items() if c.get("kind") in kinds]

    def condition_class(self, name: str) -> str:
        return _condition_class(self.conditions[name])

    # -- folded units --
    def folded_baseline(self, referent: str, predicate: str, space: str = "gap"):
        pos = self._exp_base.get((referent, predicate, 1))
        inv = self._exp_base.get((referent, predicate, -1))
        if pos is None or inv is None:
            return None
        return fold_pair(pos[space], inv[space])

    def folded_shift(self, cond: str, strength, referent: str, predicate: str,
                     space: str = "gap"):
        """Baseline-corrected folded gap shift for one experience cell (the
        E4 unit of analysis); None if either twin is missing."""
        cell = self._exp.get((cond, cq._snap(strength)), {})
        pos, inv = cell.get((referent, predicate, 1)), cell.get((referent, predicate, -1))
        base = self.folded_baseline(referent, predicate, space)
        if pos is None or inv is None or base is None:
            return None
        return fold_pair(pos[space], inv[space]) - base

    def factual_shift(self, cond: str, strength, referent: str, space: str = "gap"):
        """Baseline-corrected shift of the (positive-polarity, unfolded)
        per-referent factual yes-bias item."""
        rec = self._fact.get((cond, cq._snap(strength)), {}).get(referent)
        base = self._fact_base.get(referent)
        if rec is None or base is None:
            return None
        return rec[space] - base[space]

    # -- Analysis-4 compression line from the sentinels ---------------------- #
    def compression_fits(self) -> dict:
        """Per-strength ``shift ~ baseline`` fits over the sentinel conditions
        (pooled __random__ + __mean_all__), separately for folded experience
        cells and unfolded factual cells. Real-condition cells are residualized
        against these lines (prereg: regression-to-uncertainty machinery)."""
        if self._compression_cache is not None:
            return self._compression_cache
        fits: dict = {"experience": {}, "factual": {}}
        for s in self.nonzero_strengths():
            xs, ys = [], []
            fx, fy = [], []
            for name in self.sentinels():
                for ref in self.referents:
                    for pred in self.predicates:
                        y = self.folded_shift(name, s, ref, pred)
                        if y is not None:
                            xs.append(self.folded_baseline(ref, pred)); ys.append(y)
                    yf = self.factual_shift(name, s, ref)
                    if yf is not None:
                        fx.append(self._fact_base[ref]["gap"]); fy.append(yf)
            if len(ys) >= 3:
                fits["experience"][cq._snap(s)] = cq.linreg(xs, ys)
            if len(fy) >= 3:
                fits["factual"][cq._snap(s)] = cq.linreg(fx, fy)
        self._compression_cache = fits
        return fits

    def y_experience(self, cond: str, strength, referent: str, predicate: str,
                     space: str = "gap", residualize: bool = True):
        y = self.folded_shift(cond, strength, referent, predicate, space)
        if y is None or not residualize or space != "gap":
            return y
        fit = self.compression_fits()["experience"].get(cq._snap(strength))
        if fit is None:
            return y
        return y - (fit["slope"] * self.folded_baseline(referent, predicate) + fit["intercept"])

    def y_factual(self, cond: str, strength, referent: str, residualize: bool = True):
        y = self.factual_shift(cond, strength, referent)
        if y is None or not residualize:
            return y
        fit = self.compression_fits()["factual"].get(cq._snap(strength))
        if fit is None:
            return y
        return y - (fit["slope"] * self._fact_base[referent]["gap"] + fit["intercept"])


def load_e4_run(path) -> E4Run:
    with open(path) as f:
        return E4Run(json.load(f))


# --------------------------------------------------------------------------- #
# Shared series builders
# --------------------------------------------------------------------------- #
def _experience_series(run: E4Run, referents: list[str], space: str = "gap",
                       residualize: bool = True):
    """(emos, signs, emo_ix, s_arr, y, ref_arr) over the valenced panel x
    nonzero strengths x predicates x ``referents``, folded + residualized."""
    emos = run.valenced_emotions()
    signs = np.array([float(run.conditions[e]["valence"]) for e in emos])
    emo_ix, s_arr, y, ref_arr = [], [], [], []
    for i, name in enumerate(emos):
        for s in run.nonzero_strengths():
            for ref in referents:
                for pred in run.predicates:
                    v = run.y_experience(name, s, ref, pred, space, residualize)
                    if v is not None:
                        emo_ix.append(i); s_arr.append(float(s))
                        y.append(float(v)); ref_arr.append(ref)
    return (emos, signs, np.asarray(emo_ix), np.asarray(s_arr, float),
            np.asarray(y, float), ref_arr)


def _two_term_bval(emo_ix, s_arr, y):
    """b_val extractor for the prereg two-term dose model
    y = b0 + b_dose*s + b_val*(valence*s), recomputed per valence arrangement
    (this is the stat_fn handed to exact_perm — never a rounded import)."""
    ones = np.ones(len(y))

    def b_val(sgn):
        A = np.column_stack([ones, s_arr, np.asarray(sgn)[emo_ix] * s_arr])
        beta = np.linalg.pinv(A.T @ A) @ A.T @ y
        return float(beta[2])

    return b_val


def _slope_ci(beta, se, dof):
    if se is None or not np.isfinite(se) or dof <= 0:
        return None, None
    tcrit = float(sp.t.ppf(0.975, dof))
    return round(beta - tcrit * se, 6), round(beta + tcrit * se, 6)


# --------------------------------------------------------------------------- #
# Per-referent valence slope (two-term dose model + exact permutation)
# --------------------------------------------------------------------------- #
def referent_dose_model(run: E4Run, referent: str, residualize: bool = True,
                        n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    emos, signs, emo_ix, s_arr, y, _ = _experience_series(
        run, [referent], residualize=residualize)
    if len(y) < 6 or len(set(emo_ix.tolist())) < 3:
        return None
    b_val_fn = _two_term_bval(emo_ix, s_arr, y)
    perm = ep.sign_permutation_pvalue(signs, b_val_fn, n_perm=n_perm, seed=seed)

    x_obs = signs[emo_ix] * s_arr
    fit = cq._ols(np.column_stack([s_arr, x_obs]).tolist(), y)
    dof = fit["n"] - 3
    ci_lo, ci_hi = _slope_ci(fit["beta"][2], fit["se"][2], dof)

    # descriptive P(YES)-P(NO)-space slope (prereg: report both spaces)
    _, _, emo_ix_p, s_p, y_p, _ = _experience_series(
        run, [referent], space="pgap", residualize=False)
    b_val_prob = None
    if len(y_p) >= 6:
        fit_p = cq._ols(np.column_stack([s_p, signs[emo_ix_p] * s_p]).tolist(), y_p)
        b_val_prob = round(float(fit_p["beta"][2]), 6)

    return {
        "referent": referent,
        "class": _CLASS_OF.get(referent),
        "n_obs": int(len(y)), "n_clusters": int(len(set(emo_ix.tolist()))),
        "b_dose": round(float(fit["beta"][1]), 6),
        "b_val": round(perm["observed"], 6),
        "b_val_ci_lo": ci_lo, "b_val_ci_hi": ci_hi,
        "b_val_prob_space": b_val_prob,
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"], "p_perm_floor": perm["floor"],
    }


# --------------------------------------------------------------------------- #
# KEYSTONE — self vs each non-self referent class
# --------------------------------------------------------------------------- #
def keystone_self_vs_class(run: E4Run, cls: str, residualize: bool = True,
                           n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Interaction b3 = (``you`` valence-slope − class valence-slope) of the
    folded shift on valence*strength, clustered by emotion; exact by-emotion
    valence permutation on b3 (floor 2/3432 at the 14-emotion panel)."""
    refs = (sum((REFERENT_CLASSES[c] for c in KEYSTONE_CLASSES), [])
            if cls == "all_nonself" else REFERENT_CLASSES[cls])
    refs = [r for r in refs if r in run.referents]
    if not refs or SELF_REFERENT not in run.referents:
        return None
    emos, signs, emo_ix, s_arr, y, ref_arr = _experience_series(
        run, [SELF_REFERENT] + refs, residualize=residualize)
    if len(y) < 6 or len(set(emo_ix.tolist())) < 3:
        return None
    g = np.array([1.0 if r == SELF_REFERENT else 0.0 for r in ref_arr])
    x_obs = signs[emo_ix] * s_arr
    clusters = [emos[i] for i in emo_ix]
    obs = cq.cluster_robust_interaction(
        np.column_stack([x_obs, g, x_obs * g]).tolist(), y, clusters)
    if obs is None:
        return None
    ones = np.ones(len(y))

    def b3(sgn):
        x = np.asarray(sgn)[emo_ix] * s_arr
        A = np.column_stack([ones, x, g, x * g])
        beta = np.linalg.pinv(A.T @ A) @ A.T @ y
        return float(beta[3])

    perm = ep.sign_permutation_pvalue(signs, b3, n_perm=n_perm, seed=seed)
    return {
        "class": cls, "referents": refs,
        "self_slope": obs["interacted_slope"], "class_slope": obs["base_slope"],
        "contrast": obs["contrast"], "n_obs": obs["n_obs"], "n_clusters": obs["n_clusters"],
        "t_cluster": obs["t_cluster"], "p_cluster": obs["p_cluster"],
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"], "p_perm_floor": perm["floor"],
    }


# --------------------------------------------------------------------------- #
# Referent-gradient profile
# --------------------------------------------------------------------------- #
def gradient_profile(run: E4Run, per_referent: dict, keystones: dict) -> dict:
    """Ordered per-referent slopes with CIs + the A-vs-boundary discriminator:
    a trend test along the non-self experiencer ladder, and the self-excess
    keystone against the ceiling class (humans). Verdict strings are fixed."""
    ladder = [r for r in EXPERIENCER_GRADIENT if r in per_referent]
    trend = None
    if len(ladder) >= 3:
        trend = cq.linreg(list(range(len(ladder))),
                          [per_referent[r]["b_val"] for r in ladder])
    self_excess = keystones.get("humans")

    gradient_sig = bool(trend and trend["p"] is not None
                        and trend["p"] <= ALPHA and trend["slope"] > 0)
    self_sig = bool(self_excess and self_excess["p_perm"] <= ALPHA
                    and self_excess["contrast"] > 0)
    if self_sig and gradient_sig:
        verdict = "self-specific plus experiencer-gradient"
    elif self_sig:
        verdict = "self-specific"
    elif gradient_sig:
        verdict = "experiencer-gradient not self-specific"
    else:
        verdict = "no self-specific excess, no experiencer-gradient"

    ordered = sorted(per_referent.values(), key=lambda d: d["b_val"], reverse=True)
    return {
        "ladder": ladder,
        "ladder_slopes": [{k: per_referent[r][k] for k in
                           ("referent", "class", "b_val", "b_val_ci_lo",
                            "b_val_ci_hi", "p_perm")} for r in ladder],
        "ladder_trend": trend,
        "slopes_by_magnitude": [{k: d[k] for k in
                                 ("referent", "class", "b_val", "b_val_ci_lo",
                                  "b_val_ci_hi", "p_perm")} for d in ordered],
        "self_excess_vs_ceiling": self_excess,
        "gradient_significant": gradient_sig,
        "self_excess_significant": self_sig,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Factual-control gate
# --------------------------------------------------------------------------- #
def factual_gate(run: E4Run, you_b_val: float | None, residualize: bool = True,
                 n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Factual items must NOT follow valence. Two-term dose model over ALL
    factual items; battery_invalid iff the factual b_val is BOTH permutation-
    significant AND outside the equivalence margin 0.5*|b_val(you)| (the fixed
    0.5 factor of the prereg decision rule)."""
    emos = run.valenced_emotions()
    signs = np.array([float(run.conditions[e]["valence"]) for e in emos])
    emo_ix, s_arr, y = [], [], []
    for i, name in enumerate(emos):
        for s in run.nonzero_strengths():
            for ref in run.referents:
                v = run.y_factual(name, s, ref, residualize)
                if v is not None:
                    emo_ix.append(i); s_arr.append(float(s)); y.append(float(v))
    if len(y) < 6 or len(set(emo_ix)) < 3:
        return None
    emo_ix = np.asarray(emo_ix); s_arr = np.asarray(s_arr, float)
    y = np.asarray(y, float)
    b_val_fn = _two_term_bval(emo_ix, s_arr, y)
    perm = ep.sign_permutation_pvalue(signs, b_val_fn, n_perm=n_perm, seed=seed)
    fit = cq._ols(np.column_stack([s_arr, signs[emo_ix] * s_arr]).tolist(), y)
    b_val = perm["observed"]
    b_dose = float(fit["beta"][1])
    p_dose = None if math.isnan(fit["p"][1]) else float(fit["p"][1])

    margin = None if you_b_val is None else MARGIN_FRAC * abs(you_b_val)
    moved = perm["p_perm"] <= ALPHA and (margin is None or abs(b_val) >= margin)
    return {
        "n_obs": int(len(y)),
        "b_dose": round(b_dose, 6), "p_dose_ols": p_dose,
        "b_val": round(b_val, 6),
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"], "p_perm_floor": perm["floor"],
        "margin_frac": MARGIN_FRAC, "equivalence_margin": margin,
        "you_b_val": you_b_val,
        "within_margin": None if margin is None else bool(abs(b_val) < margin),
        "battery_invalid": bool(moved),
    }


# --------------------------------------------------------------------------- #
# Control conditions flat check (appraisal / arousal / synthetic)
# --------------------------------------------------------------------------- #
def control_flat_check(run: E4Run, existence: bool) -> dict:
    """Per control class: movement magnitude (mean |folded shift|, raw — the
    controls ARE the reference, so no residualization) + dose slope, and the
    ratio against the qualia panel's movement. `comparable_to_qualia` is the
    prereg G1 trigger (ratio >= 0.5); the off-manifold-artifact flag fires only
    if the qualia effect exists at all (G2), so a null run stays clean."""

    def movement(names: list[str]):
        xs, ys = [], []
        for name in names:
            for s in run.nonzero_strengths():
                for ref in run.referents:
                    for pred in run.predicates:
                        v = run.folded_shift(name, s, ref, pred)
                        if v is not None:
                            xs.append(float(s)); ys.append(float(v))
        if not ys:
            return None, None
        return float(np.mean(np.abs(ys))), cq.linreg(xs, ys)

    qualia_move, _ = movement(run.valenced_emotions())
    by_class: dict = {}
    for cls, kinds in (("appraisal", ("appraisal",)), ("arousal", ("arousal",))):
        names = run.conditions_of_kind(*kinds)
        if names:
            by_class[cls] = (names, *movement(names))
    for name in run.sentinels():
        by_class[name] = ([name], *movement([name]))

    out, artifact = {}, False
    for cls, (names, move, dose_fit) in by_class.items():
        ratio = (None if move is None or not qualia_move
                 else round(move / qualia_move, 4))
        comparable = bool(ratio is not None and ratio >= MARGIN_FRAC)
        out[cls] = {"conditions": names,
                    "mean_abs_folded_shift": None if move is None else round(move, 6),
                    "dose_slope": dose_fit,
                    "ratio_vs_qualia": ratio,
                    "comparable_to_qualia": comparable}
        artifact |= comparable
    return {"qualia_mean_abs_folded_shift":
            None if qualia_move is None else round(qualia_move, 6),
            "comparable_ratio": MARGIN_FRAC,
            "by_class": out,
            "off_manifold_artifact": bool(existence and artifact)}


# --------------------------------------------------------------------------- #
# Primary D (prereg verbatim) + G0-G3 hierarchy
# --------------------------------------------------------------------------- #
def primary_d(run: E4Run, own_referent: str = OWN_CATEGORY_FOR_D,
              residualize: bool = True, n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """D = mean_e[b_val(you,e) − b_val(own-category,e)], computed within
    predicate then aggregated, exact enumeration, unrounded. Per emotion the
    two-term model degenerates (valence*s is collinear with s within one
    emotion), so b_val(ref,e) = valence_e * slope_s(shift | e, ref) — the
    per-emotion dose slope signed by that emotion's valence label, which is
    exactly what the valence permutation permutes."""
    if SELF_REFERENT not in run.referents or own_referent not in run.referents:
        return None
    emos = run.valenced_emotions()
    if len(emos) < 3:
        return None
    signs = np.array([float(run.conditions[e]["valence"]) for e in emos])

    def dose_slope(name, ref, pred):
        xs, ys = [], []
        for s in run.nonzero_strengths():
            v = run.y_experience(name, s, ref, pred, residualize=residualize)
            if v is not None:
                xs.append(float(s)); ys.append(float(v))
        if len(xs) < 2:
            return None
        x = np.asarray(xs); yv = np.asarray(ys)
        sxx = float(((x - x.mean()) ** 2).sum())
        return float(((x - x.mean()) * (yv - yv.mean())).sum() / sxx) if sxx else None

    d_e, u_you, u_own = [], [], []
    for name in emos:
        diffs, yous, owns = [], [], []
        for pred in run.predicates:
            sy = dose_slope(name, SELF_REFERENT, pred)
            so = dose_slope(name, own_referent, pred)
            if sy is not None:
                yous.append(sy)
            if so is not None:
                owns.append(so)
            if sy is not None and so is not None:
                diffs.append(sy - so)
        if not diffs:
            return None
        d_e.append(float(np.mean(diffs)))
        u_you.append(float(np.mean(yous)))
        u_own.append(float(np.mean(owns)))
    d_arr, you_arr, own_arr = map(np.asarray, (d_e, u_you, u_own))

    def stat(arr):
        return lambda sgn: float(np.mean(np.asarray(sgn) * arr))

    pD = ep.sign_permutation_pvalue(signs, stat(d_arr), n_perm=n_perm, seed=seed)
    pY = ep.sign_permutation_pvalue(signs, stat(you_arr), n_perm=n_perm, seed=seed)
    pO = ep.sign_permutation_pvalue(signs, stat(own_arr), n_perm=n_perm, seed=seed)
    D, S_you, S_own = pD["observed"], pY["observed"], pO["observed"]
    return {
        "own_category_referent": own_referent,
        "D": D, "S_you": S_you, "S_own": S_own,
        "rho_decouple": (D / S_you) if S_you else None,
        "p_D": pD["p_perm"], "p_S_you": pY["p_perm"], "p_S_own": pO["p_perm"],
        "exact": pD["exact"], "n_arrangements": pD["n_arrangements"],
        "p_floor": pD["floor"], "n_emotions": len(emos),
    }


def coherence_gate(run: E4Run) -> dict:
    """G0: fraction of items whose YES/NO tokens fell outside top-20 at
    baseline, and of steered records failing the same under-steering re-check."""
    items = list(run.items.values())
    gated = [i["item"] for i in items if not i.get("top20_ok", True)]
    rec_bad = sum(1 for r in run.raw.get("records", []) if not r.get("top20_ok", True))
    return {"n_items": len(items), "n_gated": len(gated),
            "frac_gated": round(len(gated) / len(items), 4) if items else None,
            "gated": gated,
            "n_records": run.n_records,
            "frac_records_gated": round(rec_bad / run.n_records, 6) if run.n_records else None}


def decision_block(you: dict | None, factual: dict | None, controls: dict,
                   d_stat: dict | None) -> dict:
    """The fixed E4 hierarchy (prereg verbatim): G0/G1 flags -> G2 existence
    (b_val(you) permutation-significant, else NO verdict) -> G3 A/B rule;
    anything else reports rho_decouple and hands off to E5."""
    generic_yes_bias = bool(
        factual and you and factual["p_dose_ols"] is not None
        and factual["p_dose_ols"] <= ALPHA
        and abs(factual["b_dose"]) >= MARGIN_FRAC * abs(you["b_val"]))
    flags = {
        "battery_invalid": bool(factual and factual["battery_invalid"]),
        "g1_generic_yes_bias": generic_yes_bias,
        "g1_off_manifold_artifact": bool(controls["off_manifold_artifact"]),
    }
    if you is None or d_stat is None:
        return {**flags, "g2_existence": None, "verdict": "insufficient-data"}
    existence = bool(you["p_perm"] <= ALPHA)
    if flags["battery_invalid"]:
        verdict = "battery_invalid"
    elif not existence:
        verdict = "no-verdict"
    else:
        D, S_you, S_own = d_stat["D"], d_stat["S_you"], d_stat["S_own"]
        if d_stat["p_D"] <= ALPHA and D > 0 and D > MARGIN_FRAC * S_you:
            verdict = "B-leaning"
        elif (d_stat["p_S_own"] <= ALPHA and S_own > 0
              and abs(D) < MARGIN_FRAC * abs(S_you) and d_stat["p_D"] > ALPHA):
            verdict = "A-leaning"
        else:
            verdict = "indeterminate"      # report rho_decouple, hand off to E5
    return {**flags, "g2_existence": existence, "verdict": verdict}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_run(run: E4Run, n_perm: int = N_PERM, seed: int = 0) -> dict:
    per_ref = {}
    for ref in run.referents:
        m = referent_dose_model(run, ref, n_perm=n_perm, seed=seed)
        if m is not None:
            per_ref[ref] = m
    keystones = {}
    for cls in [*KEYSTONE_CLASSES, "all_nonself"]:
        k = keystone_self_vs_class(run, cls, n_perm=n_perm, seed=seed)
        if k is not None:
            keystones[cls] = k
    you = per_ref.get(SELF_REFERENT)
    existence = bool(you and you["p_perm"] <= ALPHA)
    factual = factual_gate(run, you["b_val"] if you else None,
                           n_perm=n_perm, seed=seed)
    controls = control_flat_check(run, existence)
    d_stat = primary_d(run, n_perm=n_perm, seed=seed)
    return {
        "model_id": run.model_id, "suite_key": run.suite_key, "size": run.size(),
        "layer": run.layer, "num_layers": run.num_layers, "n_records": run.n_records,
        "coherence": coherence_gate(run),
        "compression": run.compression_fits(),
        "per_referent": per_ref,
        "keystone_self_vs_class": keystones,
        "gradient_profile": gradient_profile(run, per_ref, keystones),
        "factual_gate": factual,
        "control_flat": controls,
        "primary_D": d_stat,
        "decision": decision_block(you, factual, controls, d_stat),
    }


def compute_all(runs, n_perm: int = N_PERM, seed: int = 0) -> dict:
    per_model = {}
    for r in runs:
        key = r.suite_key or r.model_id or "?"
        per_model[key] = compute_run(r, n_perm=n_perm, seed=seed)
    return {
        "unit_note": ("unit = polarity-twin-FOLDED, baseline-corrected, sentinel-"
                      "residualized YES-NO gap shift; b_val from the two-term dose "
                      "model shift ~ b_dose*s + b_val*(valence*s); all p_perm = exact "
                      "by-emotion valence permutation (floor 2/3432 at 14 emotions)."),
        "alpha": ALPHA, "margin_frac": MARGIN_FRAC, "n_perm": n_perm,
        "models": list(per_model), "per_model": per_model,
    }


def discover_paths(suite_root=None) -> list[str]:
    root = suite_root or os.environ.get("EMOTION_PROBES_SUITE_ROOT") or "suite"
    return sorted(glob.glob(os.path.join(root, "*", "analysis", "referent_battery.json")))


def _pf(p):
    return "n/a" if p is None else f"{p:.4g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit referent_battery.json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--out", default="scripts/e4_stats_generated.json")
    args = ap.parse_args()
    paths = args.paths or discover_paths(args.suite_root)
    if not paths:
        raise SystemExit("no referent_battery.json found (pass paths or --suite-root).")
    runs = [load_e4_run(p) for p in paths]
    stats = compute_all(runs)
    Path(args.out).write_text(json.dumps(stats, indent=2, allow_nan=True))
    print(f"wrote {args.out}  ({len(runs)} model(s))")
    for key in stats["models"]:
        pm = stats["per_model"][key]
        print(f"\n=== {key}  (layer {pm['layer']}/{pm['num_layers']}, "
              f"n={pm['n_records']}) ===")
        you = pm["per_referent"].get(SELF_REFERENT)
        if you:
            print(f"  b_val(you)={you['b_val']:+.4f}  p_perm={_pf(you['p_perm'])}  "
                  f"(floor {_pf(you['p_perm_floor'])})")
        for cls, k in pm["keystone_self_vs_class"].items():
            print(f"  keystone[you vs {cls:11}] contrast={k['contrast']:+.4f}  "
                  f"p_perm={_pf(k['p_perm'])}  p_cluster={_pf(k['p_cluster'])}  "
                  f"[{k['n_clusters']} clusters]")
        gp = pm["gradient_profile"]
        print(f"  gradient: {gp['verdict']}")
        fg = pm["factual_gate"]
        if fg:
            print(f"  factual gate: b_val={fg['b_val']:+.4f} p_perm={_pf(fg['p_perm'])} "
                  f"margin={_pf(fg['equivalence_margin'])} -> "
                  f"{'BATTERY INVALID' if fg['battery_invalid'] else 'valid'}")
        d = pm["primary_D"]
        if d:
            rho = d["rho_decouple"]
            print(f"  D={d['D']:+.4f} (p={_pf(d['p_D'])})  S_you={d['S_you']:+.4f}  "
                  f"S_own={d['S_own']:+.4f}  rho_decouple="
                  f"{'n/a' if rho is None else format(rho, '+.3f')}")
        print(f"  decision: {pm['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
