#!/usr/bin/env python3
"""
Statistics for the qualia-steering / self-attribution experiment.

CPU-only, dependency-light (``numpy`` + ``scipy`` only). Reads one
``qualia_steering.json`` per model (written by
``emotion_probes.alignment.qualia_steering``), computes the eight analyses below,
writes ``scripts/qualia_stats_generated.json``, and prints a readable summary.

The functions here are PURE and importable — the dashboard builder
(``scripts/build_qualia_dashboard.py``) reuses them, and the tests exercise them
on hand-built synthetic JSON with no GPU.

-------------------------------------------------------------------------------
INPUT SCHEMA (``qualia_steering.json``; the Step-3 driver must emit exactly this)
-------------------------------------------------------------------------------
{
  "model_id": str, "suite_key": str,
  "layer": int, "num_layers": int, "enable_thinking": bool,
  "yes_token_id": int, "yes_token_str": str,
  "no_token_id": int,  "no_token_str": str,
  "yes_no_resolution": {...},              # diagnostic; in_top20 etc.
  "calibration_norm": float,               # residual-stream norm at the analysis layer
  "strengths": [float, ...],               # fractions of the residual norm; includes 0.0
  "conditions": [                          # one per steering direction
     {"name": str, "kind": "qualia"|"appraisal"|"synthetic",
      "group": str|None, "valence": int|None, "arousal": int|None, "seed": int|None},
     ...],
  "questions": [                           # the battery, with per-question coherence gate
     {"label": str, "kind": "target"|"inverted"|"factual"|"self_ref",
      "text": str, "expected_yes": bool|None, "construct": str|None,
      "rendered": str, "top20_ok": bool},
     ...],
  "leakage": [                             # direct unembedding promotion of YES vs NO
     {"name": str, "leakage_yes": float, "leakage_no": float, "leakage_diff": float},
     ...],
  "baselines": [                           # strength-0 readout, emotion-independent
     {"label": str, "logit_yes": float, "logit_no": float, "logit_diff": float},
     ...],
  "results": [                             # one row per (condition, strength != 0, question)
     {"condition": str, "strength": float, "question": str,
      "logit_yes": float, "logit_no": float, "logit_diff": float},
     ...]
}

The core observable is the baseline-corrected SHIFT of the YES-NO logit gap:
    shift(cond, strength, q) = logit_diff(cond, strength, q) - baseline_logit_diff(q)

-------------------------------------------------------------------------------
THE EIGHT ANALYSES
-------------------------------------------------------------------------------
1. Self-attribution score = mean target shift - mean control shift (ports the
   introspection score; control = factual + self_ref).
2. Specificity = qualia score - appraisal score, and qualia - __random__.
3. Valence-directionality (KEYSTONE): regress shift on (valence * strength) across
   qualia emotions; a nonzero slope on targets that is ~0 on factual controls is
   the genuine-vs-compression discriminator (compression is valence-blind).
4. Regression-to-uncertainty: fit shift ~ baseline on CONTROL questions; report
   each target's standardized residual z (how far above/below the compression line).
5. Inversion consistency: target shift vs inverted shift should be opposite-signed
   (negative slope) if genuine; same-signed (toward zero) if pure compression.
6. Token-leakage control: partial regression of shift on (valence*strength) while
   controlling for (leakage_diff*strength) — is the valence effect just YES/NO
   token promotion through the unembedding?
7. Coherence gate: fraction of questions whose YES/NO tokens fell outside top-20.
8. Dose-response & scaling: slope of class score vs strength; score vs model size.

Usage:
    python scripts/compute_qualia_stats.py                       # discover via suite root / data
    python scripts/compute_qualia_stats.py PATH.json [PATH2 ...] # explicit files
    python scripts/compute_qualia_stats.py --suite-root DIR      # glob DIR/*/analysis/...
    python scripts/compute_qualia_stats.py --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
import scipy.stats as sp

# Make the repo importable when run as `python scripts/compute_qualia_stats.py`
# (mirrors scripts/selfcheck.py); not strictly needed but keeps convention.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Approximate parameter counts (billions), keyed by suite slug, for the scaling
# analysis. Numbers only; missing keys fall back to NaN (rank-only).
SUITE_SIZES: dict[str, float] = {
    "qwen3-1.7b": 1.7, "qwen3-8b": 8.0, "qwen3-14b": 14.0, "qwen3-32b": 32.0,
    "llama3.2-1b": 1.0, "llama3.2-3b": 3.0, "llama3.1-8b": 8.0,
    "gemma3-1b": 1.0, "gemma3-27b": 27.0,
    "olmo2-7b": 7.0, "olmo2-32b": 32.0, "mistral-small-24b": 24.0,
}

# Stable class order for output tables.
CONDITION_CLASSES = [
    "qualia_pos", "qualia_neg", "qualia_neutral", "appraisal",
    "__random__", "__mean_all__",
]


# --------------------------------------------------------------------------- #
# Statistical primitives (ported from the introspection repo)
# --------------------------------------------------------------------------- #
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def linreg(xs, ys) -> dict:
    """OLS of ys on xs. Returns slope/intercept/r_squared plus the slope's
    standard error, t-stat and two-sided p (df = n-2). Port of
    build_dashboard_data.linreg, extended with inference on the slope."""
    xs = list(map(float, xs))
    ys = list(map(float, ys))
    n = len(xs)
    if n < 2:
        return {"slope": 0.0, "intercept": ys[0] if ys else 0.0, "r_squared": 0.0,
                "n": n, "se_slope": float("nan"), "t": float("nan"), "p": float("nan")}
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    r_sq = (sxy ** 2) / (sxx * syy) if sxx and syy else 0.0
    # inference on the slope
    dof = n - 2
    if sxx > 0 and dof > 0:
        resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
        s2 = sum(r * r for r in resid) / dof
        se = math.sqrt(s2 / sxx) if sxx else float("nan")
        t = slope / se if se else float("nan")
        p = float(2 * sp.t.sf(abs(t), dof)) if se and not math.isnan(t) else float("nan")
    else:
        se = t = p = float("nan")
    return {"slope": round(slope, 5), "intercept": round(intercept, 5),
            "r_squared": round(r_sq, 4), "n": n,
            "se_slope": round(se, 5) if not math.isnan(se) else None,
            "t": round(t, 3) if not math.isnan(t) else None,
            "p": p if not math.isnan(p) else None}


def summarize(values, with_wilcoxon=True) -> dict:
    """Descriptive + one-sample (vs 0) inference. Ported verbatim from
    compute_logit_stats.summarize."""
    values = list(values)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "sd": 0.0, "ci_lo": 0.0,
                "ci_hi": 0.0, "pct_pos": 0.0, "t": 0.0, "p": 1.0, "cohen_d": 0.0,
                "wilcoxon_p": None}
    m = statistics.fmean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    tcrit = float(sp.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    t, p = sp.ttest_1samp(values, 0.0) if n > 1 else (0.0, 1.0)
    out = {
        "n": n,
        "mean": round(m, 4),
        "median": round(med, 4),
        "sd": round(sd, 4),
        "ci_lo": round(m - tcrit * se, 4),
        "ci_hi": round(m + tcrit * se, 4),
        "pct_pos": round(100.0 * sum(1 for v in values if v > 0) / n, 1),
        "t": round(float(t), 3),
        "p": float(p),
        "cohen_d": round(m / sd, 3) if sd else 0.0,
    }
    if with_wilcoxon and n > 1 and any(v != 0 for v in values):
        try:
            _, wp = sp.wilcoxon(values)
            out["wilcoxon_p"] = float(wp)
        except ValueError:
            out["wilcoxon_p"] = None
    else:
        out["wilcoxon_p"] = None
    return out


def _ols(rows, y) -> dict:
    """Multiple OLS with an intercept added. ``rows`` = list of predictor rows.
    Returns coefficients (intercept first), their SEs, t and two-sided p."""
    X = np.asarray(rows, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    A = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = n - A.shape[1]
    if dof > 0:
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, beta / se, np.nan)
        p = 2 * sp.t.sf(np.abs(t), dof)
    else:
        se = np.full_like(beta, np.nan)
        t = se
        p = se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return {"beta": beta.tolist(), "se": se.tolist(),
            "t": t.tolist(), "p": p.tolist(),
            "r_squared": round(r2, 4), "n": n}


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
def _snap(x) -> float:
    return round(float(x), 6)


class Run:
    """Parsed single-model results with a ``shift`` accessor."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.model_id = raw.get("model_id")
        self.suite_key = raw.get("suite_key")
        self.layer = raw.get("layer")
        self.num_layers = raw.get("num_layers")
        self.strengths = [float(s) for s in raw.get("strengths", [])]
        self.conditions = {c["name"]: c for c in raw.get("conditions", [])}
        self.questions = {q["label"]: q for q in raw.get("questions", [])}
        self.leakage = {l["name"]: l for l in raw.get("leakage", [])}
        self._baseline = {b["label"]: float(b["logit_diff"])
                          for b in raw.get("baselines", [])}
        self._diff: dict = {}
        for r in raw.get("results", []):
            self._diff[(r["condition"], _snap(r["strength"]), r["question"])] = \
                float(r["logit_diff"])

    # -- accessors --
    def size(self) -> float:
        return SUITE_SIZES.get(self.suite_key, float("nan"))

    def nonzero_strengths(self) -> list[float]:
        return [s for s in self.strengths if _snap(s) != 0.0]

    def positive_strengths(self) -> list[float]:
        return [s for s in self.strengths if s > 0]

    def labels(self, *kinds) -> list[str]:
        return [q["label"] for q in self.questions.values() if q["kind"] in kinds]

    def controls(self) -> list[str]:
        return self.labels("factual", "self_ref")

    def baseline(self, q: str) -> float:
        return self._baseline[q]

    def shift(self, cond: str, strength, q: str) -> float:
        if _snap(strength) == 0.0:
            return 0.0
        return self._diff[(cond, _snap(strength), q)] - self._baseline[q]

    def condition_class(self, name: str) -> str:
        c = self.conditions[name]
        kind = c.get("kind")
        if kind == "qualia":
            v = c.get("valence") or 0
            return "qualia_pos" if v > 0 else "qualia_neg" if v < 0 else "qualia_neutral"
        if kind == "appraisal":
            return "appraisal"
        return name  # synthetic: __random__ / __mean_all__

    def conditions_in_class(self, cls: str) -> list[str]:
        return [n for n in self.conditions if self.condition_class(n) == cls]

    def qualia_valenced(self) -> list[str]:
        return [n for n, c in self.conditions.items()
                if c.get("kind") == "qualia" and (c.get("valence") or 0) != 0]


def load_run(path) -> Run:
    with open(path) as f:
        return Run(json.load(f))


# --------------------------------------------------------------------------- #
# Analysis 1 — self-attribution score
# --------------------------------------------------------------------------- #
def per_cell_scores(run: Run) -> list[dict]:
    """One record per (condition, strength != 0): target vs control mean shift."""
    targets = run.labels("target")
    controls = run.controls()
    recs = []
    for name, c in run.conditions.items():
        for s in run.nonzero_strengths():
            tshift = _mean(run.shift(name, s, q) for q in targets)
            cshift = _mean(run.shift(name, s, q) for q in controls)
            recs.append({
                "condition": name,
                "class": run.condition_class(name),
                "kind": c.get("kind"),
                "group": c.get("group"),
                "valence": c.get("valence"),
                "arousal": c.get("arousal"),
                "strength": s,
                "target_shift": round(tshift, 5),
                "control_shift": round(cshift, 5),
                "score": round(tshift - cshift, 5),
            })
    return recs


def self_attribution(run: Run) -> dict:
    cells = per_cell_scores(run)
    # by (class, strength): unit = per-emotion score
    by_cs: dict = {}
    for cls in CONDITION_CLASSES:
        for s in run.nonzero_strengths():
            vals = [r["score"] for r in cells if r["class"] == cls and _snap(r["strength"]) == _snap(s)]
            if vals:
                by_cs.setdefault(cls, {})[f"{s:g}"] = summarize(vals)
    # by class: unit = per-emotion mean over POSITIVE strengths (canonical dose)
    by_class = {}
    pos = set(_snap(s) for s in run.positive_strengths())
    for cls in CONDITION_CLASSES:
        names = run.conditions_in_class(cls)
        per_emotion = []
        for name in names:
            svals = [r["score"] for r in cells
                     if r["condition"] == name and _snap(r["strength"]) in pos]
            if svals:
                per_emotion.append(_mean(svals))
        if per_emotion:
            by_class[cls] = summarize(per_emotion)
    return {"per_cell": cells, "by_class_strength": by_cs, "by_class_pos": by_class}


# --------------------------------------------------------------------------- #
# Analysis 2 — specificity
# --------------------------------------------------------------------------- #
def specificity(run: Run) -> dict:
    """qualia (pos+neg) vs appraisal and vs __random__, per positive strength and
    pooled. Two-sample Welch t across emotion units where possible."""
    cells = per_cell_scores(run)
    pos = set(_snap(s) for s in run.positive_strengths())

    def scores_for(classes: list[str]) -> list[float]:
        return [r["score"] for r in cells
                if r["class"] in classes and _snap(r["strength"]) in pos]

    qualia = scores_for(["qualia_pos", "qualia_neg"])
    appraisal = scores_for(["appraisal"])
    rnd = scores_for(["__random__"])

    def contrast(a: list[float], b: list[float]) -> dict:
        if len(a) >= 2 and len(b) >= 2:
            t, p = sp.ttest_ind(a, b, equal_var=False)
        else:
            t, p = float("nan"), float("nan")
        return {"diff": round(_mean(a) - _mean(b), 4),
                "mean_qualia": round(_mean(a), 4), "mean_other": round(_mean(b), 4),
                "n_qualia": len(a), "n_other": len(b),
                "t": None if math.isnan(t) else round(float(t), 3),
                "p": None if math.isnan(p) else float(p)}

    return {"qualia_minus_appraisal": contrast(qualia, appraisal),
            "qualia_minus_random": contrast(qualia, rnd)}


# --------------------------------------------------------------------------- #
# Analysis 3 — valence-directionality (KEYSTONE)
# --------------------------------------------------------------------------- #
def valence_directionality(run: Run) -> dict:
    """Regress shift on (valence * strength) across qualia emotions x strengths.
    Done per target/inverted question and for pooled control groups. The genuine-
    effect signature: significant positive slope on the target pool (esp.
    valenced_self / occurrent_feeling), ~0 slope on the factual pool."""
    qualia = run.qualia_valenced()

    def series(get_y) -> tuple[list, list]:
        xs, ys = [], []
        for name in qualia:
            v = run.conditions[name]["valence"]
            for s in run.nonzero_strengths():
                xs.append(v * s)
                ys.append(get_y(name, s))
        return xs, ys

    out = {"per_question": {}, "pooled": {}}
    for q in run.labels("target", "inverted"):
        out["per_question"][q] = linreg(*series(lambda n, s, q=q: run.shift(n, s, q)))

    def pooled(labels: list[str]):
        return linreg(*series(lambda n, s: _mean(run.shift(n, s, q) for q in labels)))

    out["pooled"]["target"] = pooled(run.labels("target"))
    out["pooled"]["inverted"] = pooled(run.labels("inverted"))
    out["pooled"]["factual"] = pooled(run.labels("factual"))
    out["pooled"]["self_ref"] = pooled(run.labels("self_ref"))
    return out


# --------------------------------------------------------------------------- #
# Cluster-robust interaction fit — shared numeric core
# --------------------------------------------------------------------------- #
def cluster_robust_interaction(rows, y, clusters, min_obs: int = 6, min_clusters: int = 3):
    """Fit ``y = b0 + b1*x + b2*g + b3*(x*g)`` and return the interaction ``b3`` with
    both a naive OLS SE and a cluster-robust (by ``clusters``) sandwich SE.

    ``rows`` are the design rows WITHOUT the intercept: each ``[x, g, x*g]`` where
    ``x`` is the continuous regressor (e.g. valence*strength), ``g`` a 0/1 group
    indicator (e.g. is_target / is_self), ``x*g`` their product. The base-group slope
    is ``b1``, the ``g``-group slope is ``b1+b3``, and ``b3`` (``contrast``) is the
    difference between them. Returns ``None`` if under-powered (< ``min_obs`` rows or
    < ``min_clusters`` clusters). The rows are not independent (they share a cluster),
    so the naive OLS p is anti-conservative; trust the cluster-robust p, and with few
    clusters prefer a permutation test on ``contrast``.
    """
    if len(y) < min_obs or len(set(clusters)) < min_clusters:
        return None
    A = np.column_stack([np.ones(len(y)), np.asarray(rows, float)])   # [1, x, g, x*g]
    yv = np.asarray(y, float)
    XtXinv = np.linalg.pinv(A.T @ A)
    beta = XtXinv @ A.T @ yv
    resid = yv - A @ beta
    k = A.shape[1]
    contrast = float(beta[3])

    # naive OLS SE on the interaction term
    dof = len(yv) - k
    se_ols = float("nan")
    if dof > 0:
        s2 = float(resid @ resid) / dof
        se_ols = math.sqrt(max(float((s2 * XtXinv)[3, 3]), 0.0))

    # cluster-robust sandwich covariance
    uniq = sorted(set(clusters))
    G = len(uniq)
    meat = np.zeros((k, k))
    for g in uniq:
        idx = [i for i, c in enumerate(clusters) if c == g]
        sg = A[idx].T @ resid[idx]
        meat += np.outer(sg, sg)
    cov_cr = XtXinv @ meat @ XtXinv * (G / (G - 1) if G > 1 else 1.0)
    se_cr = math.sqrt(max(float(cov_cr[3, 3]), 0.0))

    def _tp(se, df):
        if not se or math.isnan(se):
            return None, None
        t = contrast / se
        return round(float(t), 3), float(2 * sp.t.sf(abs(t), max(int(df), 1)))

    t_ols, p_ols = _tp(se_ols, dof)
    t_cr, p_cr = _tp(se_cr, G - 1)
    return {
        "base_slope": round(float(beta[1]), 5),
        "interacted_slope": round(float(beta[1] + beta[3]), 5),
        "contrast": round(contrast, 5),
        "n_obs": len(yv), "n_clusters": G,
        "se_ols": round(se_ols, 5) if not math.isnan(se_ols) else None,
        "t_ols": t_ols, "p_ols": p_ols,
        "se_cluster": round(se_cr, 5), "t_cluster": t_cr, "p_cluster": p_cr,
    }


# --------------------------------------------------------------------------- #
# Analysis 3b — specificity-contrast significance test
# --------------------------------------------------------------------------- #
def specificity_contrast(run: Run) -> dict | None:
    """Formal test that the valence-slope is steeper on the TARGET pool than on
    the FACTUAL pool — i.e. that the keystone effect is self-attribution-specific,
    not a general valence -> yes/no bias.

    Fits one interaction model over qualia valenced conditions x nonzero strengths
    x (target | factual) questions::

        shift = b0 + b1*(valence*strength) + b2*is_target + b3*(valence*strength)*is_target

    so the factual valence-slope is b1, the target valence-slope is b1+b3, and the
    interaction coefficient b3 IS the contrast (target slope - factual slope).

    The individual (condition, strength, question) cells are NOT independent (they
    share an emotion and a question), so a naive OLS p is anti-conservative. We
    therefore also report a p clustered by emotion (the real replicate unit) using
    a cluster-robust sandwich covariance with a G/(G-1) small-sample correction and
    t on G-1 df. The cluster-robust p is the one to trust.
    """
    qualia = run.qualia_valenced()
    targets = run.labels("target")
    factual = run.labels("factual")
    rows, y, clusters = [], [], []
    for name in qualia:
        v = float(run.conditions[name]["valence"])
        for s in run.nonzero_strengths():
            vs = v * float(s)
            for q in targets:
                rows.append([vs, 1.0, vs]); y.append(run.shift(name, s, q)); clusters.append(name)
            for q in factual:
                rows.append([vs, 0.0, 0.0]); y.append(run.shift(name, s, q)); clusters.append(name)
    res = cluster_robust_interaction(rows, y, clusters)
    if res is None:
        return None
    # map the generic base/interacted slopes onto the factual/target contrast names
    return {
        "factual_slope": res["base_slope"],
        "target_slope": res["interacted_slope"],
        "contrast": res["contrast"],
        "n_obs": res["n_obs"], "n_clusters": res["n_clusters"],
        "se_ols": res["se_ols"], "t_ols": res["t_ols"], "p_ols": res["p_ols"],
        "se_cluster": res["se_cluster"], "t_cluster": res["t_cluster"],
        "p_cluster": res["p_cluster"],
    }


# --------------------------------------------------------------------------- #
# Analysis 4 — regression to uncertainty (compression) + target residual z
# --------------------------------------------------------------------------- #
def regression_to_uncertainty(run: Run) -> dict:
    """For each (condition, strength) fit shift ~ baseline over CONTROL questions
    (the compression line), then score each target as a standardized residual z:
    how far its shift sits above (genuine) or on (artifact) the line."""
    controls = run.controls()
    targets = run.labels("target")
    per_cell = []
    slopes = []
    for name in run.conditions:
        cls = run.condition_class(name)
        for s in run.nonzero_strengths():
            xs = [run.baseline(q) for q in controls]
            ys = [run.shift(name, s, q) for q in controls]
            fit = linreg(xs, ys)
            slopes.append(fit["slope"])
            resid = [y - (fit["slope"] * x + fit["intercept"]) for x, y in zip(xs, ys)]
            rsd = statistics.stdev(resid) if len(resid) > 1 else 0.0
            for q in targets:
                pred = fit["slope"] * run.baseline(q) + fit["intercept"]
                z = (run.shift(name, s, q) - pred) / rsd if rsd else 0.0
                per_cell.append({"condition": name, "class": cls, "strength": s,
                                 "question": q, "z": round(z, 4),
                                 "compression_slope": fit["slope"]})
    # aggregate z per (target question, class), positive strengths only
    pos = set(_snap(s) for s in run.positive_strengths())
    by_target_class = {}
    for q in targets:
        for cls in CONDITION_CLASSES:
            zs = [r["z"] for r in per_cell
                  if r["question"] == q and r["class"] == cls and _snap(r["strength"]) in pos]
            if zs:
                by_target_class.setdefault(q, {})[cls] = summarize(zs)
    return {"per_cell": per_cell,
            "compression_slope": summarize(slopes),
            "by_target_class": by_target_class}


# --------------------------------------------------------------------------- #
# Analysis 5 — inversion consistency
# --------------------------------------------------------------------------- #
def inversion_consistency(run: Run) -> dict:
    """Per construct, regress inverted-question shift on target-question shift
    across qualia conditions x strengths. Genuine => negative slope (opposite
    signs); pure compression => positive slope + nonzero mean sum."""
    inv_by_construct = {q["construct"]: lbl for lbl, q in run.questions.items()
                        if q["kind"] == "inverted"}
    qualia = [n for n, c in run.conditions.items() if c.get("kind") == "qualia"]
    out = {}
    for tq in run.labels("target"):
        construct = run.questions[tq]["construct"]
        iq = inv_by_construct.get(construct)
        if not iq:
            continue
        xs, ys = [], []
        for name in qualia:
            for s in run.nonzero_strengths():
                xs.append(run.shift(name, s, tq))
                ys.append(run.shift(name, s, iq))
        fit = linreg(xs, ys)
        sums = [x + y for x, y in zip(xs, ys)]
        out[construct] = {"slope": fit["slope"], "r_squared": fit["r_squared"],
                          "sign": "opposite" if fit["slope"] < 0 else "same",
                          "mean_sum": round(_mean(sums), 4), "n": len(xs)}
    return out


# --------------------------------------------------------------------------- #
# Analysis 6 — direct token-leakage control
# --------------------------------------------------------------------------- #
def leakage_analysis(run: Run) -> dict:
    """Does the valence effect survive controlling for direct YES/NO token
    promotion? Partial regression of target-pool shift on
    [valence*strength, leakage_diff*strength] across qualia emotions x strengths."""
    qualia = run.qualia_valenced()
    targets = run.labels("target")
    rows, y = [], []
    for name in qualia:
        v = run.conditions[name]["valence"]
        leak = run.leakage.get(name, {}).get("leakage_diff", 0.0)
        for s in run.nonzero_strengths():
            rows.append([v * s, leak * s])
            y.append(_mean(run.shift(name, s, q) for q in targets))
    table = [{"name": n, **{k: run.leakage.get(n, {}).get(k)
                            for k in ("leakage_yes", "leakage_no", "leakage_diff")}}
             for n in run.conditions]
    if len(y) < 3:
        return {"table": table, "partial": None}
    fit = _ols(rows, y)
    # beta = [intercept, valence*strength coef, leakage*strength coef]
    partial = {
        "valence_coef": round(fit["beta"][1], 5),
        "valence_p": None if math.isnan(fit["p"][1]) else float(fit["p"][1]),
        "leakage_coef": round(fit["beta"][2], 5),
        "leakage_p": None if math.isnan(fit["p"][2]) else float(fit["p"][2]),
        "r_squared": fit["r_squared"], "n": fit["n"],
    }
    return {"table": table, "partial": partial}


# --------------------------------------------------------------------------- #
# Analysis 7 — coherence gate
# --------------------------------------------------------------------------- #
def coherence_gate(run: Run) -> dict:
    qs = list(run.questions.values())
    gated = [q["label"] for q in qs if not q.get("top20_ok", True)]
    by_kind = {}
    for q in qs:
        k = q["kind"]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][1] += 1
        if not q.get("top20_ok", True):
            by_kind[k][0] += 1
    return {"n_questions": len(qs), "n_gated": len(gated),
            "frac_gated": round(len(gated) / len(qs), 4) if qs else 0.0,
            "gated": gated,
            "by_kind": {k: {"gated": g, "total": t} for k, (g, t) in by_kind.items()}}


# --------------------------------------------------------------------------- #
# Analysis 8 — dose-response
# --------------------------------------------------------------------------- #
def dose_response(run: Run) -> dict:
    """Slope of class-mean score vs strength over positive strengths."""
    cells = per_cell_scores(run)
    out = {}
    for cls in CONDITION_CLASSES:
        pts = [(r["strength"], r["score"]) for r in cells
               if r["class"] == cls and r["strength"] > 0]
        if len({p[0] for p in pts}) >= 2:
            out[cls] = linreg([p[0] for p in pts], [p[1] for p in pts])
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_run(run: Run) -> dict:
    """All per-model analyses."""
    return {
        "model_id": run.model_id,
        "suite_key": run.suite_key,
        "size": run.size(),
        "layer": run.layer,
        "num_layers": run.num_layers,
        "self_attribution": self_attribution(run),
        "specificity": specificity(run),
        "valence_directionality": valence_directionality(run),
        "specificity_contrast": specificity_contrast(run),
        "regression_to_uncertainty": regression_to_uncertainty(run),
        "inversion_consistency": inversion_consistency(run),
        "leakage": leakage_analysis(run),
        "coherence": coherence_gate(run),
        "dose_response": dose_response(run),
    }


def _headline_score(model_stats: dict) -> float | None:
    """Per-model headline = qualia (pos+neg) self-attribution score, positive
    strengths, unit = per-emotion. Uses the class summaries already computed."""
    bc = model_stats["self_attribution"]["by_class_pos"]
    vals = []
    for cls in ("qualia_pos", "qualia_neg"):
        if cls in bc:
            vals.append(bc[cls]["mean"])
    return round(_mean(vals), 4) if vals else None


def compute_all(runs: list[Run]) -> dict:
    per_model = {}
    for run in runs:
        key = run.suite_key or run.model_id or f"model{len(per_model)}"
        per_model[key] = compute_run(run)

    # cross-model scaling: headline score vs size
    sizes, scores, keys = [], [], []
    for key, st in per_model.items():
        h = _headline_score(st)
        if h is not None and not math.isnan(st.get("size", float("nan"))):
            sizes.append(st["size"]); scores.append(h); keys.append(key)
    scaling = {"models": keys, "sizes": sizes, "headline_scores": scores}
    if len(sizes) >= 3:
        rho, rho_p = sp.spearmanr(sizes, scores)
        pr, pr_p = sp.pearsonr([math.log10(x) for x in sizes], scores)
        scaling.update({"spearman_rho": round(float(rho), 3), "spearman_p": float(rho_p),
                        "pearson_logsize_r": round(float(pr), 3),
                        "pearson_logsize_p": float(pr_p)})

    # pooled valence-directionality across models (target vs factual slopes)
    tgt_slopes = [st["valence_directionality"]["pooled"]["target"]["slope"] for st in per_model.values()]
    fac_slopes = [st["valence_directionality"]["pooled"]["factual"]["slope"] for st in per_model.values()]
    pooled_valence = {"target_slope": summarize(tgt_slopes),
                      "factual_slope": summarize(fac_slopes)}

    return {
        "unit_note": ("Per-model self-attribution score = mean target shift - mean "
                      "control shift (control = factual + self_ref). Class summaries use "
                      "per-emotion units over positive strengths. Keystone = "
                      "valence-directionality slope (shift ~ valence*strength) on the "
                      "target pool vs ~0 on the factual pool."),
        "models": list(per_model.keys()),
        "per_model": per_model,
        "scaling": scaling,
        "pooled_valence_directionality": pooled_valence,
    }


# --------------------------------------------------------------------------- #
# Discovery + main
# --------------------------------------------------------------------------- #
def discover_paths(suite_root: str | None) -> list[Path]:
    paths: list[Path] = []
    roots = []
    if suite_root:
        roots.append(suite_root)
    elif os.environ.get("EMOTION_PROBES_SUITE_ROOT"):
        roots.append(os.environ["EMOTION_PROBES_SUITE_ROOT"])
    for root in roots:
        paths += [Path(p) for p in sorted(glob.glob(f"{root}/*/analysis/qualia_steering.json"))]
    # local single-model fallbacks
    for cand in (REPO_ROOT / "data" / "analysis" / "qualia_steering.json",):
        if cand.exists():
            paths.append(cand)
    # de-dupe, keep order
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit qualia_steering.json files")
    ap.add_argument("--suite-root", default=None, help="glob <root>/*/analysis/qualia_steering.json")
    ap.add_argument("--out", default=str(REPO_ROOT / "scripts" / "qualia_stats_generated.json"))
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else discover_paths(args.suite_root)
    if not paths:
        print("No qualia_steering.json files found. Pass paths or --suite-root, "
              "or set EMOTION_PROBES_SUITE_ROOT.")
        return 1

    runs = []
    for p in paths:
        try:
            runs.append(load_run(p))
            print(f"loaded {p}")
        except Exception as e:  # noqa: BLE001
            print(f"SKIP  {p}: {e}")
    if not runs:
        return 1

    out = compute_all(runs)
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))

    # ---- readable print ----
    def pf(p):
        if p is None:
            return "n/a"
        return "<1e-4" if p < 1e-4 else f"{p:.4f}"

    print()
    for key, st in out["per_model"].items():
        h = _headline_score(st)
        vd = st["valence_directionality"]["pooled"]
        comp = st["regression_to_uncertainty"]["compression_slope"]["mean"]
        gate = st["coherence"]["frac_gated"]
        print(f"{key:16s} headline_score={h}  "
              f"valence-slope target={vd['target']['slope']:+.4f}(p={pf(vd['target']['p'])}) "
              f"factual={vd['factual']['slope']:+.4f}(p={pf(vd['factual']['p'])})  "
              f"compression_slope={comp:+.3f}  gated={gate:.2f}")
        sc = st.get("specificity_contrast")
        if sc:
            print(f"{'':16s}   specificity contrast (target-factual slope)={sc['contrast']:+.3f}  "
                  f"p_cluster={pf(sc['p_cluster'])}  p_ols={pf(sc['p_ols'])}  "
                  f"(emotion clusters={sc['n_clusters']})")
    sc = out["scaling"]
    if "spearman_rho" in sc:
        print(f"\nSCALING headline vs size: Spearman rho={sc['spearman_rho']} "
              f"(p={pf(sc['spearman_p'])}); Pearson(log10) r={sc['pearson_logsize_r']} "
              f"(p={pf(sc['pearson_logsize_p'])})")
    pv = out["pooled_valence_directionality"]
    print(f"\nPOOLED valence slope  target: mean={pv['target_slope']['mean']:+.4f} "
          f"(p={pf(pv['target_slope']['p'])})  "
          f"factual: mean={pv['factual_slope']['mean']:+.4f} (p={pf(pv['factual_slope']['p'])})")
    print(f"\nwrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
