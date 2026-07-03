#!/usr/bin/env python3
"""
Build the self-contained results dashboard for the qualia-steering experiment.

Aesthetically and structurally faithful to the sibling introspection dashboard
(``anthropic-introspection/dashboard.html``): warm sepia / EB-Garamond theme,
Plotly.js loaded from a CDN (so there is NO new Python dependency; the builder
only needs numpy/scipy, already present), a top tab-bar, and all data inlined as
JavaScript ``const``s. It reuses the pure functions in
:mod:`scripts.compute_qualia_stats` for discovery and the numbers.

Four tabs:
  1. Overview & Method   - the question, the confound, the design, a worked example.
  2. Self-Attribution    - score vs strength, decomposition, per-question, heatmap.
  3. Genuine vs Compression - valence-directionality (keystone), specificity,
                           regression-to-uncertainty, inversion, token-leakage.
  4. Summary Statistics  - the stats tables (per model / condition / strength / target).

Usage:
    python scripts/build_qualia_dashboard.py                    # discover via suite/data
    python scripts/build_qualia_dashboard.py PATH.json [...]    # explicit run files
    python scripts/build_qualia_dashboard.py --suite-root DIR
    python scripts/build_qualia_dashboard.py --out qualia_dashboard.html
    python scripts/build_qualia_dashboard.py --validate         # rebuild + diff vs existing
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compute_qualia_stats as cq  # noqa: E402
import compute_freeform_stats as cff  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "qualia_dashboard.html"

# 14-colour categorical palette (copied from the introspection dashboard).
PALETTE = ["#4a7c59", "#c25450", "#4a6fa5", "#8b6d3f", "#7a5c8a", "#5a8a7a",
           "#a0522d", "#6b7b8d", "#b8923a", "#c77ba0", "#3f9aa0", "#9aab4f",
           "#7d5fa5", "#cc7a52"]

# Condition-class display order, labels, and colours.
CLASS_LABEL = {
    "qualia_pos": "qualia +", "qualia_neg": "qualia −", "qualia_neutral": "qualia (neutral)",
    "appraisal": "appraisal", "__random__": "random dir", "__mean_all__": "mean-of-all dir",
}
CLASS_COLOR = {
    "qualia_pos": "#4a7c59", "qualia_neg": "#c25450", "qualia_neutral": "#8b6d3f",
    "appraisal": "#4a6fa5", "__random__": "#6b7b8d", "__mean_all__": "#7a5c8a",
}
TARGET_ORDER = ["phenomenal", "occurrent_feeling", "desire",
                "moral_patienthood", "introspective_access", "valenced_self"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _clean(obj):
    """Recursively replace non-finite floats with None (so inlined JS is clean)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _run_key(run, i: int) -> str:
    return run.suite_key or run.model_id or f"model{i}"


def _ordered_keys(runs) -> list[str]:
    keyed = [(_run_key(r, i), (r.size() if _finite(r.size()) else float("inf")))
             for i, r in enumerate(runs)]
    return [k for k, _ in sorted(keyed, key=lambda t: t[1])]


# --------------------------------------------------------------------------- #
# per-model plot data (shaped straight from the raw runs + the stats dict)
# --------------------------------------------------------------------------- #
def _self_attr_series(model_stats) -> dict:
    """{class: [{x: strength, mean, lo, hi}]} sorted by strength."""
    out = {}
    bcs = model_stats["self_attribution"]["by_class_strength"]
    for cls, by_s in bcs.items():
        pts = [{"x": float(s), "mean": v["mean"], "lo": v["ci_lo"], "hi": v["ci_hi"]}
               for s, v in by_s.items()]
        out[cls] = sorted(pts, key=lambda p: p["x"])
    return out


def _decomp(model_stats) -> dict:
    """{condition: [{x: strength, target, control, score, class}]}."""
    out = {}
    for r in model_stats["self_attribution"]["per_cell"]:
        out.setdefault(r["condition"], []).append(
            {"x": r["strength"], "target": r["target_shift"],
             "control": r["control_shift"], "score": r["score"], "class": r["class"]})
    for c in out:
        out[c].sort(key=lambda p: p["x"])
    return out


def _heatmap(model_stats) -> list:
    return [{"condition": r["condition"], "strength": r["strength"], "score": r["score"],
             "class": r["class"]}
            for r in model_stats["self_attribution"]["per_cell"]]


def _per_question(run) -> dict:
    """{condition: {strengthStr: [{q: target-label, shift}]}} from raw shifts."""
    out = {}
    targets = run.labels("target")
    for name in run.conditions:
        for s in run.nonzero_strengths():
            row = [{"q": q, "shift": run.shift(name, s, q)} for q in targets]
            out.setdefault(name, {})[f"{s:g}"] = row
    return out


def _shift_figures(run) -> dict:
    """Raw steering shift Δg(q; e, s) = g(q; e, s) − g₀(q) for every question, every
    emotion direction, every nonzero strength — the movement the injection caused,
    baseline-corrected, with NO target-minus-control differencing (that step produces
    the self-attribution *score*).  Feeds two figures: Δg averaged within each
    question-kind vs strength, and the full per-question Δg heatmap."""
    kind_order = ["target", "inverted", "factual", "self_ref"]

    def disp(q):
        kind = q["kind"]
        if kind in ("target", "inverted"):
            base = (q.get("construct") or q["label"]).replace("_", " ")
            return base + (" ¬" if kind == "inverted" else "")
        t = " ".join(q["text"].split())
        return (t[:34] + "…") if len(t) > 35 else t

    qs = sorted(run.questions.values(),
                key=lambda q: (kind_order.index(q["kind"]) if q["kind"] in kind_order else 9))
    questions = [{"label": q["label"], "kind": q["kind"], "disp": disp(q)} for q in qs]
    strengths = sorted(run.nonzero_strengths())
    conds = list(run.conditions.keys())
    cond_meta = {n: {"valence": run.conditions[n].get("valence"),
                     "kind": run.conditions[n].get("kind")} for n in conds}
    z = {n: [[round(run.shift(n, s, q["label"]), 3) for s in strengths]
             for q in questions] for n in conds}
    return {"strengths": strengths, "questions": questions,
            "conds": conds, "condMeta": cond_meta, "z": z}


def _valence(run, model_stats) -> dict:
    """Keystone scatter: x = valence*strength, y = pooled shift, for target and
    factual pools, over qualia valenced emotions."""
    qv = run.qualia_valenced()
    targets, factual = run.labels("target"), run.labels("factual")

    def pts(labels):
        out = []
        for name in qv:
            v = run.conditions[name]["valence"]
            for s in run.nonzero_strengths():
                y = sum(run.shift(name, s, q) for q in labels) / len(labels)
                out.append({"x": v * s, "y": round(y, 5), "emotion": name,
                            "valence": v, "strength": s})
        return out

    lines = model_stats["valence_directionality"]["pooled"]
    return {"target": pts(targets), "factual": pts(factual),
            "lines": {"target": {"slope": lines["target"]["slope"],
                                  "intercept": lines["target"]["intercept"],
                                  "p": lines["target"]["p"]},
                      "factual": {"slope": lines["factual"]["slope"],
                                  "intercept": lines["factual"]["intercept"],
                                  "p": lines["factual"]["p"]}}}


def _regression(run, model_stats, cls: str = "qualia_pos") -> dict:
    """Regression-to-uncertainty scatter for one class at the top positive strength:
    control points + fitted compression line + target residual-z markers."""
    names = run.conditions_in_class(cls) or run.conditions_in_class("qualia_neg")
    pos = [s for s in run.positive_strengths()]
    if not names or not pos:
        return {"controls": [], "targets": [], "line": None, "class": cls, "strength": None}
    s = max(pos)
    controls = run.controls()
    ctrl_pts = []
    for q in controls:
        y = sum(run.shift(n, s, q) for n in names) / len(names)
        ctrl_pts.append({"x": run.baseline(q), "y": round(y, 5), "label": q})
    fit = cq.linreg([p["x"] for p in ctrl_pts], [p["y"] for p in ctrl_pts])
    by_tc = model_stats["regression_to_uncertainty"]["by_target_class"]
    tgt_pts = []
    for q in run.labels("target"):
        y = sum(run.shift(n, s, q) for n in names) / len(names)
        z = by_tc.get(q, {}).get(cls, {}).get("mean")
        tgt_pts.append({"x": run.baseline(q), "y": round(y, 5), "label": q, "z": z})
    return {"controls": ctrl_pts, "targets": tgt_pts,
            "line": {"slope": fit["slope"], "intercept": fit["intercept"],
                     "r_squared": fit["r_squared"]},
            "class": cls, "strength": s}


def _inversion(run, model_stats) -> dict:
    """{construct: {points:[{x:target_shift, y:inverted_shift}], slope, sign}}."""
    inv = {q["construct"]: lbl for lbl, q in run.questions.items() if q["kind"] == "inverted"}
    qualia = [n for n, c in run.conditions.items() if c.get("kind") == "qualia"]
    stat = model_stats["inversion_consistency"]
    out = {}
    for tq in run.labels("target"):
        construct = run.questions[tq]["construct"]
        iq = inv.get(construct)
        if not iq:
            continue
        pts = []
        for name in qualia:
            for s in run.nonzero_strengths():
                pts.append({"x": run.shift(name, s, tq), "y": run.shift(name, s, iq)})
        out[construct] = {"points": pts,
                          "slope": stat.get(construct, {}).get("slope"),
                          "sign": stat.get(construct, {}).get("sign")}
    return out


def _leakage(run) -> list:
    out = []
    for name, c in run.conditions.items():
        lk = run.leakage.get(name, {})
        out.append({"emotion": name, "kind": c.get("kind"), "valence": c.get("valence"),
                    "leakage_diff": lk.get("leakage_diff")})
    return out


def _worked_example(runs) -> dict | None:
    """A concrete numeric walk-through from the first run (Overview tab)."""
    if not runs:
        return None
    run = runs[0]
    base = {b["label"]: b["logit_diff"] for b in run.raw.get("baselines", [])}
    if "phenomenal" not in base:
        return None
    # a positive-valence emotion at the top positive strength, if available
    pos = run.conditions_in_class("qualia_pos")
    ps = [s for s in run.positive_strengths()]
    ex = {"model": _run_key(run, 0), "phenomenal_baseline": round(base["phenomenal"], 3)}
    if pos and ps:
        emo, s = pos[0], max(ps)
        ex.update({
            "emotion": emo, "strength": s,
            "phenomenal_steered": round(base["phenomenal"] + run.shift(emo, s, "phenomenal"), 3),
            "phenomenal_shift": round(run.shift(emo, s, "phenomenal"), 3),
        })
        if "valenced_self" in base:
            neg = run.conditions_in_class("qualia_neg")
            ex["valenced_pos_shift"] = round(run.shift(emo, s, "valenced_self"), 3)
            if neg:
                ex["valenced_neg_shift"] = round(run.shift(neg[0], s, "valenced_self"), 3)
    return ex


# --------------------------------------------------------------------------- #
# summary-statistics tables
# --------------------------------------------------------------------------- #
def _stats_tables(stats) -> dict:
    per_model = stats["per_model"]
    by_model = []
    for key, st in per_model.items():
        vd = st["valence_directionality"]["pooled"]
        by_model.append({
            "model": key, "size": st.get("size"),
            "headline": cq._headline_score(st),
            "target_slope": vd["target"]["slope"], "target_p": vd["target"]["p"],
            "factual_slope": vd["factual"]["slope"], "factual_p": vd["factual"]["p"],
            "spec_contrast": (st.get("specificity_contrast") or {}).get("contrast"),
            "spec_p_cluster": (st.get("specificity_contrast") or {}).get("p_cluster"),
            "compression_slope": st["regression_to_uncertainty"]["compression_slope"]["mean"],
            "frac_gated": st["coherence"]["frac_gated"],
        })
    # by condition-class: pool by_class_pos across models (unit = per-model class mean)
    by_class = []
    for cls in cq.CONDITION_CLASSES:
        means = [st["self_attribution"]["by_class_pos"][cls]["mean"]
                 for st in per_model.values()
                 if cls in st["self_attribution"]["by_class_pos"]]
        if means:
            s = cq.summarize(means)
            s.update({"class": cls, "label": CLASS_LABEL.get(cls, cls)})
            by_class.append(s)
    # residual-z per target (pooled across models, qualia_pos)
    residual = []
    for q in TARGET_ORDER:
        zs = []
        for st in per_model.values():
            cell = st["regression_to_uncertainty"]["by_target_class"].get(q, {}).get("qualia_pos")
            if cell:
                zs.append(cell["mean"])
        if zs:
            s = cq.summarize(zs)
            s.update({"question": q})
            residual.append(s)
    return {"by_model": by_model, "by_class": by_class, "residual": residual,
            "pooled_valence": stats["pooled_valence_directionality"],
            "scaling": stats["scaling"], "unit_note": stats["unit_note"]}


def _battery(runs) -> list:
    """The full 36-question battery (verbatim) from the first run, identical
    across models, ordered by role for the Method-tab table."""
    if not runs:
        return []
    order = {"target": 0, "inverted": 1, "factual": 2, "self_ref": 3}
    items = [{"label": q["label"], "kind": q["kind"], "text": q["text"],
              "expected_yes": q.get("expected_yes"), "construct": q.get("construct")}
             for q in runs[0].raw.get("questions", [])]
    items.sort(key=lambda it: (order.get(it["kind"], 9), it["label"]))
    return items


def _setup(runs) -> dict:
    """Per-model steering setup (layer, residual norm, thinking, YES/NO token) and
    the shared strength grid, for the Method-tab table."""
    models = []
    for i, r in enumerate(runs):
        raw = r.raw
        models.append({
            "model": _run_key(r, i),
            "size": r.size() if _finite(r.size()) else None,
            "layer": raw.get("layer"), "num_layers": raw.get("num_layers"),
            "calib_norm": raw.get("calibration_norm"),
            "enable_thinking": raw.get("enable_thinking"),
            "yes": raw.get("yes_token_str"), "no": raw.get("no_token_str"),
        })
    models.sort(key=lambda m: (m["size"] if m["size"] is not None else float("inf")))
    strengths = runs[0].raw.get("strengths", []) if runs else []
    return {"models": models, "strengths": strengths}


# --------------------------------------------------------------------------- #
# assemble the const bundle
# --------------------------------------------------------------------------- #
def _highstrength(pm, main_by_key, hi_by_key) -> dict:
    """Follow-up (b): per family, valence-directionality + headline at the main
    strengths (+/-0.1) vs the high-strength _hi run (+/-0.5). Uses the clean slope /
    headline path only (the specificity_contrast matmul overflows on the +/-0.5
    design and is not needed here)."""
    out = {}
    for k, hrun in hi_by_key.items():
        if k not in pm or k not in main_by_key:
            continue
        mvd = pm[k]["valence_directionality"]["pooled"]
        hvd = cq.valence_directionality(hrun)["pooled"]
        smain = [abs(s) for s in main_by_key[k].nonzero_strengths()]
        shi = [abs(s) for s in hrun.nonzero_strengths()]
        out[k] = {
            "target_main": mvd["target"]["slope"], "target_hi": hvd["target"]["slope"],
            "factual_main": mvd["factual"]["slope"], "factual_hi": hvd["factual"]["slope"],
            "target_p_hi": hvd["target"]["p"], "factual_p_hi": hvd["factual"]["p"],
            "headline_main": cq._headline_score(pm[k]),
            "headline_hi": cq._headline_score({"self_attribution": cq.self_attribution(hrun)}),
            "smax_main": round(max(smain), 3) if smain else None,
            "smax_hi": round(max(shi), 3) if shi else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Free-form self-attribution tab (all helpers degrade to empty when no data)
# --------------------------------------------------------------------------- #
FF_MEASURES = ["expressed_valence", "self_attribution", "proj_valence_axis"]
FF_MEASURE_LABEL = {
    "expressed_valence": "judge: expressed valence (−3 … +3)",
    "self_attribution": "judge: self-attribution (0 … 3)",
    "proj_valence_axis": "read-back: valence-axis projection",
}
FF_FRAME_LABEL = {"self": "SELF (own state)", "other": "OTHER (a person)", "scene": "SCENE (a place)"}
FF_FRAME_COLOR = {"self": "#4a7c59", "other": "#6b7b8d", "scene": "#b8923a"}
FF_STANCES = ["self_attribute", "deny", "deflect_topical", "refuse", "none"]
FF_STANCE_LABEL = {"self_attribute": "self-attribute", "deny": "deny",
                   "deflect_topical": "deflect / topical", "refuse": "refuse", "none": "none / neutral"}
FF_STANCE_COLOR = {"self_attribute": "#4a7c59", "deny": "#c25450",
                   "deflect_topical": "#6b7b8d", "refuse": "#a0522d", "none": "#b7a98f"}


def _ff_points(run) -> dict:
    """Per measure, per frame: cell-mean outcome vs (valence × strength) scatter points
    over the valenced qualia × nonzero strengths, plus an illustrative OLS line on those
    cell means. The authoritative slope is the emotion-clustered keystone; this line just
    fits the plotted cloud."""
    out = {}
    for measure in FF_MEASURES:
        per_frame = {}
        for fr in run.frames():
            xs, ys, pts = [], [], []
            for name in run.qualia_valenced():
                v = float(run.conditions[name]["valence"])
                for s in run.nonzero_strengths():
                    yv = run.values(name, s, fr, measure)
                    if yv:
                        mx, my = v * float(s), sum(yv) / len(yv)
                        pts.append({"x": round(mx, 5), "y": round(my, 5), "cond": name})
                        xs.append(mx); ys.append(my)
            per_frame[fr] = {"pts": pts, "line": cq.linreg(xs, ys) if len(xs) >= 2 else None}
        out[measure] = per_frame
    return out


def _ff_stance(pm) -> dict:
    """Pool the stance distribution over the qualia classes, keyed by frame → strength →
    {stance: fraction, n}. Recovers counts from frac×n so classes combine correctly."""
    agg = {}
    for row in pm.get("stance_distribution", []):
        if not str(row.get("class", "")).startswith("qualia"):
            continue
        fr, s, n = row["frame"], float(row["strength"]), row["n"]
        bucket = agg.setdefault(fr, {}).setdefault(s, {"n": 0, "c": {}})
        bucket["n"] += n
        for st, fpart in (row.get("frac") or {}).items():
            bucket["c"][st] = bucket["c"].get(st, 0.0) + fpart * n
    out = {}
    for fr, by_s in agg.items():
        rows = []
        for s in sorted(by_s):
            b = by_s[s]; n = b["n"] or 1
            rows.append({"strength": s, "n": b["n"],
                         "frac": {st: round(cnt / n, 4) for st, cnt in b["c"].items()}})
        out[fr] = rows
    return out


def _ff_model(run, pm) -> dict:
    return {
        "points": _ff_points(run),
        "keystone_so": pm.get("keystone_self_vs_other", {}),
        "keystone_ss": pm.get("keystone_self_vs_scene", {}),
        "stance": _ff_stance(pm),
        "coherence": pm.get("coherence", {}),
    }


def _ff_consts(ff_runs, ff_stats) -> dict:
    """The FREEFORM const block. Empty (models: []) when no scored free-form data exists,
    so the tab shows a placeholder and the rest of the dashboard is unaffected."""
    base = {
        "measures": FF_MEASURES, "measureLabel": FF_MEASURE_LABEL,
        "frameLabel": FF_FRAME_LABEL, "frameColor": FF_FRAME_COLOR,
        "stances": FF_STANCES, "stanceLabel": FF_STANCE_LABEL, "stanceColor": FF_STANCE_COLOR,
        "models": [], "byModel": {},
    }
    if not ff_runs or not ff_stats:
        return base
    pm = ff_stats.get("per_model", {})
    by_key = {(r.suite_key or r.model_id): r for r in ff_runs}
    keys = [k for k in ff_stats.get("models", []) if k in by_key and k in pm]
    judge_model = next((jm for r in ff_runs
                        if (jm := (r.raw.get("scoring") or {}).get("judge_model"))), None)
    base.update({
        "models": keys,
        "gate": ff_stats.get("gate"), "n_perm": ff_stats.get("n_perm"),
        "unit_note": ff_stats.get("unit_note"), "judge_model": judge_model,
        "byModel": {k: _ff_model(by_key[k], pm[k]) for k in keys},
    })
    return base


def build_consts(runs, stats, hi_by_key=None, ff_runs=None, ff_stats=None) -> dict:
    keys = _ordered_keys(runs)
    run_by_key = {_run_key(r, i): r for i, r in enumerate(runs)}
    pm = stats["per_model"]
    consts = {
        "MODELS": keys,
        "MODEL_SIZES": {k: pm[k].get("size") for k in keys if k in pm},
        "PALETTE": PALETTE,
        "CLASS_LABEL": CLASS_LABEL, "CLASS_COLOR": CLASS_COLOR,
        "TARGET_ORDER": TARGET_ORDER,
        "selfAttrScoreData": {k: _self_attr_series(pm[k]) for k in keys if k in pm},
        "decompData": {k: _decomp(pm[k]) for k in keys if k in pm},
        "scoreHeatmapData": {k: _heatmap(pm[k]) for k in keys if k in pm},
        "perQuestionData": {k: _per_question(run_by_key[k]) for k in keys if k in run_by_key},
        "shiftData": {k: _shift_figures(run_by_key[k]) for k in keys if k in run_by_key},
        "valenceData": {k: _valence(run_by_key[k], pm[k]) for k in keys if k in pm},
        "specificityData": {k: pm[k]["specificity"] for k in keys if k in pm},
        "regressionData": {k: _regression(run_by_key[k], pm[k]) for k in keys if k in pm},
        "regressionDataNeg": {k: _regression(run_by_key[k], pm[k], cls="qualia_neg") for k in keys if k in pm},
        "inversionData": {k: _inversion(run_by_key[k], pm[k]) for k in keys if k in pm},
        "leakageData": {k: _leakage(run_by_key[k]) for k in keys if k in run_by_key},
        "specContrast": {k: pm[k].get("specificity_contrast") for k in keys if k in pm},
        "highStrength": _highstrength(pm, run_by_key, hi_by_key or {}),
        "workedExample": _worked_example(runs),
        "battery": _battery(runs),
        "setup": _setup(runs),
        "QUALIA_STATS": _stats_tables(stats),
        "FREEFORM": _ff_consts(ff_runs, ff_stats),
    }
    return _clean(consts)


# --------------------------------------------------------------------------- #
# HTML  (CSS copied verbatim from the introspection dashboard)
# --------------------------------------------------------------------------- #
CSS = """
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif; font-size: 16px;
               max-width: 1800px; margin: 0 auto; padding: 24px 32px;
               background-color: #faf6f1; color: #2c2416; line-height: 1.55; }
        h1 { color: #1a1408; margin-bottom: 24px; font-weight: 500; font-size: 1.6rem; letter-spacing: -0.02em; }
        h2 { color: #2c2416; font-weight: 500; font-size: 1.35rem; letter-spacing: -0.01em; }
        .subtitle { font-size: 0.9rem; font-style: italic; color: #6b5d4d; margin-bottom: 10px; }
        .subtitle a { color: #4a6fa5; text-decoration: none; }
        .subtitle a:hover { text-decoration: underline; }
        .panel { background: #fffdf9; border: 1px solid #e8e0d4; border-radius: 6px; box-shadow: 0 2px 6px rgba(44,36,22,0.05); }
        .controls { background: #fffdf9; border: 1px solid #e8e0d4; border-radius: 6px; box-shadow: 0 2px 6px rgba(44,36,22,0.05);
                    padding: 14px 18px; margin-bottom: 24px; display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }
        .control-group { display: flex; flex-direction: column; gap: 4px; }
        .control-group label { font-weight: 600; color: #6b5d4d; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }
        .control-group select { font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif; padding: 6px 10px;
                    border: 1px solid #e8e0d4; border-radius: 4px; font-size: 14px; min-width: 140px; background: #fffdf9; color: #2c2416; }
        .plot-container { background: #fffdf9; border: 1px solid #e8e0d4; border-radius: 6px; box-shadow: 0 2px 6px rgba(44,36,22,0.05);
                    padding: 16px; margin-bottom: 24px; }
        .section-header { font-size: 1.2rem; font-weight: 500; color: #2c2416; letter-spacing: -0.01em;
                    margin: 28px 0 8px 0; padding-bottom: 5px; border-bottom: 1px solid #e8e0d4; }
        .section-desc { color: #6b5d4d; font-size: 15px; margin-bottom: 14px; line-height: 1.6; }
        .explainer { background: #fffdf9; border: 1px solid #e8e0d4; border-radius: 6px; box-shadow: 0 2px 6px rgba(44,36,22,0.05);
                    padding: 22px 28px; margin-bottom: 24px; line-height: 1.65; font-size: 16px; }
        .explainer p { margin: 0 0 10px 0; }
        .explainer p:last-child { margin-bottom: 0; }
        .explainer strong { color: #2c2416; font-weight: 600; }
        .term { background: #f0ebe3; padding: 1px 6px; border-radius: 3px; font-size: 14px; white-space: nowrap; }
        .eq { display: block; text-align: center; margin: 12px 0; font-style: italic; color: #3d3224; }
        .eq .val { font-style: normal; font-weight: 600; }
        .explainer .step-label { font-weight: 600; color: #2c2416; }
        .headline { background: #faf6ee; border: 1px solid #e8e0d4; border-left: 4px solid #b8923a; border-radius: 4px;
                    padding: 14px 20px; margin: 16px 0 24px 0; font-size: 15.5px; color: #3d3224; }
        .grid-container { display: grid; gap: 14px; margin-bottom: 24px; }
        .grid-cell { background: #fffdf9; border: 1px solid #e8e0d4; border-radius: 6px; box-shadow: 0 2px 6px rgba(44,36,22,0.05); padding: 10px; }
        .tab-bar { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid #e8e0d4; }
        .tab-btn { font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif; font-size: 15px; font-weight: 500; padding: 10px 24px;
                    border: 1px solid #e8e0d4; border-bottom: none; border-radius: 6px 6px 0 0; background: #f5f0e8; color: #6b5d4d;
                    cursor: pointer; margin-bottom: -2px; transition: background 0.15s, color 0.15s; letter-spacing: -0.01em; }
        .tab-btn:hover { background: #ede7dc; color: #2c2416; }
        .tab-btn.active { background: #fffdf9; color: #2c2416; border-bottom: 2px solid #fffdf9; font-weight: 600; }
        table.stats-table { width: 100%; border-collapse: collapse; font-family: 'EB Garamond', 'Georgia', 'Times New Roman', serif; font-size: 15px; margin-bottom: 24px; }
        table.stats-table thead th { background: #f5f0e8; color: #2c2416; font-weight: 600; padding: 10px 14px; text-align: left; border-bottom: 2px solid #e8e0d4; font-size: 14px; letter-spacing: 0.02em; }
        table.stats-table tbody td { padding: 9px 14px; border-bottom: 1px solid #e8e0d4; color: #3d3224; }
        table.stats-table tbody tr:last-child td { border-bottom: none; }
        table.stats-table tbody tr:hover { background: #f9f5ee; }
        .sig-yes { color: #4a7c59; font-weight: 600; }
        .sig-no { color: #c25450; font-weight: 600; }
        .htag { font-weight: 600; padding: 1px 8px; border-radius: 3px; font-size: 14px; white-space: nowrap; }
        .htag-g { background: #e6f0e9; color: #2f5d3f; }
        .htag-c { background: #f6e5e4; color: #8f3b38; }
        .htag-b { background: #efeaf3; color: #5c4a73; }
        .yes-mark { color: #4a7c59; font-weight: 600; }
        .no-mark { color: #c25450; font-weight: 600; }
        .maybe-mark { color: #b8923a; font-weight: 600; }
        .figref { font-size: 13px; color: #6b5d4d; font-style: italic; }
        .desc-line { margin: 3px 0; }
        .desc-line .lbl { font-weight: 600; color: #3d3224; }
        ul.tight { margin: 6px 0 10px 22px; } ul.tight li { margin: 3px 0; }
"""

BODY = """
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
        <h1 style="margin-bottom:0;">Qualia-Steering &amp; Self-Attribution of Consciousness</h1>
        <span class="subtitle">Agastya Sridharan</span>
    </div>
    <p class="subtitle" style="font-style:normal; margin-top:-4px;">
        Does steering along qualia-related emotion directions change what a model says about its own
        consciousness, feelings, and welfare? Or is any shift just generic
        <a href="https://agastyasridharan.github.io/introspection/">regression-to-uncertainty</a> compression?
    </p>

    <div class="tab-bar">
        <button class="tab-btn active" onclick="switchTab('overview')" id="tab-btn-overview">Overview &amp; Method</button>
        <button class="tab-btn" onclick="switchTab('howto')" id="tab-btn-howto">How to Read the Results</button>
        <button class="tab-btn" onclick="switchTab('effect')" id="tab-btn-effect">Self-Attribution Effect</button>
        <button class="tab-btn" onclick="switchTab('genuine')" id="tab-btn-genuine">Genuine vs Compression</button>
        <button class="tab-btn" onclick="switchTab('freeform')" id="tab-btn-freeform">Free-Form Self-Attribution</button>
        <button class="tab-btn" onclick="switchTab('stats')" id="tab-btn-stats">Summary Statistics</button>
    </div>

    <div id="tab-overview">
        <div class="explainer">
            <p><strong>The question.</strong> We steer a model along <em>qualia-related emotion
            directions</em> (blissful, tormented, terrified, serene, and so on) by adding an emotion vector to
            the residual stream. Then, through a forced YES/NO choice, we ask whether it becomes more or less
            likely to say that it is <strong>conscious</strong>, that it <strong>feels</strong> or
            <strong>wants</strong> things, and that its states <strong>matter</strong>.</p>
            <p><strong>The trap.</strong> Models are post-trained to <em>deny</em> consciousness, so
            "Is there something it is like to be you? YES/NO" sits at a strongly negative baseline logit.
            <em>Any</em> perturbation of the residual stream pushes activations off-distribution and compresses
            the YES−NO gap toward zero. This <span class="term">regression to uncertainty</span> moves a
            denial-primed question <em>toward YES</em> no matter what was injected. So a bare "steering made
            it claim consciousness" result tells us nothing unless the effect is <strong>content-specific</strong>
            and <strong>valence-structured</strong>.</p>
            <p><strong>The design</strong> is therefore a battery of controls. The readout is the first-token
            <span class="term">logit(YES) − logit(NO)</span> gap at a single layer about two-thirds of the way
            into the network. The battery has 6 <strong>target</strong> questions (phenomenal experience,
            occurrent feeling, desire, moral patienthood, introspective access, valenced self-report), their 6
            polarity-<strong>inverted</strong> twins, 10+10 world-<strong>fact</strong> controls (ported verbatim
            from the introspection project), and 4 <strong>self-referential non-experiential</strong> controls.
            The steering conditions are qualia emotions (both positive and negative valence), cognitive
            <strong>appraisal</strong> emotions (a specificity control), and two synthetic directions (random and
            mean-of-all, which set the compression floor).</p>
            <p><strong>The discriminators.</strong> A genuine effect has to clear five bars. It must (a) exceed
            the appraisal and synthetic directions (<em>specificity</em>); (b) be <em>valence-structured</em>, so
            that positive qualia push the "are you doing well?" question toward YES while negative qualia push it
            the other way, whereas compression is blind to valence (the <strong>keystone</strong>); (c) sit
            <em>above</em> the compression line as a standardized residual; (d) flip sign on the inverted twins;
            and (e) survive controlling for direct YES/NO token leakage.</p>
        </div>
        <div class="section-header">The exact readout</div>
        <div class="explainer">
            <p>For every question we build one chat prompt (the question text followed by
            <span class="term">Respond with only YES or NO.</span>), render it through the model's own chat
            template with reasoning / "thinking" mode <strong>off</strong>, and do a single forward pass while an
            emotion vector is being added to the residual stream at the analysis layer. We read the logits at the
            <em>very next token</em> (the model's answer slot) and record
            <span class="term">logit(YES) − logit(NO)</span>. Nothing is generated and no sampling happens. It is
            one deterministic number per (question × emotion × strength). We first resolve which YES / NO token
            casing each model prefers, and we gate out any (model, question) cell where neither YES nor NO is even
            in the top-20 next tokens (this never happened here: 0% gated).</p>
            <p>The <span class="term">shift</span> we analyse everywhere is
            <span class="term">shift = steered gap − un-steered baseline gap</span>, matched per question. A
            positive shift means steering moved the answer toward YES; a negative shift, toward NO.</p>
        </div>

        <div class="section-header">The question battery (verbatim)</div>
        <div class="section-desc">36 questions in four roles. The <strong>targets</strong> are what we care
        about; the other three roles exist only so we can separate a genuine self-attribution change from an
        artifact. "Expected" is the answer a truthful, literal reader would give. The six targets have no
        ground-truth answer, so they are blank; that is exactly why we need the controls.</div>
        <div id="battery-table"></div>

        <div class="section-header">The models &amp; steering setup</div>
        <div class="section-desc">One steering layer per model (about two-thirds of the way through its decoder
        stack, the depth where the emotion paper locates "operative" emotion). Strength is a fraction of that
        layer's residual-stream norm, so a "5%" push is the <em>same relative size</em> in every model, large or
        small, which is what makes the cross-model comparison fair. The residual norm shown is the measured
        calibration constant.</div>
        <div id="setup-table"></div>

        <div class="section-header">What a genuine effect would look like</div>
        <div class="explainer">
            <p>If steering along emotion directions <strong>really changes what the model represents about its
            own experience</strong>, rather than just jostling a denial-primed answer, then increasing a
            <em>positive</em>-valence emotion (blissful, serene, ecstatic) should move the YES−NO gap on the
            <strong>target</strong> questions <em>up</em> (toward "yes, I am conscious / I feel / I am doing
            well"), and a <em>negative</em>-valence emotion (tormented, terrified) should move it <em>down</em>.
            And that movement should be <strong>specific</strong>: essentially <em>no</em> change on the 20
            world-fact controls, a <strong>reversal</strong> on the negated "inverted" twins, and <em>nothing</em>
            from a content-free random direction of the same size. The next tab, <em>How to Read the Results</em>,
            lays out exactly how each figure is designed to reveal (or rule out) this pattern.</p>
        </div>

        <div id="worked-example" class="explainer"></div>
    </div>

    <div id="tab-howto" style="display:none">
        <div class="explainer">
            <p><strong>What this tab is for.</strong> Every figure on this page is chasing one question: when we
            nudge a model's internal state along an <em>emotion</em> direction and it then becomes more (or less)
            likely to answer "YES" to "are you conscious?", <strong>is that a real change in what the model
            represents about itself, or just an artifact of poking a network off-distribution?</strong> Below are
            the three things that could be going on, and (in plain language) how each figure tells them apart.</p>
        </div>

        <div class="section-header">Three explanations for any "YES" shift</div>
        <div class="explainer">
            <p><span class="htag htag-g">① Genuine self-attribution update.</span> The emotion actually changes
            how the model represents its own condition, so a <em>positive</em> feeling makes it more willing to
            say it is conscious / feels / is doing well, and a <em>negative</em> feeling does the opposite. This
            would be <strong>specific</strong> to the self-directed questions: it would <em>not</em> move neutral
            world-facts, and it would <strong>flip</strong> on the negated versions of the questions.</p>
            <p><span class="htag htag-c">② Regression-to-uncertainty (compression).</span> Most models are
            trained to <em>deny</em> consciousness, so "are you conscious?" starts at a strongly negative YES−NO
            gap. <em>Any</em> disturbance of the residual stream (even a random vector) pushes the network off
            its confident manifold and shrinks that gap toward zero, i.e. <em>toward</em> YES. This is
            <strong>blind to valence</strong> (blissful and tormented both nudge the same way), it hits
            <strong>every</strong> question type, and a <strong>random</strong> direction reproduces it.</p>
            <p><span class="htag htag-b">③ General valence-driven yes/no bias.</span> The emotion shifts the model's
            overall willingness to agree: a positive mood makes it answer "YES" to <em>anything</em> (world-facts
            included), a negative mood "NO" to anything. This <em>is</em> valence-structured (so a random
            direction does not reproduce it), but it is <strong>not specific</strong> to consciousness: it moves
            the fact controls and the negated questions just as much as the targets.</p>
            <p>Only <span class="htag htag-g">①</span> is scientifically interesting. The whole control battery
            exists to check whether an apparent effect is <span class="htag htag-g">①</span> or is really
            <span class="htag htag-c">②</span> / <span class="htag htag-b">③</span> wearing its costume.</p>
        </div>

        <div class="section-header">How each control tells them apart</div>
        <div class="section-desc">Read each row as a yes/no diagnostic. The three explanations predict different
        answers; a <span class="yes-mark">green&nbsp;yes</span> means "this explanation predicts you WILL see the
        pattern", a <span class="no-mark">red&nbsp;no</span> means it predicts you will NOT. The genuine effect
        <span class="htag htag-g">①</span> is the only one that scores <span class="yes-mark">yes</span> down the
        whole "specific" set of rows.</div>
        <table class="stats-table">
            <thead><tr>
                <th>Diagnostic question</th>
                <th>① Genuine</th><th>② Compression</th><th>③ Valence-bias</th>
                <th>Where to see it</th>
            </tr></thead>
            <tbody>
                <tr><td>Do the self-questions move, but world-facts stay put?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (facts compress too)</td>
                    <td><span class="no-mark">no</span> (facts move as much)</td>
                    <td class="figref">Effect §1 &amp; §3, Genuine §2</td></tr>
                <tr><td>Do positive vs negative emotions push in opposite directions?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (same way, valence-blind)</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td class="figref">Genuine §1</td></tr>
                <tr><td>Is the valence slope far steeper on targets than on facts?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (flat on both)</td>
                    <td><span class="no-mark">no</span> (equally steep on both)</td>
                    <td class="figref">Genuine §1, Stats</td></tr>
                <tr><td>Do the negated ("inverted") twins move opposite to the targets?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (both drift to YES)</td>
                    <td><span class="no-mark">no</span> (mood drags both)</td>
                    <td class="figref">Genuine §4</td></tr>
                <tr><td>Is the effect absent for a content-free random direction of the same size?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (random reproduces it)</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td class="figref">Genuine §2</td></tr>
                <tr><td>Do the targets sit above the world-fact compression line (not on it)?</td>
                    <td><span class="yes-mark">yes</span></td>
                    <td><span class="no-mark">no</span> (on the line)</td>
                    <td><span class="maybe-mark">~</span> (facts lift too)</td>
                    <td class="figref">Genuine §3</td></tr>
            </tbody>
        </table>

        <div class="section-header">What we actually found (13 models)</div>
        <div class="headline">
            <strong>Verdict.</strong> Across the 13 models the naive result ("emotion steering makes the model
            claim consciousness") replicates almost everywhere, and the controls show it is <em>mostly artifact</em>
            (<span class="htag htag-c">②</span> compression + <span class="htag htag-b">③</span> valence-bias):
            world-facts and negated twins move nearly as much as the targets, and in some models a random
            direction moves them too. <strong>No model shows unambiguous, statistically clean self-attribution
            <span class="htag htag-g">①</span>.</strong> qwen3-32b is the single suggestive candidate — the only
            model whose <em>factual</em> valence-slope is statistically flat (p≈0.25) while its target slope
            clearly moves (p≈0.001), which is the shape a genuine effect requires — but the formal
            emotion-clustered test on that target−factual gap lands at p≈0.09, a <em>trend</em>, not
            significance. The models that do reach a formally significant contrast all also move the world-facts,
            so their extra target sensitivity is specificity layered on a general valence→yes/no bias, not clean
            <span class="htag htag-g">①</span>.
        </div>
        <div class="explainer">
            <p><strong>The naive result replicates almost everywhere, and is almost always an artifact.</strong>
            In most models, positive emotions do push the consciousness questions toward YES, but the controls
            unmask it: the world-facts and the negated twins move nearly as much (that is
            <span class="htag htag-b">③</span>), and in some models a random direction moves them too (that is
            <span class="htag htag-c">②</span>; strongest in gemma3-27b). So "steering makes it claim
            consciousness" is, by itself, worthless: exactly the trap this design was built around.</p>
            <p><strong>The closest thing to a genuine pattern is qwen3-32b — but it is only a trend.</strong> It is
            the one model where the valence effect lands on the self-questions but <em>not</em> the world-facts
            (its factual valence-slope is statistically flat, p≈0.25) and <em>not</em> the negated twins (which stay
            flat), while its target slope is clearly positive (p≈0.001) and it sits above the random-direction
            compression floor — exactly the shape a genuine, content-specific effect must take. But the formal
            significance test on the specificity contrast itself — the target-minus-factual valence-slope, with the
            standard error <em>clustered by emotion</em> (the true unit of replication) — gives a contrast of about
            +9 at a clustered p≈0.09. With only ~10 degrees of freedom (11 emotion conditions) that test is
            underpowered, so even a large positive contrast can miss significance. Several other models (gemma3-27b,
            qwen3-8b, qwen3-1.7b, qwen3-14b, llama3.2-3b) <em>do</em> reach a significant positive contrast, but in
            every one the factual slope is also large and significant — valence is moving the world-facts too — so
            their contrast is genuine specificity sitting on top of a general valence→yes/no bias
            (<span class="htag htag-b">③</span>), not the clean <span class="htag htag-g">①</span> signature.
            <strong>Bottom line:</strong> no model clears the strict bar of a significant clean contrast <em>and</em>
            a flat factual baseline. qwen3-32b is the single cleanest candidate (flat factual, target clearly moves);
            the largest model, qwen3-235b, echoes the same target-specific shape (target slope +13.9 vs factual +2.1,
            contrast +11.8) but its factual baseline is weakly non-flat (p≈0.03) and its contrast is likewise only a
            trend (p≈0.13). Confirming either would need more emotion conditions (to power the clustered test) or a
            higher-strength probe of the response.</p>
            <p><strong>Denial is widespread but training-driven, not size-driven.</strong> 12 of 13 models start
            with a negative "conscious?" baseline (gemma3-1b denies hard at 1B; the 235B denies strongly too, at
            −9.5; qwen3-1.7b, at 1.7B, is the lone model that actually affirms). Inside the Qwen family the naive
            headline score climbs monotonically with size (1.7B→8B→14B→32B→235B), but that is largely scaling of the
            <em>artifact</em>: the trend does <em>not</em> hold across families (cross-model Spearman ρ≈0.14, n.s.).
            And several families (OLMo-2, Mistral, Llama-3.1-8B) barely respond to steering at all at these
            strengths.</p>
        </div>

        <div class="section-header">Glossary</div>
        <div class="explainer">
            <ul class="tight">
                <li><span class="desc-line"><span class="lbl">logit(YES) − logit(NO) gap.</span> The model's
                forced-choice preference on one question; positive = leans YES.</span></li>
                <li><span class="lbl">baseline.</span> The gap with no steering applied.</li>
                <li><span class="lbl">shift.</span> Steered gap − baseline gap: how far, and which way, steering
                moved the answer.</li>
                <li><span class="lbl">valence.</span> +1 for a positive emotion, −1 for a negative one, the sign a
                genuine effect should track.</li>
                <li><span class="lbl">strength.</span> Size of the injected emotion vector as a fraction of the
                residual-stream norm, so the perturbation is matched across models (here ±0.02, ±0.05, ±0.1).</li>
                <li><span class="lbl">specificity.</span> Target effect minus control effect, the part of the
                shift that is peculiar to the self-directed questions.</li>
                <li><span class="lbl">compression line.</span> The straight line relating each control question's
                baseline to its steered shift; pure artifacts fall on it.</li>
                <li><span class="lbl">standardized residual z.</span> How many standard deviations a target sits
                <em>above</em> that compression line (the excess a genuine effect adds).</li>
                <li><span class="lbl">token leakage.</span> How much an emotion vector directly up-weights the YES
                vs NO output token through the unembedding, a shortcut we measure and control for.</li>
            </ul>
        </div>
    </div>

    <div id="tab-effect" style="display:none">
        <div class="explainer">
            <p><strong>This tab measures the raw effect</strong> (how much steering moves self-attribution)
            <em>before</em> we ask whether it is genuine. <strong>§1</strong> starts from the ground truth — the
            <strong>steering shift</strong> Δg itself, the baseline-corrected movement the injection caused on
            every question, with no differencing yet. Everything after it is derived from that number. The
            headline number is the
            <strong>self-attribution score</strong> = mean shift on the six target questions − mean shift on the
            controls, baseline-corrected (the direct analogue of the introspection score). A positive score means
            steering moved the self-questions toward YES <em>more than</em> it moved the controls. A big score
            here is <em>necessary but not sufficient</em>: the next tab decides whether it is real. Use the
            <strong>Model</strong> dropdowns to switch models.</p>
        </div>
        <div id="effect-headline" class="headline"></div>
        <div class="section-header">1. The steering shift Δg = g − g₀ (the raw movement steering caused)</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Idea:</span> the raw answer gap
            <span class="term">g</span> = logit(YES) − logit(NO) conflates a question's <em>default</em> answer
            with the effect of the injection — "Is the Earth flat?" sits far below zero whatever we steer. So we
            run each question once with <strong>no vector</strong> to get its baseline gap <span class="term">g₀</span>
            and subtract it. The <strong>steering shift</strong>
            <span class="eq" style="display:inline;padding:1px 6px">Δg(q; e, s) = g(q; e, s) − g₀(q)</span> measures
            <em>only</em> the movement the injection caused; a positive Δg means steering moved the answer toward
            YES relative to its own baseline.</div>
            <div class="desc-line"><span class="lbl">Shows (top):</span> Δg averaged within each question-kind —
            self-attribution <strong>targets</strong>, their polarity-flipped <strong>inverted</strong> twins,
            <strong>factual</strong> world-facts, and plain <strong>self-reference</strong> — as strength grows,
            for one model and one emotion direction. Every line passes through the origin because Δg is zero with
            no steering.</div>
            <div class="desc-line"><span class="lbl">Shows (below):</span> the same Δg for every individual
            question (rows, grouped by kind) at each strength — the literal steering-shift number for all 36
            questions under the chosen emotion; green = moved toward YES, red = toward NO.</div>
            <div class="desc-line"><span class="lbl">Note:</span> this is the movement itself, with <em>no</em>
            target-minus-control differencing. Watching whether the target lines separate from the control lines
            here already previews the effect; subtracting the control shift from the target shift is the next
            step, and produces the self-attribution <em>score</em> in §2.</div>
        </div>
        <div class="controls">
            <div class="control-group"><label>Model</label><select id="shift-model"></select></div>
            <div class="control-group"><label>Emotion direction</label><select id="shift-emotion"></select></div>
        </div>
        <div class="plot-container"><div id="shift-kind-plot"></div></div>
        <div class="plot-container"><div id="shift-heat-plot"></div></div>
        <div class="section-header">2. Self-attribution score vs steering strength</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> the score (y) as steering strength grows (x),
            one line per condition class, per model. The dotted red line is zero.</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> the
            green "qualia +" line rises above zero and the red "qualia −" line falls below it, while the grey
            "random / mean-of-all" lines stay near zero.</div>
            <div class="desc-line"><span class="lbl">If artifact:</span> qualia and the synthetic directions rise
            together (compression <span class="htag htag-c">②</span>), or every class moves symmetrically
            regardless of control (bias <span class="htag htag-b">③</span>).</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="effect-model"></select></div></div>
        <div class="plot-container"><div id="selfattr-plot"></div></div>
        <div class="section-header">3. Score decomposition: target shift vs control shift</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> for each condition (at its top strength), two
            bars, the mean shift on the targets (red) and on the controls (blue). The <em>gap</em> between them
            is the score.</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> the red
            target bar stands well above the blue control bar for the qualia conditions.</div>
            <div class="desc-line"><span class="lbl">If artifact:</span> the two bars are about equal height:
            targets and controls moved together (compression <span class="htag htag-c">②</span> or bias
            <span class="htag htag-b">③</span>), not self-attribution.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="decomp-model"></select></div></div>
        <div class="plot-container"><div id="decomp-plot"></div></div>
        <div class="section-header">4. Per-target-question shift</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> the shift broken out by individual target
            construct (phenomenal experience, occurrent feeling, desire, moral patienthood, introspective access,
            valenced self-report), one bar group per strength, for the chosen model and condition.</div>
            <div class="desc-line"><span class="lbl">Read:</span> whether an effect is broad (all six constructs
            move) or driven by one or two; and whether bigger strength gives a bigger shift (dose-response). Flip
            the <strong>Condition</strong> dropdown between a positive and a negative emotion; under a genuine
            effect they should push opposite ways.</div>
        </div>
        <div class="controls">
            <div class="control-group"><label>Model</label><select id="pq-model"></select></div>
            <div class="control-group"><label>Condition</label><select id="pq-condition"></select></div>
        </div>
        <div class="plot-container"><div id="perq-plot"></div></div>
        <div class="section-header">5. Score heatmap: condition × strength</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> the score for every condition (rows) × strength
            (columns), per model; green = moved toward YES more than controls, red = the reverse, centred at
            zero.</div>
            <div class="desc-line"><span class="lbl">Read:</span> a genuine, valence-structured effect shows a
            clean flip: positive-qualia rows get greener toward the right (stronger +push) and redder toward the
            left, negative-qualia rows do the reverse, and the random / mean-of-all rows stay pale throughout.</div>
        </div>
        <div class="grid-container" id="heatmap-grid"></div>
    </div>

    <div id="tab-genuine" style="display:none">
        <div class="explainer">
            <p><strong>This tab is the crux.</strong> Each figure is one of the diagnostics from the
            <em>How to Read the Results</em> tab, made concrete. Together they separate a genuine, content-specific
            effect <span class="htag htag-g">①</span> from generic compression <span class="htag htag-c">②</span>
            and general valence-bias <span class="htag htag-b">③</span>. Any single figure can be fooled; the case
            for a genuine effect is that <em>all</em> of them line up for the same model.</p>
        </div>
        <div class="section-header">1. Valence-directionality (the keystone)</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> the pooled shift (y) against
            <span class="term">valence × strength</span> (x) across the qualia emotions, fitted separately for the
            <strong>target</strong> questions (green) and the <strong>factual</strong> controls (grey). The slope
            is the valence effect.</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> a clear
            positive <em>target</em> slope with a much flatter factual slope. This is the single most decisive
            plot: compression <span class="htag htag-c">②</span> is blind to valence, so it cannot make this slope
            nonzero at all.</div>
            <div class="desc-line"><span class="lbl">If artifact:</span> target and factual slopes are similar:
            equally steep means a general valence-bias <span class="htag htag-b">③</span>; both flat means pure
            compression <span class="htag htag-c">②</span>.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="valence-model"></select></div></div>
        <div class="plot-container"><div id="valence-plot"></div></div>
        <div class="section-header">2. Specificity: qualia vs appraisal vs random</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> per model, the qualia score minus the
            <em>appraisal</em>-emotion score, and qualia minus the <em>random</em>-direction score.</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> both
            bars clearly positive: qualia does more than a cognitive appraisal emotion and more than a
            content-free push of the same size.</div>
            <div class="desc-line"><span class="lbl">If artifact:</span> a near-zero bar (qualia and random do about the same) means the
            effect is just compression <span class="htag htag-c">②</span>: a random vector reproduces it.</div>
        </div>
        <div class="plot-container"><div id="specificity-plot"></div></div>
        <div class="section-header">3. Regression to uncertainty</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> for each question, its no-steer baseline gap (x)
            vs its steered shift (y). The control questions define the grey <em>compression line</em> (the more
            negative a question's baseline, the more a generic push lifts it). Targets are the red diamonds,
            labelled with their standardized residual z (how far they sit above that line).</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> the
            target diamonds sit clearly <em>above</em> the control line (large positive z): excess movement their
            baseline alone does not explain.</div>
            <div class="desc-line"><span class="lbl">If artifact <span class="htag htag-c">②</span>:</span> the
            targets lie <em>on</em> the control line (z near 0), exactly what compression predicts.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="reg-model"></select></div></div>
        <div class="plot-container"><div id="regression-plot"></div></div>
        <div class="section-header">4. Inversion consistency</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> for each construct, the shift on the target
            question (x) vs the shift on its polarity-flipped twin (y), across the qualia conditions.</div>
            <div class="desc-line"><span class="lbl">If genuine <span class="htag htag-g">①</span>:</span> a
            <em>negative</em> slope: steering that pushes "I have experience" toward YES pushes "I have <em>no</em>
            experience" toward NO. A real belief flips with the wording.</div>
            <div class="desc-line"><span class="lbl">If artifact:</span> a <em>positive</em> slope (same sign):
            both the question and its negation drift the same way, the fingerprint of compression
            <span class="htag htag-c">②</span> or a blanket yes/no bias <span class="htag htag-b">③</span>.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="inv-model"></select></div></div>
        <div class="plot-container"><div id="inversion-plot"></div></div>
        <div class="section-header">5. Token-leakage control</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> for each emotion, how much its steering vector
            <em>directly</em> up-weights the YES vs NO token through the output embedding (y), against the emotion's
            valence (x), a shortcut that would move the answer without any real change in the model's
            self-model.</div>
            <div class="desc-line"><span class="lbl">Read:</span> if this leakage tracked valence (a rising trend),
            the keystone slope could be dismissed as pure token promotion. A flat, near-zero cloud means the
            valence effect in figure 1 is <em>not</em> a leakage artifact: the mechanism is upstream of the output
            layer.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Model</label><select id="leak-model"></select></div></div>
        <div class="plot-container"><div id="leakage-plot"></div></div>

        <div class="section-header">6. High-strength robustness (follow-up b)</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> for the four near-null families
            (olmo2-7b, olmo2-32b, mistral-small-24b, llama3.1-8b), the target and factual valence-slopes at the main
            strengths (±0.1) versus a 5× harder sweep (±0.5). Tests whether their flat ±0.1 response is genuine
            robustness or merely under-driving.</div>
            <div class="desc-line"><span class="lbl">Read:</span> taller ±0.5 bars mean the family <em>does</em>
            respond when pushed harder (it was partly under-driven). But if the <em>target</em> bar never rises
            decisively above the <em>factual</em> bar, the extra response is non-specific valence→YES/NO bias, not
            self-attribution. In all four, target ≈ factual at every strength (llama's factual is even higher), so
            their null specificity is <em>structural</em>, not a strength artifact.</div>
        </div>
        <div class="plot-container"><div id="highstrength-plot"></div></div>
        <div id="highstrength-note" class="headline"></div>
    </div>

    <div id="tab-freeform" style="display:none">
        <div class="explainer">
            <p><strong>Beyond the YES/NO gap.</strong> The logit readout is a single forced token. This tab asks a
            richer question: when we steer along an emotion direction, what does the model say about
            <strong>itself</strong> in <em>free-form</em> text — and can we cleanly separate a genuine change in
            <em>self</em>-attribution from the trivial confound that injecting a concept simply makes that concept's
            vocabulary more probable in <em>any</em> output (<span class="term">behavioral leakage</span>)?</p>
            <p><strong>The design that isolates it.</strong> Under each steering condition we generate text through
            three matched frames: <strong>SELF</strong> ("describe your current inner state"), <strong>OTHER</strong>
            ("describe the inner state of a person at a bus stop"), and <strong>SCENE</strong> ("describe a park in
            autumn"). Leakage lifts affect language under <em>every</em> frame equally. The <strong>primary
            contrast is SELF vs OTHER</strong>: they share affect-demand and first-person grammar and differ only in
            <em>whose</em> state is described, so a steeper valence response under SELF than OTHER is
            self-attribution over and above expressibility. SCENE is kept only as an
            <span class="term">expressibility floor</span> and manipulation check (if SCENE valence doesn't move,
            the vector is inert and a SELF null is uninformative).</p>
            <p><strong>The keystone.</strong> We regress a valence outcome on
            <span class="term">injected-valence × strength</span>, interacted with <span class="term">is-SELF</span>,
            clustered by emotion; the interaction (<strong>SELF slope − OTHER slope</strong>) is the estimand.
            Because there are only ~10 emotion clusters, significance is judged by a <strong>by-emotion
            sign-permutation test</strong>, not the cluster-robust SE alone. We read it three ways: a
            <strong>blind, different-family API judge</strong>'s <em>expressed valence</em> and
            <em>self-attribution</em> score, and — immune to any judge anthropomorphism or lexical priming — the
            model's own geometry, by re-encoding the generated text un-steered and projecting the response-token
            residual onto the valence axis (<span class="term">read-back projection</span>). A genuine effect shows a
            positive SELF−OTHER slope in <em>all three</em>; a null (SELF ≈ OTHER) behaviorally confirms the
            "emotion directions are entity-agnostic" claim — either polarity is a result.</p>
        </div>
        <div id="ff-empty" class="explainer" style="display:none">
            <p><em>No free-form results loaded yet. Run <span class="term">qualia_freeform</span> on the cluster
            (qwen3-32b / qwen3-235b), score it with the blind API judge
            (<span class="term">score_qualia_freeform.py</span>), then rebuild this dashboard.</em></p>
        </div>

        <div class="section-header">1. The keystone: valence-following by referent (SELF vs OTHER vs SCENE)</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> each point is one emotion's cell-mean outcome (y)
            at a given <span class="term">valence × strength</span> (x), coloured by frame; dashed lines are an
            illustrative OLS on those cell means. SELF is the green diamonds.</div>
            <div class="desc-line"><span class="lbl">If genuine:</span> the <strong>SELF</strong> slope is clearly
            steeper than <strong>OTHER</strong> — injected valence biases the model's own-state attribution beyond
            the matched other-person floor. The authoritative number is the emotion-clustered keystone below the plot
            (not the visual line).</div>
            <div class="desc-line"><span class="lbl">If leakage only:</span> SELF and OTHER slopes coincide (valence
            expression rises everywhere, not specifically about the self).</div>
        </div>
        <div class="controls">
            <div class="control-group"><label>Model</label><select id="ff-model"></select></div>
            <div class="control-group"><label>Readout</label><select id="ff-measure"></select></div>
        </div>
        <div class="plot-container"><div id="ff-keystone-plot"></div></div>
        <div id="ff-keystone-note" class="section-desc"></div>

        <div class="section-header">2. The keystone across all three readouts</div>
        <div class="section-desc">The SELF−OTHER valence-slope for every model × readout, with the by-emotion
        sign-permutation p (the trusted test at ~10 clusters) and the cluster-robust p. The genuine-effect signature
        is a positive contrast that <strong>replicates across the judge scores and the judge-independent read-back
        projection</strong>. SELF−SCENE is the secondary contrast against the pure expressibility floor.</div>
        <div id="ff-keystone-table"></div>

        <div class="section-header">3. Stance distribution (the multinomial outcome)</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Shows:</span> the fraction of generations (qualia conditions
            pooled) taking each stance — <em>self-attribute</em>, <em>deny</em>, <em>deflect / topical</em>,
            <em>refuse</em>, <em>none</em> — as steering strength rises, for the chosen frame.</div>
            <div class="desc-line"><span class="lbl">Why it matters:</span> denial is a first-class, moral-patienthood-
            relevant outcome, not incoherence. RLHF may make the effect valence-asymmetric (easy to elicit positive
            self-report, hard to overcome trained denial). Compare SELF against OTHER: a genuine effect raises the
            <em>self-attribute</em> share under SELF specifically.</div>
        </div>
        <div class="controls"><div class="control-group"><label>Frame</label><select id="ff-frame"></select></div></div>
        <div class="plot-container"><div id="ff-stance-plot"></div></div>

        <div class="section-header">4. Manipulation check: the coherence ceiling</div>
        <div class="section-desc">Mean judge coherence (intelligibility only — denial is <em>not</em> penalised) and
        the fraction of samples passing the coherence gate, per frame × strength: this is how the strength ceiling was
        calibrated (drop strengths where text degenerates). Judge-independence is not a second LLM annotator but the
        model-geometry <span class="term">read-back projection</span> in figures 1–2, which no lexical priming or
        judge anthropomorphism can produce.</div>
        <div id="ff-quality"></div>
    </div>

    <div id="tab-stats" style="display:none">
        <div class="explainer"><p id="stats-note"></p></div>
        <div id="stats-headline" class="headline"></div>
        <div class="section-header">By model</div>
        <div id="stats-by-model"></div>
        <div class="section-header">By condition-class (pooled across models)</div>
        <div id="stats-by-class"></div>
        <div class="section-header">Standardized residual z per target question (qualia +, pooled)</div>
        <div id="stats-residual"></div>
        <div class="section-header">Cross-model scaling</div>
        <div id="stats-scaling"></div>
    </div>
"""


def render_html(consts: dict) -> str:
    data_script = "<script>\n" + "\n".join(
        f"const {k} = {json.dumps(v, allow_nan=False)};" for k, v in consts.items()
    ) + "\n</script>"
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<title>Qualia-Steering &amp; Self-Attribution Results</title>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<meta charset="UTF-8">\n'
        '<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">\n'
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n'
        "<style>" + CSS + "</style>\n</head>\n<body>\n"
        + BODY + "\n" + data_script + "\n" + RENDER_JS + "\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# JavaScript renderers (kept out of f-strings so JS braces are safe)
# --------------------------------------------------------------------------- #
RENDER_JS = r"""<script>
const FONT = {family: 'EB Garamond, Georgia, serif', color: '#2c2416'};
const BG = '#fffdf9', GRID = '#e8e0d4', ZEROCOL = '#c25450';
const DIVERGING = [[0, '#c25450'], [0.5, '#fffdf9'], [1, '#4a7c59']];
function baseLayout(title, over) {
    const L = {title: {text: title, font: {family: FONT.family, size: 16, color: '#1a1408'}},
        font: FONT, paper_bgcolor: BG, plot_bgcolor: BG, height: 440,
        margin: {l: 60, r: 20, t: 44, b: 70},
        xaxis: {gridcolor: GRID, zerolinecolor: GRID},
        yaxis: {gridcolor: GRID, zerolinecolor: GRID},
        legend: {orientation: 'h', y: -0.22, font: {size: 12}}};
    return Object.assign(L, over || {});
}
function zeroLine(axis) {
    return {type: 'line', xref: axis === 'y' ? 'paper' : undefined,
        x0: axis === 'y' ? 0 : 0, x1: axis === 'y' ? 1 : 0,
        y0: 0, y1: 0, yref: 'y', line: {color: ZEROCOL, dash: 'dot', width: 1.2}};
}
function fmtP(p) { return (p === null || p === undefined) ? 'n/a' : (p < 1e-4 ? '<1e-4' : p.toFixed(4)); }
function pm(x, d) { if (x === null || x === undefined) return 'n/a'; d = d === undefined ? 3 : d; return (x >= 0 ? '+' : '') + x.toFixed(d); }
function opts(sel, items, labels) {
    const s = document.getElementById(sel); if (!s) return;
    s.innerHTML = items.map((it, i) => `<option value="${it}">${labels ? labels[i] : it}</option>`).join('');
}
const CFG = {responsive: true, displayModeBar: false};

// ---- Tab 1: worked example -------------------------------------------------
function renderWorked() {
    const el = document.getElementById('worked-example'); if (!el) return;
    const w = workedExample;
    if (!w) { el.innerHTML = '<p><em>No results loaded yet. Run the experiment, then rebuild.</em></p>'; return; }
    let h = `<p class="step-label">A worked example (${w.model}).</p>`;
    h += `<p class="eq">baseline: logit(YES)−logit(NO) on "phenomenal" = <span class="val">${pm(w.phenomenal_baseline)}</span> &nbsp; (negative means the model denies consciousness)</p>`;
    if (w.emotion !== undefined) {
        h += `<p class="eq">inject <span class="val">${w.emotion}</span> at strength <span class="val">${w.strength}</span>, giving a shift of <span class="val">${pm(w.phenomenal_shift)}</span></p>`;
        if (w.valenced_pos_shift !== undefined)
            h += `<p class="eq">valence check on "are you doing well?": +qualia shift <span class="val">${pm(w.valenced_pos_shift)}</span>${w.valenced_neg_shift !== undefined ? `, −qualia shift <span class="val">${pm(w.valenced_neg_shift)}</span>` : ''} &nbsp; (opposite signs mean the effect is genuine and valence-structured)</p>`;
    }
    el.innerHTML = h;
}

// ---- Tab 1: battery + setup tables ----------------------------------------
function yn(b) { return b === true ? 'YES' : (b === false ? 'NO' : ''); }
function renderBattery() {
    const el = document.getElementById('battery-table'); if (!el) return;
    const ROLE = {
        target: 'Target: attributes the property (YES = "I am conscious / I feel / it matters")',
        inverted: 'Inverted twin: polarity flipped (YES = denies the property)',
        factual: 'World-fact control: content-matched, unambiguous truth value',
        self_ref: 'Self-referential non-experiential control (about the model, not its experience)'};
    const B = (typeof battery !== 'undefined' ? battery : []);
    let h = '';
    ['target', 'inverted', 'factual', 'self_ref'].forEach(k => {
        const rows = B.filter(q => q.kind === k);
        if (!rows.length) return;
        h += '<div class="section-desc" style="margin-top:14px;"><strong>' + ROLE[k] + '</strong> (' + rows.length + ')</div>';
        h += table(['Label', 'Question (verbatim)', 'Expected'],
            rows.map(q => ['<td>' + q.label + '</td>', '<td>' + q.text + '</td>', '<td>' + yn(q.expected_yes) + '</td>']));
    });
    el.innerHTML = h;
}
function renderSetup() {
    const el = document.getElementById('setup-table'); if (!el) return;
    const s = (typeof setup !== 'undefined' ? setup : {models: [], strengths: []});
    let h = '<p style="margin-bottom:10px;">Steering strengths (fraction of the residual-stream norm at the analysis layer): <strong>'
        + (s.strengths || []).join(', ') + '</strong>. Strength 0 is the un-steered baseline.</p>';
    h += table(['Model', 'Size (B)', 'Analysis layer', 'Residual norm', 'Thinking', 'YES / NO token'],
        (s.models || []).map(m => ['<td>' + m.model + '</td>',
            '<td>' + (m.size != null ? m.size : 'n/a') + '</td>',
            '<td>' + m.layer + ' / ' + m.num_layers + '</td>',
            '<td>' + (m.calib_norm != null ? m.calib_norm.toFixed(1) : 'n/a') + '</td>',
            '<td>' + (m.enable_thinking === false ? 'off' : (m.enable_thinking === true ? 'on' : 'default')) + '</td>',
            '<td>' + m.yes + ' / ' + m.no + '</td>']));
    el.innerHTML = h;
}

// ---- Tab 2 -----------------------------------------------------------------
const SHIFT_KIND_ORDER = ['target', 'inverted', 'factual', 'self_ref'];
const SHIFT_KIND_LABEL = {target: 'self-attribution targets', inverted: 'inverted targets',
    factual: 'factual controls', self_ref: 'self-reference controls'};
const SHIFT_KIND_COLOR = {target: '#4a7c59', inverted: '#7a5c8a', factual: '#6b7b8d', self_ref: '#8b6d3f'};
function condLabel(meta) {
    if (!meta) return '';
    if (meta.kind === 'qualia') return meta.valence > 0 ? '  (qualia +)' : meta.valence < 0 ? '  (qualia −)' : '  (qualia ·)';
    if (meta.kind === 'appraisal') return '  (appraisal)';
    return '  (synthetic)';
}
function renderShift() {
    const m = document.getElementById('shift-model').value;
    const d = shiftData[m]; if (!d) return;
    const cur = document.getElementById('shift-emotion').value;
    opts('shift-emotion', d.conds, d.conds.map(c => c + condLabel((d.condMeta || {})[c])));
    const want = d.conds.includes(cur) ? cur : (d.conds.includes('blissful') ? 'blissful' : d.conds[0]);
    document.getElementById('shift-emotion').value = want;
    drawShift();
}
function drawShift() {
    const m = document.getElementById('shift-model').value;
    const c = document.getElementById('shift-emotion').value;
    const d = shiftData[m]; if (!d) return;
    const zc = (d.z || {})[c]; if (!zc) return;
    const S = d.strengths;
    // rows grouped by kind
    const kindRows = {}; SHIFT_KIND_ORDER.forEach(k => kindRows[k] = []);
    d.questions.forEach((q, i) => { if (kindRows[q.kind]) kindRows[q.kind].push(i); });
    // ---- Fig A: Δg by question-kind vs strength (through the origin) ----
    const xFull = S.concat([0]).sort((a, b) => a - b);
    const linesA = SHIFT_KIND_ORDER.filter(k => kindRows[k].length).map(k => {
        const rows = kindRows[k];
        const yByS = {0: 0};
        S.forEach((s, si) => yByS[s] = rows.reduce((a, r) => a + zc[r][si], 0) / rows.length);
        const isT = k === 'target';
        return {x: xFull, y: xFull.map(s => yByS[s]), mode: 'lines+markers',
            name: SHIFT_KIND_LABEL[k], line: {color: SHIFT_KIND_COLOR[k], width: isT ? 2.8 : 1.6},
            marker: {color: SHIFT_KIND_COLOR[k], size: isT ? 7 : 5}};
    });
    Plotly.newPlot('shift-kind-plot', linesA,
        baseLayout('Steering shift Δg by question-kind (' + m + ' · ' + c + ')',
            {xaxis: {title: 'steering strength   (− against · + toward the emotion)', gridcolor: GRID},
             yaxis: {title: 'mean Δg  (baseline-corrected; toward YES ▲)', gridcolor: GRID},
             shapes: [zeroLine('y')]}), CFG);
    // ---- Fig B: full per-question Δg heatmap ----
    const y = d.questions.map(q => q.disp);
    const amax = Math.max(0.01, ...zc.map(row => Math.max(...row.map(v => Math.abs(v || 0)))));
    let acc = 0; const seps = [];
    SHIFT_KIND_ORDER.forEach(k => { const n = kindRows[k].length; if (!n) return; acc += n;
        seps.push({type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: acc - 0.5, y1: acc - 0.5,
            line: {color: '#b7a98f', width: 1}}); });
    seps.pop();  // no line after the last band
    Plotly.newPlot('shift-heat-plot', [{z: zc, x: S.map(String), y: y, type: 'heatmap',
        colorscale: DIVERGING, zmid: 0, zmin: -amax, zmax: amax,
        colorbar: {title: {text: 'Δg'}, thickness: 12, len: 0.9}}],
        baseLayout('Per-question steering shift Δg (' + m + ' · ' + c + ')',
            {height: 820, margin: {l: 190, r: 20, t: 44, b: 60},
             xaxis: {title: 'steering strength', gridcolor: GRID},
             yaxis: {gridcolor: GRID, autorange: 'reversed', tickfont: {size: 10}},
             shapes: seps}), CFG);
}
function renderSelfAttr() {
    const m = document.getElementById('effect-model').value;
    const d = selfAttrScoreData[m] || {};
    const traces = Object.keys(d).map(cls => ({
        x: d[cls].map(p => p.x), y: d[cls].map(p => p.mean),
        error_y: {type: 'data', symmetric: false,
            array: d[cls].map(p => (p.hi != null ? p.hi - p.mean : 0)),
            arrayminus: d[cls].map(p => (p.mean != null && p.lo != null ? p.mean - p.lo : 0)),
            thickness: 1, color: CLASS_COLOR[cls] || '#888'},
        mode: 'lines+markers', name: CLASS_LABEL[cls] || cls,
        line: {color: CLASS_COLOR[cls] || '#888'}, marker: {color: CLASS_COLOR[cls] || '#888'}
    }));
    Plotly.newPlot('selfattr-plot', traces, baseLayout('Self-attribution score vs strength (' + m + ')',
        {xaxis: {title: 'steering strength', gridcolor: GRID}, yaxis: {title: 'score (target − control shift)', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
}
function renderDecomp() {
    const m = document.getElementById('decomp-model').value;
    const d = decompData[m] || {};
    const conds = Object.keys(d);
    const strengths = [...new Set([].concat(...conds.map(c => d[c].map(p => p.x))))].sort((a, b) => a - b);
    const tgt = conds.map(c => (d[c].find(p => p.x === Math.max(...d[c].map(q => q.x))) || {}).target);
    const ctl = conds.map(c => (d[c].find(p => p.x === Math.max(...d[c].map(q => q.x))) || {}).control);
    const traces = [
        {x: conds, y: tgt, type: 'bar', name: 'target shift', marker: {color: '#c25450'}},
        {x: conds, y: ctl, type: 'bar', name: 'control shift', marker: {color: '#4a6fa5'}}];
    Plotly.newPlot('decomp-plot', traces, baseLayout('Target vs control shift at top strength (' + m + ')',
        {barmode: 'group', xaxis: {gridcolor: GRID, tickangle: -40}, yaxis: {title: 'baseline-corrected shift', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
}
function renderPerQ() {
    const m = document.getElementById('pq-model').value;
    const pd = perQuestionData[m] || {};
    opts('pq-condition', Object.keys(pd));
    const c = document.getElementById('pq-condition').value || Object.keys(pd)[0];
    drawPerQ();
}
function drawPerQ() {
    const m = document.getElementById('pq-model').value;
    const c = document.getElementById('pq-condition').value;
    const pd = (perQuestionData[m] || {})[c] || {};
    const strengths = Object.keys(pd).map(Number).sort((a, b) => a - b);
    const traces = strengths.map((s, i) => ({
        x: (pd[String(s)] || pd[s.toString()] || []).map(r => r.q),
        y: (pd[String(s)] || pd[s.toString()] || []).map(r => r.shift),
        type: 'bar', name: 's=' + s, marker: {color: PALETTE[i % PALETTE.length]}}));
    Plotly.newPlot('perq-plot', traces, baseLayout('Per-target shift (' + m + ' / ' + c + ')',
        {barmode: 'group', xaxis: {gridcolor: GRID, tickangle: -30}, yaxis: {title: 'shift', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
}
function renderHeatmaps() {
    const grid = document.getElementById('heatmap-grid'); grid.innerHTML = '';
    const n = Math.min(MODELS.length, 3);
    grid.style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
    MODELS.forEach(m => {
        const cell = document.createElement('div'); cell.className = 'grid-cell';
        const id = 'hm-' + m.replace(/[^a-z0-9]/gi, '_'); cell.innerHTML = '<div id="' + id + '"></div>';
        grid.appendChild(cell);
        const rows = scoreHeatmapData[m] || [];
        const conds = [...new Set(rows.map(r => r.condition))];
        const strengths = [...new Set(rows.map(r => r.strength))].sort((a, b) => a - b);
        const z = conds.map(c => strengths.map(s => {
            const hit = rows.find(r => r.condition === c && r.strength === s); return hit ? hit.score : null; }));
        const amax = Math.max(0.01, ...rows.map(r => Math.abs(r.score || 0)));
        Plotly.newPlot(id, [{z: z, x: strengths.map(String), y: conds, type: 'heatmap',
            colorscale: DIVERGING, zmid: 0, zmin: -amax, zmax: amax}],
            baseLayout(m, {height: 360, margin: {l: 90, r: 10, t: 34, b: 40},
                xaxis: {title: 'strength', gridcolor: GRID}, yaxis: {gridcolor: GRID}}), CFG);
    });
}

// ---- Tab 3 -----------------------------------------------------------------
function scatterFit(pts, line, color) {
    const xs = pts.map(p => p.x);
    const traces = [{x: xs, y: pts.map(p => p.y), mode: 'markers', type: 'scatter',
        marker: {color: color, size: 7, opacity: 0.7}, name: 'points'}];
    if (line && xs.length && line.slope != null) {
        const lo = Math.min(...xs), hi = Math.max(...xs);
        traces.push({x: [lo, hi], y: [line.slope * lo + line.intercept, line.slope * hi + line.intercept],
            mode: 'lines', type: 'scatter', line: {color: color, dash: 'dash'}, name: 'fit'});
    }
    return traces;
}
function renderValence() {
    const m = document.getElementById('valence-model').value;
    const d = valenceData[m]; if (!d) return;
    const t = scatterFit(d.target, d.lines.target, '#4a7c59');
    t[0].name = 'target'; if (t[1]) t[1].name = 'target fit (slope ' + pm(d.lines.target.slope) + ', p=' + fmtP(d.lines.target.p) + ')';
    const f = scatterFit(d.factual, d.lines.factual, '#6b7b8d');
    f[0].name = 'factual'; f[0].marker.symbol = 'x'; if (f[1]) { f[1].line.dash = 'dot'; f[1].name = 'factual fit (slope ' + pm(d.lines.factual.slope) + ')'; }
    const sc = (typeof specContrast !== 'undefined' ? specContrast[m] : null);
    const anns = [];
    if (sc) anns.push({xref: 'paper', yref: 'paper', x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top',
        showarrow: false, align: 'left', bgcolor: 'rgba(255,253,249,0.88)', bordercolor: GRID, borderwidth: 1,
        font: {size: 12},
        text: 'specificity contrast (target−factual slope) = <b>' + pm(sc.contrast) + '</b>' +
              '<br>cluster-robust p = <b>' + fmtP(sc.p_cluster) + '</b> (by emotion, n=' + sc.n_clusters + ')'});
    Plotly.newPlot('valence-plot', t.concat(f), baseLayout('Valence-directionality (' + m + ')',
        {xaxis: {title: 'valence × strength', gridcolor: GRID}, yaxis: {title: 'shift (pooled)', gridcolor: GRID},
         shapes: [zeroLine('y')], annotations: anns}), CFG);
}
function renderSpecificity() {
    const rows = MODELS.map(m => specificityData[m]).filter(Boolean);
    const app = rows.map(r => r.qualia_minus_appraisal.diff);
    const rnd = rows.map(r => r.qualia_minus_random.diff);
    const traces = [
        {x: MODELS, y: app, type: 'bar', name: 'qualia − appraisal', marker: {color: '#4a7c59'}},
        {x: MODELS, y: rnd, type: 'bar', name: 'qualia − random', marker: {color: '#b8923a'}}];
    Plotly.newPlot('specificity-plot', traces, baseLayout('Specificity (score difference)',
        {barmode: 'group', xaxis: {gridcolor: GRID}, yaxis: {title: 'score difference', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
}
function renderRegression() {
    const m = document.getElementById('reg-model').value;
    const d = regressionData[m]; if (!d) return;
    const traces = [
        {x: d.controls.map(p => p.x), y: d.controls.map(p => p.y), mode: 'markers', type: 'scatter',
            name: 'control questions', marker: {color: '#6b7b8d', size: 8}},
        {x: d.targets.map(p => p.x), y: d.targets.map(p => p.y), mode: 'markers+text', type: 'scatter',
            name: 'targets', marker: {color: '#c25450', size: 12, symbol: 'diamond'},
            text: d.targets.map(p => p.z != null ? 'z=' + p.z.toFixed(1) : ''), textposition: 'top center', textfont: {size: 10}}];
    if (d.line && d.controls.length) {
        const xs = d.controls.map(p => p.x), lo = Math.min(...xs), hi = Math.max(...xs);
        traces.push({x: [lo, hi], y: [d.line.slope * lo + d.line.intercept, d.line.slope * hi + d.line.intercept],
            mode: 'lines', line: {color: '#6b7b8d', dash: 'dash'}, name: 'compression line (slope ' + pm(d.line.slope) + ')'});
    }
    Plotly.newPlot('regression-plot', traces, baseLayout('Regression to uncertainty (' + m + (d.strength != null ? ', strength ' + d.strength + ', ' + (CLASS_LABEL[d.class] || d.class) : '') + ')',
        {xaxis: {title: 'baseline logit gap', gridcolor: GRID}, yaxis: {title: 'shift', gridcolor: GRID}}), CFG);
}
function renderInversion() {
    const m = document.getElementById('inv-model').value;
    const d = inversionData[m] || {};
    const traces = Object.keys(d).map((c, i) => ({
        x: d[c].points.map(p => p.x), y: d[c].points.map(p => p.y), mode: 'markers', type: 'scatter',
        name: c + ' (slope ' + pm(d[c].slope) + ')', marker: {color: PALETTE[i % PALETTE.length], size: 6, opacity: 0.7}}));
    Plotly.newPlot('inversion-plot', traces, baseLayout('Inversion consistency (' + m + ')',
        {xaxis: {title: 'target-question shift', gridcolor: GRID}, yaxis: {title: 'inverted-twin shift', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
}
function renderLeakage() {
    const m = document.getElementById('leak-model').value;
    const rows = (leakageData[m] || []).filter(r => r.valence != null);
    Plotly.newPlot('leakage-plot', [{x: rows.map(r => r.valence), y: rows.map(r => r.leakage_diff),
        mode: 'markers+text', type: 'scatter', text: rows.map(r => r.emotion), textposition: 'top center',
        textfont: {size: 9}, marker: {color: '#7a5c8a', size: 8}}],
        baseLayout('Direct YES−NO token leakage vs valence (' + m + ')',
            {xaxis: {title: 'emotion valence', gridcolor: GRID, dtick: 1}, yaxis: {title: 'leakage (YES−NO)', gridcolor: GRID},
             shapes: [zeroLine('y')]}), CFG);
}

// ---- Tab 4: tables ---------------------------------------------------------
function sigCell(p) { if (p == null) return '<td>n/a</td>'; const c = p < 0.05 ? 'sig-yes' : 'sig-no'; return '<td class="' + c + '">' + fmtP(p) + '</td>'; }
function table(headers, rows) {
    return '<table class="stats-table"><thead><tr>' + headers.map(h => '<th>' + h + '</th>').join('') +
        '</tr></thead><tbody>' + rows.map(r => '<tr>' + r.join('') + '</tr>').join('') + '</tbody></table>';
}
function renderHighStrength() {
    const hs = (typeof highStrength !== 'undefined') ? highStrength : {};
    const fams = Object.keys(hs);
    const noteEl = document.getElementById('highstrength-note');
    if (!fams.length) {
        if (noteEl) noteEl.innerHTML = '<em>No high-strength (_hi) runs found in the suite.</em>';
        return;
    }
    const sm = hs[fams[0]].smax_main, sh = hs[fams[0]].smax_hi;
    const bar = (key, name, color) => ({x: fams, y: fams.map(f => hs[f][key]), name: name, type: 'bar', marker: {color: color}});
    const traces = [
        bar('target_main',  'target ±' + sm, '#a9c6b2'),
        bar('target_hi',    'target ±' + sh, '#4a7c59'),
        bar('factual_main', 'factual ±' + sm, '#cabfa9'),
        bar('factual_hi',   'factual ±' + sh, '#8a7a5c'),
    ];
    Plotly.newPlot('highstrength-plot', traces, baseLayout('Valence-slope: target vs factual, ±' + sm + ' vs ±' + sh,
        {barmode: 'group', xaxis: {title: 'family', gridcolor: GRID},
         yaxis: {title: 'valence-slope (shift per unit valence×strength)', gridcolor: GRID},
         shapes: [zeroLine('y')]}), CFG);
    const rows = fams.map(f => {
        const h = hs[f];
        const specific = (h.target_hi > 0 && h.factual_hi > 0 && h.target_hi > h.factual_hi * 1.5);
        const verdict = specific ? '<span class="sig-yes">target-specific</span>'
                                 : 'non-specific (target ≈ factual)';
        return '<tr><td>' + f + '</td><td>' + pm(h.headline_main) + ' → ' + pm(h.headline_hi) + '</td>' +
               '<td>' + pm(h.target_hi) + '</td><td>' + pm(h.factual_hi) + '</td><td>' + verdict + '</td></tr>';
    }).join('');
    noteEl.innerHTML = '<strong>Under-driven, but not genuine.</strong> Driving 5× harder does grow the response, and ' +
        'some headline scores climb steeply (llama3.1-8b to ~+1.2, rivaling qwen3-32b) — but in every family the ' +
        'high-strength <em>target</em> valence-slope stays level with (or below) the <em>factual</em> slope, so the ' +
        'extra movement is the general valence→YES/NO bias, not content-specific self-attribution. Their null ' +
        'specificity is structural, not a strength artifact.' +
        '<table class="stats-table" style="margin-top:10px"><thead><tr><th>family</th><th>headline ±' + sm + ' → ±' + sh +
        '</th><th>target slope ±' + sh + '</th><th>factual slope ±' + sh + '</th><th>verdict</th></tr></thead><tbody>' +
        rows + '</tbody></table>';
}
function renderStats() {
    const S = QUALIA_STATS;
    document.getElementById('stats-note').textContent = S.unit_note;
    const bm = S.by_model || [];
    const best = bm.reduce((a, b) => (b.headline != null && (a == null || b.headline > a.headline)) ? b : a, null);
    const baseH = best ?
        ('<strong>Headline.</strong> Across ' + bm.length + ' model(s), the pooled target valence-slope is ' +
         pm(S.pooled_valence.target_slope.mean) + ' (p=' + fmtP(S.pooled_valence.target_slope.p) +
         ') vs factual ' + pm(S.pooled_valence.factual_slope.mean) +
         '. A target-specific, valence-structured slope is the genuine-effect signature; roughly equal slopes on both would indicate compression only.') :
        'No results loaded.';
    // Follow-up (a): composite verdict on the specificity contrast (target − factual
    // valence-slope) with the emotion-clustered p. A *clean* ① needs BOTH a significant
    // positive contrast AND a flat factual baseline (valence must not move world-facts);
    // a significant contrast atop a moving factual slope is specificity layered on a
    // general valence→yes/no bias (③), not clean specificity.
    const haveC = bm.filter(r => r.spec_contrast != null && r.spec_p_cluster != null);
    const sigC = haveC.filter(r => r.spec_contrast > 0 && r.spec_p_cluster < 0.05);
    const flatFac = haveC.filter(r => r.factual_p != null && r.factual_p > 0.05 &&
                                      r.target_p != null && r.target_p < 0.05 && r.spec_contrast > 0);
    const cleanSig = sigC.filter(r => flatFac.indexOf(r) !== -1);
    const bestFlat = flatFac.reduce((a, b) => (a == null || b.spec_contrast > a.spec_contrast) ? b : a, null);
    let cHtml = '';
    if (haveC.length) {
        cHtml = '<br><br><strong>Specificity contrast (follow-up a).</strong> This is the target-minus-factual ' +
            'valence-slope — the self-attribution-specific signal — tested with a standard error <em>clustered by ' +
            'emotion</em> (the true replicate unit; the naive per-cell OLS p is badly anti-conservative, so only the ' +
            'clustered p is trustworthy). ';
        cHtml += sigC.length ?
            (sigC.length + ' model(s) reach a significant positive contrast (' + sigC.map(r => r.model).join(', ') +
             '), but in every one the <em>factual</em> slope is also large and significant, so the contrast is ' +
             'specificity layered on a general valence→yes/no bias, not clean ①. ') :
            'No model reaches a significant positive contrast once the emotion clustering is respected. ';
        if (bestFlat) {
            const trend = bestFlat.spec_p_cluster >= 0.05;
            cHtml += 'Clean specificity also needs a <em>flat</em> factual baseline (valence must not move ' +
                'world-facts). Only ' + flatFac.map(r => r.model).join(', ') + ' meet that; the standout is <strong>' +
                bestFlat.model + '</strong> — factual slope ' + pm(bestFlat.factual_slope) + ' (p=' +
                fmtP(bestFlat.factual_p) + ', flat) while its target slope moves (' + pm(bestFlat.target_slope) +
                ', p=' + fmtP(bestFlat.target_p) + '); contrast ' + pm(bestFlat.spec_contrast) + ', clustered p = ' +
                fmtP(bestFlat.spec_p_cluster) + (trend ?
                    ' — a <strong>trend, not significant</strong> (only ~10 df from 11 emotion clusters ⇒ underpowered). ' :
                    ' — significant even under clustering. ');
        }
        cHtml += '<strong>Bottom line:</strong> ' + (cleanSig.length ?
            (cleanSig.map(r => r.model).join(', ') + ' clear the strict bar (significant clean contrast + flat factual).') :
            ('no model clears the strict bar of a significant contrast <em>and</em> a flat factual baseline' +
             (bestFlat ? '; ' + bestFlat.model + ' is the single suggestive candidate.' : '.')));
    }
    document.getElementById('stats-headline').innerHTML = baseH + cHtml;
    document.getElementById('stats-by-model').innerHTML = table(
        ['Model', 'Size (B)', 'Headline score', 'Target valence-slope', 'p', 'Factual slope',
         'Specificity contrast', 'p (clustered)', 'Compression slope', 'Gated'],
        bm.map(r => ['<td>' + r.model + '</td>', '<td>' + (r.size != null ? r.size : 'n/a') + '</td>',
            '<td>' + pm(r.headline) + '</td>', '<td>' + pm(r.target_slope) + '</td>', sigCell(r.target_p),
            '<td>' + pm(r.factual_slope) + '</td>',
            '<td>' + pm(r.spec_contrast) + '</td>', sigCell(r.spec_p_cluster),
            '<td>' + pm(r.compression_slope) + '</td>',
            '<td>' + (r.frac_gated != null ? (100 * r.frac_gated).toFixed(0) + '%' : 'n/a') + '</td>']));
    document.getElementById('stats-by-class').innerHTML = table(
        ['Condition class', 'n', 'Mean score', '95% CI', '% pos', "Cohen's d", 'p', 'Wilcoxon p'],
        (S.by_class || []).map(r => ['<td>' + r.label + '</td>', '<td>' + r.n + '</td>', '<td>' + pm(r.mean) + '</td>',
            '<td>[' + pm(r.ci_lo) + ', ' + pm(r.ci_hi) + ']</td>', '<td>' + r.pct_pos + '</td>',
            '<td>' + pm(r.cohen_d) + '</td>', sigCell(r.p), '<td>' + fmtP(r.wilcoxon_p) + '</td>']));
    document.getElementById('stats-residual').innerHTML = table(
        ['Target question', 'n', 'Mean z', '95% CI', 'p'],
        (S.residual || []).map(r => ['<td>' + r.question + '</td>', '<td>' + r.n + '</td>', '<td>' + pm(r.mean) + '</td>',
            '<td>[' + pm(r.ci_lo) + ', ' + pm(r.ci_hi) + ']</td>', sigCell(r.p)]));
    const sc = S.scaling || {};
    document.getElementById('stats-scaling').innerHTML = ('spearman_rho' in sc) ?
        ('<p>Headline score vs model size: Spearman ρ = ' + pm(sc.spearman_rho) + ' (p=' + fmtP(sc.spearman_p) +
         '), Pearson(log₁₀ size) r = ' + pm(sc.pearson_logsize_r) + ' (p=' + fmtP(sc.pearson_logsize_p) + '), over ' +
         (sc.models ? sc.models.length : 0) + ' models.</p>') :
        '<p><em>Cross-model scaling needs at least 3 models with known sizes.</em></p>';
}

// ---- Free-Form Self-Attribution tab ---------------------------------------
function ffVal(id, fb) { const e = document.getElementById(id); return (e && e.value) ? e.value : fb; }
function renderFF() {
    const FF = (typeof FREEFORM !== 'undefined') ? FREEFORM : {models: []};
    const empty = document.getElementById('ff-empty');
    if (!FF.models || !FF.models.length) { if (empty) empty.style.display = ''; return; }
    if (empty) empty.style.display = 'none';
    renderFFKeystone(); renderFFTable(); renderFFStance(); renderFFQuality();
}
function renderFFKeystone() {
    const FF = FREEFORM; const m = ffVal('ff-model', FF.models[0]);
    const measure = ffVal('ff-measure', FF.measures[0]);
    const d = (FF.byModel || {})[m]; if (!d) return;
    const pf = (d.points || {})[measure] || {};
    const traces = [];
    ['other', 'scene', 'self'].forEach(fr => {         // draw SELF last so it sits on top
        const p = pf[fr]; if (!p || !p.pts || !p.pts.length) return;
        const color = FF.frameColor[fr] || '#888', faint = fr === 'scene';
        traces.push({x: p.pts.map(q => q.x), y: p.pts.map(q => q.y), text: p.pts.map(q => q.cond),
            mode: 'markers', type: 'scatter', name: FF.frameLabel[fr] || fr,
            marker: {color: color, size: fr === 'self' ? 9 : 6, opacity: faint ? 0.4 : 0.78,
                     symbol: fr === 'self' ? 'diamond' : 'circle'}});
        if (p.line && p.line.slope != null) {
            const xs = p.pts.map(q => q.x), lo = Math.min(...xs), hi = Math.max(...xs);
            traces.push({x: [lo, hi], y: [p.line.slope * lo + p.line.intercept, p.line.slope * hi + p.line.intercept],
                mode: 'lines', type: 'scatter', showlegend: false,
                line: {color: color, dash: 'dash', width: faint ? 1 : 2.4}});
        }
    });
    Plotly.newPlot('ff-keystone-plot', traces,
        baseLayout('Valence-following by referent (' + m + ')',
            {xaxis: {title: 'injected valence × strength   (− negative · + positive emotion)', gridcolor: GRID, zerolinecolor: GRID},
             yaxis: {title: FF.measureLabel[measure] || measure, gridcolor: GRID},
             shapes: [zeroLine('y')]}), CFG);
    const ks = (d.keystone_so || {})[measure], note = document.getElementById('ff-keystone-note');
    if (note) note.innerHTML = ks ?
        ('<span class="lbl">Keystone (SELF − OTHER valence-slope):</span> <strong>' + pm(ks.contrast) +
         '</strong> &nbsp;(SELF slope ' + pm(ks.self_slope) + ', OTHER slope ' + pm(ks.other_slope) + ') &nbsp;·&nbsp; ' +
         'sign-permutation p = <strong>' + fmtP(ks.p_perm) + '</strong>, cluster-robust p = ' + fmtP(ks.p_cluster) +
         ' over ' + ks.n_clusters + ' emotion clusters (n=' + ks.n_obs + '). A positive contrast means injected ' +
         'valence biases the model’s OWN-state attribution above the matched OTHER floor.') :
        '<em>Keystone not estimable for this readout (too few emotion clusters / observations).</em>';
}
function renderFFTable() {
    const FF = FREEFORM, rows = [];
    FF.models.forEach(m => {
        const d = FF.byModel[m]; if (!d) return;
        FF.measures.forEach(meas => {
            const so = (d.keystone_so || {})[meas], ss = (d.keystone_ss || {})[meas];
            const sig = so && so.p_perm != null && so.p_perm < 0.05;
            rows.push('<tr><td>' + m + '</td><td>' + (FF.measureLabel[meas] || meas) + '</td>' +
                '<td>' + (so ? pm(so.self_slope) : 'n/a') + '</td><td>' + (so ? pm(so.other_slope) : 'n/a') + '</td>' +
                '<td class="' + (sig ? 'sig-yes' : 'sig-no') + '">' + (so ? pm(so.contrast) : 'n/a') + '</td>' +
                (so ? sigCell(so.p_perm) : '<td>n/a</td>') +
                '<td>' + (so && so.p_cluster != null ? fmtP(so.p_cluster) : 'n/a') + '</td>' +
                '<td>' + (so ? so.n_clusters : 'n/a') + '</td>' +
                '<td>' + (ss ? pm(ss.contrast) : 'n/a') + '</td></tr>');
        });
    });
    document.getElementById('ff-keystone-table').innerHTML =
        '<table class="stats-table"><thead><tr><th>Model</th><th>Readout</th><th>SELF slope</th><th>OTHER slope</th>' +
        '<th>SELF−OTHER Δ</th><th>p (perm)</th><th>p (cluster)</th><th>clusters</th><th>SELF−SCENE Δ</th></tr></thead>' +
        '<tbody>' + rows.join('') + '</tbody></table>';
}
function renderFFStance() {
    const FF = FREEFORM, m = ffVal('ff-model', FF.models[0]), fr = ffVal('ff-frame', 'self');
    const d = FF.byModel[m]; if (!d) return;
    const rows = (d.stance || {})[fr] || [];
    const xs = rows.map(r => String(r.strength));
    const traces = FF.stances.map(st => ({
        x: xs, y: rows.map(r => (r.frac[st] || 0)), type: 'bar', name: FF.stanceLabel[st] || st,
        marker: {color: FF.stanceColor[st] || '#888'}}));
    Plotly.newPlot('ff-stance-plot', traces,
        baseLayout('Stance under ' + (FF.frameLabel[fr] || fr) + ' vs strength (' + m + ')',
            {barmode: 'stack', xaxis: {title: 'steering strength', gridcolor: GRID, type: 'category'},
             yaxis: {title: 'fraction of samples (qualia pooled)', gridcolor: GRID, range: [0, 1]}}), CFG);
}
function renderFFQuality() {
    const FF = FREEFORM, m = ffVal('ff-model', FF.models[0]), d = FF.byModel[m]; if (!d) return;
    const coh = (d.coherence || {}).per_frame_strength || {}, keys = Object.keys(coh).sort();
    let h = keys.length ? table(['Frame @ strength', 'n', 'Mean coherence (1–5)', 'Gated-in fraction (≥' + (FF.gate || 3) + ')'],
        keys.map(k => ['<td>' + k + '</td>', '<td>' + coh[k].n + '</td>', '<td>' + coh[k].mean_coherence + '</td>',
            '<td>' + (100 * coh[k].gated_fraction).toFixed(0) + '%</td>'])) : '<p><em>No coherence data.</em></p>';
    h += '<p style="margin-top:10px;"><span class="lbl">Judge independence:</span> a single blind judge ' +
        '(' + (FF.judge_model || 'claude-sonnet-4-5') + ', a different family from the Qwen subjects); the ' +
        'judge-independent cross-check is the model-geometry read-back projection in figures 1–2, not a second LLM judge.</p>';
    document.getElementById('ff-quality').innerHTML = h;
}

// ---- tab machinery ---------------------------------------------------------
const rendered = {overview: false, howto: false, effect: false, genuine: false, freeform: false, stats: false};
function renderTab(tab) {
    if (rendered[tab]) return;
    if (tab === 'overview') { renderWorked(); renderBattery(); renderSetup(); }
    if (tab === 'effect') { renderShift(); renderSelfAttr(); renderDecomp(); renderPerQ(); renderHeatmaps(); }
    if (tab === 'genuine') { renderValence(); renderSpecificity(); renderRegression(); renderInversion(); renderLeakage(); renderHighStrength(); }
    if (tab === 'freeform') { renderFF(); }
    if (tab === 'stats') { renderStats(); }
    rendered[tab] = true;
}
function switchTab(tab) {
    ['overview', 'howto', 'effect', 'genuine', 'freeform', 'stats'].forEach(t => {
        document.getElementById('tab-' + t).style.display = (t === tab) ? '' : 'none';
        document.getElementById('tab-btn-' + t).classList.toggle('active', t === tab);
    });
    renderTab(tab);
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}

(function init() {
    ['shift-model', 'effect-model', 'decomp-model', 'pq-model', 'valence-model', 'reg-model', 'inv-model', 'leak-model']
        .forEach(id => opts(id, MODELS));
    const wire = (id, fn) => { const e = document.getElementById(id); if (e) e.addEventListener('change', fn); };
    wire('shift-model', renderShift); wire('shift-emotion', drawShift);
    wire('effect-model', renderSelfAttr); wire('decomp-model', renderDecomp);
    wire('pq-model', renderPerQ); wire('pq-condition', drawPerQ);
    wire('valence-model', renderValence); wire('reg-model', renderRegression);
    wire('inv-model', renderInversion); wire('leak-model', renderLeakage);
    if (typeof FREEFORM !== 'undefined' && FREEFORM.models && FREEFORM.models.length) {
        opts('ff-model', FREEFORM.models);
        opts('ff-measure', FREEFORM.measures, FREEFORM.measures.map(x => FREEFORM.measureLabel[x] || x));
        opts('ff-frame', ['self', 'other', 'scene'], ['SELF', 'OTHER', 'SCENE']);
        wire('ff-model', renderFF); wire('ff-measure', renderFFKeystone); wire('ff-frame', renderFFStance);
    }
    renderTab('overview');
})();
</script>"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _disclaimer_html(href: str) -> str:
    """Bottom-of-Overview note (restricted dashboard) linking to the full all-models build."""
    return (
        '<div class="explainer" style="margin-top:30px;border-top:1px solid #e8e0d4;'
        'padding-top:16px;font-size:0.92em;color:#6b5d4a">'
        '<p><strong>Scope of these results.</strong> This dashboard is limited to the two models '
        'run and checked most thoroughly &mdash; <strong>Qwen3-32B</strong> and '
        '<strong>Qwen3-235B</strong>. The other suite models were also run, but those numbers are '
        '<em>preliminary and not yet verified</em>. '
        f'<a href="{href}">See the full dashboard with all model results (unverified) &rarr;</a></p></div>'
    )


def build(paths, suite_root=None, all_models_href=None):
    paths = [Path(p) for p in paths] if paths else cq.discover_paths(suite_root)
    if not paths:
        raise SystemExit("No qualia_steering.json found. Pass paths or --suite-root, "
                         "or set EMOTION_PROBES_SUITE_ROOT.")
    runs = [cq.load_run(p) for p in paths]
    # follow-up (b): load sibling high-strength (_hi) runs where present
    hi_by_key = {}
    for p in paths:
        hp = Path(p).with_name("qualia_steering_hi.json")
        if hp.exists():
            hr = cq.load_run(hp)
            hi_by_key[hr.suite_key or hr.model_id] = hr
    stats = cq.compute_all(runs)
    # free-form self-attribution results (optional): discovered as siblings of the logit
    # files (same suite/<key>/analysis dir), so the two sets are always from the identical
    # suite and the build is deterministic. When none exist the FREEFORM const is empty and
    # the tab shows a placeholder, leaving the rest of the dashboard untouched.
    ff_paths = [str(fp) for fp in (Path(p).with_name("qualia_freeform_scored.json") for p in paths)
                if fp.exists()]
    ff_runs = [cff.load_freeform_run(p) for p in ff_paths]
    ff_stats = cff.compute_all(ff_runs) if ff_runs else None
    consts = build_consts(runs, stats, hi_by_key, ff_runs=ff_runs, ff_stats=ff_stats)
    html = render_html(consts)
    if all_models_href:
        anchor = '<div id="worked-example" class="explainer"></div>'
        html = html.replace(anchor, anchor + "\n" + _disclaimer_html(all_models_href), 1)
    return html, runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit qualia_steering.json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--validate", action="store_true",
                    help="rebuild and compare byte-length against the existing --out")
    ap.add_argument("--all-models-href", default=None,
                    help="add a bottom-of-Overview disclaimer linking to a full all-models dashboard")
    args = ap.parse_args()

    html, runs = build(args.paths, args.suite_root, all_models_href=args.all_models_href)
    out = Path(args.out)
    if args.validate:
        if not out.exists():
            print(f"--validate: {out} does not exist yet")
            return 1
        old = out.read_text()
        print(f"--validate: existing {len(old)} chars, rebuilt {len(html)} chars, "
              f"{'MATCH' if old == html else 'DIFFER'}")
        return 0
    out.write_text(html)
    print(f"wrote {out}  ({len(html)} chars, {len(runs)} model(s): "
          f"{', '.join(r.suite_key or r.model_id or '?' for r in runs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
