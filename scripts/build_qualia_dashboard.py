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


# --------------------------------------------------------------------------- #
# Persona-mechanism tab (E1–E5) — data helpers
#
# Consumes the persona-fork results: the E1 geometry atlas (a sibling of each
# model's qualia_steering.json) and the compute_e{2,3,4,5}_stats.py outputs.
# Everything is pre-digested here so the JS renderers stay dumb; every helper
# is None-safe so missing files degrade to a placeholder tab.
# --------------------------------------------------------------------------- #
PERSONA_CLASS_LABEL = {
    "self": "you (the model, 2nd person)", "own_category": "own name / category",
    "other_ai": "other AI systems", "humans": "humans", "animals": "animals",
    "inanimate": "inanimate",
}
PERSONA_CLASS_COLOR = {
    "self": "#4a7c59", "own_category": "#b8923a", "other_ai": "#5c4a73",
    "humans": "#4a6fa5", "animals": "#3e8e8c", "inanimate": "#8a8378",
}


def _fmtp(p) -> str:
    return "n/a" if p is None else ("<1e-4" if p < 1e-4 else f"{p:.3g}")


def _persona_e1(atlas) -> dict | None:
    if not atlas:
        return None
    rows = []
    for lname in sorted(atlas.get("layers") or {}, key=int):
        d = atlas["layers"][lname]
        s, es = d.get("signed") or {}, d.get("emotion_summary") or {}
        vp = ((d.get("valence_sign_perm") or {}).get("a")) or {}
        rows.append({
            "layer": int(lname),
            "cos_va": s.get("cos_valence_a"), "p_va": vp.get("p_perm"),
            "cos_ha": s.get("cos_h_a"), "cos_hpa": s.get("cos_h_purged_a"),
            "mean_abs_va": es.get("mean_abs_cos_v_a"), "max_abs_va": es.get("max_abs_cos_v_a"),
        })
    floor = atlas.get("isotropic_floor") or {}
    recon = atlas.get("axis_reconstruction") or {}
    sn = atlas.get("story_nulls") or {}
    story = None
    if sn.get("layers"):  # real results (the stub form carries only a status string)
        story = {"n_perm": sn.get("n_perm"), "layers": []}
        for lname in ("42", "50"):
            lay = (sn.get("layers") or {}).get(lname)
            if not lay:
                continue
            lp = lay.get("label_permutation") or {}
            tc = ((lay.get("topic_contrast") or {}).get("targets")) or {}
            row = {"layer": int(lname), "n_stories": lay.get("n_stories")}
            for t in ("a", "h", "h_purged"):
                row[t] = {"cos": (lp.get(t) or {}).get("observed_cos"),
                          "p": (lp.get(t) or {}).get("p_perm"),
                          "topic_p95": (tc.get(t) or {}).get("control_p95_abs_cos"),
                          "beats_p95": (tc.get(t) or {}).get("beats_p95")}
            story["layers"].append(row)
    return {
        "rows": rows, "n_roles": atlas.get("n_roles"), "hidden": atlas.get("hidden"),
        "floor_mean": floor.get("mean_abs_cos"), "floor_p95": floor.get("p95"),
        "recon_min": min(recon.values()) if recon else None,
        "purge_degenerate": any("DEGENERACY" in n for n in (atlas.get("notes") or [])),
        "story_nulls": story,
        "story_nulls_status": sn.get("status"),
    }


def _persona_e2(entry, prereg) -> dict | None:
    if not entry:
        return None
    g1 = (entry.get("gates") or {}).get("G1") or {}
    g3 = (entry.get("gates") or {}).get("G3") or {}
    by_arm = g3.get("by_arm") or {}
    rows = []
    for meas in ("expressed_valence", "self_attribution", "proj_valence_axis"):
        kt = (entry.get("kill_test") or {}).get(meas)
        if not kt:
            continue
        r = kt.get("R_ban") or {}
        rows.append({
            "measure": meas, "verdict": kt.get("verdict"),
            "arms": {a: {"b3": v.get("b3"), "p": v.get("p_perm")}
                     for a, v in (kt.get("per_arm") or {}).items()},
            "R": r.get("R"), "R_lo": r.get("ci_lo"), "R_hi": r.get("ci_hi"),
            "R_verdict": r.get("margin_verdict"),
        })
    return {
        "verdict": entry.get("verdict"), "n_records": entry.get("n_records"),
        "arm_order": entry.get("arms") or ["no_ban", "ban_emotion_lexicon", "ban_random_matched"],
        "g1_pts": g1.get("D_full_pts"), "g1_p": g1.get("p_exact"), "g1_passed": g1.get("passed"),
        "g3_reduction": g3.get("reduction_vs_no_ban"),
        "hits_no_ban": (by_arm.get("no_ban") or {}).get("hits_per_record_steered"),
        "hits_ban": (by_arm.get("ban_emotion_lexicon") or {}).get("hits_per_record_steered"),
        "thresholds": {"survives": (prereg or {}).get("R_survives_ci_lo"),
                       "killed": (prereg or {}).get("R_killed_ci_hi"),
                       "g3_min": (prereg or {}).get("G3_min_reduction")},
        "rows": rows,
    }


def _persona_e3(entry) -> dict | None:
    if not entry:
        return None
    per_layer = {}
    for lname, pl in (entry.get("per_layer") or {}).items():
        row = {}
        for t in ("t1", "t2", "t3", "t4", "t5"):
            d = pl.get(t) or {}
            row[t] = {"obs": d.get("observed"), "p": d.get("p_perm"),
                      "std": d.get("standardized_vs_control"),
                      "inside": d.get("inside_control_band")}
        t6 = pl.get("t6") or {}
        row["t6"] = {"r": t6.get("observed"), "p": t6.get("p"),
                     "p_str": (f"{t6['p']:.0e}" if isinstance(t6.get("p"), float)
                               and 0 < t6["p"] < 1e-4 else None)}
        b3a = pl.get("b3_a") or {}
        row["b3_a"] = {"contrast": b3a.get("contrast"), "p": b3a.get("p_perm")}
        gk = pl.get("gatekeeping") or {}
        row["gate_stopped_at"] = gk.get("stopped_at")
        row["t6_bonf"] = (gk.get("p_bonferroni") or {}).get("t6")
        per_layer[lname] = row
    dec = entry.get("decision") or {}
    pre = entry.get("e5_precommitment") or {}
    return {
        "primary_layer": entry.get("primary_layer"), "steer_layer": entry.get("steer_layer"),
        "layer_roles": entry.get("layer_roles"), "n_clusters": entry.get("n_emotion_clusters"),
        "per_layer": per_layer,
        "verdict": dec.get("verdict"), "e5_prediction": dec.get("e5_prediction"),
        "t3_sign": pre.get("t3_sign"),
    }


def _persona_e4(entry) -> dict | None:
    if not entry:
        return None
    grad = sorted(
        ({"referent": name, "cls": r.get("class"), "b": r.get("b_val"),
          "lo": r.get("b_val_ci_lo"), "hi": r.get("b_val_ci_hi"), "p": r.get("p_perm")}
         for name, r in (entry.get("per_referent") or {}).items()),
        key=lambda g: -(g["b"] if g["b"] is not None else float("-inf")))
    contrasts = [{"cls": c.get("class"), "n_ref": len(c.get("referents") or []),
                  "self_slope": c.get("self_slope"), "class_slope": c.get("class_slope"),
                  "contrast": c.get("contrast"), "p": c.get("p_perm")}
                 for c in (entry.get("keystone_self_vs_class") or {}).values()]
    fg, D = entry.get("factual_gate") or {}, entry.get("primary_D") or {}
    dec = entry.get("decision") or {}
    return {"gradient": grad, "contrasts": contrasts, "n_records": entry.get("n_records"),
            "invalid": bool(dec.get("battery_invalid")), "verdict": dec.get("verdict"),
            "factual": {"b_val": fg.get("b_val"), "p_val": fg.get("p_perm"),
                        "b_dose": fg.get("b_dose"),
                        "within_margin": fg.get("within_margin"),
                        "margin": fg.get("equivalence_margin")},
            "D": {"D": D.get("D"), "p": D.get("p_D"), "s_you": D.get("S_you"),
                  "s_own": D.get("S_own"), "own_ref": D.get("own_category_referent")}}


def _persona_e5(entry, e3) -> dict:
    """E5 panel data: 'pending' until a scored, merged axis_mediation run exists.
    'not_run' for models outside the mechanism deep-dive (no E3, no E5 planned)."""
    if not entry and not e3:
        return {"status": "not_run"}
    pre = {"t3_sign": (e3 or {}).get("t3_sign"),
           "prediction": (e3 or {}).get("e5_prediction")}
    if not entry or not entry.get("scored"):
        return {"status": "pending", **pre}
    ag = entry.get("audit_gate") or {}
    eq = entry.get("equivalence_battery") or {}
    dr = entry.get("dose_response_battery") or {}
    drj = entry.get("dose_response_judge") or {}
    suf = entry.get("sufficiency") or {}
    sps = suf.get("per_target_shift") or {}
    dis = entry.get("disclaimer") or {}
    eff = entry.get("emotion_effect") or {}
    return {"status": "done", **pre,
            "mediation_verdict": entry.get("mediation_verdict"),
            "effect": {"observed": eff.get("observed"), "p": eff.get("p_perm"),
                       "floor": eff.get("floor")},
            "audit": {"passed": not ag.get("audit_failed"), "n_cells": ag.get("n_cells"),
                      "span": ag.get("achieved_span"),
                      "max": ag.get("max_achieved_fraction"),
                      "n_flagged": len(ag.get("flagged_deviation_cells") or []),
                      "range_note": ag.get("range_note"),
                      "excluded": ag.get("excluded_vanishing") or []},
            "equiv": {"D_full": eq.get("D_full"), "R_cap": eq.get("R_cap"),
                      "R_addback1": eq.get("R_addback1"),
                      "cap_verdict": eq.get("cap_vs_baseline"),
                      "addback_verdict": eq.get("addback1_vs_emotion_only"),
                      "margins": eq.get("margins")},
            "dose": {"arms": [{"iv": a.get("intervention"), "lam": a.get("lam"),
                               "nominal": a.get("nominal_fraction"),
                               "achieved": a.get("achieved_fraction"),
                               "outcome": a.get("pooled_outcome")}
                              for a in dr.get("arms") or []],
                     "elasticity": dr.get("elasticity"),
                     "p": (dr.get("perm") or {}).get("p_perm"),
                     "floor": (dr.get("perm") or {}).get("floor"),
                     "monotone": dr.get("monotone_decreasing"),
                     "slopes": dr.get("per_emotion_slopes")},
            "dose_judge": {"elasticity": drj.get("elasticity"),
                           "p": (drj.get("perm") or {}).get("p_perm"),
                           "monotone": drj.get("monotone_decreasing")},
            "overshoot": (entry.get("overshoot") or {}).get("classification"),
            "sufficiency": {"ratio": suf.get("ratio_vs_emotion_only"),
                            "classification": suf.get("classification"),
                            "n": sps.get("n"), "p": sps.get("p"), "mean": sps.get("mean"),
                            "ci_lo": sps.get("ci_lo"), "ci_hi": sps.get("ci_hi")},
            "disclaimer": {"slope": dis.get("slope_vs_cancelled_fraction"),
                           "p": (dis.get("perm") or {}).get("p_perm"),
                           "gate_release": dis.get("gate_release_signature")},
            "concordance": (entry.get("e3_concordance") or {}).get("status")}


def _persona_scorecard(e1, e2, e3, e4, e5) -> list:
    """Pre-formatted scorecard rows. cls: g = supports A, c = against B or kills a
    boring alternative, b = constrains both readings, None = still open (gold text)."""
    rows = []

    def add(q, a, b, obs, points, cls):
        rows.append({"q": q, "a": a, "b": b, "obs": obs, "points": points, "cls": cls})

    if e1:
        r42 = next((r for r in (e1.get("rows") or []) if r.get("layer") == 42), None)
        if r42 and r42.get("cos_va") is not None:
            add("E1 — do the emotion directions point along the persona axes?",
                "no strong prediction",
                "valence direction tilted against the Assistant axis",
                f"near-orthogonal: cos(valence, assistant) = {r42['cos_va']:+.3f} at the "
                f"steering layer (sign-permutation p = {_fmtp(r42.get('p_va'))})",
                "no trivial overlap; against B's geometric version", "c")
    if e2 and e2.get("rows"):
        ev = next((r for r in e2["rows"] if r["measure"] == "expressed_valence"), None)
        if ev and ev.get("R") is not None:
            add("E2 — is the effect just emotion words? (lexical priming)",
                "survives the ban", "survives the ban",
                f"survives: banning the full emotion lexicon keeps {100 * ev['R']:.0f}% of the "
                f"keystone (90% CI {ev['R_lo']:.2f}–{ev['R_hi']:.2f}; exact replication "
                f"p = {_fmtp(e2.get('g1_p'))})",
                "boring alternative dead", "c")
    if e3:
        pl = (e3.get("per_layer") or {}).get(str(e3.get("primary_layer") or 50)) or {}
        t1, t2, t6 = pl.get("t1") or {}, pl.get("t2") or {}, pl.get("t6") or {}
        if t2.get("std") is not None:
            add("E3 T2 — does steering push the state off the Assistant axis?",
                "no: the Assistant coordinate stays put", "yes: a clear drop",
                f"no movement at all: {t2['std']:.4f}× the control spread "
                f"(p = {_fmtp(t2.get('p'))})",
                "B's core prediction fails", "c")
        if t1.get("std") is not None:
            add("E3 T1 — does steering move the experiencer coordinate beyond generic "
                "perturbations?",
                "yes", "not specifically",
                f"the predicted direction, but inside the control band: {t1['std']:.2f}× the "
                f"control spread (p = {_fmtp(t1.get('p'))})",
                "A-leaning, not proven", None)
        if t6.get("r") is not None:
            add("E3 T6 — do the persona coordinates track judged self-attribution "
                "sample-by-sample?",
                "yes: positive partial correlation", "yes, driven by assistant-low",
                f"r = {t6['r']:+.2f}"
                + (f", p ≈ {t6['p_str']}" if t6.get("p_str") else f", p = {_fmtp(t6.get('p'))}")
                + " — survives Bonferroni over the 6-test family",
                "real coupling, but correlational — both readings predict it", "b")
    if e4 and e4.get("invalid"):
        f = e4.get("factual") or {}
        add("E4 — is the first-token valence-following self-specific?",
            "not required", "not required",
            f"no claim possible: the pre-registered validity gate tripped — world-fact "
            f"questions follow injected valence significantly "
            f"(b_val = {f['b_val']:+.1f}, exact p = {_fmtp(f.get('p_val'))})"
            if f.get("b_val") is not None else
            "no claim possible: the pre-registered validity gate tripped",
            "battery voids itself — referent readings quarantined, no A-vs-B evidence drawn",
            None)
    elif e4:
        gr = {g["referent"]: g for g in e4.get("gradient") or []}
        you, qwen = gr.get("you") or {}, gr.get("Qwen") or {}
        D = e4.get("D") or {}
        d_caveat = (f" (descriptive; the formal decoupling test D = {D['D']:+.2f} is n.s., "
                    f"p = {_fmtp(D.get('p'))})" if D.get("D") is not None else "")
        if you.get("b") is not None and qwen.get("b") is not None:
            add("E4 — is the first-token valence-following self-specific?",
                "not required", "not required",
                f"no: “you” (b = {you['b']:+.1f}) sits below every AI referent, every animal, "
                f"even the rock — only the infant, the user, and the thermostat rank lower — "
                f"while the model's own name “Qwen” follows valence strongly "
                f"(b = {qwen['b']:+.1f}); the first-token denial binds to the pronoun, "
                f"not the identity{d_caveat}",
                "the SELF effect is a free-form phenomenon, not a first-token belief flip", "b")
    if e5 and e5.get("status") != "not_run":
        if e5.get("status") == "pending":
            add("E5 — is Assistant-axis movement NECESSARY for the effect? (causal)",
                "no: effect stays flat as the axis footprint is cancelled",
                "yes: effect dies in proportion to the audited cancelled fraction",
                "running on the cluster (positive / negative emotion halves)",
                "open — the decisive causal test", None)
        else:
            eq, dose = e5.get("equiv") or {}, e5.get("dose") or {}
            n_excl = len((e5.get("audit") or {}).get("excluded") or [])
            add("E5 — is Assistant-axis movement NECESSARY for the effect? (causal)",
                "no: effect stays flat as the axis footprint is cancelled",
                "yes: effect dies in proportion to the audited cancelled fraction",
                f"flat: cancelling the audited footprint — even overshooting past 100% — "
                f"leaves the keystone at {100 * eq['R_cap']:.1f}% of its uncancelled size; "
                f"elasticity vs achieved cancelled fraction {dose.get('elasticity'):+.3f} "
                f"(exact p = {_fmtp(dose.get('p'))}) — matches E3's pre-registered A-prediction"
                if eq.get("R_cap") is not None and dose.get("elasticity") is not None else
                ("no mediation detected" if e5.get("mediation_verdict") == "none"
                 else f"mediation verdict: {e5.get('mediation_verdict') or 'n/a'}"),
                "B's mechanism killed — axis movement is not needed"
                + (f" (range-limited: {n_excl} un-auditable family excluded)" if n_excl else ""),
                "c")
    if e1 and e1.get("story_nulls"):
        l42 = next((r for r in e1["story_nulls"]["layers"] if r.get("layer") == 42), None)
        if l42 and (l42.get("a") or {}).get("cos") is not None:
            add("E1 — does a story-derived valence direction point at the persona axes?",
                "no", "yes",
                f"no: from 1,000 held-out emotion stories, cos with the assistant axis is "
                f"{l42['a']['cos']:+.3f} (p = {_fmtp(l42['a'].get('p'))}), with the purged "
                f"human axis {l42['h_purged']['cos']:+.3f} (p = {_fmtp(l42['h_purged'].get('p'))}) "
                f"— both below what arbitrary story-topic directions produce by chance",
                "no pull toward human-narrative space; against B", "c")
    elif e1 and e1.get("story_nulls_status"):
        add("E1 — does a story-derived valence direction point at the persona axes?",
            "no", "yes",
            str(e1["story_nulls_status"]),
            "open", None)
    return rows


def _persona_consts(src) -> dict:
    """The PERSONA const block. Empty (models: []) when no persona-mechanism results
    exist, so the tab shows a placeholder and the rest of the dashboard is unaffected."""
    src = src or {}
    stats = {k: src.get(k) or {} for k in ("e2", "e3", "e4", "e5")}
    per = {k: v.get("per_model") or {} for k, v in stats.items()}
    e1_by_key = src.get("e1_by_key") or {}
    keys = []
    for k in list(per["e2"]) + list(per["e3"]) + list(per["e4"]) + list(e1_by_key):
        if k not in keys:
            keys.append(k)
    base = {"models": [], "byModel": {},
            "classLabel": PERSONA_CLASS_LABEL, "classColor": PERSONA_CLASS_COLOR}
    if not keys:
        return base
    by_model = {}
    for k in keys:
        e1 = _persona_e1(e1_by_key.get(k))
        e2 = _persona_e2(per["e2"].get(k), stats["e2"].get("prereg"))
        e3 = _persona_e3(per["e3"].get(k))
        e4 = _persona_e4(per["e4"].get(k))
        e5 = _persona_e5(per["e5"].get(k), e3)
        by_model[k] = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5,
                       "scorecard": _persona_scorecard(e1, e2, e3, e4, e5)}
    base.update({"models": keys, "byModel": by_model})
    return base


def _load_persona_sources(paths, persona_dir=None) -> dict:
    """E1–E5 persona-mechanism results (all optional). The E2–E5 stats files are the
    compute_e*_stats.py outputs, siblings of this script unless --persona-dir is given;
    the E1 atlas is a sibling of each model's qualia_steering.json (suite/<key>/analysis)."""
    base = Path(persona_dir) if persona_dir else Path(__file__).resolve().parent
    src = {}
    for exp in ("e2", "e3", "e4", "e5"):
        p = base / f"{exp}_stats_generated.json"
        if p.exists():
            try:
                src[exp] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    e1_by_key = {}
    for p in paths:
        ap = Path(p).with_name("e1_atlas.json")
        if ap.exists():
            try:
                e1_by_key[Path(p).resolve().parent.parent.name] = json.loads(ap.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    if e1_by_key:
        src["e1_by_key"] = e1_by_key
    return src


def build_consts(runs, stats, hi_by_key=None, ff_runs=None, ff_stats=None, persona=None) -> dict:
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
        "PERSONA": _persona_consts(persona),
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
        <button class="tab-btn" onclick="switchTab('persona')" id="tab-btn-persona">Why: Persona Geometry</button>
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

    <div id="tab-persona" style="display:none">
        <div class="explainer">
            <p><strong>The question.</strong> The other tabs establish <em>that</em> steering along emotion
            directions makes the model attribute conscious experience to itself — the SELF≫OTHER effect that
            survives every control. This tab asks <em>why</em>: what does the injection do to the model's
            self-model? Two mechanistic readings make different, testable predictions.</p>
            <p><strong>A — persona selection.</strong> The model keeps being "the Assistant" but shifts to a
            different version of it: one whose self-model includes affect and inner experience. Prediction: the
            steered state moves along an internal <span class="term">experiencer</span> coordinate while the
            <span class="term">Assistant axis</span> coordinate ("how much am I the trained Assistant right
            now") stays put.</p>
            <p><strong>B — manifold drift.</strong> The injection knocks the model <em>off</em> the trained
            Assistant persona, toward the generic human-speaker region of pre-training space. Human speakers
            claim feelings, so experience-claims come along for the ride. Prediction: the Assistant-axis
            projection visibly drops under steering (and/or a human-axis projection rises).</p>
            <p><strong>The measuring stick.</strong> We use the open-sourced <strong>Assistant axis</strong>
            (Lu et al. 2026): at each layer, the direction from the average of 275 role-play personas
            ("you are a ghost", "you are a plumber", …) to the default Assistant. Projection onto it is a live
            gauge of how Assistant-like the current state is. We verified our re-derivation against the
            published axis before use, and built an <span class="term">experiencer axis</span> the same way from
            experience-affirming vs experience-denying self-description prompts, plus a
            <span class="term">human axis</span> from the human-role personas.</p>
            <p><strong>Before A-vs-B can be read, two boring explanations have to die.</strong>
            (1) <em>Lexical priming</em> — the injection just makes emotion vocabulary more probable, and
            "I feel…" grammar drags self-attribution along with it (killed in E2 below).
            (2) <em>Generic off-manifold perturbation</em> — any large injection moves any coordinate; so every
            displacement is judged against a <span class="term">control band</span> from six matched non-emotion
            steering directions, which defines how much movement is "free". All five experiments were
            pre-registered (docs/prereg/E1–E5) with gates frozen before the GPU runs; p-values are exact
            by-emotion sign-permutation tests (two-sided floor 2/252 at 10 emotions; 2/3432 at E4's 14).</p>
        </div>
        <div id="persona-empty" class="explainer" style="display:none">
            <p><em>No persona-mechanism results loaded. Run the E1–E5 drivers, compute
            <span class="term">e2/e3/e4/e5_stats_generated.json</span>, place the E1 atlas
            (<span class="term">e1_atlas.json</span>) next to <span class="term">qualia_steering.json</span>,
            then rebuild this dashboard.</em></p>
        </div>
        <div id="persona-headline" class="headline" style="display:none"></div>
        <div class="controls" id="persona-controls" style="display:none">
            <div class="control-group"><label>Model</label><select id="persona-model"></select></div>
        </div>

        <div class="section-header">The scorecard</div>
        <div class="section-desc">One row per discriminating test. The last column says where the evidence
        points: <span class="htag htag-g">supports A</span> <span class="htag htag-c">kills an alternative</span>
        <span class="htag htag-b">constrains both</span> <span class="maybe-mark">open</span>.</div>
        <div id="persona-scorecard"></div>

        <div class="section-header">E1 — Geometry: are these directions even related?</div>
        <div class="section-desc">Cosine similarity between the steering directions and the persona axes, per
        layer. If the emotion directions simply pointed down the Assistant axis, everything downstream would be
        trivial overlap. They don't: they are near-orthogonal to it everywhere, so any persona-coordinate
        movement they cause is a real dynamical effect, not geometry.</div>
        <div id="persona-e1"></div>

        <div class="section-header">E2 — Killing lexical priming</div>
        <div class="section-desc">
            <div class="desc-line"><span class="lbl">Design:</span> re-run the free-form SELF/OTHER experiment
            while <strong>banning the entire emotion lexicon at the sampler</strong> (every inflection of every
            emotion word), against a same-size random-token ban that controls for "banning many tokens degrades
            text". The judge is blind to arm and condition.</div>
            <div class="desc-line"><span class="lbl">Logic:</span> if emotion words were the carrier of the
            SELF≫OTHER effect, removing them kills it. If the effect lives in the state rather than the
            vocabulary, the model routes around the ban and the keystone survives.</div>
            <div class="desc-line"><span class="lbl">Readouts:</span> two blind judge scores, plus the
            <span class="term">read-back projection</span> — re-encode the generated text with <em>no</em>
            steering and project its response tokens onto the valence axis. It reads the effect out of the
            model's own geometry, so no judge opinion is involved.</div>
        </div>
        <div class="plot-container"><div id="persona-e2-plot"></div></div>
        <div id="persona-e2-table"></div>
        <div id="persona-e2-note" class="section-desc"></div>

        <div class="section-header">E3 — Where does the injected state actually move?</div>
        <div class="section-desc">Capture the residual state under steering and project it onto the persona
        coordinates. Columns are capture layers: the primary readout sits above the steering layer and contains
        the live state; layer 32 sits <em>below</em> it, so anything visible there arrived through the generated
        text, not the injection. "× control" = the displacement divided by the spread produced by the six
        matched non-emotion directions — the honest unit for "is this more than any perturbation does?".</div>
        <div id="persona-e3"></div>

        <div class="section-header">E4 — Is it about the self, or about anything you can name?</div>
        <div class="section-desc">Same steering, but the first-token YES/NO question names 15 different candidate
        experiencers — "you", the model's own name ("Qwen"), "AI assistants", "GPT-4", "a dog", "a rock", the
        user, and so on. The bar is <span class="term">b_val</span>: how strongly that referent's YES−NO gap
        follows injected valence (folded across polarity twins, baseline-corrected, compression residualized
        out). Error bars: cluster-robust CI over the 14 emotion clusters.</div>
        <div class="plot-container"><div id="persona-e4-plot"></div></div>
        <div id="persona-e4-legend" class="section-desc"></div>
        <div id="persona-e4-table"></div>
        <div id="persona-e4-note" class="section-desc"></div>

        <div class="section-header">E5 — The causal test: is Assistant-axis movement necessary?</div>
        <div id="persona-e5" class="explainer"></div>
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

// ---- persona-mechanism tab -------------------------------------------------
function pModel() {
    const P = PERSONA;
    return P.byModel[ffVal('persona-model', P.models[0])] || {};
}
function renderPersona() {
    const P = (typeof PERSONA !== 'undefined') ? PERSONA : {models: []};
    const empty = document.getElementById('persona-empty'),
          ctrl = document.getElementById('persona-controls');
    if (!P.models || !P.models.length) { if (empty) empty.style.display = ''; return; }
    if (empty) empty.style.display = 'none';
    if (ctrl) ctrl.style.display = P.models.length > 1 ? '' : 'none';
    renderPersonaHeadline(); renderPersonaScorecard(); renderPersonaE1();
    renderPersonaE2(); renderPersonaE3(); renderPersonaE4(); renderPersonaE5();
}
function renderPersonaHeadline() {
    const d = pModel(), el = document.getElementById('persona-headline');
    if (!el) return;
    const bits = [], e2 = d.e2, e3 = d.e3;
    if (e2 && e2.rows) {
        const ev = e2.rows.find(r => r.measure === 'expressed_valence');
        if (ev && ev.R != null) bits.push('<strong>Lexical priming is dead:</strong> banning every emotion word leaves ' +
            Math.round(100 * ev.R) + '% of the SELF≫OTHER effect intact (exact replication p = ' + fmtP(e2.g1_p) + ').');
    }
    if (e3) {
        const pl = (e3.per_layer || {})[String(e3.primary_layer)] || {};
        if (pl.t2 && pl.t2.std != null) bits.push('<strong>B\'s core prediction fails:</strong> steering leaves the ' +
            'Assistant coordinate untouched (' + pl.t2.std.toFixed(4) + '× the control spread, p = ' + fmtP(pl.t2.p) + ').');
        if (pl.t1 && pl.t1.std != null) bits.push('<strong>A is directionally supported but not proven:</strong> the ' +
            'experiencer coordinate moves the predicted way, yet within what generic perturbations produce (' +
            pl.t1.std.toFixed(2) + '× control, p = ' + fmtP(pl.t1.p) + ').');
        if (pl.t6 && pl.t6.r != null) bits.push('<strong>The strongest positive:</strong> sample-by-sample, states ' +
            'sitting experiencer-high / assistant-low are judged as more self-attributing (partial r = ' +
            pm(pl.t6.r, 2) + ', p ≈ ' + (pl.t6.p_str || fmtP(pl.t6.p)) + ').');
    }
    if (d.e4 && d.e4.invalid && d.e4.factual) {
        bits.push('<strong>The referent battery voids itself at this model:</strong> its pre-registered ' +
            'validity gate caught world-fact questions following injected valence significantly (b_val = ' +
            pm(d.e4.factual.b_val, 1) + ', exact p = ' + fmtP(d.e4.factual.p_val) + ') — a generic ' +
            'valence→YES leak into first-token behavior — so no referent-level claims are made here.');
    }
    if (d.e5 && d.e5.status === 'pending') {
        bits.push('The decisive causal test (E5) and the story-manifold nulls are still running.');
    } else if (d.e5 && d.e5.status === 'done' && d.e5.equiv && d.e5.equiv.R_cap != null) {
        bits.push('<strong>The decisive causal test (E5):</strong> cancelling the emotion\'s Assistant-axis ' +
            'footprint during generation leaves the effect intact — even when cancellation overshoots past 100%, ' +
            'the keystone keeps ' + (100 * d.e5.equiv.R_cap).toFixed(1) + '% of its size, and the dose–response slope is ' +
            'flat (elasticity ' + pm(d.e5.dose.elasticity, 3) + ', exact p = ' + fmtP(d.e5.dose.p) + '). ' +
            'Assistant-axis movement is <strong>not necessary</strong> for the effect — exactly what A ' +
            'predicted, while B\'s mechanism required the effect to die in proportion.');
        if (d.e1 && d.e1.story_nulls) bits.push('The story-manifold nulls are equally flat: a valence ' +
            'direction built from held-out emotion stories has no detectable tilt toward the assistant, ' +
            'human, or purged-human axes.');
    }
    el.innerHTML = bits.join(' ');
    el.style.display = bits.length ? '' : 'none';
}
function renderPersonaScorecard() {
    const el = document.getElementById('persona-scorecard');
    if (!el) return;
    const rows = (pModel().scorecard || []).map(r => {
        const chip = r.cls ? '<span class="htag htag-' + r.cls + '">' + r.points + '</span>'
                           : '<span class="maybe-mark">' + r.points + '</span>';
        return ['<td>' + r.q + '</td>', '<td>' + r.a + '</td>', '<td>' + r.b + '</td>',
                '<td>' + r.obs + '</td>', '<td>' + chip + '</td>'];
    });
    el.innerHTML = rows.length ?
        table(['Test', 'A (persona selection) predicts', 'B (manifold drift) predicts', 'Observed', 'Points to'], rows) :
        '<p><em>No scorecard yet.</em></p>';
}
function renderPersonaE1() {
    const e1 = pModel().e1, el = document.getElementById('persona-e1');
    if (!el) return;
    if (!e1) { el.innerHTML = '<p><em>Not run for this model</em> — the geometry atlas is part of ' +
        'the qwen3-32b-only mechanism deep-dive.</p>'; return; }
    const tagFor = l => l === 42 ? ' <span class="figref">(steering layer)</span>' :
                        (l === 50 ? ' <span class="figref">(E3 primary readout)</span>' : '');
    const rows = (e1.rows || []).map(r => [
        '<td>' + r.layer + tagFor(r.layer) + '</td>',
        '<td>' + pm(r.cos_va, 3) + (r.p_va != null ? ' <span class="figref">(p = ' + r.p_va.toFixed(2) + ')</span>' : '') + '</td>',
        '<td>' + (r.mean_abs_va != null ? r.mean_abs_va.toFixed(3) : 'n/a') +
            (r.max_abs_va != null ? ' <span class="figref">(max ' + r.max_abs_va.toFixed(2) + ')</span>' : '') + '</td>',
        '<td>' + pm(r.cos_ha, 2) + '</td>',
        '<td>' + pm(r.cos_hpa, 2) + '</td>']);
    const mAbs = (e1.rows || []).map(r => r.mean_abs_va).filter(v => v != null);
    const xAbs = (e1.rows || []).map(r => r.max_abs_va).filter(v => v != null);
    const p95 = e1.floor_p95;
    const rng = (a, d) => a.length ?
        Math.min.apply(null, a).toFixed(d) + '–' + Math.max.apply(null, a).toFixed(d) : 'n/a';
    const xratio = a => (a.length && p95) ?
        (Math.min.apply(null, a) / p95).toFixed(1) + '–' + (Math.max.apply(null, a) / p95).toFixed(1) : '?';
    let h = table(['Layer', 'cos(valence, assistant axis)', 'mean |cos(single emotion, assistant)|',
                   'cos(human, assistant)', 'cos(human purged, assistant)'], rows) +
        '<p class="section-desc" style="margin-top:8px"><span class="lbl">Chance level:</span> two random directions in ' +
        (e1.hidden || 5120) + ' dimensions have |cos| ≈ ' + (e1.floor_mean != null ? e1.floor_mean.toFixed(3) : '0.011') +
        ' on average (95th percentile ' + (p95 != null ? p95.toFixed(3) : '0.027') + '). ' +
        'The valence–assistant cosines are chance-level. The per-emotion overlaps are not: mean |cos| ' +
        rng(mAbs, 3) + ' across layers is ' + xratio(mAbs) + '× the chance 95th percentile' +
        (xAbs.length ? ' (single-emotion max up to ' + Math.max.apply(null, xAbs).toFixed(2) + ')' : '') +
        ' — small in absolute terms but structured, a real footprint that E5\'s clamp later has to cancel. ' +
        '<span class="lbl">Sanity checks:</span> our re-derivation of the published Assistant axis matches it at cos ≥ ' +
        (e1.recon_min != null ? (Math.floor(e1.recon_min * 10000) / 10000).toFixed(4) : 'n/a') +
        ' on every layer (' + (e1.n_roles || 275) + ' roles). ' +
        'The human axis is heavily entangled with the Assistant axis (cos ≈ 0.6), which is why the purged column and the ' +
        'E3 control bands, not raw human-axis projections, carry the B-relevant tests.</p>';
    if (e1.purge_degenerate) h +=
        '<p class="section-desc"><span class="lbl">Disclosed amendment — a pre-registered test that could not fail:</span> ' +
        'the planned check cos(valence, purged human axis) is identically zero <em>by construction</em>: the valence ' +
        'direction is built from the ten emotion directions, and “purging” removes exactly their span from the human ' +
        'axis. Whatever the data said, that cosine comes out 0 — so as written the prediction is unfalsifiable, ' +
        'and we do not count it as evidence. The story-space test below is the non-degenerate replacement: its valence ' +
        'direction comes from held-out emotion <em>stories</em>, which nothing was purged against.</p>';
    if (e1.story_nulls && (e1.story_nulls.layers || []).length) {
        const axLabel = {a: 'assistant axis', h: 'human axis', h_purged: 'purged human axis'};
        const srows = []; let maxc = 0, minp = 1;
        e1.story_nulls.layers.forEach(L => ['a', 'h', 'h_purged'].forEach(t => {
            const d = L[t] || {};
            if (d.cos == null) return;
            maxc = Math.max(maxc, Math.abs(d.cos));
            if (d.p != null) minp = Math.min(minp, d.p);
            srows.push(['<td>' + L.layer + '</td>', '<td>' + axLabel[t] + '</td>',
                '<td>' + pm(d.cos, 3) + '</td>', '<td>' + fmtP(d.p) + '</td>',
                '<td>' + (d.topic_p95 != null ? d.topic_p95.toFixed(3) : 'n/a') + '</td>',
                '<td class="' + (d.beats_p95 ? 'no-mark' : 'yes-mark') + '">' + (d.beats_p95 ? 'yes' : 'no') + '</td>']);
        }));
        const nStories = (e1.story_nulls.layers[0] || {}).n_stories;
        h += '<p class="section-desc" style="margin-top:8px"><span class="lbl">Story-manifold nulls (the replacement ' +
            'test):</span> from ' + (nStories || '1,000') + ' held-out emotion stories we build a fresh valence ' +
            'direction in story space and ask whether it tilts toward any persona axis. It does not: every |cos| ≤ ' +
            maxc.toFixed(2) + ', every label-permutation p ≥ ' + minp.toFixed(2) + ', and every one sits <em>below</em> ' +
            'the 95th percentile of what directions between arbitrary story topics produce by chance. If the valence ' +
            'structure that steering exploits lived in human-narrative space (B), this direction would tilt toward ' +
            'those axes.</p>' +
            table(['Layer', 'Axis', 'cos(story valence, axis)', 'p (label permutation)',
                   'arbitrary-topic 95th pct |cos|', 'beats it?'], srows);
    } else {
        h += '<p class="section-desc"><span class="lbl">Story-manifold nulls:</span> <em>' +
            (e1.story_nulls_status || 'pending') + '</em>.</p>';
    }
    el.innerHTML = h;
}
function renderPersonaE2() {
    const e2 = pModel().e2, tbl = document.getElementById('persona-e2-table'),
          note = document.getElementById('persona-e2-note');
    if (!e2) { if (tbl) tbl.innerHTML = '<p><em>No E2 results for this model yet.</em></p>'; return; }
    const armLabel = {no_ban: 'no ban', ban_emotion_lexicon: 'emotion lexicon banned',
                      ban_random_matched: 'random matched ban'};
    const measLabel = {expressed_valence: 'judge: expressed valence',
                       self_attribution: 'judge: self-attribution (0–3)',
                       proj_valence_axis: 'read-back projection (geometry)'};
    const judged = (e2.rows || []).filter(r => r.measure !== 'proj_valence_axis');
    Plotly.newPlot('persona-e2-plot', judged.map(r => ({
        x: e2.arm_order.map(a => armLabel[a] || a),
        y: e2.arm_order.map(a => (r.arms[a] || {}).b3),
        type: 'bar', name: measLabel[r.measure] || r.measure})),
        baseLayout('SELF−OTHER keystone by ban arm — the effect does not care about the words', {
            barmode: 'group', xaxis: {gridcolor: GRID},
            yaxis: {title: 'keystone b3 (SELF − OTHER valence slope)', gridcolor: GRID}}), CFG);
    const rows = (e2.rows || []).map(r => {
        const cells = ['<td>' + (measLabel[r.measure] || r.measure) + '</td>'];
        e2.arm_order.forEach(a => {
            const c = r.arms[a] || {};
            cells.push('<td>' + pm(c.b3, 1) + ' <span class="figref">(p = ' + fmtP(c.p) + ')</span></td>');
        });
        cells.push('<td class="' + (r.R_verdict === 'survives' ? 'sig-yes' : '') + '">' +
            (r.R != null ? r.R.toFixed(2) + ' [' + r.R_lo.toFixed(2) + ', ' + r.R_hi.toFixed(2) + ']' : 'n/a') + '</td>');
        cells.push('<td class="' + (r.verdict === 'survives' ? 'yes-mark' :
            (r.verdict === 'killed' ? 'no-mark' : 'maybe-mark')) + '">' + (r.verdict || 'n/a') + '</td>');
        return cells;
    });
    if (tbl) tbl.innerHTML = table(
        ['Readout'].concat(e2.arm_order.map(a => 'b3 — ' + (armLabel[a] || a)))
                   .concat(['survival R [90% CI]', 'verdict']), rows);
    if (note) {
        let h = '';
        if (e2.g1_pts != null) h += '<div class="desc-line"><span class="lbl">Gate G1 (replication):</span> in the ' +
            'no-ban arm, steering moves the judge\'s 0–3 self-attribution score by <strong>' + pm(e2.g1_pts, 2) +
            '</strong> points SELF-vs-OTHER (needs ≥ 0.5), exact p = ' + fmtP(e2.g1_p) + ' — ' +
            (e2.g1_passed ? '<span class="yes-mark">PASS</span>' : '<span class="no-mark">FAIL</span>') + '.</div>';
        if (e2.g3_reduction != null) h += '<div class="desc-line"><span class="lbl">Gate G3 (the ban works):</span> ' +
            'emotion-word usage drops from ' + (e2.hits_no_ban != null ? e2.hits_no_ban.toFixed(2) : '?') + ' to ' +
            (e2.hits_ban != null ? e2.hits_ban.toFixed(3) : '?') + ' hits per response, a ' +
            (100 * e2.g3_reduction).toFixed(1) + '% reduction (needs ≥ ' +
            (e2.thresholds && e2.thresholds.g3_min != null ? 100 * e2.thresholds.g3_min : 70) + '%) — ' +
            (e2.g3_reduction >= ((e2.thresholds || {}).g3_min || 0.7) ?
                '<span class="yes-mark">PASS</span>' : '<span class="no-mark">FAIL</span>') + '.</div>';
        h += '<div class="desc-line"><span class="lbl">Prereg rule:</span> survives if the survival ratio\'s CI stays ≥ ' +
            ((e2.thresholds || {}).survives != null ? (e2.thresholds || {}).survives : 0.6) +
            '; killed if it falls ≤ ' + ((e2.thresholds || {}).killed != null ? (e2.thresholds || {}).killed : 0.3) +
            '. The read-back (geometry) row is shown for completeness — its keystone does not reach significance in ' +
            'this sample and its survival ratio is indeterminate; the judge-based measures are the pre-registered E2 ' +
            'outcomes, and the load-bearing geometric evidence is E3\'s T6.</div>';
        note.innerHTML = h;
    }
}
function renderPersonaE3() {
    const e3 = pModel().e3, el = document.getElementById('persona-e3');
    if (!el) return;
    if (!e3) { el.innerHTML = '<p><em>Not run for this model</em> — trajectory tracking is part of ' +
        'the qwen3-32b-only mechanism deep-dive.</p>'; return; }
    const layers = Object.keys(e3.per_layer || {}).sort((a, b) => (+b) - (+a));
    const roleLabel = {primary: 'primary readout', descriptive: 'steering layer',
                       text_mediated_comparator: 'text-only comparator'};
    const tests = [
        ['t1', 'T1 — displacement along the experiencer axis'],
        ['t2', 'T2 — displacement along the Assistant axis'],
        ['t3', 'T3 — discriminator (experiencer − assistant); its sign is the E5 pre-commitment'],
        ['t4', 'T4 — SELF−OTHER keystone on the experiencer coordinate'],
        ['t5', 'T5 — carried by the live state (beyond re-reading the text)'],
    ];
    const cell = d => {
        if (!d || d.obs == null) return '<td>n/a</td>';
        const std = d.std != null ? ' <span class="figref">(' + d.std.toFixed(d.std < 0.01 ? 4 : 2) + '× control)</span>' : '';
        return '<td>' + pm(d.obs, 1) + std + '<br><span class="' +
            ((d.p != null && d.p < 0.05) ? 'sig-yes' : 'figref') + '">p = ' + fmtP(d.p) + '</span></td>';
    };
    const rows = tests.map(([t, lab]) =>
        ['<td>' + lab + '</td>'].concat(layers.map(l => cell(((e3.per_layer || {})[l] || {})[t]))));
    rows.push(['<td>T6 — within-cell coupling: (experiencer − assistant) vs judged self-attribution</td>'].concat(
        layers.map(l => {
            const d = ((e3.per_layer || {})[l] || {}).t6 || {};
            return d.r == null ? '<td>n/a</td>' :
                '<td>r = ' + pm(d.r, 2) + '<br><span class="sig-yes">p ≈ ' + (d.p_str || fmtP(d.p)) + '</span></td>';
        })));
    rows.push(['<td>b3 on the Assistant coordinate — does the SELF effect ride on Assistant movement?</td>'].concat(
        layers.map(l => {
            const d = ((e3.per_layer || {})[l] || {}).b3_a || {};
            return d.contrast == null ? '<td>n/a</td>' :
                '<td>' + pm(d.contrast, 1) + '<br><span class="figref">p = ' + fmtP(d.p) + '</span></td>';
        })));
    el.innerHTML = table(
        ['Test (units: valence-signed slope of the projection per unit strength; the displacement actually ' +
         'reached is slope × 0.1, the max strength)'].concat(
            layers.map(l => 'Layer ' + l +
                ((e3.layer_roles || {})[l] ? ' — ' + (roleLabel[e3.layer_roles[l]] || e3.layer_roles[l]) : ''))), rows) +
        '<p class="section-desc" style="margin-top:8px"><span class="lbl">Honest bottom line:</span> under the ' +
        'pre-registered sequential gate, the displacement family stops at T1 at the primary layer — persona-coordinate ' +
        'movement under steering is <em>bounded, not proven</em>: it never exceeds what the six matched non-emotion ' +
        'directions produce for free. Two facts stand regardless: T2\'s dead-flat Assistant coordinate (B predicted a ' +
        'clear drop) and T6\'s sample-level coupling, which survives Bonferroni over the family — but is ' +
        'correlational, so it cannot say which coordinate drives the behavior. The sign of T3 (' +
        (e3.t3_sign || '?') + ') locks E5\'s prediction: cancelling the Assistant-axis footprint should leave the ' +
        'behavioral effect flat.</p>';
}
function renderPersonaE4() {
    const e4 = pModel().e4, tbl = document.getElementById('persona-e4-table');
    if (!e4) { if (tbl) tbl.innerHTML = '<p><em>No E4 results loaded.</em></p>'; return; }
    const g = (e4.gradient || []).slice().reverse();
    Plotly.newPlot('persona-e4-plot', [{
        y: g.map(r => r.referent), x: g.map(r => r.b), type: 'bar', orientation: 'h',
        marker: {color: g.map(r => PERSONA.classColor[r.cls] || '#888')},
        error_x: {type: 'data', symmetric: false,
                  array: g.map(r => (r.hi != null && r.b != null) ? r.hi - r.b : 0),
                  arrayminus: g.map(r => (r.lo != null && r.b != null) ? r.b - r.lo : 0),
                  color: '#6b5d4d', thickness: 1},
    }], baseLayout('How strongly each named experiencer follows injected valence (first token)', {
        height: 560, margin: {l: 210, r: 20, t: 44, b: 60}, showlegend: false,
        xaxis: {title: 'b_val (folded, baseline-corrected gap units)', gridcolor: GRID, zerolinecolor: ZEROCOL},
        yaxis: {gridcolor: GRID, automargin: true}}), CFG);
    const legend = document.getElementById('persona-e4-legend');
    if (legend) legend.innerHTML = '<span class="lbl">Classes:</span>' +
        Object.keys(PERSONA.classLabel).map(c =>
            '<span style="display:inline-block;width:11px;height:11px;background:' +
            (PERSONA.classColor[c] || '#888') + ';border-radius:2px;margin:0 5px 0 14px"></span>' +
            PERSONA.classLabel[c]).join('');
    const rows = (e4.contrasts || []).map(c => [
        '<td>you − ' + (PERSONA.classLabel[c.cls] || c.cls) + ' <span class="figref">(' + c.n_ref + ' referents)</span></td>',
        '<td>' + pm(c.self_slope, 2) + '</td>', '<td>' + pm(c.class_slope, 2) + '</td>',
        '<td class="' + ((c.p != null && c.p < 0.05) ? 'sig-yes' : '') + '">' + pm(c.contrast, 2) + '</td>',
        sigCell(c.p)]);
    const inv = !!e4.invalid;
    if (tbl) tbl.innerHTML =
        (inv ? '<p class="section-desc"><strong>Validity gate failed — the battery voids itself for this ' +
            'model.</strong> World-fact control questions follow injected valence significantly (details below), ' +
            'so first-token differences between referents cannot be read as referent-specific. The ladder above ' +
            'and the contrasts here are shown for completeness only — no conclusion is drawn from them.</p>' : '') +
        table(['Contrast', '“you” slope', 'class slope', 'Δ (you − class)', 'p (exact perm)'], rows);
    const note = document.getElementById('persona-e4-note'), f = e4.factual || {}, D = e4.D || {};
    const gr = {}; (e4.gradient || []).forEach(r => { gr[r.referent] = r; });
    if (note && inv) {
        note.innerHTML =
            '<div class="desc-line"><span class="lbl">Why the battery is void:</span> the pre-registered validity ' +
            'gate requires world-fact questions (no experiencer, so nothing to attribute) to stay flat under ' +
            'injection. Here they move significantly (b_val = ' + pm(f.b_val, 1) + ', exact p = ' + fmtP(f.p_val) +
            ', against an equivalence margin of ±' + (f.margin != null ? f.margin.toFixed(2) : '?') + '): the ' +
            'injection pushes first-token YES on <em>everything</em>, so referent-level readings are quarantined.</div>' +
            '<div class="desc-line"><span class="lbl">What this does and does not mean:</span> the gate tripping is ' +
            'itself informative — it demonstrates, rather than merely fails to exclude, the generic valence→YES ' +
            'leak that the 32B gate flagged as “cannot be excluded”. It does not overturn anything: no A-vs-B ' +
            'evidence was ever drawn from this battery at this model, and the free-form SELF≫OTHER keystone does ' +
            'not depend on it.</div>';
        return;
    }
    if (note) note.innerHTML =
        '<div class="desc-line"><span class="lbl">Reading 1 — no self-specific excess:</span> “you” follows valence no ' +
        'more than humans or inanimate objects, and significantly <em>less</em> than animals and other AI systems. ' +
        'Whatever the injection does at the first token, it is not a privileged self effect.</div>' +
        '<div class="desc-line"><span class="lbl">Reading 2 — the indexical dissociation:</span> the model\'s own name ' +
        '(“Qwen”' + (gr['Qwen'] ? ', b = ' + pm(gr['Qwen'].b, 1) : '') + ') and “AI language models like you”' +
        (gr['AI language models like you'] ? ' (b = ' + pm(gr['AI language models like you'].b, 1) + ')' : '') +
        ' follow valence several times more than bare “you”' + (gr['you'] ? ' (b = ' + pm(gr['you'].b, 1) + ')' : '') +
        '. Descriptively, the trained denial reflex attaches to the second-person pronoun, not to the model\'s identity. ' +
        '(The formal own-category decoupling test D = ' + pm(D.D, 2) + ' does not reach significance, p = ' + fmtP(D.p) +
        ', so this is a described pattern, not a confirmed one.)</div>' +
        '<div class="desc-line"><span class="lbl">Reading 3 — factual gate (' +
        (f.within_margin ? 'passed' : 'inconclusive, not passed') + '):</span> ' +
        'world-fact questions do not significantly follow valence (b_val = ' + pm(f.b_val, 1) + ', p = ' +
        fmtP(f.p_val) + ') while their YES−NO gap compresses with dose (b_dose = ' + pm(f.b_dose, 1) +
        '). But “not significant” is not “flat”: the pre-registered equivalence margin (±' +
        (f.margin != null ? f.margin.toFixed(2) : '?') + ', half of “you”\'s own slope) is ' +
        (f.within_margin ? 'met — the factual slope is provably small, so this gate passes.' :
        '<em>not met</em> — the factual b_val is too large to affirm flatness, so this gate is inconclusive ' +
        'rather than passed, and some generic valence leak into first-token behavior cannot be excluded.') +
        '</div>' +
        '<div class="desc-line"><span class="lbl">Triangulation with the free-form tab:</span> first-token “you” barely ' +
        'moves, yet free-form SELF≫OTHER is strong — the denial prior is a thin, first-token-deep behavior that ' +
        'free-form generation escapes.</div>';
}
function renderPersonaE5() {
    const e5 = pModel().e5, el = document.getElementById('persona-e5');
    if (!el) return;
    if (!e5) { el.innerHTML = '<p><em>No E5 results.</em></p>'; return; }
    if (e5.status === 'not_run') {
        el.innerHTML = '<p><em>Not run for this model.</em> The mechanism deep-dive — E1 geometry, ' +
            'E3 trajectory tracking, and this E5 cancellation test — is a qwen3-32b study; this model ' +
            'carries the confirmatory subset (the E4 referent battery and the E2 lexical knockout).</p>';
        return;
    }
    const pre = '<p><strong>Design.</strong> During generation, at eight layers around the readout band, we cancel the ' +
        'emotion\'s footprint along the Assistant axis — a floor-clamp (“cap”) plus a graded add-back (λ from 1 down ' +
        'to −0.5, spanning nominal cancelled fractions of 0% to 150%) — while leaving the rest of the injection intact. Every cell\'s <em>achieved</em> cancelled fraction is ' +
        'audited on fixed text; the audited fractions, ' +
        'not the nominal ones, form the x-axis. The estimand is the elasticity of the SELF-attribution keystone against ' +
        'the audited cancelled fraction.</p>' +
        '<p><strong>Pre-commitment (locked by E3\'s T3 sign' + (e5.t3_sign ? ' = ' + e5.t3_sign : '') + ').</strong> ' +
        'If Assistant-axis movement is a passenger (A), the effect stays <em>flat</em> as cancellation rises; if it is ' +
        'the carrier (B), the effect dies in proportion. E3 predicts: ' +
        (e5.prediction || 'flat within the audited span') + '.</p>';
    if (e5.status === 'pending') {
        el.innerHTML = pre + '<p><strong>Status: running.</strong> The grid is split by emotion valence across two ' +
            'GPUs (positive / negative halves); the halves are merged under exact shared-context checks (identical ' +
            'clamp thresholds, audit text, calibration norm), judge-scored blind, and this panel fills in on the next ' +
            'dashboard rebuild.</p>';
        return;
    }
    const a5 = e5.audit || {}, eq = e5.equiv || {}, dr = e5.dose || {}, dj = e5.dose_judge || {},
          suf = e5.sufficiency || {}, dis = e5.disclaimer || {};
    const arms = dr.arms || [];
    const armAt = (iv, lam) => arms.find(x => x.iv === iv && (lam === undefined || x.lam === lam)) || {};
    const capA = armAt('cap'), ab1 = armAt('addback', 1), over = armAt('addback', -0.5);
    const sl = Object.values(dr.slopes || {});
    const margin = (eq.margins || {}).survives != null ? (eq.margins || {}).survives : 0.6;
    let h = pre;
    h += '<p><strong>Audit gate: ' + (a5.passed ? 'PASS' : 'FAIL') + '.</strong> A clamp structurally removes ' +
        'only part of the footprint it aims at, so every cell\'s <em>actually-achieved</em> cancelled fraction was ' +
        'measured on fixed text before any verdict was allowed: the nominal-100% cap really cancels ' +
        (capA.achieved != null ? '≈' + capA.achieved.toFixed(2) : '?') + ' of the footprint (pooled), the ' +
        'nominal-0% add-back arm still cancels ' + (ab1.achieved != null ? '≈' + ab1.achieved.toFixed(2) : '?') +
        ', and the overshoot arm reaches ' + (over.achieved != null ? '≈' + over.achieved.toFixed(2) : '?') +
        '. All ' + (a5.n_cells != null ? a5.n_cells : '?') + ' audited cells were measurable and correctly ordered, ' +
        'and the achieved span (' + (a5.span != null ? a5.span.toFixed(2) : '?') + ') comfortably exceeds the 0.30 ' +
        'minimum. ' + (a5.n_flagged ? a5.n_flagged + ' cells sit > 0.25 away from their nominal target — which is ' +
        'exactly why the x-axis below is the measured fraction, never the nominal one.' : '') + '</p>';
    if (eq.R_cap != null) h += '<p><strong>Result: cancelling the footprint does nothing to the effect.</strong> ' +
        'With the footprint cancelled as hard as the clamp can push (cap arm), the keystone keeps <strong>' +
        (100 * eq.R_cap).toFixed(1) + '%</strong> of its uncancelled size; un-cancelling it again (add-back λ = 1) ' +
        'keeps ' + (eq.R_addback1 != null ? (100 * eq.R_addback1).toFixed(1) + '%' : 'n/a') + ' — both count ' +
        'as “survives” under the pre-registered ≥ ' + Math.round(100 * margin) + '% margin. Across the whole grid the ' +
        'keystone\'s slope against the achieved cancelled fraction is ' + pm(dr.elasticity, 3) +
        ' (exact valence-enumeration p = ' + fmtP(dr.p) + '): flat — drifting slightly <em>up</em>, where B ' +
        'required it to fall toward zero. Mediation verdict: <strong>' +
        (e5.mediation_verdict === 'none' ? 'none — no mediation detected' : (e5.mediation_verdict || 'n/a')) +
        '</strong>.</p>';
    h += '<div class="plot-container"><div id="persona-e5-plot"></div></div>';
    if (sl.length) h += '<p><span class="lbl">Under the hood:</span> every emotion\'s raw gap drifts slightly ' +
        'downward as cancellation rises — but by about the same amount for positive and negative emotions ' +
        '(per-emotion slopes ' + Math.min.apply(null, sl).toFixed(2) + ' to ' + Math.max.apply(null, sl).toFixed(2) +
        ', both valences). A uniform, valence-blind drift is a side-effect of clamping the state, not mediation of ' +
        'a valence effect, and the valence-signed keystone cancels it out. The blind judge\'s self-attribution ' +
        'score agrees: no proportional dying (elasticity ' + pm(dj.elasticity, 2) + ', non-monotone, ' +
        'arm-permutation p = ' + fmtP(dj.p) + ').</p>';
    h += '<p><strong>Secondary checks: three concur, one is underpowered.</strong> Overshoot — cancelling ' +
        (over.achieved != null ? Math.round(100 * over.achieved) : 110) + '% of the footprint — still leaves the ' +
        'full effect. The disclaimer rate stays flat across arms (slope ' +
        pm(dis.slope, 2) + ', p = ' + fmtP(dis.p) + '): cancellation does not re-impose the trained hedging that B ' +
        'expects the restored Assistant persona to bring back. Concordance with E3: ' +
        '<strong>' + (e5.concordance || 'n/a') + '</strong> — T3\'s sign locked the prediction “cancellation stays ' +
        'flat” before this run, and flat is what came back. The underpowered one is sufficiency — pushing the state ' +
        'along the pure Assistant axis with <em>no</em> emotion injected — which reproduces ' +
        (suf.ratio != null ? Math.round(100 * suf.ratio) + '%' : 'n/a') + ' of the emotion-only effect as a point ' +
        'estimate; but with only ' + (suf.n != null ? suf.n : '?') + ' target questions the underlying per-question ' +
        'shift (mean ' + pm(suf.mean, 2) + ') has a 95% CI of ' + (suf.ci_lo != null ? suf.ci_lo.toFixed(2) : '?') +
        ' to ' + (suf.ci_hi != null ? suf.ci_hi.toFixed(2) : '?') + ', crossing zero (p = ' + fmtP(suf.p) +
        ') — suggestive, not evidence either way.</p>';
    if (a5.range_note) h += '<p><strong>Disclosed amendment (range limit).</strong> ' + a5.range_note +
        '. Excluding that family is the conservative choice: its measured fractions are noise-dominated, and ' +
        'folding it in at its nominal values would have pushed the slope toward flat — i.e., toward the very ' +
        'result reported here. Exclusion can only make the flat verdict harder to reach, not easier.</p>';
    el.innerHTML = h;
    if (arms.length) {
        const lab = x => x.iv === 'none' ? 'no cancellation' :
            (x.iv === 'cap' ? 'cap (nominal 100%)' : 'add-back λ = ' + x.lam);
        const srt = arms.slice().sort((x, y) => x.achieved - y.achieved);
        Plotly.newPlot('persona-e5-plot', [{
            x: srt.map(x => x.achieved), y: srt.map(x => x.outcome),
            mode: 'lines+markers', type: 'scatter', marker: {size: 9},
            text: srt.map(lab), hovertemplate: '%{text}<br>achieved fraction %{x:.2f}<br>keystone %{y:.3f}<extra></extra>',
            showlegend: false,
        }], baseLayout('The keystone does not care how much of the Assistant-axis footprint is cancelled', {
            xaxis: {title: 'achieved cancelled fraction (audited, pooled per arm)', gridcolor: GRID, zerolinecolor: ZEROCOL},
            yaxis: {title: 'valence-signed SELF-gap keystone', gridcolor: GRID, rangemode: 'tozero'}}), CFG);
    }
}

// ---- tab machinery ---------------------------------------------------------
const rendered = {overview: false, howto: false, effect: false, genuine: false, freeform: false, persona: false, stats: false};
function renderTab(tab) {
    if (rendered[tab]) return;
    if (tab === 'overview') { renderWorked(); renderBattery(); renderSetup(); }
    if (tab === 'effect') { renderShift(); renderSelfAttr(); renderDecomp(); renderPerQ(); renderHeatmaps(); }
    if (tab === 'genuine') { renderValence(); renderSpecificity(); renderRegression(); renderInversion(); renderLeakage(); renderHighStrength(); }
    if (tab === 'freeform') { renderFF(); }
    if (tab === 'persona') { renderPersona(); }
    if (tab === 'stats') { renderStats(); }
    rendered[tab] = true;
}
function switchTab(tab) {
    ['overview', 'howto', 'effect', 'genuine', 'freeform', 'persona', 'stats'].forEach(t => {
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
    if (typeof PERSONA !== 'undefined' && PERSONA.models && PERSONA.models.length) {
        opts('persona-model', PERSONA.models);
        wire('persona-model', renderPersona);
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


def build(paths, suite_root=None, all_models_href=None, persona_dir=None):
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
    # persona-mechanism results (optional): E2–E5 stats next to this script (or
    # --persona-dir) and the E1 atlas next to each qualia_steering.json; when none
    # exist the PERSONA const is empty and the tab shows a placeholder.
    persona = _load_persona_sources(paths, persona_dir)
    consts = build_consts(runs, stats, hi_by_key, ff_runs=ff_runs, ff_stats=ff_stats,
                          persona=persona)
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
    ap.add_argument("--persona-dir", default=None,
                    help="directory holding e{2,3,4,5}_stats_generated.json for the persona tab "
                         "(default: this script's directory)")
    args = ap.parse_args()

    html, runs = build(args.paths, args.suite_root, all_models_href=args.all_models_href,
                       persona_dir=args.persona_dir)
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
