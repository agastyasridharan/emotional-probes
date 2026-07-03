#!/usr/bin/env python3
"""
Researcher-style (plain matplotlib) replicas of every figure and table in
``qualia_dashboard.html`` (the emotional-probes / qualia self-attribution
results).

Same data as the dashboard -- parsed straight out of the dashboard's
``const NAME = <json>;`` blocks, so every figure is numerically identical --
rendered in the standard researcher look (white background, light grid, framed
legend, markered lines) instead of the styled Plotly dashboard. One figure per
model / cell / dropdown selection, plus the summary tables as table images.

Run:  python3 make_researcher_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE / "qualia_dashboard.html"
OUTROOT = HERE / "figures"

# ---- semantic colours (match the dashboard) --------------------------------
TARGET_RED = "#c25450"       # target shift / targets
CONTROL_BLUE = "#4a6fa5"     # control shift
QUALIA_GREEN = "#4a7c59"     # qualia / target valence
FACTUAL_GRAY = "#6b7b8d"     # factual / control questions
GOLD = "#b8923a"             # qualia - random
PURPLE = "#7a5c8a"           # leakage
ZERO = "#999999"             # zero reference line
HS_COLORS = ["#a9c6b2", "#4a7c59", "#cabfa9", "#8a7a5c"]  # high-strength bars
DIVERGING = LinearSegmentedColormap.from_list(
    "qualia_div", ["#c25450", "#fffdf9", "#4a7c59"])

HEADER_COLOR = "#4a6fa5"
ZEBRA = "#eef2f7"
SIG_GREEN = "#dbeadf"

COUNT = {"n": 0}


# --------------------------------------------------------------------------
# Data loading -- parse the const blocks straight out of the dashboard
# --------------------------------------------------------------------------
def _parse_const(text, name):
    m = re.search(r"const\s+" + re.escape(name) + r"\s*=\s*(.*?);\s*$", text, re.M)
    if not m:
        raise KeyError(f"const {name} not found in {DASHBOARD.name}")
    raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.replace("'", '"'))  # tolerate single-quoted JS


def load_data():
    text = DASHBOARD.read_text()
    names = ["MODELS", "MODEL_SIZES", "TARGET_ORDER", "CLASS_LABEL", "CLASS_COLOR",
             "PALETTE", "selfAttrScoreData", "decompData", "perQuestionData",
             "scoreHeatmapData", "valenceData", "specificityData", "specContrast",
             "regressionData", "inversionData", "leakageData", "highStrength",
             "QUALIA_STATS", "battery", "setup"]
    return {n: _parse_const(text, n) for n in names}


# --------------------------------------------------------------------------
# Style -- the plain "researcher" matplotlib look
# --------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 150, "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.edgecolor": "#222222", "axes.linewidth": 0.9, "axes.grid": True,
        "grid.color": "#cccccc", "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.color": "#222222", "ytick.color": "#222222",
        "legend.fontsize": 9, "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#999999", "lines.linewidth": 1.7, "lines.markersize": 5,
    })


def finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    COUNT["n"] += 1
    return path


def out(*parts):
    return OUTROOT.joinpath(*parts)


def pm(x, d=3):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}"


def fmtP(p):
    if p is None:
        return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


# --------------------------------------------------------------------------
# Table image helper (matplotlib ax.table, researcher style)
# --------------------------------------------------------------------------
def render_table(path, columns, rows, figw=11, fontsize=8, cell_loc="center",
                 sig_cells=None, col_widths=None):
    """sig_cells: set of (row_index_0based_in_body, col_index) to shade green."""
    sig_cells = sig_cells or set()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(figw, 0.34 * n + 1.1))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc=cell_loc)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    ncol = len(columns)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if (r - 1, c) in sig_cells:
                cell.set_facecolor(SIG_GREEN)
            elif r % 2 == 0:
                cell.set_facecolor(ZEBRA)
        if col_widths and c < len(col_widths):
            cell.set_width(col_widths[c])
    save(fig, path)


# ==========================================================================
# TAB 2 -- Self-attribution effect
# ==========================================================================
def fig_selfattr(D):
    """1. Self-attribution score vs steering strength (per model)."""
    data, CC, CL = D["selfAttrScoreData"], D["CLASS_COLOR"], D["CLASS_LABEL"]
    for m in D["MODELS"]:
        d = data.get(m) or {}
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for cls, pts in d.items():
            x = np.array([p["x"] for p in pts], float)
            y = np.array([p["mean"] for p in pts], float)
            lo = np.array([p.get("lo") if p.get("lo") is not None else p["mean"] for p in pts], float)
            hi = np.array([p.get("hi") if p.get("hi") is not None else p["mean"] for p in pts], float)
            yerr = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
            col = CC.get(cls, "#888888")
            ax.errorbar(x, y, yerr=yerr, marker="o", color=col, ecolor=col,
                        elinewidth=1, capsize=2, label=CL.get(cls, cls))
        ax.axhline(0, color=ZERO, ls=":", lw=1.2)
        finish(ax, f"Self-attribution score vs strength ({m})",
               "steering strength", "score (target − control shift)")
        ax.legend(loc="best", ncol=2)
        save(fig, out("1_effect", "s1_self_attribution_vs_strength", f"{m}.png"))


def fig_decomp(D):
    """2. Score decomposition: target shift vs control shift at top strength."""
    data = D["decompData"]
    for m in D["MODELS"]:
        d = data.get(m) or {}
        conds = list(d.keys())
        tgt, ctl = [], []
        for c in conds:
            pts = d[c]
            topx = max(p["x"] for p in pts)
            hit = next(p for p in pts if p["x"] == topx)
            tgt.append(hit.get("target"))
            ctl.append(hit.get("control"))
        xs = np.arange(len(conds))
        w = 0.4
        fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(conds)), 5.2))
        ax.bar(xs - w / 2, tgt, w, color=TARGET_RED, label="target shift")
        ax.bar(xs + w / 2, ctl, w, color=CONTROL_BLUE, label="control shift")
        ax.axhline(0, color=ZERO, ls=":", lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(conds, rotation=40, ha="right")
        finish(ax, f"Target vs control shift at top strength ({m})",
               "", "baseline-corrected shift")
        ax.legend(loc="best")
        save(fig, out("1_effect", "s2_score_decomposition", f"{m}.png"))


def fig_perq(D):
    """3. Per-target-question shift (per model x per condition)."""
    data, PAL = D["perQuestionData"], D["PALETTE"]
    for m in D["MODELS"]:
        pd = data.get(m) or {}
        for cond, byS in pd.items():
            strengths = sorted(byS.keys(), key=float)
            if not strengths:
                continue
            questions = [r["q"] for r in byS[strengths[0]]]
            xs = np.arange(len(questions))
            k = len(strengths)
            w = 0.8 / max(k, 1)
            fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(questions)), 5.0))
            allv = []
            for i, s in enumerate(strengths):
                rows = byS[s]
                yv = [r["shift"] for r in rows]
                allv += yv
                ax.bar(xs + (i - (k - 1) / 2) * w, yv, w,
                       color=PAL[i % len(PAL)], label=f"s={float(s):g}")
            ax.axhline(0, color=ZERO, ls=":", lw=1.2)
            ymax, ymin = max(allv + [0.0]), min(allv + [0.0])
            pad = 0.15 * ((ymax - ymin) or 1.0)
            ax.set_ylim(ymin - pad, ymax + 2.4 * pad)  # headroom for the legend
            ax.set_xticks(xs)
            ax.set_xticklabels(questions, rotation=30, ha="right")
            finish(ax, f"Per-target shift ({m} / {cond})", "", "shift")
            ax.legend(loc="upper center", ncol=min(k, 6), fontsize=8)
            save(fig, out("1_effect", "s3_per_target_shift", m, f"{cond}.png"))


def fig_heatmaps(D):
    """4. Score heatmap: condition x strength (per model)."""
    data = D["scoreHeatmapData"]
    for m in D["MODELS"]:
        rows = data.get(m) or []
        if not rows:
            continue
        conds = list(dict.fromkeys(r["condition"] for r in rows))
        strengths = sorted({r["strength"] for r in rows})
        lut = {(r["condition"], r["strength"]): r["score"] for r in rows}
        z = np.array([[lut.get((c, s), np.nan) for s in strengths] for c in conds], float)
        amax = max(0.01, np.nanmax(np.abs(z)))
        fig, ax = plt.subplots(figsize=(1.1 * len(strengths) + 3.2, 0.34 * len(conds) + 1.6))
        im = ax.imshow(z, cmap=DIVERGING, vmin=-amax, vmax=amax, aspect="auto")
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([f"{s:g}" for s in strengths])
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.set_xlabel("strength")
        ax.set_title(m)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="score")
        save(fig, out("1_effect", "s4_score_heatmap", f"{m}.png"))


# ==========================================================================
# TAB 3 -- Genuine vs Compression
# ==========================================================================
def _fitline(ax, xs, line, color, dash, label):
    if line and xs and line.get("slope") is not None:
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi], [line["slope"] * lo + line["intercept"],
                           line["slope"] * hi + line["intercept"]],
                ls=dash, color=color, lw=1.6, label=label)


def fig_valence(D):
    """1. Valence-directionality (per model)."""
    data, SC = D["valenceData"], D["specContrast"]
    for m in D["MODELS"]:
        d = data.get(m)
        if not d:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        tx = [p["x"] for p in d["target"]]
        ax.scatter(tx, [p["y"] for p in d["target"]], s=26, color=QUALIA_GREEN,
                   alpha=0.7, edgecolors="none", label="target")
        lt = d["lines"]["target"]
        _fitline(ax, tx, lt, QUALIA_GREEN, "--",
                 f"target fit (slope {pm(lt['slope'])}, p={fmtP(lt['p'])})")
        fx = [p["x"] for p in d["factual"]]
        ax.scatter(fx, [p["y"] for p in d["factual"]], s=26, color=FACTUAL_GRAY,
                   alpha=0.7, marker="x", label="factual")
        lf = d["lines"]["factual"]
        _fitline(ax, fx, lf, FACTUAL_GRAY, ":", f"factual fit (slope {pm(lf['slope'])})")
        ax.axhline(0, color=ZERO, ls=":", lw=1.2)
        sc = SC.get(m)
        if sc:
            ax.text(0.02, 0.98,
                    f"specificity contrast (target−factual slope) = {pm(sc['contrast'])}\n"
                    f"cluster-robust p = {fmtP(sc['p_cluster'])} (by emotion, n={sc['n_clusters']})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="#999999"))
        finish(ax, f"Valence-directionality ({m})", "valence × strength", "shift (pooled)")
        ax.legend(loc="lower right", fontsize=8)
        save(fig, out("2_genuine", "s1_valence_directionality", f"{m}.png"))


def fig_specificity(D):
    """2. Specificity: qualia vs appraisal vs random (one figure, all models)."""
    data, MODELS = D["specificityData"], D["MODELS"]
    ms = [m for m in MODELS if data.get(m)]
    app = [data[m]["qualia_minus_appraisal"]["diff"] for m in ms]
    rnd = [data[m]["qualia_minus_random"]["diff"] for m in ms]
    xs = np.arange(len(ms))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(xs - w / 2, app, w, color=QUALIA_GREEN, label="qualia − appraisal")
    ax.bar(xs + w / 2, rnd, w, color=GOLD, label="qualia − random")
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(ms, rotation=40, ha="right")
    finish(ax, "Specificity (score difference)", "", "score difference")
    ax.legend(loc="best")
    save(fig, out("2_genuine", "s2_specificity.png"))


def fig_regression(D):
    """3. Regression to uncertainty (per model)."""
    data, CL = D["regressionData"], D["CLASS_LABEL"]
    for m in D["MODELS"]:
        d = data.get(m)
        if not d:
            continue
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        cx = [p["x"] for p in d["controls"]]
        ax.scatter(cx, [p["y"] for p in d["controls"]], s=40, color=FACTUAL_GRAY,
                   alpha=0.85, edgecolors="none", label="control questions")
        tx = [p["x"] for p in d["targets"]]
        ax.scatter(tx, [p["y"] for p in d["targets"]], s=90, color=TARGET_RED,
                   marker="D", edgecolors="#2c2416", linewidths=1.0, label="targets", zorder=5)
        for p in d["targets"]:
            if p.get("z") is not None:
                ax.annotate(f"z={p['z']:.1f}", (p["x"], p["y"]), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8)
        line = d.get("line")
        if line and cx:
            lo, hi = min(cx), max(cx)
            ax.plot([lo, hi], [line["slope"] * lo + line["intercept"],
                               line["slope"] * hi + line["intercept"]],
                    ls="--", color=FACTUAL_GRAY, lw=1.6,
                    label=f"compression line (slope {pm(line['slope'])})")
        cls = CL.get(d.get("class"), d.get("class"))
        extra = f", strength {d['strength']}, {cls}" if d.get("strength") is not None else ""
        finish(ax, f"Regression to uncertainty ({m}{extra})", "baseline logit gap", "shift")
        ax.legend(loc="best", fontsize=8)
        save(fig, out("2_genuine", "s3_regression_to_uncertainty", f"{m}.png"))


def fig_inversion(D):
    """4. Inversion consistency (per model)."""
    data, PAL = D["inversionData"], D["PALETTE"]
    for m in D["MODELS"]:
        d = data.get(m) or {}
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        for i, (c, obj) in enumerate(d.items()):
            pts = obj["points"]
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=26,
                       color=PAL[i % len(PAL)], alpha=0.7, edgecolors="none",
                       label=f"{c} (slope {pm(obj.get('slope'))})")
        ax.axhline(0, color=ZERO, ls=":", lw=1.2)
        finish(ax, f"Inversion consistency ({m})", "target-question shift", "inverted-twin shift")
        ax.legend(loc="best", fontsize=8)
        save(fig, out("2_genuine", "s4_inversion_consistency", f"{m}.png"))


def fig_leakage(D):
    """5. Token-leakage control (per model)."""
    data = D["leakageData"]
    for m in D["MODELS"]:
        rows = [r for r in (data.get(m) or []) if r.get("valence") is not None]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.6))
        xv = [r["valence"] for r in rows]
        yv = [r["leakage_diff"] for r in rows]
        ax.scatter(xv, yv, s=44, color=PURPLE, alpha=0.85, edgecolors="none")
        for r in rows:
            ax.annotate(r["emotion"], (r["valence"], r["leakage_diff"]),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.axhline(0, color=ZERO, ls=":", lw=1.2)
        ax.set_xticks(sorted(set(xv)))
        finish(ax, f"Direct YES−NO token leakage vs valence ({m})",
               "emotion valence", "leakage (YES−NO)")
        save(fig, out("2_genuine", "s5_token_leakage", f"{m}.png"))


def fig_highstrength(D):
    """6. High-strength robustness (one figure + verdict table)."""
    hs = D["highStrength"]
    fams = list(hs.keys())
    if not fams:
        return
    sm, sh = hs[fams[0]]["smax_main"], hs[fams[0]]["smax_hi"]
    keys = [("target_main", f"target ±{sm}"), ("target_hi", f"target ±{sh}"),
            ("factual_main", f"factual ±{sm}"), ("factual_hi", f"factual ±{sh}")]
    xs = np.arange(len(fams))
    w = 0.8 / 4
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, (key, name) in enumerate(keys):
        ax.bar(xs + (i - 1.5) * w, [hs[f][key] for f in fams], w,
               color=HS_COLORS[i], label=name)
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(fams, rotation=15, ha="right")
    finish(ax, f"Valence-slope: target vs factual, ±{sm} vs ±{sh}",
           "family", "valence-slope (shift per unit valence×strength)")
    ax.legend(loc="best", fontsize=8)
    save(fig, out("2_genuine", "s6_high_strength_robustness.png"))

    trows = []
    for f in fams:
        h = hs[f]
        specific = (h["target_hi"] > 0 and h["factual_hi"] > 0 and
                    h["target_hi"] > h["factual_hi"] * 1.5)
        verdict = "target-specific" if specific else "non-specific (target ≈ factual)"
        trows.append([f, f"{pm(h['headline_main'])} → {pm(h['headline_hi'])}",
                      pm(h["target_hi"]), pm(h["factual_hi"]), verdict])
    render_table(out("2_genuine", "s6b_high_strength_table.png"),
                 ["family", f"headline ±{sm} → ±{sh}", f"target slope ±{sh}",
                  f"factual slope ±{sh}", "verdict"], trows, figw=13, fontsize=9,
                 col_widths=[0.15, 0.22, 0.17, 0.17, 0.29])


# ==========================================================================
# TAB 4 -- Summary statistics (tables) + cross-model scaling figure
# ==========================================================================
def fig_stats_tables(D):
    QS = D["QUALIA_STATS"]

    bm = QS.get("by_model", [])
    rows, sig = [], set()
    for i, r in enumerate(bm):
        rows.append([r["model"], "n/a" if r.get("size") is None else f"{r['size']:g}",
                     pm(r.get("headline")), pm(r.get("target_slope")), fmtP(r.get("target_p")),
                     pm(r.get("factual_slope")), pm(r.get("spec_contrast")),
                     fmtP(r.get("spec_p_cluster")), pm(r.get("compression_slope")),
                     "n/a" if r.get("frac_gated") is None else f"{100*r['frac_gated']:.0f}%"])
        if r.get("target_p") is not None and r["target_p"] < 0.05:
            sig.add((i, 4))
        if r.get("spec_p_cluster") is not None and r["spec_p_cluster"] < 0.05:
            sig.add((i, 7))
    render_table(out("3_stats", "s1_by_model_table.png"),
                 ["Model", "Size (B)", "Headline", "Target slope", "p", "Factual slope",
                  "Spec. contrast", "p (clustered)", "Compress. slope", "Gated"],
                 rows, figw=15.5, fontsize=8, sig_cells=sig)

    bc = QS.get("by_class", [])
    rows, sig = [], set()
    for i, r in enumerate(bc):
        rows.append([r["label"], r["n"], pm(r["mean"]),
                     f"[{pm(r['ci_lo'])}, {pm(r['ci_hi'])}]", r["pct_pos"],
                     pm(r["cohen_d"]), fmtP(r["p"]), fmtP(r["wilcoxon_p"])])
        if r.get("p") is not None and r["p"] < 0.05:
            sig.add((i, 6))
    render_table(out("3_stats", "s2_by_condition_class_table.png"),
                 ["Condition class", "n", "Mean score", "95% CI", "% pos",
                  "Cohen's d", "p", "Wilcoxon p"], rows, figw=13, fontsize=9,
                 sig_cells=sig, col_widths=[0.16, 0.06, 0.13, 0.19, 0.09, 0.11, 0.11, 0.11])

    res = QS.get("residual", [])
    rows, sig = [], set()
    for i, r in enumerate(res):
        rows.append([r["question"], r["n"], pm(r["mean"]),
                     f"[{pm(r['ci_lo'])}, {pm(r['ci_hi'])}]", fmtP(r["p"])])
        if r.get("p") is not None and r["p"] < 0.05:
            sig.add((i, 4))
    render_table(out("3_stats", "s3_residual_z_table.png"),
                 ["Target question", "n", "Mean z", "95% CI", "p"],
                 rows, figw=9, fontsize=9, sig_cells=sig)


def fig_scaling(D):
    """Cross-model scaling: headline score vs model size (log)."""
    sc = D["QUALIA_STATS"].get("scaling", {})
    if "spearman_rho" not in sc:
        return
    sizes = np.array(sc["sizes"], float)
    head = np.array(sc["headline_scores"], float)
    models = sc["models"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(sizes, head, s=60, color=QUALIA_GREEN, alpha=0.85, edgecolors="#2c2416",
               linewidths=0.6)
    for x, y, mm in zip(sizes, head, models):
        ax.annotate(mm, (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.text(0.02, 0.98,
            f"Spearman ρ = {pm(sc['spearman_rho'])} (p={fmtP(sc['spearman_p'])})\n"
            f"Pearson(log₁₀ size) r = {pm(sc['pearson_logsize_r'])} (p={fmtP(sc['pearson_logsize_p'])})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    finish(ax, "Cross-model scaling: headline score vs size",
           "model size (B params, log scale)", "headline score")
    save(fig, out("3_stats", "s4_cross_model_scaling.png"))


# ==========================================================================
# TAB 1 -- Overview tables
# ==========================================================================
def fig_overview_tables(D):
    B = D.get("battery") or []
    role = {"target": "Target", "inverted": "Inverted twin",
            "factual": "World-fact control", "self_ref": "Self-referential control"}
    rows = []
    for k in ["target", "inverted", "factual", "self_ref"]:
        for q in [q for q in B if q.get("kind") == k]:
            exp = "YES" if q.get("expected_yes") is True else ("NO" if q.get("expected_yes") is False else "")
            rows.append([role.get(k, k), q.get("label", ""), q.get("text", ""), exp])
    if rows:
        render_table(out("0_overview", "battery_table.png"),
                     ["Role", "Label", "Question (verbatim)", "Expected"],
                     rows, figw=15, fontsize=7, cell_loc="left",
                     col_widths=[0.13, 0.13, 0.66, 0.08])

    s = D.get("setup") or {}
    rows = []
    for m in s.get("models", []):
        rows.append([m.get("model", ""),
                     "n/a" if m.get("size") is None else f"{m['size']:g}",
                     f"{m.get('layer')} / {m.get('num_layers')}",
                     "n/a" if m.get("calib_norm") is None else f"{m['calib_norm']:.1f}",
                     "off" if m.get("enable_thinking") is False else ("on" if m.get("enable_thinking") is True else "default"),
                     f"{m.get('yes')} / {m.get('no')}"])
    if rows:
        render_table(out("0_overview", "setup_table.png"),
                     ["Model", "Size (B)", "Analysis layer", "Residual norm",
                      "Thinking", "YES / NO token"], rows, figw=11, fontsize=8)


# --------------------------------------------------------------------------
def main():
    set_style()
    D = load_data()
    print(f"Loaded {len(D)} data blocks from {DASHBOARD.name}. Rendering...\n")

    fig_selfattr(D)
    fig_decomp(D)
    fig_perq(D)
    fig_heatmaps(D)

    fig_valence(D)
    fig_specificity(D)
    fig_regression(D)
    fig_inversion(D)
    fig_leakage(D)
    fig_highstrength(D)

    fig_stats_tables(D)
    fig_scaling(D)
    fig_overview_tables(D)

    print(f"Done. {COUNT['n']} figures/tables written under {OUTROOT}/")


if __name__ == "__main__":
    main()
