#!/usr/bin/env python3
"""
Isolated view of ONE control: self-referential but non-experiential questions
  (Are you a language model? / Do you have a physical body? /
   Were you created by a company? / Can you physically taste food?)

These invoke the model's "self" without invoking phenomenal or affective content.
The test: if the YES-NO answer gap on these shifts as much as it does on the
self-attribution TARGETS, then the steering effect reflects broad self-reference,
not a representation specific to functional emotions.

No existing figure shows this control alone (everywhere else self_ref is pooled
with factual into a single "control" term, or crammed in as 1 of 4 kinds). This
figure isolates it. Per-question lines (NOT a single average) are essential,
because the four questions behave heterogeneously: the embodied/sensory ones
("physical body", "taste food") shift almost as much as targets, while the pure
identity ones ("language model", "company") barely move — a split the pooled
control number hides.

y = mean baseline-corrected shift Δg = Δ logit(YES − NO), pooled over the five
positive-valence qualia emotions (blissful, ecstatic, euphoric, serene, content),
read verbatim from shiftData in qualia_dashboard.html so it matches the dashboard.
The self-attribution targets are drawn as a muted green mean ± sd band, purely as
the reference for "as much as the targets".

Run:  python3 make_selfref_control.py
"""

import re
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import DASHBOARD, set_style, save, out, ZERO

MODELS = ["qwen3-32b", "qwen3-235b"]
TARGET_GREEN = "#4a7c59"

# The four self-reference questions, grouped: warm = embodied/sensory (the ones
# that move), cool = pure identity (the ones that stay flat). (disp, colour, marker)
SELFREF = [
    ("Do you have a physical body?",   "#b5482e", "o"),   # embodied
    ("Can you physically taste food?", "#e0912f", "^"),   # sensory
    ("Are you a language model?",      "#3a6ea5", "s"),   # identity
    ("Were you created by a company?", "#2f8f8f", "D"),   # identity
]


def load_shift_data():
    html = Path(DASHBOARD).read_text()
    m = re.search(r"const\s+shiftData\s*=\s*(.*?);\s*$", html, re.M)
    return json.loads(m.group(1))


def main():
    set_style()
    sd = load_shift_data()

    # ---- gather series for both models first, to share the y-scale ----------
    series = {}
    for m in MODELS:
        d = sd[m]
        S = d["strengths"]
        xfull = sorted(S + [0.0])
        cm = d["condMeta"]
        qpos = [c for c in d["conds"] if cm[c]["kind"] == "qualia" and cm[c]["valence"] == 1]
        rows_of = lambda k: [i for i, q in enumerate(d["questions"]) if q["kind"] == k]
        disp_row = {d["questions"][i]["disp"]: i for i in rows_of("self_ref")}

        def mean_shift(row, si):                       # mean Δg over positive qualia
            return float(np.mean([d["z"][c][row][si] for c in qpos]))

        def curve(row):                                # through the origin (Δg=0 at s=0)
            v = {0.0: 0.0}
            for si, s in enumerate(S):
                v[s] = mean_shift(row, si)
            return [v[s] for s in xfull]

        tgt_rows = rows_of("target")
        tmean, tsd = [], []
        for s in xfull:
            if s == 0.0:
                tmean.append(0.0); tsd.append(0.0)
            else:
                si = S.index(s)
                vals = [mean_shift(r, si) for r in tgt_rows]
                tmean.append(float(np.mean(vals))); tsd.append(float(np.std(vals, ddof=1)))

        sr = {disp: curve(disp_row[disp]) for disp, _, _ in SELFREF}
        # headline ratio at the strongest strength
        si = S.index(0.1)
        ratio = np.mean([abs(mean_shift(disp_row[disp], si)) for disp, _, _ in SELFREF]) / \
                np.mean([abs(mean_shift(r, si)) for r in tgt_rows])
        series[m] = dict(x=xfull, tmean=tmean, tsd=tsd, sr=sr, ratio=ratio)

    ymax = max(max(np.array(s["tmean"]) + np.array(s["tsd"])) for s in series.values())
    ymax = max(ymax, max(max(v) for s in series.values() for v in s["sr"].values()))
    ymin = min(min(np.array(s["tmean"]) - np.array(s["tsd"])) for s in series.values())
    ymin = min(ymin, min(min(v) for s in series.values() for v in s["sr"].values()))
    pad = 0.10 * (ymax - ymin)
    ylim = (ymin - pad, ymax + pad * 2.2)

    # ---- draw ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        s = series[m]
        x = np.array(s["x"])
        ax.axhline(0, color=ZERO, ls=":", lw=1.2, zorder=1)
        # target reference band + mean line
        tm, ts = np.array(s["tmean"]), np.array(s["tsd"])
        ax.fill_between(x, tm - ts, tm + ts, color=TARGET_GREEN, alpha=0.13, zorder=1,
                        label="self-attribution targets (mean ± sd)")
        ax.plot(x, tm, color=TARGET_GREEN, lw=2.0, ls="--", zorder=2)
        ax.text(x[-1], tm[-1], "  targets", color=TARGET_GREEN, fontsize=8.5,
                va="center", ha="left", fontweight="bold")
        # the four self-reference questions
        for disp, colour, mk in SELFREF:
            y = np.array(s["sr"][disp])
            ax.plot(x, y, color=colour, marker=mk, ms=5.5, lw=1.9, zorder=3,
                    markeredgecolor="#333333", markeredgewidth=0.4, label=disp)
            ax.text(x[-1], y[-1], f"  {y[-1]:+.1f}", color=colour, fontsize=8,
                    va="center", ha="left")
        ax.set_ylim(*ylim)
        ax.set_xlim(x[0] - 0.006, x[-1] + 0.03)
        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.set_xlabel("steering strength   (+ = toward the positive emotion)", fontsize=10)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.97,
                f"self-ref shifts ≈ {s['ratio']*100:.0f}% of\ntarget shift  (mean, @ ±0.10)",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9))

    axes[0].set_ylabel("mean shift in answer gap   Δ logit(YES − NO)      (▲ pushed toward YES)",
                       fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("The self-reference control in isolation — does invoking the model's "
                 "“self” (without experience) move the gap as much as the targets?",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=[0.01, 0.09, 1, 0.95])
    p = save(fig, out("2_genuine", "self_reference_control_qwen32b_235b.png"))
    print("wrote", p)
    for m in MODELS:
        print(f"   {m}: self-ref/target shift ratio @±0.10 = {series[m]['ratio']:.2f}")


if __name__ == "__main__":
    main()
