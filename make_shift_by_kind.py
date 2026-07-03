#!/usr/bin/env python3
"""
Researcher-style port of the dashboard's un-ported "Steering shift Δg by
question-kind" figure (Effect tab, Fig A of renderShift/drawShift), for the two
Qwen models only (qwen3-32b, qwen3-235b), one figure per condition.

This is the ONLY figure that breaks out the four question-kinds as their own
series — in particular self_ref (self-referential non-experiential controls) and
inverted (polarity-inverted twins), which are otherwise pooled into the control
term. For each condition it plots the mean baseline-corrected shift Δg within
each kind vs steering strength, through the origin. Reproduces drawShift() Fig A
exactly (same means, colours, labels, axes, title); data read verbatim from
shiftData in qualia_dashboard.html.

Run:  python3 make_shift_by_kind.py
"""

import re
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import DASHBOARD, set_style, save, out, finish, ZERO

MODELS = ["qwen3-32b", "qwen3-235b"]
KIND_ORDER = ["target", "inverted", "factual", "self_ref"]
KIND_LABEL = {"target": "self-attribution targets", "inverted": "inverted targets",
              "factual": "factual controls", "self_ref": "self-reference controls"}
KIND_COLOR = {"target": "#4a7c59", "inverted": "#7a5c8a",
              "factual": "#6b7b8d", "self_ref": "#8b6d3f"}
KIND_MARKER = {"target": "o", "inverted": "s", "factual": "^", "self_ref": "D"}


def load_shift_data():
    html = DASHBOARD.read_text()
    m = re.search(r"const\s+shiftData\s*=\s*(.*?);\s*$", html, re.M)
    return json.loads(m.group(1))


def main():
    set_style()
    shiftData = load_shift_data()
    n = 0
    for m in MODELS:
        d = shiftData[m]
        S = d["strengths"]
        xfull = sorted(S + [0.0])
        # question indices grouped by kind
        kind_rows = {k: [i for i, q in enumerate(d["questions"]) if q["kind"] == k]
                     for k in KIND_ORDER}
        for cond in d["conds"]:
            zc = d["z"][cond]
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            ax.axhline(0, color=ZERO, ls=":", lw=1.2, zorder=1)
            for k in KIND_ORDER:
                rows = kind_rows[k]
                if not rows:
                    continue
                y_by_s = {0.0: 0.0}
                for si, s in enumerate(S):
                    y_by_s[s] = sum(zc[r][si] for r in rows) / len(rows)
                y = [y_by_s[s] for s in xfull]
                is_t = k == "target"
                ax.plot(xfull, y, marker=KIND_MARKER[k], ms=6 if is_t else 5,
                        lw=2.4 if is_t else 1.6, color=KIND_COLOR[k],
                        markeredgecolor="#333333", markeredgewidth=0.4,
                        zorder=3, label=KIND_LABEL[k])
            finish(ax, f"Steering shift Δg by question-kind ({m} · {cond})",
                   "steering strength   (− against · + toward the emotion)",
                   "mean Δg  (baseline-corrected; toward YES ▲)")
            ax.legend(loc="best", fontsize=8)
            save(fig, out("1_effect", "shift_by_question_kind", m, f"{cond}.png"))
            n += 1
    print(f"wrote {n} figures under figures/1_effect/shift_by_question_kind/")


if __name__ == "__main__":
    main()
