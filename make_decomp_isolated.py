#!/usr/bin/env python3
"""
Isolated score-decomposition figures (dashboard "Self-attribution effect" tab,
section 2 / s2) for qwen3-32b and qwen3-235b, restricted to a fixed subset of
ten emotion conditions. Identical style, axes, legend, and title to the full
`s2_score_decomposition/<model>.png` figures -- only the set of conditions shown
on the x-axis is reduced. Reuses the data loader and researcher style from
make_researcher_figures.py, so values stay numerically identical to the
dashboard.

Run:  python3 make_decomp_isolated.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import (load_data, set_style, finish, save, out,
                                      TARGET_RED, CONTROL_BLUE, ZERO)

MODELS = ["qwen3-32b", "qwen3-235b"]
EMOTIONS = ["blissful", "ecstatic", "euphoric", "serene", "content",
            "tormented", "terrified", "panicked", "hurt", "overwhelmed"]


def main():
    set_style()
    data = load_data()["decompData"]
    for m in MODELS:
        d = data.get(m) or {}
        conds = [c for c in EMOTIONS if c in d]          # requested subset, requested order
        tgt, ctl = [], []
        for c in conds:
            pts = d[c]
            topx = max(p["x"] for p in pts)              # same "top strength" selection as fig_decomp
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
        p = save(fig, out("1_effect", "s2_score_decomposition_isolated", f"{m}.png"))
        print("wrote", p, "(%d conditions)" % len(conds))


if __name__ == "__main__":
    main()
