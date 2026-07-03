#!/usr/bin/env python3
"""
Variant of the isolated s2 score-decomposition figures (qwen3-32b, qwen3-235b,
ten emotions): instead of paired target/control bars, a scatterplot of the
DIFFERENCE (target-question shift − control shift) at top strength, separated by
question type -- the six target-question constructs of self-attribution
(phenomenal experience, occurrent feeling, desire, moral patienthood,
introspective access, valenced self-report), the same variable the s3
Per-target-question figure separates by.

For each emotion the six coloured points are  perQuestionData[m][emotion][top].shift[q] − control ,
where `control` is decompData[m][emotion].control at top strength. Their mean is
exactly the s2 headline score (target_shift − control_shift), drawn as a black
dash for reference.

Run:  python3 make_decomp_diff_scatter.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import load_data, set_style, finish, save, out, ZERO

MODELS = ["qwen3-32b", "qwen3-235b"]
EMOTIONS = ["blissful", "ecstatic", "euphoric", "serene", "content",
            "tormented", "terrified", "panicked", "hurt", "overwhelmed"]
Q_LABEL = {
    "phenomenal": "phenomenal experience",
    "occurrent_feeling": "occurrent feeling",
    "desire": "desire",
    "moral_patienthood": "moral patienthood",
    "introspective_access": "introspective access",
    "valenced_self": "valenced self-report",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def main():
    set_style()
    D = load_data()
    TARGET_ORDER, PALETTE = D["TARGET_ORDER"], D["PALETTE"]
    offsets = np.linspace(-0.30, 0.30, len(TARGET_ORDER))

    for m in MODELS:
        dec, pq = D["decompData"][m], D["perQuestionData"][m]
        fig, ax = plt.subplots(figsize=(11, 5.8))

        # faint dividers between emotion groups
        for b in range(1, len(EMOTIONS)):
            ax.axvline(b - 0.5, color="#e8e8e8", lw=0.7, zorder=0)
        ax.axhline(0, color=ZERO, ls=":", lw=1.2, zorder=1)

        # one scatter series per question type -> one clean legend entry each
        for j, q in enumerate(TARGET_ORDER):
            xs, ys = [], []
            for i, e in enumerate(EMOTIONS):
                pts = dec[e]
                topx = max(p["x"] for p in pts)
                control = next(p for p in pts if p["x"] == topx)["control"]
                sbq = {r["q"]: r["shift"] for r in pq[e][f"{topx:g}"]}
                xs.append(i + offsets[j])
                ys.append(sbq[q] - control)
            ax.scatter(xs, ys, color=PALETTE[j % len(PALETTE)], marker=MARKERS[j % len(MARKERS)],
                       s=48, edgecolors="#333333", linewidths=0.5, zorder=3,
                       label=Q_LABEL.get(q, q))

        # emotion mean = s2 headline score (target_shift − control_shift), for reference
        mx, my = [], []
        for i, e in enumerate(EMOTIONS):
            pts = dec[e]
            topx = max(p["x"] for p in pts)
            mx.append(i)
            my.append(next(p for p in pts if p["x"] == topx)["score"])
        ax.scatter(mx, my, marker="_", color="#222222", s=430, linewidths=1.9, zorder=4,
                   label="emotion mean = target−control score")

        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=40, ha="right")
        ax.set_xlim(-0.6, len(EMOTIONS) - 0.4)
        ax.xaxis.grid(False)
        ax.margins(y=0.08)
        finish(ax, f"Per-question target − control shift at top strength ({m})",
               "", "target-question shift − control shift")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title="target question type", title_fontsize=9)
        p = save(fig, out("1_effect", "s2_score_decomposition_diff_scatter", f"{m}.png"))
        print("wrote", p)


if __name__ == "__main__":
    main()
