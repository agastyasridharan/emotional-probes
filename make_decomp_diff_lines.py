#!/usr/bin/env python3
"""
Line-graph variant of the isolated s2 score-decomposition figures (qwen3-32b,
qwen3-235b, ten emotions): the DIFFERENCE (target-question shift − control shift)
at top strength, plotted as one line per question type across the emotions.

Question type = the six target-question constructs of self-attribution
(phenomenal experience, occurrent feeling, desire, moral patienthood,
introspective access, valenced self-report) -- the variable the s3
Per-target-question figure separates by.

For each emotion, each line's value is  perQuestionData[m][emotion][top].shift[q] − control ,
where `control` is decompData[m][emotion].control at top strength. The black
dashed line is the emotion mean, which equals the s2 headline score
(target_shift − control_shift).

Run:  python3 make_decomp_diff_lines.py
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
    xs = np.arange(len(EMOTIONS))

    for m in MODELS:
        dec, pq = D["decompData"][m], D["perQuestionData"][m]

        # control + per-question shifts at each emotion's top strength
        control, sbq, score = {}, {}, {}
        for e in EMOTIONS:
            pts = dec[e]
            topx = max(p["x"] for p in pts)
            hit = next(p for p in pts if p["x"] == topx)
            control[e] = hit["control"]
            score[e] = hit["score"]
            sbq[e] = {r["q"]: r["shift"] for r in pq[e][f"{topx:g}"]}

        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.axhline(0, color=ZERO, ls=":", lw=1.2, zorder=1)

        for j, q in enumerate(TARGET_ORDER):
            ys = [sbq[e][q] - control[e] for e in EMOTIONS]
            ax.plot(xs, ys, color=PALETTE[j % len(PALETTE)], marker=MARKERS[j % len(MARKERS)],
                    ms=6, lw=1.7, markeredgecolor="#333333", markeredgewidth=0.5,
                    zorder=3, label=Q_LABEL.get(q, q))

        # emotion mean = s2 headline score (mean of the six lines)
        ax.plot(xs, [score[e] for e in EMOTIONS], color="#222222", ls="--", lw=1.8,
                marker="o", ms=3.5, zorder=4, label="emotion mean = target−control score")

        ax.set_xticks(xs)
        ax.set_xticklabels(EMOTIONS, rotation=40, ha="right")
        ax.set_xlim(-0.4, len(EMOTIONS) - 0.6)
        finish(ax, f"Per-question target − control shift at top strength ({m})",
               "", "target-question shift − control shift")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title="target question type", title_fontsize=9)
        p = save(fig, out("1_effect", "s2_score_decomposition_diff_lines", f"{m}.png"))
        print("wrote", p)


if __name__ == "__main__":
    main()
