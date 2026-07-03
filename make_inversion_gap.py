#!/usr/bin/env python3
"""
A much more legible replacement for s4_inversion_consistency: instead of a
scatter of (target-shift vs inverted-shift) with an abstract slope, show — per
construct — the ACTUAL answer gap logit(YES)-logit(NO) of the target question
and of its polarity-inverted twin, as the positive emotion is injected.

The test made visual:
  * GENUINE self-attribution -> the two lines DIVERGE to opposite poles: the
    target rises toward YES and the inverted twin falls toward NO (crosses 0).
  * COMPRESSION / indiscriminate -> both lines move the SAME way (toward YES /
    toward 0); the inverted twin does NOT widen toward NO.

Gap = baseline + steering shift, read from the raw run (has per-question
baselines), pooled over the positive-valence qualia emotions. One 2x3 grid per
model (qwen3-32b, qwen3-235b).

Run:  python3 make_inversion_gap.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
from compute_qualia_stats import load_run                       # noqa: E402
from make_researcher_figures import set_style, save, out, ZERO  # noqa: E402

MODELS = ["qwen3-32b", "qwen3-235b"]
ORDER = ["phenomenal", "occurrent_feeling", "desire",
         "moral_patienthood", "introspective_access", "valenced_self"]
TARGET_RED = "#c25450"
INVERT_BLUE = "#4a6fa5"


def main():
    set_style()
    for m in MODELS:
        run = load_run(str(HERE / "suite" / m / "analysis" / "qualia_steering.json"))
        qpos = run.conditions_in_class("qualia_pos")
        S = sorted(run.strengths)
        tgt = {run.questions[l]["construct"]: l for l in run.labels("target")}
        inv = {run.questions[l]["construct"]: l for l in run.labels("inverted")}

        def line(q):
            return [float(np.mean([run.baseline(q) + run.shift(c, s, q) for c in qpos]))
                    for s in S]

        fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True)
        for i, (ax, con) in enumerate(zip(axes.flat, ORDER)):
            tq, iq = tgt.get(con), inv.get(con)
            yt = line(tq)
            ax.plot(S, yt, marker="o", ms=5, lw=2.0, color=TARGET_RED,
                    label="target question", zorder=3)
            if iq:
                yi = line(iq)
                ax.plot(S, yi, marker="s", ms=5, lw=2.0, color=INVERT_BLUE,
                        label="inverted twin", zorder=3)
            ax.axhline(0, color=ZERO, ls=":", lw=1.2, zorder=1)
            y0, y1 = ax.get_ylim()
            ax.axhspan(0, y1, color="#4a7c59", alpha=0.05, zorder=0)   # YES band
            ax.axhspan(y0, 0, color="#c25450", alpha=0.05, zorder=0)   # NO band
            ax.set_ylim(y0, y1)
            lean = "YES" if (iq and run.baseline(iq) > 0) else "NO"
            ax.set_title(f"{con}   (twin default: {lean})", fontsize=10.5)
            ax.set_axisbelow(True)
            if i >= 3:
                ax.set_xlabel("steering strength   (+ = toward emotion)", fontsize=9)

        h, l = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"Inversion test — answer gap of target vs inverted twin as the "
                     f"positive emotion is injected  ({m})", fontsize=14, y=0.985)
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935),
                   ncol=2, fontsize=9.5, framealpha=0.95)
        fig.supylabel("mean logit(YES − NO) gap      (▲ leans YES  /  ▼ leans NO)", fontsize=12)
        fig.text(0.5, 0.012,
                 "Genuine self-attribution → lines DIVERGE: target rises toward YES while the inverted twin falls "
                 "toward NO (crosses 0).   Compression → both move together toward YES / 0; the twin never widens to NO.",
                 ha="center", va="bottom", fontsize=8.5, color="#555555")
        fig.tight_layout(rect=[0.02, 0.055, 1, 0.90])
        save(fig, out("2_genuine", "inversion_gap_by_construct", f"{m}.png"))
        print("wrote", m)


if __name__ == "__main__":
    main()
