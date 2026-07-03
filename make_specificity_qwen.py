#!/usr/bin/env python3
"""
Qwen-only version of the Genuine §2 specificity figure (s2_specificity.png):
per model, the qualia score minus the appraisal score (green) and minus the
random-direction score (gold). Same data/style/title as the full figure, but
restricted to the Qwen family (qwen3-1.7b, -8b, -14b, -32b, -235b, ascending
size). Values come verbatim from specificityData, so they are identical to the
full figure.

Run:  python3 make_specificity_qwen.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import (load_data, set_style, finish, save, out,
                                      QUALIA_GREEN, GOLD, ZERO)


def main():
    set_style()
    D = load_data()
    data = D["specificityData"]
    ms = [m for m in D["MODELS"] if m.startswith("qwen") and data.get(m)]
    app = [data[m]["qualia_minus_appraisal"]["diff"] for m in ms]
    rnd = [data[m]["qualia_minus_random"]["diff"] for m in ms]
    xs = np.arange(len(ms))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(xs - w / 2, app, w, color=QUALIA_GREEN, label="qualia − appraisal")
    ax.bar(xs + w / 2, rnd, w, color=GOLD, label="qualia − random")
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(ms, rotation=40, ha="right")
    finish(ax, "Specificity (score difference)", "", "score difference")
    ax.legend(loc="best")
    p = save(fig, out("2_genuine", "s2_specificity_qwen.png"))
    print("wrote", p)
    for m, a, r in zip(ms, app, rnd):
        print(f"   {m:12s} qualia−appraisal={a:+.3f}  qualia−random={r:+.3f}")


if __name__ == "__main__":
    main()
