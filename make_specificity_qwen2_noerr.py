#!/usr/bin/env python3
"""
Specificity figure (Genuine §2) for qwen3-32b and qwen3-235b only, WITHOUT error
bars. Bars: qualia score minus appraisal (green) and minus random (gold); same
data/style/title as the full s2_specificity figure.

Run:  python3 make_specificity_qwen2_noerr.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_researcher_figures import (load_data, set_style, finish, save, out,
                                      QUALIA_GREEN, GOLD, ZERO)

MODELS = ["qwen3-32b", "qwen3-235b"]


def main():
    set_style()
    D = load_data()
    data = D["specificityData"]
    app = [data[m]["qualia_minus_appraisal"]["diff"] for m in MODELS]
    rnd = [data[m]["qualia_minus_random"]["diff"] for m in MODELS]
    xs = np.arange(len(MODELS))
    w = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.bar(xs - w / 2, app, w, color=QUALIA_GREEN, label="qualia − appraisal")
    ax.bar(xs + w / 2, rnd, w, color=GOLD, label="qualia − random")
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    finish(ax, "Specificity (score difference)", "", "score difference")
    ax.legend(loc="best")
    p = save(fig, out("2_genuine", "s2_specificity_qwen32b_235b_no_errorbars.png"))
    print("wrote", p)


if __name__ == "__main__":
    main()
