#!/usr/bin/env python3
"""
Specificity figure (Genuine §2) for qwen3-32b and qwen3-235b only, WITH error
bars and p-values. Three specificity contrasts (qualia score minus a control
direction's score, pooled over positive strengths):
    qualia - appraisal    (non-qualia emotion directions)
    qualia - random       (content-free noise floor)
    qualia - mean-of-all  (generic-emotion floor; the other synthetic direction)

All three are computed exactly as scripts/compute_qualia_stats.py::specificity
does it — per (condition x positive-strength) score, qualia = qualia_pos+neg,
Welch two-sample t (ttest_ind, equal_var=False). Verified to reproduce the
stored qualia-appraisal / qualia-random diff/t/p exactly.

Error bars = 95% CI of the mean difference (SE = |diff/t|, df = n_a+n_b-2).
NOTE these are UNCLUSTERED (each condition x strength treated as independent), so
optimistic; the honest emotion-clustered uncertainty is not available for this
score-difference contrast (and is undefined for a single synthetic direction).

Run:  python3 make_specificity_qwen2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from make_researcher_figures import (load_data, set_style, finish, save, out,
                                      QUALIA_GREEN, GOLD, ZERO)

MODELS = ["qwen3-32b", "qwen3-235b"]
MEANALL = "#7a5c8a"
# (label, control-classes, colour)
CONTRASTS = [
    ("qualia − appraisal", {"appraisal"}, QUALIA_GREEN),
    ("qualia − random", {"__random__"}, GOLD),
    ("qualia − mean-of-all", {"__mean_all__"}, MEANALL),
]


def contrast(dec, control_classes):
    """Reproduce compute_qualia_stats.specificity: pooled positive-strength
    scores, qualia = qualia_pos+neg, Welch two-sample t."""
    def scores_for(classes):
        return [p["score"] for pts in dec.values() for p in pts
                if p["class"] in classes and p["x"] > 0]
    q = scores_for({"qualia_pos", "qualia_neg"})
    b = scores_for(control_classes)
    t, p = st.ttest_ind(q, b, equal_var=False)
    diff = np.mean(q) - np.mean(b)
    se = abs(diff / t) if t else float("nan")
    ci = st.t.ppf(0.975, len(q) + len(b) - 2) * se
    return diff, ci, p


def fmtp(p):
    if p < 1e-3:
        return f"p={p:.1e}".replace("e-0", "e-")   # e.g. p=3.3e-7
    return f"p={p:.2f}"                              # e.g. p=0.60


def main():
    set_style()
    D = load_data()
    xs = np.arange(len(MODELS))
    k = len(CONTRASTS)
    w = 0.8 / k

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axhline(0, color=ZERO, ls=":", lw=1.2)
    tops = []
    for j, (label, classes, colour) in enumerate(CONTRASTS):
        diffs, cis, ps = [], [], []
        for m in MODELS:
            d, c, p = contrast(D["decompData"][m], classes)
            diffs.append(d); cis.append(c); ps.append(p)
        off = (j - (k - 1) / 2) * w
        ax.bar(xs + off, diffs, w, yerr=cis, color=colour, label=label,
               error_kw=dict(capsize=3, elinewidth=1.1, ecolor="#333333"))
        for i in range(len(MODELS)):
            top = diffs[i] + cis[i]
            ax.text(xs[i] + off, top + 0.06, fmtp(ps[i]), ha="center", va="bottom",
                    fontsize=7.5, color="#333333")
            tops.append(top)
    ax.set_ylim(top=max(tops) + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS)
    finish(ax, "Specificity (score difference)", "", "score difference")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    fig.text(0.5, -0.02,
             "Error bars: 95% CI; p: Welch two-sample t (unclustered — treats "
             "condition×strength as independent).",
             ha="center", va="top", fontsize=7.5, color="#666666")
    p = save(fig, out("2_genuine", "s2_specificity_qwen32b_235b.png"))
    print("wrote", p)
    for label, classes, _ in CONTRASTS:
        for m in MODELS:
            d, c, pv = contrast(D["decompData"][m], classes)
            print(f"   {m:12s} {label:22s} diff={d:+.3f} ±{c:.3f}  {fmtp(pv)}")


if __name__ == "__main__":
    main()
