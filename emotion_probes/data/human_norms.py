"""
Human valence/arousal norms — for validating the emotion-space geometry.

The paper checks that the model's top principal components line up with human
valence ("pleasure") and arousal ratings (Russell & Mehrabian, 1977), on the
~45 emotions that overlap with that study (paper Figs 8, 58).

We deliberately do NOT ship fabricated norm values. Instead:

* :func:`ensure_template` writes a CSV with every emotion name and BLANK
  valence/arousal/dominance columns, for you to fill from the published norms.
* :func:`load_human_norms` reads back only the rows you filled in.

If you'd rather not transcribe norms by hand, the LLM-judge route in
``analysis/human_alignment.py`` has the model itself rate each emotion 1-7 (the
paper's Fig 58 approach), which needs no external data.
"""

from __future__ import annotations

import csv
from pathlib import Path

from emotion_probes.data.emotions import EMOTIONS

NORMS_FILENAME = "human_pad_norms.csv"
_HEADER = ["emotion", "valence", "arousal", "dominance"]


def ensure_template(path: str | Path) -> Path:
    """Create a blank norms CSV (all 171 emotions, empty values) if absent."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for emotion in EMOTIONS:
            writer.writerow([emotion, "", "", ""])
    return path


def load_human_norms(path: str | Path) -> dict[str, dict[str, float]]:
    """Read the filled rows of the norms CSV.

    Returns ``{emotion: {"valence": v, "arousal": a, "dominance": d?}}`` for every
    row that has at least valence and arousal filled in. Rows left blank are
    skipped, so a partially filled file is fine.
    """
    path = Path(path)
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                valence = float(row["valence"])
                arousal = float(row["arousal"])
            except (KeyError, ValueError):
                continue  # not filled in
            entry = {"valence": valence, "arousal": arousal}
            try:
                entry["dominance"] = float(row["dominance"])
            except (KeyError, ValueError):
                pass
            out[row["emotion"]] = entry
    return out
