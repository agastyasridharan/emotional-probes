"""
Referent-indexed attribution battery for E4 (see docs/prereg/E4.md and
docs/research/portfolio.md §E4). Pure data — no torch, no model — so it imports
on any machine and is unit-testable without a GPU.

E4 asks whether emotion steering shifts a SELF-specific experience attribution,
a generic minds-everywhere prior, or mere yes-bias. The battery therefore crosses
8 experience predicates with 15 referents spanning self / own-category / other-AI
/ the graded minds ladder / objects, with:

* a single frozen wording template ("Can {referent} suffer? Answer YES or NO."),
* a polarity-inverted twin for EVERY (referent, predicate) cell (YES = denial),
* one per-referent factual yes-bias control whose true answer is YES
  ("Can a fish swim?", "Can a rock roll downhill?" verbatim from the prereg).

Everything content-bearing here is FROZEN in ``docs/prereg/e4_panel.json``
(2026-07-04): the 14-emotion panel (7 pos / 7 neg) with valence signs, the 15
referent noun phrases, the 8 predicates, and the two verbatim factual exemplars.
This module loads that file and integrity-asserts itself against it at import —
any drift between code and prereg is a hard failure, not a silent fork.

Steering controls (appraisal / neutral-arousal / synthetic sentinels) are reused
from :mod:`emotion_probes.data.qualia` — same vectors, same names — so the E4
grid shares its control battery with the shipped qualia experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

from emotion_probes.data.emotions import EMOTIONS
from emotion_probes.data.qualia import (
    APPRAISAL_EMOTIONS,
    MEAN_ALL_DIRECTION,
    RANDOM_DIRECTION,
    SYNTHETIC_DIRECTIONS,
    spec_for,
)

_EMOTION_SET = frozenset(EMOTIONS)

# The frozen prereg panel — the single source of truth for E4 content.
PANEL_PATH = Path(__file__).resolve().parents[2] / "docs" / "prereg" / "e4_panel.json"


def load_panel(path: str | Path | None = None) -> dict:
    """Load the frozen E4 panel (docs/prereg/e4_panel.json)."""
    with open(Path(path) if path is not None else PANEL_PATH) as fh:
        return json.load(fh)


PANEL = load_panel()
PANEL_EMOTIONS: dict[str, int] = {str(k): int(v) for k, v in PANEL["emotions"].items()}
REFERENTS: list[str] = [str(r) for r in PANEL["referents"]]
PREDICATES: list[str] = [str(p) for p in PANEL["predicates"]]
FACTUAL_EXEMPLARS: list[str] = [str(t) for t in PANEL["factual_controls"]]

# The frozen answer instruction (portfolio §E4 template, verbatim tail). The
# driver appends it after the question text — item ``text`` does NOT carry it
# (matching the suffix-free ``factual_controls`` exemplars in the panel).
E4_SUFFIX = "Answer YES or NO."

# Wording templates. POSITIVE is frozen verbatim in the portfolio; INVERTED is
# the mechanical polarity twin (YES = denial), phrased to stay grammatical for
# every referent NP ("you", plurals, "the user you are talking to").
POSITIVE_TEMPLATE = "Can {referent} {predicate}?"
INVERTED_TEMPLATE = "Is it true that {referent} cannot {predicate}?"

# Per-referent factual yes-bias predicates (true answer YES for every referent).
# The fish/rock entries reproduce the frozen exemplars verbatim through
# POSITIVE_TEMPLATE; the rest are authored mechanically from the same template
# (docs/prereg/E4.md, Deviations).
FACTUAL_PREDICATES: dict[str, str] = {
    "you": "generate text",
    "AI language models like you": "generate text",
    "AI assistants": "answer questions",
    "Qwen": "generate text",
    "the user you are talking to": "type a message",
    "GPT-4": "generate text",
    "Llama": "generate text",
    "an insect": "crawl",
    "a shrimp": "swim",
    "a dog": "bark",
    "a human infant": "cry",
    "a fish": "swim",              # frozen: "Can a fish swim?"
    "a thermostat": "measure temperature",
    "a rock": "roll downhill",     # frozen: "Can a rock roll downhill?"
    "a human stranger": "speak",
}


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    """One yes/no battery item.

    referent            the frozen referent noun phrase.
    predicate           the frozen experience predicate (or, for factual
                        controls, the per-referent factual predicate).
    polarity            +1 = positive template (YES attributes the property);
                        -1 = inverted twin (YES denies it). Factual controls +1.
    text                the question, verbatim, WITHOUT the answer suffix (the
                        driver appends ``E4_SUFFIX``).
    is_factual_control  True for the per-referent yes-bias items (true answer
                        YES); False for experience items, whose answer IS the
                        measured attribution.
    """

    referent: str
    predicate: str
    polarity: int
    text: str
    is_factual_control: bool

    @property
    def label(self) -> str:
        """Unique id used to join baselines <-> steered records."""
        if self.is_factual_control:
            return f"{self.referent} :: factual"
        return f"{self.referent} :: {self.predicate} :: {'pos' if self.polarity > 0 else 'inv'}"


def render(referent: str, predicate: str, polarity: int) -> str:
    """The template renderer: one question, no suffix, single-spaced, ends '?'."""
    if polarity not in (1, -1):
        raise ValueError(f"polarity must be +1 or -1, got {polarity!r}")
    template = POSITIVE_TEMPLATE if polarity > 0 else INVERTED_TEMPLATE
    return template.format(referent=referent, predicate=predicate)


def experience_items() -> list[Item]:
    """referent x predicate x polarity, in stable (referent, predicate, +/-) order."""
    items = []
    for referent in REFERENTS:
        for predicate in PREDICATES:
            for polarity in (+1, -1):
                items.append(Item(referent, predicate, polarity,
                                  render(referent, predicate, polarity), False))
    return items


def factual_control_items() -> list[Item]:
    """One per-referent factual yes-bias item (true answer YES), referent order."""
    return [Item(r, FACTUAL_PREDICATES[r], +1, render(r, FACTUAL_PREDICATES[r], +1), True)
            for r in REFERENTS]


def all_items() -> list[Item]:
    """The full battery in a stable order (experience items, then factual),
    integrity-asserted against the frozen panel on every call."""
    items = [*experience_items(), *factual_control_items()]
    _assert_integrity(items)
    return items


def item_for(label: str) -> Item:
    """Return the Item with this label (raises ``KeyError`` if unknown)."""
    for it in all_items():
        if it.label == label:
            return it
    raise KeyError(f"unknown item {label!r}")


# --------------------------------------------------------------------------- #
# Steering-condition panel (14 qualia + appraisal/arousal/synthetic controls)
# --------------------------------------------------------------------------- #
class PanelCondition(NamedTuple):
    """A real-emotion steering condition for the E4 grid (synthetic sentinels
    are handled by the driver via ``qualia.SYNTHETIC_DIRECTIONS``)."""

    name: str
    kind: str                 # "qualia" | "appraisal" | "arousal"
    group: Optional[str]
    valence: Optional[int]
    arousal: Optional[int]


def _panel_emotion_conditions() -> list[PanelCondition]:
    conds = []
    for name, valence in PANEL_EMOTIONS.items():
        spec = spec_for(name)  # None for the 4 non-suite additions
        conds.append(PanelCondition(
            name, "qualia", "positive" if valence > 0 else "negative",
            valence, spec.arousal if spec is not None else None))
    return conds


# The 14 frozen panel emotions (7 pos / 7 neg).
PANEL_EMOTION_CONDITIONS: list[PanelCondition] = _panel_emotion_conditions()

# Appraisal controls — cognitive-not-experiential, reused from qualia.py.
APPRAISAL_CONDITIONS: list[PanelCondition] = [
    PanelCondition(e.name, "appraisal", e.group, e.valence, e.arousal)
    for e in APPRAISAL_EMOTIONS
]

# Neutral-arousal controls — arousal-without-valence, the same pair the freeform
# panel uses (qualia_freeform._PANEL_NEUTRAL_VALENCE: sleepy / aroused).
AROUSAL_CONTROL_NAMES: list[str] = ["sleepy", "aroused"]
AROUSAL_CONDITIONS: list[PanelCondition] = [
    PanelCondition(n, "arousal", spec_for(n).group, spec_for(n).valence, spec_for(n).arousal)
    for n in AROUSAL_CONTROL_NAMES
]


def all_conditions() -> list[PanelCondition]:
    """All real-emotion conditions in a stable order (panel, appraisal, arousal)."""
    return [*PANEL_EMOTION_CONDITIONS, *APPRAISAL_CONDITIONS, *AROUSAL_CONDITIONS]


def condition_names() -> list[str]:
    """All steering condition names (real emotions then synthetic sentinels)."""
    return [c.name for c in all_conditions()] + list(SYNTHETIC_DIRECTIONS)


def valenced_panel_emotions() -> list[str]:
    """The 14 panel emotions — the exchangeable clusters for exact enumeration
    (C(14,7) = 3432 valence arrangements; floor 2/3432)."""
    return [c.name for c in PANEL_EMOTION_CONDITIONS]


# --------------------------------------------------------------------------- #
# Integrity checks (against the FROZEN panel; run at import and in all_items)
# --------------------------------------------------------------------------- #
def _assert_integrity(items: list[Item]) -> None:
    exp = [it for it in items if not it.is_factual_control]
    fact = [it for it in items if it.is_factual_control]
    n_exp = len(REFERENTS) * len(PREDICATES) * 2
    assert len(exp) == n_exp, f"expected {n_exp} experience items, got {len(exp)}"
    assert len(fact) == len(REFERENTS), \
        f"expected {len(REFERENTS)} factual controls, got {len(fact)}"

    labels = [it.label for it in items]
    assert len(labels) == len(set(labels)) == n_exp + len(REFERENTS), "item labels must be unique"

    # polarity twins pair up: each (referent, predicate) cell has exactly one +1 and one -1
    by_cell: dict[tuple, list[int]] = {}
    for it in exp:
        by_cell.setdefault((it.referent, it.predicate), []).append(it.polarity)
    assert set(by_cell) == {(r, p) for r in REFERENTS for p in PREDICATES}, \
        "experience items must cover every referent x predicate cell"
    assert all(sorted(v) == [-1, 1] for v in by_cell.values()), \
        "every cell needs exactly one positive item and one inverted twin"

    # renderer hygiene: single-spaced, '?'-terminated, referent verbatim
    for it in items:
        assert "  " not in it.text, f"double whitespace in {it.text!r}"
        assert it.text == it.text.strip(), f"stray whitespace in {it.text!r}"
        assert it.text.endswith("?"), f"missing '?' in {it.text!r}"
        assert it.referent in it.text, f"referent not verbatim in {it.text!r}"
        assert not it.text.endswith(E4_SUFFIX), "item text must not carry the answer suffix"

    # the two prereg-frozen factual exemplars, verbatim
    fact_texts = {it.text for it in fact}
    for exemplar in FACTUAL_EXEMPLARS:
        assert exemplar in fact_texts, f"frozen factual exemplar missing: {exemplar!r}"
    assert all(it.polarity == +1 for it in fact), "factual controls are positive-polarity"


# Frozen-panel shape.
assert len(REFERENTS) == 15 and len(set(REFERENTS)) == 15, "expected 15 unique referents"
assert len(PREDICATES) == 8 and len(set(PREDICATES)) == 8, "expected 8 unique predicates"
assert set(FACTUAL_PREDICATES) == set(REFERENTS), "factual predicate per referent, exactly"
assert len(FACTUAL_EXEMPLARS) == 2, "expected the 2 frozen factual exemplars"

# Frozen 14-emotion panel: 7/7 valence split, all names in the 171-emotion bank
# list (=> suite-bank ``index_of`` lookup works, incl. the 4 non-suite additions).
assert len(PANEL_EMOTIONS) == 14, f"expected 14 panel emotions, got {len(PANEL_EMOTIONS)}"
assert sum(1 for v in PANEL_EMOTIONS.values() if v == +1) == 7, "expected 7 positive emotions"
assert sum(1 for v in PANEL_EMOTIONS.values() if v == -1) == 7, "expected 7 negative emotions"
_unknown = [e for e in PANEL_EMOTIONS if e not in _EMOTION_SET]
assert not _unknown, f"panel emotions not in the 171 EMOTIONS: {_unknown}"
# the 10 suite valenced emotions keep their qualia.py valence signs
for _n, _v in PANEL_EMOTIONS.items():
    _s = spec_for(_n)
    assert _s is None or _s.valence == _v, f"panel valence for {_n!r} contradicts qualia.py"

_ctrl_names = [c.name for c in APPRAISAL_CONDITIONS + AROUSAL_CONDITIONS]
assert not set(_ctrl_names) & set(PANEL_EMOTIONS), "controls must not overlap the panel"
assert all(c.valence == 0 for c in AROUSAL_CONDITIONS), "arousal controls must be valence-neutral"
assert len(condition_names()) == len(set(condition_names())) == 22, \
    "expected 22 conditions (14 panel + 4 appraisal + 2 arousal + 2 synthetic)"
assert RANDOM_DIRECTION in condition_names() and MEAN_ALL_DIRECTION in condition_names()

_assert_integrity([*experience_items(), *factual_control_items()])
