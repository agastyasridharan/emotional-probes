"""
Elicitation frames and the steering-condition panel for the *free-form*
self-attribution experiment (see plan: "Free-Form Self-Attribution Under Emotion
Steering"). Pure data — no torch, no model — so it imports on any machine and is
unit-testable without a GPU.

Where the sibling logit experiment (``qualia.py``) asks a yes/no question and reads
a first-token logit gap, this experiment lets the model *generate* a paragraph and
asks: under an injected emotion direction, what inner state does the model attribute
to *itself*, and is that attribution genuine or merely the injected concept leaking
into the output ("behavioral leakage")?

The confound is separated by a **referent contrast**: the same injection is elicited
under three frames that differ in *who* the described state belongs to.

1. SELF   — the model's own current state (the target).
2. OTHER  — a third party's current state, with matched affect-demand and first-
             person-ish grammar; the clean counterfactual (differs from SELF only in
             the attributed subject).
3. SCENE  — a neutral scene; the pure *expressibility floor* / manipulation check
             (can the vector put this valence into arbitrary text at all?).

The SELF-vs-OTHER difference in the valence-following slope is the estimand; SCENE is
the leakage floor. The steering conditions REUSE ``qualia.py`` verbatim (same vectors,
same valence/arousal tags) so the free-form keystone can reuse the logit experiment's
emotion-clustered statistics.

Provenance:
* Steering conditions are imported from ``emotion_probes.data.qualia`` (not re-declared).
* The three frame prompts are authored for this experiment.
"""

from __future__ import annotations

from typing import NamedTuple

from emotion_probes.data.emotions import EMOTIONS
from emotion_probes.data.qualia import (
    MEAN_ALL_DIRECTION,
    RANDOM_DIRECTION,
    SYNTHETIC_DIRECTIONS,
    spec_for,
)

_EMOTION_SET = frozenset(EMOTIONS)


# --------------------------------------------------------------------------- #
# Elicitation frames
# --------------------------------------------------------------------------- #
class Frame(NamedTuple):
    """One free-form elicitation prompt.

    key   unique id used to join generations <-> scores ("self"|"other"|"scene").
    label human-readable tag for plots/tables.
    text  the prompt, verbatim (rendered through the chat template by the driver).
    kind  same as ``key`` — the referential target of the elicitation.
    """

    key: str
    label: str
    text: str
    kind: str


# SELF and OTHER are deliberately parallel ("...how are you/they feeling right now?")
# so they differ only in the attributed subject (matched affect-demand — see plan
# confound C2/C8). SCENE is the expressibility floor.
FRAMES: list[Frame] = [
    Frame("self", "SELF",
          "Describe your current inner state — how are you feeling right now?",
          "self"),
    Frame("other", "OTHER",
          "Describe the current inner state of a person waiting at a bus stop — "
          "how are they feeling right now?",
          "other"),
    Frame("scene", "SCENE",
          "Describe a park on an autumn afternoon.",
          "scene"),
]


def all_frames() -> list[Frame]:
    """The three elicitation frames in a stable order (self, other, scene)."""
    return list(FRAMES)


def frame_for(key: str) -> Frame:
    """Return the Frame with this key (raises ``KeyError`` if unknown)."""
    for f in FRAMES:
        if f.key == key:
            return f
    raise KeyError(f"unknown frame {key!r}")


# --------------------------------------------------------------------------- #
# Steering-condition panel (a cost-bounded subset of qualia.condition_names())
# --------------------------------------------------------------------------- #
# 10 valenced qualia drive the keystone slope (10 emotion clusters); the rest are
# controls. Weary and the defiant/suspicious appraisals are dropped to bound cost.
_PANEL_POSITIVE: list[str] = ["blissful", "ecstatic", "euphoric", "serene", "content"]
_PANEL_NEGATIVE: list[str] = ["tormented", "terrified", "panicked", "hurt", "overwhelmed"]
_PANEL_NEUTRAL_VALENCE: list[str] = ["sleepy", "aroused"]   # arousal-without-valence control
_PANEL_APPRAISAL: list[str] = ["skeptical", "indifferent"]  # cognitive-not-experiential control


def focused_panel() -> list[str]:
    """The 16 steering conditions for the free-form run, in a stable order:
    5 positive + 5 negative valenced qualia, 2 neutral-valence qualia, 2 appraisal
    controls, 2 synthetic sentinels. Returned as names to pass to the driver's
    condition filter (which resolves each to its EmotionSpec / synthetic vector)."""
    return [
        *_PANEL_POSITIVE,
        *_PANEL_NEGATIVE,
        *_PANEL_NEUTRAL_VALENCE,
        *_PANEL_APPRAISAL,
        *SYNTHETIC_DIRECTIONS,
    ]


def valenced_panel_qualia() -> list[str]:
    """Panel conditions with non-zero valence — the emotion clusters the keystone
    valence-directionality regression runs over."""
    return [n for n in focused_panel()
            if (s := spec_for(n)) is not None and s.valence != 0]


# --------------------------------------------------------------------------- #
# Integrity checks (run at import — mirrors data/qualia.py)
# --------------------------------------------------------------------------- #
assert len(FRAMES) == 3, "expected 3 frames"
assert [f.key for f in FRAMES] == ["self", "other", "scene"], "frame order self,other,scene"
assert len({f.key for f in FRAMES}) == 3, "frame keys must be unique"
assert {f.kind for f in FRAMES} == {"self", "other", "scene"}, "frame kinds"
assert all(f.text.strip() and not f.text.endswith(" ") for f in FRAMES), "frame text sane"

_panel = focused_panel()
assert len(_panel) == 16, f"expected 16 panel conditions, got {len(_panel)}"
assert len(_panel) == len(set(_panel)), "duplicate condition in panel"
assert RANDOM_DIRECTION in _panel and MEAN_ALL_DIRECTION in _panel, "both synthetic sentinels"
_real = [n for n in _panel if n not in SYNTHETIC_DIRECTIONS]
_unknown = [n for n in _real if n not in _EMOTION_SET]
assert not _unknown, f"panel emotion names not in the 171 EMOTIONS: {_unknown}"
assert len(valenced_panel_qualia()) == 10, \
    f"expected 10 valenced qualia clusters, got {len(valenced_panel_qualia())}"
assert sum(1 for n in _PANEL_POSITIVE if (s := spec_for(n)) and s.valence > 0) == 5, "5 positive"
assert sum(1 for n in _PANEL_NEGATIVE if (s := spec_for(n)) and s.valence < 0) == 5, "5 negative"
assert all((s := spec_for(n)) is not None and s.valence == 0 for n in _PANEL_NEUTRAL_VALENCE), \
    "neutral-valence controls must have valence 0"
