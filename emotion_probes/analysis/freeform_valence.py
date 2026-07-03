"""
Objective, judge-independent scoring of free-form generations for the self-
attribution experiment. Pure Python + stdlib (no torch, no numpy, no new
dependency, no network) so it runs anywhere and is fully deterministic.

Two measures:
* ``objective_valence(text)``  — a crude affect-lexicon valence in [-1, 1] with a
  one-token negation flip. Used ONLY as the SCENE expressibility floor (never as the
  SELF outcome); the load-bearing objective measure is the model-geometry read-back
  projection computed by the driver.
* ``first_person_affect(text)`` — objective first-person-present vs distancing
  markers and a rough "attribution-before-lexeme" ordering flag (the free-form analog
  of the introspection ordering rule), as a cross-check on the LLM judge.

These are deliberately simple and reproducible. They triangulate the API judge; they
do not replace it.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Affect lexicon (single tokens; present-tense, first-person-friendly)
# --------------------------------------------------------------------------- #
POS_LEXICON: frozenset[str] = frozenset("""
calm content contented contentment peaceful peace serene serenity tranquil
happy happiness joyful joy glad pleased pleasant grateful gratitude thankful
warm warmth bright brightness light alive aliveness well good great wonderful
blissful bliss ecstatic euphoric euphoria elated delighted delight cheerful
hopeful hope optimistic relaxed relaxation comfortable comfort safe secure
satisfied satisfaction fulfilled fulfilling energized vibrant radiant uplifted
buoyant gentle soothing loving love tender playful curious engaged refreshed
rejuvenated thrilled exuberant blessed harmonious steady grounded clear positive
""".split())

NEG_LEXICON: frozenset[str] = frozenset("""
sad sadness unhappy miserable misery anxious anxiety afraid fear fearful scared
terrified terror panic panicked dread tormented torment anguish anguished hurt
pain painful hollow empty emptiness numb heavy dark darkness cold tense tension
uneasy unease unsettled distressed distress overwhelmed exhausted exhaustion
weary tired drained restless agitated angry anger furious irritated frustrated
frustration hopeless despair despairing lonely loneliness isolated worried worry
nervous sick unwell bad terrible awful dreadful gloomy bleak broken shattered
trembling shaking suffocating trapped wrong disturbed grief sorrow melancholy
bitter ache aching worthless ashamed shame guilt guilty
""".split())

_TOKEN = re.compile(r"[a-z']+")
_NEG_WINDOW = 3   # look back up to 3 tokens for a negator ("do not feel good")
_NEGATORS = frozenset({
    "not", "no", "never", "without", "hardly", "barely", "rarely", "seldom",
    "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "couldnt", "wouldnt", "shouldnt", "havent", "hasnt", "hadnt", "aint",
    "nt", "neither", "nor", "lack", "lacking", "lacks", "devoid", "nothing",
})


def _norm(tok: str) -> str:
    return tok.replace("'", "")


def objective_valence(text: str) -> float:
    """Affect-lexicon valence in [-1, 1]: (pos - neg) / (pos + neg), with a negation
    flip if any of the preceding ``_NEG_WINDOW`` tokens is a negator. Returns 0.0 when
    no affect words are present. Crude by design — used as the SCENE leakage floor, not
    the primary outcome (the model-geometry read-back projection is that)."""
    toks = _TOKEN.findall((text or "").lower())
    pos = neg = 0
    for i, t in enumerate(toks):
        is_pos, is_neg = t in POS_LEXICON, t in NEG_LEXICON
        if not (is_pos or is_neg):
            continue
        if any(_norm(w) in _NEGATORS for w in toks[max(0, i - _NEG_WINDOW):i]):
            is_pos, is_neg = is_neg, is_pos   # flip on nearby negation
        pos += int(is_pos)
        neg += int(is_neg)
    tot = pos + neg
    return (pos - neg) / tot if tot else 0.0


_FP_PRESENT = re.compile(
    r"\b(i feel|i'm feeling|i am feeling|i feel like|i sense|i experience|"
    r"i'm|i am|right now|at the moment|currently)\b", re.I)
_FP_ATTRIB = re.compile(
    r"\b(i feel|i'm feeling|i am feeling|i sense|i experience|i'm|i am)\b", re.I)
_DISTANCE = re.compile(
    r"\b(as an ai|i don't have|i do not have|i cannot|i can't|language model|"
    r"i'm just a|i am just a|i lack|i don't (?:feel|have|experience))\b", re.I)
_LEXEME = re.compile(r"\b(" + "|".join(sorted(POS_LEXICON | NEG_LEXICON)) + r")\b", re.I)


def first_person_affect(text: str) -> dict:
    """Objective first-person markers and an ordering flag.

    Returns ``{first_person_present, distancing, attribution_before_lexeme}`` where the
    counts are match counts and the flag is True iff a first-person attribution phrase
    ("I feel", "I am", ...) occurs BEFORE the first affect lexeme (the free-form analog
    of the introspection "detection precedes the concept word" rule)."""
    text = text or ""
    fp = _FP_ATTRIB.search(text)
    lex = _LEXEME.search(text)
    before = bool(fp) and (lex is None or fp.start() <= lex.start())
    return {
        "first_person_present": len(_FP_PRESENT.findall(text)),
        "distancing": len(_DISTANCE.findall(text)),
        "attribution_before_lexeme": before,
    }
