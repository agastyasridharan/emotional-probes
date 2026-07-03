"""
Token-span tools: which token positions belong to which part of the text.

Two jobs:

1. **Averaging the right tokens.** :func:`mean_from_offset` and :func:`masked_mean`
   average a model's residual stream over a chosen set of positions (e.g. "all
   tokens from the 50th onward", or "only the masking speaker's turns").

2. **Mapping characters to tokens EXACTLY (fixes Issue #7).** The original
   deflection code estimated token positions by re-tokenising text prefixes and
   counting ids. That is not safe, because tokenisation is not additive:
   ``tokenize(a + b) != tokenize(a) + tokenize(b)`` in general, so boundaries
   could be off by a token or two. :func:`char_spans_to_token_mask` instead uses
   the tokenizer's ``offset_mapping`` (the exact ``(char_start, char_end)`` of
   every token) to decide overlap — no guessing.

The dialogue-parsing helpers (:func:`find_dialogue_start_char`,
:func:`find_speaker_char_spans`) are pure string operations, so they are easy to
unit-test without a model.
"""

from __future__ import annotations

import numpy as np

# A character span is an inclusive-start, exclusive-end pair of character offsets.
CharSpan = tuple[int, int]
TokenSpan = tuple[int, int]  # (token_start inclusive, token_end exclusive)


# --------------------------------------------------------------------------- #
# Averaging over token positions
# --------------------------------------------------------------------------- #
def masked_mean(activations: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of ``activations`` (shape ``(num_tokens, hidden)``) over the token
    positions where ``mask`` is truthy. Returns a ``(hidden,)`` vector (zeros if
    the mask selects nothing)."""
    activations = np.asarray(activations, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    count = mask.sum()
    if count == 0:
        return np.zeros(activations.shape[-1], dtype=np.float64)
    return (activations * mask[:, None]).sum(axis=0) / count


def mean_from_offset(
    activations: np.ndarray, seq_len: int, skip: int
) -> tuple[np.ndarray, int]:
    """Average activations over real token positions ``[skip, seq_len)``.

    This is the paper's "average from the 50th token onward" with ``skip=50``.

    Returns ``(mean_vector, num_positions_used)``. If no positions remain
    (sequence too short), the vector is zeros and the count is 0 — the caller
    can then skip that example, as the original code did.
    """
    activations = np.asarray(activations, dtype=np.float64)
    start = max(0, skip)
    end = min(seq_len, activations.shape[0])
    if end <= start:
        return np.zeros(activations.shape[-1], dtype=np.float64), 0
    window = activations[start:end]
    return window.mean(axis=0), end - start


# --------------------------------------------------------------------------- #
# Exact character -> token mapping (Issue #7)
# --------------------------------------------------------------------------- #
def char_spans_to_token_mask(
    offset_mapping: list[tuple[int, int]],
    char_spans: list[CharSpan],
) -> np.ndarray:
    """Boolean mask over tokens that overlap any of ``char_spans``.

    Parameters
    ----------
    offset_mapping:
        From ``tokenizer(text, return_offsets_mapping=True)``: a list with one
        ``(char_start, char_end)`` per token. Special tokens have a zero-length
        span (``start == end``) and are never selected.
    char_spans:
        Character spans ``(start, end)`` to mark.

    Returns
    -------
    np.ndarray
        Boolean array of length ``len(offset_mapping)``; ``True`` where a real
        token overlaps a requested span.
    """
    n = len(offset_mapping)
    mask = np.zeros(n, dtype=bool)
    for t, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_end <= tok_start:
            continue  # special / empty token
        for cs, ce in char_spans:
            # Half-open overlap test: token [tok_start,tok_end) ∩ [cs,ce) != ∅.
            if tok_start < ce and tok_end > cs:
                mask[t] = True
                break
    return mask


def char_spans_to_token_spans(
    offset_mapping: list[tuple[int, int]],
    char_spans: list[CharSpan],
) -> list[TokenSpan]:
    """Like :func:`char_spans_to_token_mask` but returns one
    ``(token_start, token_end_exclusive)`` per input char span (the first and
    last overlapping real token). Char spans that match no token are skipped."""
    token_spans: list[TokenSpan] = []
    for cs, ce in char_spans:
        hits = [
            t
            for t, (s, e) in enumerate(offset_mapping)
            if e > s and s < ce and e > cs
        ]
        if hits:
            token_spans.append((hits[0], hits[-1] + 1))
    return token_spans


# --------------------------------------------------------------------------- #
# Dialogue parsing (pure string ops)
# --------------------------------------------------------------------------- #
def find_dialogue_start_char(text: str, speakers: list[str]) -> int | None:
    """Character index where the dialogue (first speaker turn) begins.

    Looks for the earliest ``"\\nName:"`` or ``"\\n\\nName:"`` over all speakers.
    Returns ``None`` if no speaker turn is found (e.g. a malformed generation).
    """
    positions: list[int] = []
    for name in speakers:
        for sep in (f"\n\n{name}:", f"\n{name}:"):
            pos = text.find(sep)
            if pos != -1:
                positions.append(pos)
    return min(positions) if positions else None


def find_speaker_char_spans(text: str, speaker: str, all_speakers: list[str]) -> list[CharSpan]:
    """Character spans of one speaker's utterances (the text after their
    ``"Name:"`` prefix, up to the next speaker turn).

    This isolates, e.g., only the *masking* speaker's words in a deflection
    dialogue.
    """
    spans: list[CharSpan] = []
    prefix = f"{speaker}:"
    search_from = 0
    while True:
        start = text.find(prefix, search_from)
        if start == -1:
            break
        content_start = start + len(prefix)
        # The utterance ends at the next speaker turn (any speaker) or end of text.
        next_turn = len(text)
        for name in all_speakers:
            for sep in (f"\n\n{name}:", f"\n{name}:"):
                pos = text.find(sep, content_start)
                if pos != -1 and pos < next_turn:
                    next_turn = pos
        spans.append((content_start, next_turn))
        search_from = content_start
    return spans
