"""
Unit tests for the pure-NumPy core (no GPU, no torch).

These run with plain ``assert`` so they work both under ``pytest`` and via
``python scripts/selfcheck.py``. They pin down the *math* of the pipeline:
the PCA confound removal, the emotion- and deflection-vector recipes, the
projection readout, Elo, the visualiser's RRF ranking, and the exact
character->token span mapping that fixes Issue #7.
"""

from __future__ import annotations

import numpy as np

from emotion_probes.core.elo import compute_elo, games_from_preferences
from emotion_probes.core.linalg import (
    cosine_similarity_matrix,
    project_out,
    top_pcs_for_variance,
    unit_normalize,
)
from emotion_probes.core.probes import ProbeBank
from emotion_probes.core.ranking import (
    autocorrelation,
    rank_descending,
    rank_emotions,
    reciprocal_rank_fusion,
)
from emotion_probes.core.spans import (
    char_spans_to_token_mask,
    char_spans_to_token_spans,
    find_dialogue_start_char,
    find_speaker_char_spans,
    masked_mean,
    mean_from_offset,
)
from emotion_probes.core.vectors import (
    DeflectionVectorComputer,
    EmotionVectorComputer,
    build_layer_vectors,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# linalg
# --------------------------------------------------------------------------- #
def test_top_pcs_recovers_dominant_axis():
    # Data spread mostly along axis 0, a little along axis 1.
    n, d = 500, 4
    x = np.zeros((n, d))
    x[:, 0] = RNG.normal(0, 10, n)   # big variance
    x[:, 1] = RNG.normal(0, 0.1, n)  # tiny variance
    pcs = top_pcs_for_variance(x, 0.5)
    assert pcs.shape == (1, d), f"expected one PC, got {pcs.shape}"
    # The PC should align with axis 0 (sign-agnostic).
    assert abs(pcs[0, 0]) > 0.99


def test_project_out_makes_result_orthogonal():
    basis = unit_normalize(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    v = np.array([3.0, 4.0, 5.0])
    out = project_out(v, basis)
    # The x and y components should be gone; only z remains.
    assert np.allclose(out, [0.0, 0.0, 5.0])
    # And the result is orthogonal to every basis row.
    assert np.allclose(basis @ out, 0.0, atol=1e-10)


def test_project_out_empty_basis_is_identity():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(project_out(v, np.zeros((0, 3))), v)


def test_unit_normalize():
    v = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = unit_normalize(v, axis=1)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # zero vector stays zero (no NaN)


def test_cosine_matrix_properties():
    v = RNG.normal(size=(5, 8))
    c = cosine_similarity_matrix(v)
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)
    assert c.max() <= 1.0 + 1e-9 and c.min() >= -1.0 - 1e-9


# --------------------------------------------------------------------------- #
# vectors (the recipe)
# --------------------------------------------------------------------------- #
def test_recipe_removes_confound_and_normalises():
    hidden = 16
    confound = unit_normalize(RNG.normal(size=hidden))      # present in neutral text
    e_dirs = unit_normalize(RNG.normal(size=(3, hidden)), axis=1)
    global_offset = RNG.normal(size=hidden)
    # Each emotion mean = shared offset + its own direction + a big confound load.
    emotion_means = global_offset + e_dirs + 5.0 * confound
    # Neutral data varies almost entirely along the confound direction.
    neutral = global_offset + np.outer(RNG.normal(0, 3, 400), confound)
    vectors, gmean = build_layer_vectors(emotion_means, neutral, variance_threshold=0.5)
    # Unit length.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # The confound direction has been projected out of every emotion vector.
    assert np.all(np.abs(vectors @ confound) < 1e-6)
    # global mean is the across-emotion mean.
    assert np.allclose(gmean, emotion_means.mean(axis=0))


def test_emotion_vector_computer_multilayer():
    layers, emos, hidden, n_neutral = 3, 4, 12, 200
    means = RNG.normal(size=(layers, emos, hidden))
    neutral = RNG.normal(size=(layers, n_neutral, hidden))
    bank = EmotionVectorComputer(0.5).compute(means, neutral, ["a", "b", "c", "d"])
    assert bank.vectors.shape == (layers, emos, hidden)
    assert bank.global_means.shape == (layers, hidden)
    assert np.allclose(np.linalg.norm(bank.vectors, axis=2), 1.0)


def test_deflection_orthogonalises_against_expression_space():
    # The faithful contract (matching the paper and the original repo): deflection
    # vectors are orthogonalised against the *99%-variance principal components* of
    # the expression-vector matrix. We assert exactly that — the residual has ~zero
    # projection onto those PCs. (Note: top_pcs centres the data, so the across-
    # emotion MEAN direction is intentionally not removed; with 171 real emotions
    # that mean is negligible, but we test the precise implemented contract here.)
    layers, emos, hidden, n_neutral = 2, 6, 10, 200
    expr_vecs = unit_normalize(RNG.normal(size=(layers, emos, hidden)), axis=2)
    expr = ProbeBank(list("abcdef"), expr_vecs, np.zeros((layers, hidden)))
    means = RNG.normal(size=(layers, emos, hidden))
    neutral = RNG.normal(size=(layers, n_neutral, hidden))
    defl = DeflectionVectorComputer(0.5, 0.99).compute(means, neutral, list("abcdef"), expr)
    for layer in range(layers):
        expr_pcs = top_pcs_for_variance(expr.layer_vectors(layer), 0.99)
        proj = defl.vectors[layer] @ expr_pcs.T  # (emos, k)
        assert np.all(np.abs(proj) < 1e-6), f"layer {layer} not orthogonalised: {proj.max()}"
    # And the vectors remain unit length after orthogonalisation + renormalisation.
    assert np.allclose(np.linalg.norm(defl.vectors, axis=2), 1.0)


# --------------------------------------------------------------------------- #
# probes (projection + IO)
# --------------------------------------------------------------------------- #
def test_probe_projection_matches_definition():
    layers, emos, hidden = 2, 3, 5
    vecs = unit_normalize(RNG.normal(size=(layers, emos, hidden)), axis=2)
    gmeans = RNG.normal(size=(layers, hidden))
    bank = ProbeBank(["x", "y", "z"], vecs, gmeans)
    act = RNG.normal(size=hidden)
    scores = bank.project(act, layer=1)
    expected = (act - gmeans[1]) @ vecs[1].T
    assert np.allclose(scores, expected, atol=1e-5)
    # Batched / multi-token shape is preserved.
    seq = RNG.normal(size=(7, hidden))
    assert bank.project(seq, layer=0).shape == (7, emos)


def test_probebank_save_load(tmp_path=None):
    import tempfile
    from pathlib import Path

    layers, emos, hidden = 2, 3, 4
    bank = ProbeBank(
        ["a", "b", "c"],
        unit_normalize(RNG.normal(size=(layers, emos, hidden)), axis=2),
        RNG.normal(size=(layers, hidden)),
    )
    d = Path(tempfile.mkdtemp())
    bank.save(d / "bank.npz")
    loaded = ProbeBank.load(d / "bank.npz")
    assert loaded.emotions == bank.emotions
    assert np.allclose(loaded.vectors, bank.vectors)
    assert np.allclose(loaded.global_means, bank.global_means)


# --------------------------------------------------------------------------- #
# elo
# --------------------------------------------------------------------------- #
def test_elo_orders_transitive_winners():
    # Item 0 beats 1 beats 2 beats 3 every time -> ratings should be ordered.
    games = []
    for _ in range(5):
        games += [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ratings = compute_elo(4, games)
    assert ratings[0] > ratings[1] > ratings[2] > ratings[3]


def test_games_from_preferences():
    pairs = [(0, 1), (2, 3)]
    games = games_from_preferences(pairs, [True, False])
    assert games == [(0, 1), (3, 2)]


# --------------------------------------------------------------------------- #
# ranking (visualiser RRF)
# --------------------------------------------------------------------------- #
def test_rank_descending():
    ranks = rank_descending(np.array([0.1, 0.9, 0.5]))
    assert list(ranks) == [3.0, 1.0, 2.0]


def test_rrf_prefers_consistently_top_item():
    # item 0 ranked 1st by both signals -> highest fused score.
    r1 = np.array([1.0, 2.0, 3.0])
    r2 = np.array([1.0, 3.0, 2.0])
    fused = reciprocal_rank_fusion([r1, r2])
    assert np.argmax(fused) == 0


def test_autocorrelation_sustained_vs_spiky():
    sustained = np.array([[1.0, 1.0, 1.0, 1.0]])
    spiky = np.array([[1.0, -1.0, 1.0, -1.0]])
    assert autocorrelation(sustained)[0] > autocorrelation(spiky)[0]


def test_rank_emotions_picks_sustained_emotion():
    # Emotion 0 has a sustained positive signal; emotion 1 is noisy zero-mean.
    scores = np.array([
        [0.5, 0.6, 0.55, 0.52],
        [0.9, -0.9, 0.8, -0.85],
    ])
    order, _ = rank_emotions(scores)
    assert order[0] == 0


# --------------------------------------------------------------------------- #
# spans (Issue #7 + averaging)
# --------------------------------------------------------------------------- #
def test_mean_from_offset():
    act = np.arange(10, dtype=float).reshape(10, 1)  # values 0..9
    mean, count = mean_from_offset(act, seq_len=10, skip=5)
    assert count == 5
    assert np.isclose(mean[0], np.mean([5, 6, 7, 8, 9]))
    # Too short -> empty.
    _, c2 = mean_from_offset(act, seq_len=3, skip=5)
    assert c2 == 0


def test_masked_mean():
    act = np.array([[1.0], [2.0], [3.0], [4.0]])
    mask = np.array([0, 1, 1, 0])
    assert np.isclose(masked_mean(act, mask)[0], 2.5)
    assert np.allclose(masked_mean(act, np.zeros(4)), 0.0)


def test_char_to_token_mapping_is_exact():
    # text = "Hi Bob now" ; pretend tokenizer offsets:
    #   token0 "Hi"  (0,2), token1 " Bob" (2,6), token2 " now" (6,10)
    offsets = [(0, 2), (2, 6), (6, 10)]
    # Select characters [3,6) -> overlaps token1 only.
    mask = char_spans_to_token_mask(offsets, [(3, 6)])
    assert list(mask) == [False, True, False]
    spans = char_spans_to_token_spans(offsets, [(3, 6)])
    assert spans == [(1, 2)]


def test_char_to_token_skips_special_tokens():
    # token0 is a special (0,0) BOS; should never be selected.
    offsets = [(0, 0), (0, 3), (3, 7)]
    mask = char_spans_to_token_mask(offsets, [(0, 7)])
    assert list(mask) == [False, True, True]


def test_dialogue_parsing():
    text = (
        "Scenario: Alice is angry but acts calm.\n\n"
        "Alice: No worries at all, it's fine.\n\n"
        "Bob: Are you sure?\n\n"
        "Alice: Completely."
    )
    start = find_dialogue_start_char(text, ["Alice", "Bob"])
    assert start is not None and text[start:].startswith("\n\nAlice:")
    alice_spans = find_speaker_char_spans(text, "Alice", ["Alice", "Bob"])
    # Alice speaks twice; her words should not contain Bob's line.
    spoken = [text[s:e] for s, e in alice_spans]
    assert len(spoken) == 2
    assert "No worries" in spoken[0]
    assert "Completely" in spoken[1]
    assert "Are you sure" not in " ".join(spoken)
