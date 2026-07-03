"""
Integration test for the vector-building pipelines (no GPU / torch).

Fabricates small synthetic activation ``.npz`` files in a temp directory, runs
both :class:`EmotionVectorPipeline` and :class:`DeflectionVectorPipeline`, and
checks the saved banks have the right shapes, unit-norm vectors, and that the
deflection "target" bank is orthogonalised against the expression space.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.linalg import top_pcs_for_variance
from emotion_probes.core.probes import ProbeBank
from emotion_probes.core.vectors import DeflectionVectors
from emotion_probes.pipeline import DeflectionVectorPipeline, EmotionVectorPipeline

RNG = np.random.default_rng(7)
L, H, N = 3, 8, 60


def _make_config() -> Config:
    return Config().with_(data_dir=Path(tempfile.mkdtemp()))


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_emotion_then_deflection_pipeline():
    cfg = _make_config()
    emotions = ["happy", "sad", "angry", "calm", "afraid", "proud"]
    E = len(emotions)

    # Emotion-story activations + neutral baseline.
    _write_npz(
        cfg.activations_dir.parent / "stories.npz",
        means=RNG.normal(size=(L, E, H)).astype(np.float32),
        emotions=np.array(emotions, dtype=object),
    )
    _write_npz(cfg.neutral_story_activations, activations=RNG.normal(size=(L, N, H)).astype(np.float32))

    bank = EmotionVectorPipeline(cfg).run()
    assert bank.emotions == emotions
    assert bank.vectors.shape == (L, E, H)
    assert np.allclose(np.linalg.norm(bank.vectors, axis=2), 1.0)
    # Round-trips from disk.
    reloaded = ProbeBank.load(cfg.emotion_vectors_path)
    assert reloaded.emotions == emotions

    # Deflection activations (target/scenario share emotions; displayed differs; pairs).
    target_emos = ["happy", "sad", "angry"]
    displayed_emos = ["calm", "proud"]
    pair_labels = ["happy->calm", "sad->proud", "angry->calm"]
    _write_npz(
        cfg.deflection_activations,
        target_means=RNG.normal(size=(L, len(target_emos), H)).astype(np.float32),
        target_emotions=np.array(target_emos, dtype=object),
        scenario_means=RNG.normal(size=(L, len(target_emos), H)).astype(np.float32),
        displayed_means=RNG.normal(size=(L, len(displayed_emos), H)).astype(np.float32),
        displayed_emotions=np.array(displayed_emos, dtype=object),
        pair_means=RNG.normal(size=(L, len(pair_labels), H)).astype(np.float32),
        pair_labels=np.array(pair_labels, dtype=object),
    )
    _write_npz(cfg.neutral_dialogue_activations, activations=RNG.normal(size=(L, N, H)).astype(np.float32))

    defl = DeflectionVectorPipeline(cfg).run()
    assert defl.target.emotions == target_emos
    assert defl.displayed.emotions == displayed_emos
    assert defl.pair.emotions == pair_labels
    assert np.allclose(np.linalg.norm(defl.target.vectors, axis=2), 1.0)

    # The "target" bank must be orthogonalised against the expression space.
    for layer in range(L):
        expr_pcs = top_pcs_for_variance(bank.layer_vectors(layer), cfg.expression_orthogonalization_threshold)
        proj = defl.target.vectors[layer] @ expr_pcs.T
        assert np.all(np.abs(proj) < 1e-5), f"target not orthogonalised at layer {layer}"

    # Round-trips from disk.
    loaded = DeflectionVectors.load(cfg.deflection_vectors_path)
    assert loaded.target.emotions == target_emos
    assert loaded.pair.emotions == pair_labels
