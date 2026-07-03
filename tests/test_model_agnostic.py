"""
Tests for the model-agnosticism guards (CPU-only, no torch).

These pin the safety net that lets an ARBITRARY open model be passed in:
  * ProbeBank.check_against — the bank<->model compatibility guard that prevents
    silently-wrong results from using a bank with the wrong model;
  * the source_model_id stamp surviving a save/load roundtrip;
  * ProbedModel._resolve_hidden_size handling the common config shapes
    (hidden_size / n_embd / d_model, top-level and nested) — a pure static method
    that needs no torch;
  * the decoder-layer path list covering the architectures we claim to support.
"""

from __future__ import annotations

import os
import tempfile
import types
import warnings

import numpy as np

from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import (
    ProbedModel,
    _HIDDEN_SIZE_ATTRS,
    _LAYER_PATHS,
)


def _bank(num_layers: int = 4, hidden: int = 8, source: str | None = "model/x") -> ProbeBank:
    return ProbeBank(
        emotions=["a", "b"],
        vectors=np.zeros((num_layers, 2, hidden), dtype=np.float32),
        global_means=np.zeros((num_layers, hidden), dtype=np.float32),
        source_model_id=source,
    )


def _fake_model(num_layers: int, hidden: int, model_id: str) -> object:
    """A stand-in exposing just what check_against reads."""
    return types.SimpleNamespace(
        num_layers=num_layers,
        hidden_size=hidden,
        config=types.SimpleNamespace(model_id=model_id),
    )


# --------------------------------------------------------------------------- #
# check_against
# --------------------------------------------------------------------------- #
def test_check_against_accepts_a_matching_model() -> None:
    _bank(4, 8, "model/x").check_against(_fake_model(4, 8, "model/x"))  # no raise, no warn


def test_check_against_rejects_layer_count_mismatch() -> None:
    # Same hidden width, different depth — the silent-wrong-results case.
    try:
        _bank(4, 8).check_against(_fake_model(6, 8, "other"))
        raise AssertionError("expected ValueError on layer-count mismatch")
    except ValueError as exc:
        assert "layers" in str(exc)


def test_check_against_rejects_hidden_size_mismatch() -> None:
    try:
        _bank(4, 8).check_against(_fake_model(4, 16, "other"))
        raise AssertionError("expected ValueError on hidden-size mismatch")
    except ValueError as exc:
        assert "hidden" in str(exc)


def test_check_against_warns_on_model_id_mismatch_when_shapes_match() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bank(4, 8, "model/x").check_against(_fake_model(4, 8, "model/y"))
    assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)


# --------------------------------------------------------------------------- #
# source_model_id persistence
# --------------------------------------------------------------------------- #
def test_source_model_id_survives_save_load_roundtrip() -> None:
    path = os.path.join(tempfile.mkdtemp(), "vectors.npz")
    _bank(4, 8, "Qwen/Qwen2.5-7B-Instruct").save(path)
    assert ProbeBank.load(path).source_model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_missing_source_model_id_loads_as_none() -> None:
    path = os.path.join(tempfile.mkdtemp(), "vectors.npz")
    _bank(4, 8, None).save(path)
    assert ProbeBank.load(path).source_model_id is None


# --------------------------------------------------------------------------- #
# hidden-size resolution (static; no torch)
# --------------------------------------------------------------------------- #
def test_resolve_hidden_size_top_level_variants() -> None:
    for attr in _HIDDEN_SIZE_ATTRS:  # hidden_size / n_embd / d_model
        model = types.SimpleNamespace(config=types.SimpleNamespace(**{attr: 1234}))
        assert ProbedModel._resolve_hidden_size(model) == 1234


def test_resolve_hidden_size_nested_text_config() -> None:
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=3072))
    )
    assert ProbedModel._resolve_hidden_size(model) == 3072


def test_resolve_hidden_size_raises_when_absent() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=100))
    try:
        ProbedModel._resolve_hidden_size(model)
        raise AssertionError("expected RuntimeError when no width field is present")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# layer-path coverage
# --------------------------------------------------------------------------- #
def test_layer_paths_cover_the_claimed_architectures() -> None:
    # The architectures the README/ARCHITECTURE docs claim to support must each
    # have their decoder-layer container path enumerated.
    for path in (
        "model.layers",                # Llama/Qwen/Mistral/Mixtral/Gemma2/Phi3
        "model.language_model.layers", # Gemma 3n multimodal nesting
        "transformer.h",               # GPT-2/GPT-J/Falcon/BLOOM
        "transformer.blocks",          # MPT/DBRX
        "gpt_neox.layers",             # GPT-NeoX/Pythia
        "model.decoder.layers",        # OPT
        "backbone.layers",             # Mamba
        "rwkv.blocks",                 # RWKV
    ):
        assert path in _LAYER_PATHS, f"missing layer path: {path}"
