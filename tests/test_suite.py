"""
Consistency checks for the 14-model suite manifest (emotion_probes/models/suite.py).

Pure-Python, no torch/network — these just assert the manifest is internally
coherent and that its env-emitter actually round-trips into a ``Config``. They
catch the easy ways a future manifest edit could go wrong (duplicate key, a
layer fraction that lands out of range, a sharded model left on device_map=cuda,
an env line that doesn't map to the field it should).
"""

from __future__ import annotations

import os

from emotion_probes.config import Config
from emotion_probes.models.suite import SUITE, by_key, env_exports, single_card_lpt


def test_keys_are_unique_and_nonempty():
    keys = [m.key for m in SUITE]
    assert len(keys) == len(set(keys)), "duplicate suite key"
    assert all(m.key and m.model_id for m in SUITE)
    assert len(SUITE) == 14


def test_analysis_layer_in_range():
    for m in SUITE:
        idx = m.analysis_layer()
        assert 0 <= idx < m.num_layers, f"{m.key}: analysis layer {idx} out of range"


def test_device_map_matches_capacity():
    # A model is sharded (auto) iff it does not fit one card; single-card otherwise.
    for m in SUITE:
        if m.fits_one_card():
            assert m.device_map == "cuda", f"{m.key} fits one card but device_map={m.device_map}"
        else:
            assert m.device_map == "auto", f"{m.key} needs sharding but device_map={m.device_map}"
    assert sum(1 for m in SUITE if not m.fits_one_card()) == 2  # 235b + mistral-large


def test_per_family_quirks():
    # Qwen3 dense force thinking off; the 235B (non-thinking already) leaves it None.
    for key in ("qwen3-1.7b", "qwen3-8b", "qwen3-14b", "qwen3-32b"):
        assert by_key(key).enable_thinking is False
    assert by_key("qwen3-235b").enable_thinking is None
    # Both Gemmas request eager attention; the 235B alone is FP8.
    assert by_key("gemma3-1b").attn_implementation == "eager"
    assert by_key("gemma3-27b").attn_implementation == "eager"
    assert [m.key for m in SUITE if m.fp8] == ["qwen3-235b"]
    # Only the 27B Gemma (VLM wrapper) is flagged multimodal.
    assert [m.key for m in SUITE if m.multimodal] == ["gemma3-27b"]


def test_single_card_lpt_is_complete_and_descending():
    lpt = single_card_lpt()
    single = [m.key for m in SUITE if m.fits_one_card()]
    assert sorted(lpt) == sorted(single)            # same set, all single-card models
    params = {m.key: m.params_b for m in SUITE}
    assert all(params[a] >= params[b] for a, b in zip(lpt, lpt[1:]))  # longest-first
    assert lpt[0] in ("qwen3-32b", "olmo2-32b")     # a ~32B model leads


def test_expected_gated_set():
    gated = {m.key for m in SUITE if m.gated}
    assert gated == {
        "llama3.2-1b", "llama3.2-3b", "llama3.1-8b",
        "gemma3-1b", "gemma3-27b", "mistral-large-123b",
    }


def test_env_exports_roundtrip_into_config():
    # The runner does `eval "$(suite env <key>)"`; emulate that and confirm the
    # exported env actually produces the intended Config for a representative model.
    m = by_key("qwen3-32b")
    saved = {k: v for k, v in os.environ.items() if k.startswith("EMOTION_PROBES_")}
    try:
        for k in list(os.environ):
            if k.startswith("EMOTION_PROBES_"):
                del os.environ[k]
        for line in env_exports(m):
            assert line.startswith("export ")
            k, _, v = line[len("export "):].partition("=")
            os.environ[k] = v
        cfg = Config()
        assert cfg.model_id == m.model_id
        assert cfg.device_map == m.device_map
        assert cfg.batch_size == m.batch_size
        assert cfg.deflection_batch_size == m.deflection_batch_size
        assert cfg.enable_thinking is False  # qwen3 dense
        assert str(cfg.data_dir).endswith("/qwen3-32b")
    finally:
        for k in list(os.environ):
            if k.startswith("EMOTION_PROBES_"):
                del os.environ[k]
        os.environ.update(saved)
