"""
GPU-free tests for the per-story residual cache (``scripts/cache_story_residuals.py``,
the E1 story-null caching pass — docs/prereg/E1.md "Deviations").

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. torch is used only
on CPU (tiny nn.Modules standing in for decoder layers); no GPU, no transformers.

Coverage:
* pooling math          — the REAL ``ProbedModel.extract_means`` (the exact code path
                          that built stories.npz and that the cache script delegates to)
                          driven by a fake tokenizer + fake torch model with KNOWN hidden
                          states: masked mean correct under right padding (poison pad id),
                          skip respected, batching-invariant, truncation, short-story
                          zero-row clamp;
* CSV loading           — order preservation within emotion, topic alignment, subset
                          filter, unknown-emotion / missing-column errors;
* encode_emotion        — fp16 output, shape, kwargs passthrough, non-finite/overflow
                          hard-fail;
* validation gate       — per-layer cosines ~1 on matched data; planted mismatch raises
                          with emotion + layer diagnostics; zero-norm layer fails; shape
                          mismatch fails;
* run_cache             — writes <out>/<emotion>.npy fp16 + topics.json FLAT (the
                          geometry.story_nulls cache contract); resume skips existing
                          files (and re-encodes only deleted ones); gate aborts BEFORE
                          writing the failing .npy; corrupt existing cache raises;
                          missing reference emotion raises; --emotions subset; subset
                          re-run merges topics.json instead of clobbering it;
* load_reference        — provenance hard-fail on the quarantined Qwen2.5-7B geometry
                          (28, E, 3584), missing keys, name/row mismatch.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cache_story_residuals as csr
from emotion_probes.persona.assets import ProvenanceError


# =========================================================================== #
# 1. Pooling math — the real ProbedModel.extract_means on a fake torch model
# =========================================================================== #
PAD_ID = 999  # poison: any unmasked pad position visibly corrupts the mean


class _Batch(dict):
    def to(self, device):
        return self


class _IntTok:
    """Texts are space-separated ints -> token ids; right-pads with the poison id."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=None, return_offsets_mapping=False, add_special_tokens=True):
        import torch

        self.calls.append({"add_special_tokens": add_special_tokens, "padding": padding,
                           "truncation": truncation, "max_length": max_length})
        ids = [[int(w) for w in t.split()][:max_length] for t in texts]
        width = max(len(r) for r in ids)
        input_ids = torch.tensor([r + [PAD_ID] * (width - len(r)) for r in ids])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in ids])
        return _Batch(input_ids=input_ids, attention_mask=mask)


def _fake_probed(n_layers=2, hidden=4):
    """object.__new__(ProbedModel) with hand-set attrs; hidden state at (b, t, d) is
    ``input_ids[b, t] + d`` and layer i's hooked output is that times (i + 1)."""
    import torch

    from emotion_probes.models.language_model import ProbedModel

    class _ScaledLayer(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = float(k)

        def forward(self, x):
            return x * self.k

    class _KnownStateLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = hidden
            self.layers = torch.nn.ModuleList(_ScaledLayer(i + 1) for i in range(n_layers))
            self.emb = torch.nn.Embedding(2, 2)  # only consulted for device discovery

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids=None, attention_mask=None, **kw):
            base = input_ids.float().unsqueeze(-1) + torch.arange(self.hidden).float()
            for layer in self.layers:
                layer(base)  # forward hooks capture each layer's output

    pm = object.__new__(ProbedModel)
    pm.config = SimpleNamespace(token_skip=50, batch_size=2, max_length=8)
    pm.tokenizer = _IntTok()
    pm.tokenizer_is_fast = True
    pm.model = _KnownStateLM()
    pm._layers = list(pm.model.layers)
    pm.num_layers = n_layers
    pm.hidden_size = hidden
    return pm


def test_pooling_masked_mean_under_padding():
    pm = _fake_probed()
    out = pm.extract_means(["3 5 7", "11 13"], skip=1, batch_size=2, max_length=8)
    assert out.shape == (2, 2, 4) and out.dtype == np.float32
    d = np.arange(4, dtype=np.float64)
    # text0: positions >= 1 are [5, 7] -> mean 6; text1: only position 1 is real -> 13.
    # text1's position 2 is the poison pad (999): if the mask were ignored the mean
    # would be 506 + d, so these equalities prove attention-mask-aware pooling.
    np.testing.assert_allclose(out[0, 0], 6 + d)
    np.testing.assert_allclose(out[0, 1], (6 + d) * 2)
    np.testing.assert_allclose(out[1, 0], 13 + d)
    np.testing.assert_allclose(out[1, 1], (13 + d) * 2)
    # recipe knobs match the original stories.npz extraction
    assert all(c["add_special_tokens"] and c["padding"] and c["truncation"]
               for c in pm.tokenizer.calls)


def test_pooling_batching_invariant():
    # padding width differs between batchings; masked pooling must not care
    pm = _fake_probed()
    texts = ["3 5 7", "11 13", "2 4 6 8"]
    a = pm.extract_means(texts, skip=1, batch_size=2, max_length=8)
    b = pm.extract_means(texts, skip=1, batch_size=3, max_length=8)
    np.testing.assert_allclose(a, b)


def test_pooling_truncation_and_short_story_clamp():
    pm = _fake_probed()
    # truncation at max_length=2: "3 5 7" -> [3, 5]; skip=1 -> mean over [5]
    out = pm.extract_means(["3 5 7", "11 13"], skip=1, batch_size=2, max_length=2)
    np.testing.assert_allclose(out[0, 0], 5 + np.arange(4))
    # a story shorter than the skip -> zero row (count clamped to 1), replicating
    # the original pipeline's behaviour exactly
    out = pm.extract_means(["9", "3 5 7"], skip=1, batch_size=2, max_length=8)
    np.testing.assert_allclose(out[0], 0.0)


# =========================================================================== #
# 2. CSV loading
# =========================================================================== #
def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["emotion", "topic", "story"])
        w.writerows(rows)


def test_load_stories_order_and_alignment():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "stories.csv")
    _write_csv(p, [
        ("joy", "cooking", "A story, with commas\nand a newline."),
        ("grief", "sports", "grief story one"),
        ("joy", "sports", "second joy story"),
        ("grief", "geography", "grief story two"),
    ])
    t = csr.load_stories(p)
    assert t.emotions == ["joy", "grief"]                      # first-appearance order
    assert t.stories["joy"] == ["A story, with commas\nand a newline.", "second joy story"]
    assert t.topics["joy"] == ["cooking", "sports"]            # row-aligned with stories
    assert t.topics["grief"] == ["sports", "geography"]
    sub = csr.load_stories(p, emotions=["grief"])
    assert sub.emotions == ["grief"] and "joy" not in sub.stories
    with pytest.raises(csr.ValidationError):
        csr.load_stories(p, emotions=["rage"])                 # unknown emotion


def test_load_stories_missing_column():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "bad.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emotion", "story"])
        w.writerow(["joy", "text"])
    with pytest.raises(csr.ValidationError, match="topic"):
        csr.load_stories(p)


# =========================================================================== #
# 3. encode_emotion + validation gate
# =========================================================================== #
class _FakeEncoder:
    """Stands in for ProbedModel: deterministic per-story means from a crc32 seed."""

    def __init__(self, n_layers=3, hidden=4):
        self.n_layers, self.hidden = n_layers, hidden
        self.encoded = []  # story lists actually encoded (resume assertions)
        self.kwargs = []

    def extract_means(self, texts, skip=None, batch_size=None, max_length=None):
        self.encoded.append(list(texts))
        self.kwargs.append({"skip": skip, "batch_size": batch_size, "max_length": max_length})
        return np.stack([self._one(t) for t in texts])

    def _one(self, text):
        rng = np.random.default_rng(zlib.crc32(text.encode()))
        return (rng.standard_normal((self.n_layers, self.hidden)) * 7.0).astype(np.float32)


def test_encode_emotion_fp16_and_kwargs():
    enc = _FakeEncoder()
    arr = csr.encode_emotion(enc, ["a", "b"], batch_size=8, skip=50, max_length=512)
    assert arr.dtype == np.float16 and arr.shape == (2, 3, 4)
    assert enc.kwargs[0] == {"skip": 50, "batch_size": 8, "max_length": 512}
    np.testing.assert_allclose(arr, enc.extract_means(["a", "b"]).astype(np.float16))


def test_encode_emotion_rejects_fp16_overflow_and_nan():
    class _Overflowing:
        def extract_means(self, texts, **kw):
            out = np.ones((len(texts), 2, 3), dtype=np.float32)
            out[0, 0, 0] = 1e5  # > 65504 -> inf in float16
            return out

    with pytest.raises(csr.ValidationError, match="65504"):
        csr.encode_emotion(_Overflowing(), ["a"], batch_size=1, skip=50, max_length=512)

    class _NaN:
        def extract_means(self, texts, **kw):
            out = np.ones((len(texts), 2, 3), dtype=np.float32)
            out[0, 1, 2] = np.nan
            return out

    with pytest.raises(csr.ValidationError, match="non-finite"):
        csr.encode_emotion(_NaN(), ["a"], batch_size=1, skip=50, max_length=512)


def test_gate_passes_on_matched_data():
    rng = np.random.default_rng(0)
    per = rng.standard_normal((10, 3, 4)).astype(np.float16)
    ref = per.astype(np.float64).mean(axis=0)
    cos = csr.layer_cosines(per, ref)
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)
    stats = csr.validate_emotion(per, ref, "joy")
    assert stats["min_cos"] > 0.999 and 0 <= stats["argmin_layer"] < 3


def test_gate_triggers_on_planted_mismatch():
    rng = np.random.default_rng(1)
    per = rng.standard_normal((10, 3, 4)).astype(np.float16)
    ref = per.astype(np.float64).mean(axis=0)
    bad = ref.copy()
    bad[1] = -bad[1]                                            # cosine -1 at layer 1
    with pytest.raises(csr.ValidationError) as ei:
        csr.validate_emotion(per, bad, "grief")
    msg = str(ei.value)
    assert "grief" in msg and "L1=" in msg and "1/3 layers" in msg
    assert "token_skip=50" in msg                               # diagnostic names the recipe


def test_gate_zero_norm_and_shape_mismatch():
    rng = np.random.default_rng(2)
    per = rng.standard_normal((5, 3, 4)).astype(np.float16)
    ref = per.astype(np.float64).mean(axis=0)
    zref = ref.copy()
    zref[0] = 0.0                                               # nan cosine -> gate fails
    with pytest.raises(csr.ValidationError):
        csr.validate_emotion(per, zref, "joy")
    with pytest.raises(csr.ValidationError, match="shape"):
        csr.validate_emotion(per, ref[:2], "joy")


# =========================================================================== #
# 4. run_cache — outputs, resume, gate abort, subset
# =========================================================================== #
def _fixture(tmp, emotions=("joy", "grief"), n=5):
    """CSV + table + reference npz arrays consistent with _FakeEncoder outputs."""
    enc = _FakeEncoder()
    rows = [(e, f"topic-{e}-{i}", f"{e} story {i}, with commas")
            for e in emotions for i in range(n)]
    csv_path = os.path.join(tmp, "stories.csv")
    _write_csv(csv_path, rows)
    table = csr.load_stories(csv_path)
    ref = np.stack(
        [enc.extract_means(table.stories[e]).astype(np.float16).astype(np.float64).mean(axis=0)
         for e in table.emotions], axis=1).astype(np.float32)   # (L, E, H)
    enc.encoded.clear()
    enc.kwargs.clear()
    return enc, table, list(table.emotions), ref, csv_path


_KW = dict(batch_size=4, skip=50, max_length=512, log=lambda m: None)


def test_run_cache_writes_outputs():
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, _ = _fixture(tmp)
    out = Path(tmp) / "cache"
    summary = csr.run_cache(enc, table, emos, ref, out, **_KW)
    # FLAT layout: <out>/<emotion>.npy beside <out>/topics.json — the exact
    # contract of emotion_probes.persona.geometry.story_nulls(--story-acts out)
    for e in emos:
        arr = np.load(out / f"{e}.npy")
        assert arr.dtype == np.float16 and arr.shape == (5, 3, 4)
    assert not (out / "stories").exists()
    topics = json.load(open(out / "topics.json"))
    assert topics == {e: table.topics[e] for e in emos}
    assert topics["joy"] == [f"topic-joy-{i}" for i in range(5)]  # row-aligned order
    assert summary["n_encoded"] == 2 and summary["n_stories"] == 10
    assert summary["min_cos"] > 0.99
    assert enc.kwargs[0] == {"skip": 50, "batch_size": 4, "max_length": 512}


def test_run_cache_resume_skips_existing():
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, _ = _fixture(tmp)
    out = Path(tmp) / "cache"
    csr.run_cache(enc, table, emos, ref, out, **_KW)
    assert len(enc.encoded) == 2
    enc.encoded.clear()
    s2 = csr.run_cache(enc, table, emos, ref, out, **_KW)       # full resume: no encoding
    assert enc.encoded == [] and s2["n_encoded"] == 0
    assert all(r["cached"] for r in s2["emotions"])
    (out / "grief.npy").unlink()                                # re-encode only the gap
    s3 = csr.run_cache(enc, table, emos, ref, out, **_KW)
    assert len(enc.encoded) == 1 and enc.encoded[0][0].startswith("grief")
    assert s3["n_encoded"] == 1


def test_run_cache_gate_aborts_before_write():
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, _ = _fixture(tmp)
    out = Path(tmp) / "cache"
    bad = ref.copy()
    bad[:, 1, :] *= -1.0                                        # plant a mismatch on grief
    with pytest.raises(csr.ValidationError, match="grief"):
        csr.run_cache(enc, table, emos, bad, out, **_KW)
    assert (out / "joy.npy").exists()                           # joy passed first
    assert not (out / "grief.npy").exists()                     # aborted BEFORE the write
    assert not (out / "grief.npy.tmp").exists()
    assert (out / "topics.json").exists()


def test_run_cache_rejects_corrupt_existing_cache():
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, _ = _fixture(tmp)
    out = Path(tmp) / "cache"
    csr.run_cache(enc, table, emos, ref, out, **_KW)
    np.save(out / "joy.npy", np.zeros((2, 3, 4), dtype=np.float16))  # wrong n
    with pytest.raises(csr.ValidationError, match="delete"):
        csr.run_cache(enc, table, emos, ref, out, **_KW)


def test_run_cache_missing_reference_emotion():
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, _ = _fixture(tmp)
    with pytest.raises(csr.ValidationError, match="grief"):
        csr.run_cache(enc, table, emos[:1], ref[:, :1, :], Path(tmp) / "cache", **_KW)


def test_run_cache_emotions_subset():
    tmp = tempfile.mkdtemp()
    enc, _, emos, ref, csv_path = _fixture(tmp)
    sub = csr.load_stories(csv_path, emotions=["grief"])
    out = Path(tmp) / "cache"
    summary = csr.run_cache(enc, sub, emos, ref, out, **_KW)
    assert summary["n_encoded"] == 1
    assert (out / "grief.npy").exists()
    assert not (out / "joy.npy").exists()
    assert list(json.load(open(out / "topics.json"))) == ["grief"]


def test_run_cache_subset_rerun_merges_topics():
    # a later --emotions subset re-run must not clobber the topics.json entries
    # of emotions already cached (their row alignment would be unrecoverable)
    tmp = tempfile.mkdtemp()
    enc, table, emos, ref, csv_path = _fixture(tmp)
    out = Path(tmp) / "cache"
    csr.run_cache(enc, table, emos, ref, out, **_KW)            # full run: joy + grief
    sub = csr.load_stories(csv_path, emotions=["grief"])
    csr.run_cache(enc, sub, emos, ref, out, **_KW)              # subset re-run
    topics = json.load(open(out / "topics.json"))
    assert set(topics) == {"joy", "grief"}                      # joy survived the re-run
    assert topics["joy"] == table.topics["joy"]
    assert topics["grief"] == table.topics["grief"]


# =========================================================================== #
# 5. load_reference provenance
# =========================================================================== #
def test_load_reference_good_and_quarantined():
    tmp = tempfile.mkdtemp()
    good = os.path.join(tmp, "stories.npz")
    np.savez(good, means=np.zeros((64, 2, 5120), dtype=np.float32),
             emotions=np.array(["joy", "grief"], dtype=object))
    emos, means = csr.load_reference(good)
    assert emos == ["joy", "grief"] and means.shape == (64, 2, 5120)
    assert means.dtype == np.float32
    # the quarantined Qwen2.5-7B geometry must hard-fail
    bad = os.path.join(tmp, "bad.npz")
    np.savez(bad, means=np.zeros((28, 2, 3584), dtype=np.float32),
             emotions=np.array(["joy", "grief"], dtype=object))
    with pytest.raises(ProvenanceError):
        csr.load_reference(bad)


def test_load_reference_bad_keys_and_row_mismatch():
    tmp = tempfile.mkdtemp()
    nokeys = os.path.join(tmp, "nokeys.npz")
    np.savez(nokeys, foo=np.zeros(3))
    with pytest.raises(ProvenanceError, match="keys"):
        csr.load_reference(nokeys)
    mism = os.path.join(tmp, "mism.npz")
    np.savez(mism, means=np.zeros((64, 2, 5120), dtype=np.float32),
             emotions=np.array(["joy", "grief", "awe"], dtype=object))
    with pytest.raises(ProvenanceError, match="names"):
        csr.load_reference(mism)
