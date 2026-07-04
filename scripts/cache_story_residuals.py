#!/usr/bin/env python3
"""
Cache PER-STORY residual means for E1's decision-grade nulls (story-label
permutation + topic-contrast; docs/research/portfolio.md §E1 "Nulls",
docs/prereg/E1.md "Deviations").

The cluster ships stories.csv (17,100 rows: emotion, topic, story) and
stories.npz holding only per-emotion MEANS (64, 171, 5120) — no per-story
residuals, so the label-permutation null cannot run. This script encodes every
story with a plain forward pass (NO generation, ``output`` via forward hooks),
mean-pools each story's token span at ALL 64 layers, and writes

    <out>/<emotion>.npy   (n_stories, 64, 5120) float16   (~11 GB total)
    <out>/topics.json     {emotion: [topic, ...]}, row-aligned with the .npy
                          rows (merged with any existing entries each run)

This FLAT layout is the cache contract of the consumer,
``emotion_probes.persona.geometry.story_nulls`` (invoked via
``run_e1_atlas.py --story-acts <out>``): one ``<emotion>.npy`` per emotion plus
``topics.json`` (and optional ``layers.json``) in the SAME directory. Row i of
each .npy is model layer i (all 64 cached, so no layers.json is needed).

ORIGINAL POOLING (discovered in-repo, replicated EXACTLY): stories.npz was
produced by ``emotion_probes/extraction/extractor.py::StoryActivationExtractor``,
which calls ``ProbedModel.extract_means(stories)`` with the Config defaults —
tokenize with ``add_special_tokens=True``, RIGHT padding, truncation at
``max_length=512``; attention-mask-aware mean over token positions
``>= token_skip=50`` (counts clamped to >= 1, so a story shorter than the skip
yields a zero row); then the per-emotion mean of those per-story means, saved
by ``combine()`` as ``means`` (64, E, 5120) float32 + ``emotions``. We call the
SAME ``extract_means`` with the SAME defaults, so the mean over each emotion's
per-story rows reproduces the npz row up to float16 rounding and kernel
nondeterminism.

VALIDATION GATE (built in): after encoding each emotion, the mean over its
per-story rows must have cosine > ``--threshold`` (default 0.99) with the
matching stories.npz row at EVERY layer, else abort with a diagnostic BEFORE
writing the .npy — this catches tokenization/pooling mismatches with the
original extraction pipeline.

Checkpoint/resume: existing ``<emotion>.npy`` files are shape-, dtype- and
gate-checked, then skipped. Writes are atomic (tmp + fsync + os.replace,
the extractor's checkpoint idiom), so a mid-write kill never leaves a
truncated cache that poisons the next resume.

Cluster-only (1 H100, ~1-2 GPU-h; 171 emotions x ~100 stories):

    python3 scripts/cache_story_residuals.py \
        --stories-csv /data/agastyas/research/data-emo-qwen32b/stories.csv \
        --ref-npz     /data/agastyas/research/data-emo-qwen32b/activations/stories.npz \
        --out         /data/agastyas/research/data-emo-qwen32b/story_residuals

(The extraction pipeline writes the combined npz to
``<data>/activations/stories.npz`` — ``StoryActivationExtractor.combine()`` —
next to the per-emotion means in ``<data>/activations/stories/``.)

The pure pieces (CSV loading, pooling delegation, gate, resume loop) are
GPU-free and unit-tested in tests/test_story_cache.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python3 scripts/cache_story_residuals.py`
    sys.path.insert(0, str(REPO_ROOT))

from emotion_probes.persona.assets import HIDDEN, N_LAYERS, ProvenanceError

CLUSTER_DATA = Path("/data/agastyas/research/data-emo-qwen32b")
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_THRESHOLD = 0.99


class ValidationError(RuntimeError):
    """The per-story cache failed the mean-reconstruction gate against stories.npz."""


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
@dataclass
class StoryTable:
    """stories.csv grouped by emotion; CSV row order preserved within each."""

    emotions: list  # unique emotions, order of first appearance
    stories: dict   # emotion -> [story, ...]
    topics: dict    # emotion -> [topic, ...], row-aligned with stories


def load_stories(csv_path: str | Path, emotions=None) -> StoryTable:
    """Read stories.csv (columns emotion, topic, story) into a :class:`StoryTable`.

    ``emotions`` optionally restricts to a subset (unknown names raise). Row
    order within each emotion is the CSV order — the .npy row order downstream.
    """
    order, stories, topics = [], {}, {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"emotion", "topic", "story"} - set(reader.fieldnames or [])
        if missing:
            raise ValidationError(f"{csv_path}: missing columns {sorted(missing)} "
                                  f"(got {reader.fieldnames})")
        for row in reader:
            e = str(row["emotion"])
            if e not in stories:
                order.append(e)
                stories[e], topics[e] = [], []
            stories[e].append(str(row["story"]))
            topics[e].append(str(row["topic"]))
    if emotions is not None:
        unknown = [e for e in emotions if e not in stories]
        if unknown:
            raise ValidationError(f"--emotions not present in {csv_path}: {unknown}")
        keep = set(emotions)
        order = [e for e in order if e in keep]
        stories = {e: stories[e] for e in order}
        topics = {e: topics[e] for e in order}
    return StoryTable(emotions=order, stories=stories, topics=topics)


# --------------------------------------------------------------------------- #
# Reference npz (per-emotion means)
# --------------------------------------------------------------------------- #
def load_reference(path: str | Path):
    """Load stories.npz (keys ``means``/``emotions``) with hard provenance checks.

    Returns ``(emotions, means)``, means (64, E, 5120) float32. Raises
    :class:`ProvenanceError` on any other geometry — the quarantined Qwen2.5-7B
    bank is (28, 171, 3584) and must never pass.
    """
    z = np.load(path, allow_pickle=True)
    if "means" not in z.files or "emotions" not in z.files:
        raise ProvenanceError(f"{path}: expected npz keys 'means'/'emotions', got {z.files}")
    means = z["means"]
    emotions = [str(e) for e in z["emotions"]]
    if means.ndim != 3 or means.shape[0] != N_LAYERS or means.shape[2] != HIDDEN:
        raise ProvenanceError(
            f"{path}: means shape {means.shape} != ({N_LAYERS}, E, {HIDDEN}) — this must be "
            "the Qwen3-32B stories.npz (the Qwen2.5-7B bank is quarantined; never load it)")
    if means.shape[1] != len(emotions):
        raise ProvenanceError(f"{path}: {len(emotions)} emotion names != {means.shape[1]} rows")
    return emotions, means.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Encoding + validation gate
# --------------------------------------------------------------------------- #
def encode_emotion(model, stories, batch_size: int, skip: int, max_length: int) -> np.ndarray:
    """Per-story mean residuals at all layers, float16 for storage.

    Delegates to ``ProbedModel.extract_means`` — the SAME code path (tokenize,
    right padding, truncation, masked mean over positions >= skip) that built
    stories.npz, so the validation gate closes the loop on any drift.
    """
    means = model.extract_means(stories, skip=skip, batch_size=batch_size,
                                max_length=max_length)  # (n, L, H) float32
    with np.errstate(over="ignore"):  # we hard-fail on non-finite ourselves below
        arr = np.asarray(means, dtype=np.float16)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        max_abs = float(np.abs(np.asarray(means, dtype=np.float32)).max())
        raise ValidationError(
            f"{n_bad} non-finite float16 values after casting per-story means "
            f"(max |float32 value| = {max_abs:.1f}; float16 overflows above 65504). "
            "Either the model produced nan/inf residuals or a residual channel "
            "exceeds float16 range — do NOT store this cache as fp16.")
    return arr


def layer_cosines(per_story: np.ndarray, ref_row: np.ndarray) -> np.ndarray:
    """Per-layer cosine between mean(per-story rows) and the reference row.

    per_story (n, L, H); ref_row (L, H) -> (L,) float64, nan where a norm is 0.
    """
    mean = per_story.astype(np.float64).mean(axis=0)  # (L, H)
    ref = ref_row.astype(np.float64)
    num = (mean * ref).sum(axis=1)
    den = np.linalg.norm(mean, axis=1) * np.linalg.norm(ref, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def validate_emotion(per_story: np.ndarray, ref_row: np.ndarray, emotion: str,
                     threshold: float = DEFAULT_THRESHOLD) -> dict:
    """THE GATE: every layer's cosine must exceed ``threshold``, else abort."""
    if per_story.ndim != 3 or per_story.shape[1:] != ref_row.shape:
        raise ValidationError(f"{emotion}: per-story array shape {per_story.shape} does not "
                              f"match reference row shape {ref_row.shape}")
    cos = layer_cosines(per_story, ref_row)
    bad = np.flatnonzero(~(cos > threshold))  # nan counts as failing
    if bad.size:
        detail = ", ".join(f"L{int(l)}={cos[l]:.6f}" for l in bad[:8])
        more = ", ..." if bad.size > 8 else ""
        raise ValidationError(
            f"validation gate FAILED for {emotion!r}: cosine <= {threshold} at "
            f"{bad.size}/{len(cos)} layers ({detail}{more}); n_stories={per_story.shape[0]}. "
            "The per-story pooling must replicate the original extraction "
            "(StoryActivationExtractor -> ProbedModel.extract_means: add_special_tokens=True, "
            "right padding, truncation at max_length=512, masked mean over positions >= "
            "token_skip=50). Check --skip / --max-length, the tokenizer revision, and that "
            "--ref-npz belongs to this stories.csv and model.")
    return {"min_cos": float(np.min(cos)), "argmin_layer": int(np.argmin(cos))}


# --------------------------------------------------------------------------- #
# Atomic writes (extractor checkpoint idiom: tmp + fsync + replace)
# --------------------------------------------------------------------------- #
def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main loop (resume per emotion)
# --------------------------------------------------------------------------- #
def run_cache(model, table: StoryTable, ref_emotions, ref_means: np.ndarray,
              out_dir: str | Path, *, batch_size: int, skip: int, max_length: int,
              threshold: float = DEFAULT_THRESHOLD, log=_log) -> dict:
    """Encode + gate + save every emotion in ``table``; resumable per emotion.

    ``model`` only needs ``.extract_means`` (a fake suffices in tests). Existing
    ``.npy`` files are shape/dtype/gate-checked, then skipped. The gate runs
    BEFORE each write, so a pooling mismatch never leaves a bad cache on disk.

    Layout is FLAT — ``<out>/<emotion>.npy`` beside ``<out>/topics.json`` — the
    contract of ``emotion_probes.persona.geometry.story_nulls``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_index = {e: i for i, e in enumerate(ref_emotions)}
    missing = [e for e in table.emotions if e not in ref_index]
    if missing:
        raise ValidationError(
            f"{len(missing)} emotions in the CSV have no row in the reference npz, so they "
            f"cannot be gate-checked: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    # topics.json is pure CSV-derived — write it first so even an interrupted
    # run leaves the row-alignment record next to the .npy files. MERGE with any
    # existing entries so a later --emotions subset re-run cannot destroy the
    # alignment record of emotions already cached.
    topics_path = out_dir / "topics.json"
    topics_payload = {}
    if topics_path.exists():
        try:
            topics_payload = dict(json.loads(topics_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            topics_payload = {}  # unreadable -> rebuild from this run's table
    topics_payload.update({e: table.topics[e] for e in table.emotions})
    _atomic_write_json(topics_path, topics_payload)

    rows = []
    for pos, emotion in enumerate(table.emotions, 1):
        stories = table.stories[emotion]
        ref_row = ref_means[:, ref_index[emotion], :]
        out_file = out_dir / f"{emotion}.npy"
        t0 = time.time()
        if out_file.exists():
            arr = np.load(out_file)
            if arr.shape != (len(stories), *ref_row.shape) or arr.dtype != np.float16:
                raise ValidationError(
                    f"{out_file}: existing cache has shape {arr.shape} dtype {arr.dtype}, "
                    f"expected ({len(stories)}, {ref_row.shape[0]}, {ref_row.shape[1]}) "
                    "float16 — delete the file to re-encode")
            stats = validate_emotion(arr, ref_row, emotion, threshold)
            cached = True
        else:
            arr = encode_emotion(model, stories, batch_size, skip, max_length)
            stats = validate_emotion(arr, ref_row, emotion, threshold)  # gate BEFORE write
            _atomic_save_npy(out_file, arr)
            cached = False
        rows.append({"emotion": emotion, "n_stories": len(stories), "cached": cached, **stats})
        tail = "cached, skipped" if cached else f"{time.time() - t0:.1f}s"
        log(f"  [{pos}/{len(table.emotions)}] {emotion}: n={len(stories)} "
            f"min_cos={stats['min_cos']:.6f} (L{stats['argmin_layer']}) {tail}")
    return {
        "out_dir": str(out_dir),
        "n_emotions": len(rows),
        "n_stories": sum(r["n_stories"] for r in rows),
        "n_encoded": sum(not r["cached"] for r in rows),
        "min_cos": min((r["min_cos"] for r in rows), default=None),
        "emotions": rows,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache per-story residual means (all 64 layers, fp16) for the "
                    "E1 story-label-permutation / topic-contrast nulls.")
    ap.add_argument("--stories-csv", default=str(CLUSTER_DATA / "stories.csv"))
    ap.add_argument("--ref-npz", default=str(CLUSTER_DATA / "activations" / "stories.npz"),
                    help="per-emotion means npz the gate validates against (the extraction "
                    "pipeline writes it to <data>/activations/stories.npz)")
    ap.add_argument("--out", required=True,
                    help="output dir: writes <out>/<emotion>.npy + <out>/topics.json "
                    "(flat — pass this same dir to run_e1_atlas.py --story-acts)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="forward-pass batch (all 64 layers are captured per pass; "
                    "lower this if activation memory is tight)")
    ap.add_argument("--emotions", nargs="+", default=None, help="subset (default: all in the CSV)")
    ap.add_argument("--device", default="cuda",
                    help='device_map: "cuda" pins one visible GPU, "auto" shards')
    ap.add_argument("--skip", type=int, default=None,
                    help="pool token positions >= this (default Config.token_skip=50 — "
                    "MUST match the original extraction)")
    ap.add_argument("--max-length", type=int, default=None,
                    help="truncation length (default Config.max_length=512 — "
                    "MUST match the original extraction)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="per-layer cosine gate vs the reference npz")
    args = ap.parse_args()

    from emotion_probes.config import Config
    from emotion_probes.models.language_model import ProbedModel

    config = Config().with_(model_id=args.model, device_map=args.device)
    skip = config.token_skip if args.skip is None else args.skip
    max_length = config.max_length if args.max_length is None else args.max_length

    ref_emotions, ref_means = load_reference(args.ref_npz)
    table = load_stories(args.stories_csv, emotions=args.emotions)
    _log(f"{args.stories_csv}: {sum(len(v) for v in table.stories.values())} stories, "
         f"{len(table.emotions)} emotions; reference {args.ref_npz}: {len(ref_emotions)} emotions")

    model = ProbedModel(config)
    if (model.num_layers, model.hidden_size) != (N_LAYERS, HIDDEN):
        raise ProvenanceError(
            f"{args.model}: num_layers={model.num_layers}, hidden={model.hidden_size} — "
            f"expected ({N_LAYERS}, {HIDDEN}); the reference npz is Qwen3-32B only")
    _log(f"model {args.model} loaded: {model.num_layers} layers, hidden {model.hidden_size}; "
         f"pooling skip={skip} max_length={max_length} batch={args.batch_size} "
         f"gate cos>{args.threshold}")

    t0 = time.time()
    summary = run_cache(model, table, ref_emotions, ref_means, args.out,
                        batch_size=args.batch_size, skip=skip, max_length=max_length,
                        threshold=args.threshold)
    mc = summary["min_cos"]
    _log(f"DONE: {summary['n_encoded']} encoded + "
         f"{summary['n_emotions'] - summary['n_encoded']} already cached of "
         f"{summary['n_emotions']} emotions ({summary['n_stories']} stories); "
         f"global min cosine {'n/a' if mc is None else f'{mc:.6f}'}; "
         f"{(time.time() - t0) / 60:.1f} min -> {summary['out_dir']}")


if __name__ == "__main__":
    main()
