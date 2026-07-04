"""
E1 geometry atlas — CPU constructions over the frozen persona assets.

Pure numpy: no torch, no GPU; every function runs on a laptop against the local
``persona_assets/`` bundle (portfolio.md §E1, docs/prereg/E1.md). E1 is a geometry
GATE — it does NOT adjudicate mechanism A vs B and must never be reported as doing
so. The GPU pieces of E1 (split-injection bridge, story-residual cache) live
elsewhere; :func:`story_nulls` merely CONSUMES the cache written by
``scripts/cache_story_residuals.py`` and returns ``None`` while it does not exist
(the atlas then notes "gated on story cache").

Constructions:

* :func:`build_directions` — per-layer unit directions: assistant axis ``a``;
  emotionality-matched human sub-axis ``h``; ``h_purged`` (the 10-cluster emotion
  span projected out of EVERY role vector, then h rebuilt — the only B-relevant
  variant, raw ``h`` is descriptive); the pre-registered ``valence`` axis
  (mean of 5 positive minus mean of 5 negative suite cluster directions); and the
  10 suite emotion directions ``v_<emotion>``.
* :func:`pca_atlas` — 8-PC persona basis of the 275 role vectors (either
  ``default_vector`` or mean-of-roles centering) with |cos(PC_i, a)|.
* :func:`cosine_table` — full pairwise |cos| Gram matrix as a nested dict.
* :func:`isotropic_floor` — empirical |cos| floor between random unit directions
  (floor ONLY, never a decision criterion; analytic E|cos| = sqrt(2/(pi*dim))).
* :func:`story_nulls` — the two decision-grade nulls (story-label permutation and
  topic-contrast directions) recomputed through the same mean-difference pipeline;
  ALL permutation inference goes through ``scripts/exact_perm.py``
  ``sign_permutation_pvalue`` (exact when tractable, add-one Monte-Carlo else).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emotion_probes.data.qualia import spec_for
from emotion_probes.data.qualia_freeform import valenced_panel_qualia
from emotion_probes.persona.assets import (
    AssistantAxis,
    EmotionBank,
    ProvenanceError,
    human_axis,
    unit,
)

DIRECTION_TARGETS = ("a", "h", "h_purged")  # story-null / permutation targets


def _exact_perm():
    """Lazy import of scripts/exact_perm.py (scripts/ is not a package)."""
    import sys
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import exact_perm
    return exact_perm


# --------------------------------------------------------------------------- #
# Suite valence panel
# --------------------------------------------------------------------------- #
def suite_valence_panel() -> tuple[list[str], list[str]]:
    """The frozen 10-cluster suite panel as (5 positive, 5 negative) name lists."""
    names = valenced_panel_qualia()
    pos = [n for n in names if spec_for(n).valence > 0]
    neg = [n for n in names if spec_for(n).valence < 0]
    if len(pos) != 5 or len(neg) != 5:
        raise AssertionError(f"expected 5/5 valenced panel, got {len(pos)}/{len(neg)}")
    return pos, neg


# --------------------------------------------------------------------------- #
# Span helpers
# --------------------------------------------------------------------------- #
def span_basis(vectors: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Orthonormal rows spanning the given (k, H) vectors (SVD, rank-truncated)."""
    v = np.asarray(vectors, dtype=np.float32)
    _, s, vt = np.linalg.svd(v, full_matrices=False)
    keep = s > rtol * float(s[0])
    return vt[keep].astype(np.float32)


def purge(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of ``x`` (..., H) inside the span of ``basis`` rows."""
    x = np.asarray(x, dtype=np.float32)
    return x - (x @ basis.T) @ basis


# --------------------------------------------------------------------------- #
# Directions
# --------------------------------------------------------------------------- #
def build_directions(bank: EmotionBank, ax: AssistantAxis, subsets: dict,
                     layer: int, seed: int = 0) -> dict:
    """All E1 unit directions at one layer: ``a``, ``h``, ``h_purged``,
    ``valence``, and ``v_<emotion>`` for the 10 suite valenced emotions.

    ``h``/``h_purged`` share the same deterministic emotionality-matched weighted
    construction (assets.human_axis; ``seed`` is accepted for API compatibility
    but has no effect), so their difference isolates the purge exactly.
    """
    pos, neg = suite_valence_panel()
    names = [*pos, *neg]
    v_rows = np.stack([unit(bank.vector(e, layer)) for e in names])   # (10, H)

    directions = {"a": ax.unit_axis(layer)}
    directions["h"] = unit(human_axis(ax, subsets, seed=seed)[layer])

    # h_purged: project the emotion span out of every role vector (at this layer),
    # rebuild h through the identical matched-subset pipeline.
    basis = span_basis(v_rows)
    purged_roles = {}
    for name, mat in ax.role_vectors.items():
        m = np.array(mat, dtype=np.float32, copy=True)
        m[layer] = purge(m[layer], basis)
        purged_roles[name] = m
    ax_purged = AssistantAxis(axis=ax.axis, default_vector=ax.default_vector,
                              role_vectors=purged_roles, path=f"{ax.path}#purged_L{layer}")
    directions["h_purged"] = unit(human_axis(ax_purged, subsets, seed=seed)[layer])

    directions["valence"] = unit(v_rows[: len(pos)].mean(axis=0)
                                 - v_rows[len(pos):].mean(axis=0))
    for name, row in zip(names, v_rows):
        directions[f"v_{name}"] = row
    return directions


# --------------------------------------------------------------------------- #
# PCA persona basis
# --------------------------------------------------------------------------- #
def pca_atlas(ax: AssistantAxis, layer: int, center: str = "default",
              n_pcs: int = 8) -> dict:
    """PCs of the role vectors at ``layer`` after centering on the default vector
    (``center='default'``) or the mean of roles (``center='mean'``); explained
    variance ratios and |cos(PC_i, a)| for i = 1..n_pcs. ``components`` is the
    (n_pcs, H) orthonormal PC matrix (kept as an array; JSON writers should strip
    or sidecar it)."""
    names = sorted(ax.role_vectors)
    rows = np.stack([ax.role_vectors[n][layer] for n in names]).astype(np.float32)
    if center == "default":
        c = ax.default_vector[layer].astype(np.float32)
    elif center == "mean":
        c = rows.mean(axis=0)
    else:
        raise ValueError(f"center must be 'default' or 'mean', got {center!r}")
    x = rows - c
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    evr = var / float(var.sum())
    k = int(min(n_pcs, vt.shape[0]))
    components = vt[:k].astype(np.float32)
    a = ax.unit_axis(layer)
    return {
        "layer": int(layer),
        "center": center,
        "n_roles": len(names),
        "n_pcs": k,
        "components": components,
        "explained_variance_ratio": [float(e) for e in evr[:k]],
        "cum_explained_variance": float(evr[:k].sum()),
        "cos_axis": {f"pc{i + 1}": abs(float(components[i] @ a)) for i in range(k)},
    }


# --------------------------------------------------------------------------- #
# Cosine table + isotropic floor
# --------------------------------------------------------------------------- #
def cosine_table(directions: dict) -> dict:
    """Full pairwise |cos| Gram matrix over the named directions (nested dict)."""
    units = {n: unit(v) for n, v in directions.items()}
    names = list(units)
    return {ni: {nj: abs(float(units[ni] @ units[nj])) for nj in names}
            for ni in names}


def isotropic_floor(dim: int = 5120, n: int = 2000, seed: int = 0) -> dict:
    """Empirical |cos| between ``n`` random unit-vector pairs in ``dim`` dims:
    mean and 95th/99th percentiles, with the analytic mean sqrt(2/(pi*dim)) as a
    cross-check. Reported as a FLOOR only — decision-grade nulls are the
    pipeline-matched ones in :func:`story_nulls`."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, dim))
    b = rng.standard_normal((n, dim))
    cos = np.abs(np.einsum("ij,ij->i", a, b)
                 / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))
    return {
        "dim": int(dim),
        "n": int(n),
        "mean_abs_cos": float(cos.mean()),
        "p95": float(np.percentile(cos, 95)),
        "p99": float(np.percentile(cos, 99)),
        "analytic_mean": float(np.sqrt(2.0 / (np.pi * dim))),
    }


# --------------------------------------------------------------------------- #
# Story-cache nulls (gated on scripts/cache_story_residuals.py output)
# --------------------------------------------------------------------------- #
def _valence_stat_fn(x: np.ndarray, pos_ids: list, neg_ids: list,
                     target_unit: np.ndarray):
    """stat(labels) = signed cos(valence direction rebuilt from the labelling, t).

    The pipeline per arrangement: per-group story means -> mean(positive groups)
    minus mean(negative groups) -> unit-normalize -> dot the unit target. O(n*H)
    per call. The identity arrangement is recomputed by the same function
    (exact_perm's rule), so no rounded import can undercut the floor."""
    def stat(labels: np.ndarray) -> float:
        means = [x[labels == k].mean(axis=0) for k in (*pos_ids, *neg_ids)]
        n_pos = len(pos_ids)
        v = (np.mean(means[:n_pos], axis=0) - np.mean(means[n_pos:], axis=0))
        nrm = float(np.linalg.norm(v))
        return float(v @ target_unit) / nrm if nrm > 0 else 0.0
    return stat


def _topic_contrast(x: np.ndarray, topics: np.ndarray, target_units: dict,
                    observed_abs: dict, min_stories: int) -> dict:
    """>=50 topic-contrast directions through the identical mean-difference
    pipeline; per target: the control 95th percentile of |cos| and whether the
    observed emotion statistic beats it."""
    uniq = sorted(set(topics.tolist()))
    dirs, pairs = [], []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            m1, m2 = topics == uniq[i], topics == uniq[j]
            if int(m1.sum()) < min_stories or int(m2.sum()) < min_stories:
                continue
            v = x[m1].mean(axis=0) - x[m2].mean(axis=0)
            nrm = float(np.linalg.norm(v))
            if nrm == 0.0:
                continue
            dirs.append(v / nrm)
            pairs.append([uniq[i], uniq[j]])
    if not dirs:
        return {"status": "no usable topic pairs", "n_directions": 0}
    mat = np.stack(dirs)                                       # (n_dirs, H)
    out = {"n_directions": len(dirs), "n_topics": len(uniq),
           "min_stories_per_topic": int(min_stories),
           "enough_directions": len(dirs) >= 50, "targets": {}}
    for name, t in target_units.items():
        abs_cos = np.abs(mat @ t)
        p95 = float(np.percentile(abs_cos, 95))
        out["targets"][name] = {
            "control_p95_abs_cos": p95,
            "control_max_abs_cos": float(abs_cos.max()),
            "observed_abs_cos": float(observed_abs[name]),
            "beats_p95": bool(observed_abs[name] > p95),
        }
    return out


def story_nulls(story_acts_dir, directions_by_layer: dict, n_perm: int = 10000,
                seed: int = 0, targets=DIRECTION_TARGETS,
                min_topic_stories: int = 8):
    """The two decision-grade E1 nulls, computed from the per-story residual cache.

    Cache contract (written by ``scripts/cache_story_residuals.py``): one
    ``<emotion>.npy`` per emotion, shape (n_stories, n_layers_cached, hidden);
    optional ``layers.json`` = list of model-layer indices for the cached layer
    rows (absent -> row i IS model layer i); optional ``topics.json`` =
    {emotion: [topic per story]} enabling the topic-contrast null.

    ``directions_by_layer`` = {model_layer: build_directions(...) output}. Only
    the requested layers are read (mmap slice) to bound memory. Returns ``None``
    when the cache directory does not exist yet — the atlas notes the gate.

    Pipeline note: the bank recipe additionally projects out neutral-text PCs;
    the cache holds no neutral acts, so the nulls use the mean-difference
    pipeline (recorded in the output for the prereg audit trail).
    """
    if story_acts_dir is None:
        return None
    root = Path(story_acts_dir)
    if not root.is_dir():
        return None

    ep = _exact_perm()
    pos, neg = suite_valence_panel()
    names = [*pos, *neg]
    missing = [e for e in names if not (root / f"{e}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"story cache {root} exists but lacks per-emotion files for {missing} "
            "— incomplete cache_story_residuals.py output, refusing to compute nulls")

    layer_rows = None
    if (root / "layers.json").exists():
        layer_rows = [int(v) for v in json.loads((root / "layers.json").read_text())]
    topics_map = None
    if (root / "topics.json").exists():
        topics_map = json.loads((root / "topics.json").read_text())

    pos_ids = list(range(len(pos)))
    neg_ids = list(range(len(pos), len(names)))
    out = {
        "cache_dir": str(root),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "panel": names,
        "pipeline_note": ("mean-difference pipeline (per-emotion story means -> "
                          "pos-minus-neg -> unit); the bank's neutral-text PC "
                          "removal step needs neutral acts absent from this cache"),
        "layers": {},
    }
    for layer in sorted(directions_by_layer):
        dirs = directions_by_layer[layer]
        if layer_rows is not None:
            if layer not in layer_rows:
                out["layers"][str(layer)] = {
                    "status": f"layer {layer} not in cache (layers.json: {layer_rows})"}
                continue
            row = layer_rows.index(layer)
        else:
            row = int(layer)

        slices, counts, topic_chunks = [], [], []
        for e in names:
            arr = np.load(root / f"{e}.npy", mmap_mode="r")
            if arr.ndim != 3:
                raise ProvenanceError(f"{root / (e + '.npy')}: expected 3-d "
                                      f"(n_stories, n_layers, hidden), got {arr.shape}")
            if layer_rows is None and row >= arr.shape[1]:
                raise ProvenanceError(
                    f"{root / (e + '.npy')}: layer {layer} out of range for cache with "
                    f"{arr.shape[1]} layer rows and no layers.json mapping")
            hidden = int(dirs["a"].shape[0])
            if arr.shape[2] != hidden:
                raise ProvenanceError(f"{root / (e + '.npy')}: hidden {arr.shape[2]} "
                                      f"!= direction dim {hidden}")
            slices.append(np.asarray(arr[:, row, :], dtype=np.float32))
            counts.append(int(arr.shape[0]))
            if topics_map is not None:
                tl = topics_map.get(e)
                if tl is None or len(tl) != arr.shape[0]:
                    raise ProvenanceError(f"topics.json entry for {e!r} missing or "
                                          f"length != n_stories {arr.shape[0]}")
                topic_chunks.append(np.array(tl, dtype=object))
        x = np.concatenate(slices, axis=0)
        labels = np.concatenate([np.full(c, k, dtype=float)
                                 for k, c in enumerate(counts)])

        target_units = {t: unit(dirs[t]) for t in targets if t in dirs}
        label_perm, observed_abs = {}, {}
        for tname, t in target_units.items():
            res = ep.sign_permutation_pvalue(
                labels, _valence_stat_fn(x, pos_ids, neg_ids, t),
                n_perm=n_perm, seed=seed)
            label_perm[tname] = {
                "observed_cos": float(res["observed"]),
                "abs_observed": abs(float(res["observed"])),
                "p_perm": float(res["p_perm"]),
                "exact": bool(res["exact"]),
                "n_arrangements": int(res["n_arrangements"]),
                "floor": float(res["floor"]),
            }
            observed_abs[tname] = abs(float(res["observed"]))

        layer_out = {
            "row_in_cache": int(row),
            "n_stories": int(x.shape[0]),
            "stories_per_emotion": {e: c for e, c in zip(names, counts)},
            "label_permutation": label_perm,
        }
        if topics_map is not None:
            topics = np.concatenate(topic_chunks)
            layer_out["topic_contrast"] = _topic_contrast(
                x, topics, target_units, observed_abs, min_topic_stories)
        else:
            layer_out["topic_contrast"] = {
                "status": "gated: topics.json not present in story cache"}
        out["layers"][str(layer)] = layer_out
    return out
