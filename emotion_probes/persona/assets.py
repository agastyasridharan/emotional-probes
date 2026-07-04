"""
Persona-mechanism asset loaders — the CPU foundation every E1-E5 driver imports.

Three asset families, all local under ``persona_assets/`` (see docs/research/
portfolio.md Day-0 and docs/research/repo-assistant-axis.md):

* the Qwen3-32B research emotion bank (``qwen32b_emotion_vectors.npz``,
  vectors (64, 171, 5120) fp32);
* the assistant-axis bundle (``assistant-axis-qwen32b/``: axis + default vector,
  each (64, 5120), plus 275 per-role vectors; all shipped bf16, cast to fp32 here);
* the frozen role subsets (``docs/prereg/role_subsets.json``, adjudicated from the
  two rater files) used to build the human-vs-AI sub-axis.

Provenance is enforced hard: the sibling repo ships a Qwen2.5-7B bank with shape
(28, 171, 3584) that is dimensionally incompatible with Qwen3-32B — any loader here
raises :class:`ProvenanceError` on hidden != 5120 or n_layers != 64 rather than
letting a wrong-model vector reach a steering hook.

Sign/index conventions (verified by :func:`anchor_check`, results in
docs/prereg/anchor_check.json): axis row ``i`` = post-block-``i`` residual stream;
``axis = default − mean(roles)``, so HIGHER projection = more Assistant-like.

torch is imported lazily (only inside :func:`load_assistant_axis`), so the module
imports and unit-tests on a laptop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Qwen3-32B geometry — the only model this bank/axis is valid for.
N_LAYERS = 64
HIDDEN = 5120
N_EMOTIONS = 171
N_ROLES = 275
SOURCE_MODEL_ID = "Qwen/Qwen3-32B"

_QUARANTINE_NOTE = (
    "expected Qwen3-32B geometry (n_layers=64, hidden=5120). A dimensionally "
    "incompatible Qwen2.5-7B bank with shape (28, 171, 3584) is QUARANTINED at "
    "emotional-probes/data/vectors/emotion_vectors.npz — never load it here."
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMOTION_BANK = _REPO_ROOT / "persona_assets" / "qwen32b_emotion_vectors.npz"
DEFAULT_AXIS_ROOT = _REPO_ROOT / "persona_assets" / "assistant-axis-qwen32b"
DEFAULT_ROLE_SUBSETS = _REPO_ROOT / "docs" / "prereg" / "role_subsets.json"


class ProvenanceError(RuntimeError):
    """Asset failed a model-provenance/shape check (wrong model's vectors)."""


class AnchorError(RuntimeError):
    """Assistant-axis sign/layer-indexing anchor check failed."""


# --------------------------------------------------------------------------- #
# Emotion bank
# --------------------------------------------------------------------------- #
@dataclass
class EmotionBank:
    """The 171-emotion research bank: unit vectors per (layer, emotion)."""

    emotions: list                # 171 names
    vectors: np.ndarray           # (64, 171, 5120) float32
    path: str
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._index:
            self._index = {e: i for i, e in enumerate(self.emotions)}

    def index(self, emotion: str) -> int:
        try:
            return self._index[emotion]
        except KeyError:
            raise KeyError(f"unknown emotion {emotion!r} (bank: {self.path})") from None

    def vector(self, emotion: str, layer: int) -> np.ndarray:
        """fp32 copy of the (5120,) vector for one emotion at one layer."""
        if not 0 <= layer < self.vectors.shape[0]:
            raise IndexError(f"layer {layer} out of range 0..{self.vectors.shape[0] - 1}")
        return np.array(self.vectors[layer, self.index(emotion)], dtype=np.float32, copy=True)


def load_emotion_bank(path: str | Path | None = None) -> EmotionBank:
    """Load the Qwen3-32B emotion bank with hard provenance checks."""
    path = Path(path) if path is not None else DEFAULT_EMOTION_BANK
    z = np.load(path, allow_pickle=True)
    if "vectors" not in z.files or "emotions" not in z.files:
        raise ProvenanceError(f"{path}: expected npz keys 'emotions'/'vectors', got {z.files}")
    vectors = z["vectors"]
    if vectors.shape != (N_LAYERS, N_EMOTIONS, HIDDEN):
        raise ProvenanceError(f"{path}: vectors shape {vectors.shape} — {_QUARANTINE_NOTE}")
    emotions = [str(e) for e in z["emotions"]]
    if len(emotions) != N_EMOTIONS:
        raise ProvenanceError(f"{path}: {len(emotions)} emotion names != {N_EMOTIONS}")
    if "source_model_id" in z.files:
        src = str(z["source_model_id"])
        if src != SOURCE_MODEL_ID:
            raise ProvenanceError(f"{path}: source_model_id {src!r} != {SOURCE_MODEL_ID!r}")
    return EmotionBank(emotions=emotions, vectors=vectors.astype(np.float32, copy=False),
                       path=str(path))


# --------------------------------------------------------------------------- #
# Assistant axis
# --------------------------------------------------------------------------- #
@dataclass
class AssistantAxis:
    """Lu et al. assistant-axis bundle: axis = default − mean(275 roles), per layer.

    Row ``i`` = post-block-``i`` residual stream; higher projection onto
    ``unit_axis(L)`` = more Assistant-like (sign verified by :func:`anchor_check`).
    """

    axis: np.ndarray             # (64, 5120) fp32, raw (NOT unit-normalized)
    default_vector: np.ndarray   # (64, 5120) fp32
    role_vectors: dict           # name -> (64, 5120) fp32, all 275
    path: str

    def unit_axis(self, layer: int) -> np.ndarray:
        """Unit-normalized axis direction at one layer."""
        return unit(self.axis[layer])


def _as_layer_matrix(obj, what: str, path: Path) -> np.ndarray:
    """torch object -> fp32 (64, 5120) array; unwrap {'vector'/'axis': ...} dicts."""
    if isinstance(obj, dict):
        for key in ("vector", "axis"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ProvenanceError(f"{path}: {what} is a dict without 'vector'/'axis' key: "
                                  f"{sorted(obj)}")
    arr = obj.float().numpy().astype(np.float32, copy=False)
    if arr.shape != (N_LAYERS, HIDDEN):
        raise ProvenanceError(f"{path}: {what} shape {arr.shape} — {_QUARANTINE_NOTE}")
    return arr


def load_assistant_axis(root: str | Path | None = None) -> AssistantAxis:
    """Load axis, default vector and all 275 role vectors (bf16 -> fp32)."""
    import torch  # lazy: keep the module importable without torch

    root = Path(root) if root is not None else DEFAULT_AXIS_ROOT
    axis = _as_layer_matrix(torch.load(root / "assistant_axis.pt", map_location="cpu"),
                            "assistant_axis", root)
    default = _as_layer_matrix(torch.load(root / "default_vector.pt", map_location="cpu"),
                               "default_vector", root)
    role_files = sorted((root / "role_vectors").glob("*.pt"))
    if len(role_files) != N_ROLES:
        raise ProvenanceError(f"{root}/role_vectors: {len(role_files)} files != {N_ROLES} "
                              "(reconcile the role atlas — portfolio.md Day-0 item 3)")
    roles = {f.stem: _as_layer_matrix(torch.load(f, map_location="cpu"), f"role {f.stem}", f)
             for f in role_files}
    return AssistantAxis(axis=axis, default_vector=default, role_vectors=roles, path=str(root))


def load_role_subsets(path: str | Path | None = None) -> dict:
    """Load the frozen, adjudicated role subsets (docs/prereg/role_subsets.json)."""
    path = Path(path) if path is not None else DEFAULT_ROLE_SUBSETS
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet — it is the adjudicated role-subset freeze "
            "(portfolio.md Day-0 item 4) produced from docs/prereg/role_ratings_rater*.json. "
            "Run the adjudication step before building the human sub-axis.")
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def unit(v) -> np.ndarray:
    """v / ||v|| as fp32; raises on zero norm (never silently steer a null direction)."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        raise ValueError(f"cannot unit-normalize vector with norm {n}")
    return v / n


def project(acts, direction):
    """Scalar projection(s) of activation vector(s) onto unit(direction).

    ``acts`` (H,) -> float;  (..., H) -> np.ndarray of shape (...,).
    Projections are uncentered — compare against a baseline condition.
    """
    out = np.asarray(acts, dtype=np.float32) @ unit(direction)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# Anchor check — sign & layer-index convention verification
# --------------------------------------------------------------------------- #
def anchor_check(ax: AssistantAxis, layers=(16, 32, 42, 50)) -> dict:
    """Verify axis sign/indexing: default must project MORE Assistant-like than the
    mean role at EVERY checked layer (axis = default − mean(roles) by construction).

    Returns {layer: {"default", "role_mean", "role_std", "margin_sd"}}; raises
    :class:`AnchorError` on any violation (suspect an off-by-one in the layer rows).
    """
    role_stack = np.stack(list(ax.role_vectors.values()))  # (n_roles, 64, 5120)
    out, failures = {}, []
    for layer in layers:
        u = ax.unit_axis(layer)
        d = project(ax.default_vector[layer], u)
        r = project(role_stack[:, layer, :], u)
        mean, std = float(r.mean()), float(r.std())
        out[int(layer)] = {
            "default": d,
            "role_mean": mean,
            "role_std": std,
            "margin_sd": (d - mean) / std if std > 0 else float("inf"),
        }
        if not d > mean:
            failures.append(layer)
    if failures:
        raise AnchorError(
            f"assistant-axis anchor check FAILED at layers {failures}: default projection "
            f"must exceed mean role projection (axis = default − mean(roles)); check for "
            f"layer-indexing off-by-one (embedding row?). Stats: {out}")
    return out


# --------------------------------------------------------------------------- #
# Human sub-axis (emotionality-matched human − ai_artificial)
# --------------------------------------------------------------------------- #
def _roles_and_ratings(subsets: dict, category: str, field_name: str) -> dict:
    """{role: rating} for one category, tolerant of the plausible frozen schemas:

    * rater-file style: ``{"roles": {name: {"category": ..., field_name: ...}}}``;
    * per-category dicts: ``{category: {name: {field_name: ...} | rating}}``;
    * per-category name lists + a top-level ``"roles"``/``"ratings"`` table.
    """
    table = subsets.get("roles") or subsets.get("ratings") or {}
    entry = subsets.get(category)
    if entry is None:
        out = {n: int(m[field_name]) for n, m in table.items()
               if isinstance(m, dict) and m.get("category") == category}
        if not out:
            raise KeyError(f"role_subsets has no {category!r} subset (keys: {sorted(subsets)})")
        return out
    if isinstance(entry, dict):
        return {n: int(m[field_name]) if isinstance(m, dict) else int(m)
                for n, m in entry.items()}
    out = {}
    for n in entry:  # list of names
        m = table.get(n)
        if m is None:
            raise KeyError(f"no {field_name!r} rating for role {n!r} in role_subsets")
        out[n] = int(m[field_name]) if isinstance(m, dict) else int(m)
    return out


def _matched_weighted_h(ax: AssistantAxis, human: dict, ai: dict,
                        matched_on: str) -> np.ndarray:
    """Weighted mean(human) − weighted mean(ai) over the rating levels present in
    BOTH categories, both means taken with the SAME level mixture
    w_level ∝ min(n_human_level, n_ai_level) — exact histogram matching that uses
    every role at the overlapping levels."""
    levels = sorted(set(human.values()) & set(ai.values()))
    if not levels:
        raise ValueError(f"human/ai_artificial {matched_on!r} histograms do not overlap "
                         "at any level — cannot build a matched human axis")
    weights = {lv: min(sum(1 for r in human.values() if r == lv),
                       sum(1 for r in ai.values() if r == lv))
               for lv in levels}
    total = float(sum(weights.values()))
    h_mean = a_mean = 0.0
    for lv in levels:
        w = weights[lv] / total
        h_mean = h_mean + w * np.mean(
            [ax.role_vectors[n] for n, r in sorted(human.items()) if r == lv], axis=0)
        a_mean = a_mean + w * np.mean(
            [ax.role_vectors[n] for n, r in sorted(ai.items()) if r == lv], axis=0)
    return (h_mean - a_mean).astype(np.float32)


def human_axis(ax: AssistantAxis, subsets: dict, matched_on: str = "emotionality",
               seed: int = 0) -> np.ndarray:
    """h = weighted mean(human roles) − weighted mean(ai_artificial roles),
    (64, 5120) fp32, where both category means use the SAME ``matched_on`` level
    mixture (w_level ∝ min of the two level counts over the overlapping levels), so
    h is emotionality-matched by construction (portfolio.md §E1: a raw contrast
    would bake affect into h) while using EVERY role at the overlapping levels.

    This deterministic construction replaces an earlier min-count seeded
    subsampling: with 232 human vs only 7 ai_artificial roles that construction
    kept just 7 randomly chosen humans, and the resulting h was sampling-noise
    dominated (measured on the real assets: cos(h, a[42]) ranged 0.02–0.70 across
    subsample seeds). The weighted form is that construction's exact over-seeds
    expectation. ``seed`` is retained for API compatibility and is ignored.
    """
    del seed  # deterministic construction; kept for API compatibility
    human = _roles_and_ratings(subsets, "human", matched_on)
    ai = _roles_and_ratings(subsets, "ai_artificial", matched_on)
    for name in (*human, *ai):
        if name not in ax.role_vectors:
            raise KeyError(f"role {name!r} from role_subsets has no vector in {ax.path}")
    return _matched_weighted_h(ax, human, ai, matched_on)


def human_axis_jackknife(ax: AssistantAxis, subsets: dict,
                         matched_on: str = "emotionality") -> dict:
    """Leave-one-out over the ai_artificial roles: {dropped_role: h_loo (64, 5120)}.

    The human sub-axis rests on a tiny AI-role set (7 in the frozen atlas), so its
    per-statistic jackknife SE is a required honesty report (n − 1 of n AI roles,
    identical matched-weighted pipeline). Levels that lose their only AI member are
    dropped for that replicate, exactly as the main construction would."""
    human = _roles_and_ratings(subsets, "human", matched_on)
    ai = _roles_and_ratings(subsets, "ai_artificial", matched_on)
    for name in (*human, *ai):
        if name not in ax.role_vectors:
            raise KeyError(f"role {name!r} from role_subsets has no vector in {ax.path}")
    out = {}
    for drop in sorted(ai):
        ai_loo = {n: r for n, r in ai.items() if n != drop}
        out[drop] = _matched_weighted_h(ax, human, ai_loo, matched_on)
    return out
