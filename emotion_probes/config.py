"""
Central configuration — the single source of truth for the whole pipeline.

Every other module reads its settings from a :class:`Config` object. That means
you change a path, a model, or a hyper-parameter in exactly ONE place, and the
whole pipeline follows.

Why this exists
---------------
The original code hard-coded, in ~8 separate files:

* the model id (``"google/gemma-4-E4B"``),
* the analysis layer (``23`` here, ``19`` there),
* the hidden size (``2560``),
* and absolute data paths (``/mnt/ssd/...``, ``~/data/...``).

That locked the project to one specific model and made it easy to get subtly
inconsistent. ``Config`` fixes that:

* **Issue #1 (model is hard-coded):** ``model_id`` is a normal field you can
  override — the pipeline runs on any HuggingFace causal LM.
* **Issue #4 (analysis layer is wrong/arbitrary):** we store ``layer_fraction``
  (a fraction of model depth, default ``2/3`` to match the paper's
  "planned/operative" emotion layer) rather than a fixed integer. The actual
  index is computed per-model by :meth:`Config.analysis_layer`, so it is correct
  whether the model has 28 layers or 80.

Reading the config from the environment
---------------------------------------
The big artifacts (tens of GB of activations) must NOT live in ``$HOME`` on the
cluster (10 GiB cap). Set ``EMOTION_PROBES_DATA=/data/<user>/emotion-probes`` and
all paths relocate there automatically. Locally it defaults to ``<repo>/data``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Repo root = the directory that contains the ``emotion_probes`` package.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """Where artifacts live. Override with ``$EMOTION_PROBES_DATA`` on the cluster."""
    env = os.environ.get("EMOTION_PROBES_DATA")
    return Path(env) if env else (REPO_ROOT / "data")


def _default_model_id() -> str:
    """The probed model. Override with ``$EMOTION_PROBES_MODEL_ID`` to switch rungs
    (see docs/MODELS.md) without editing code. Pair with a *separate*
    ``$EMOTION_PROBES_DATA`` per model so runs never overwrite each other."""
    return os.environ.get("EMOTION_PROBES_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def _default_device_map() -> str:
    """``"cuda"`` pins one visible GPU; ``"auto"`` shards across all visible GPUs.
    Override with ``$EMOTION_PROBES_DEVICE_MAP`` (e.g. ``auto`` for a 70B sharded
    over several cards). Which cards are visible is controlled externally by
    ``CUDA_VISIBLE_DEVICES`` — never hard-code it."""
    return os.environ.get("EMOTION_PROBES_DEVICE_MAP", "cuda")


def _default_batch_size() -> int:
    """Forward-pass batch for story/neutral means. Override with
    ``$EMOTION_PROBES_BATCH_SIZE`` — the biggest easy throughput win on a large
    pipeline-sharded model is a larger batch (amortizes the per-pass bubble),
    bounded by activation memory."""
    return int(os.environ.get("EMOTION_PROBES_BATCH_SIZE", "32"))


def _default_deflection_batch_size() -> int:
    """Forward-pass batch for the (longer, 2048-token) deflection dialogues.
    Override with ``$EMOTION_PROBES_DEFLECTION_BATCH_SIZE``."""
    return int(os.environ.get("EMOTION_PROBES_DEFLECTION_BATCH_SIZE", "8"))


def _default_enable_thinking() -> bool | None:
    """Whether to keep a reasoning model's "thinking" mode on when applying the
    chat template (Qwen3 dense defaults it ON). ``None`` (unset) leaves the
    model's own default; set ``$EMOTION_PROBES_ENABLE_THINKING=0`` to force a
    non-reasoning state for faithful extraction/behaviour evals. Harmless for
    models whose chat template ignores the flag."""
    v = os.environ.get("EMOTION_PROBES_ENABLE_THINKING")
    return None if v is None else (v == "1")


def _default_attn_impl() -> str | None:
    """The ``attn_implementation`` passed to ``from_pretrained`` (e.g. ``"eager"``
    for bit-exact Gemma activations). ``None`` lets transformers pick its default
    (sdpa). Override with ``$EMOTION_PROBES_ATTN_IMPL``."""
    return os.environ.get("EMOTION_PROBES_ATTN_IMPL") or None


@dataclass(frozen=True)
class Config:
    """Immutable bundle of every setting. Use :func:`default_config` for the
    standard instance, or ``replace(cfg, model_id="...")`` to make a variant.

    The defaults reproduce the paper's described methodology as closely as a
    port to an open model allows. See the field comments for the paper mapping.
    """

    # ---- The model we probe -------------------------------------------------
    # NOTE: this is an OPEN model, not Claude Sonnet 4.5 (whose weights are not
    # public). This project is therefore a faithful *methodology port*, not a
    # numerical reproduction of the paper's Sonnet results. Swap freely — the
    # default is the cheapest, dev-friendly rung of the recommended ladder
    # (and a scientific anchor: it's a Persona Vectors model). The full ladder,
    # the H100 capacity math, and per-model extraction settings are in
    # docs/MODELS.md.
    model_id: str = field(default_factory=_default_model_id)
    dtype: str = "bfloat16"
    # "cuda" pins to a single visible GPU (let `gpurun`/CUDA_VISIBLE_DEVICES pick it).
    # "auto" shards a large model across all visible GPUs (e.g. the 4x H100 NVL).
    # Both are env-overridable ($EMOTION_PROBES_MODEL_ID / $EMOTION_PROBES_DEVICE_MAP).
    device_map: str = field(default_factory=_default_device_map)

    # Per-model load/generation quirks (see emotion_probes/models/suite.py). Both
    # default to the model's own behaviour; set them per-model for the suite.
    #   enable_thinking: force a reasoning model's <think> mode on/off in the chat
    #     template (Qwen3 dense -> False for a faithful non-reasoning state). None
    #     = leave the model default; passed through to apply_chat_template only.
    #   attn_implementation: e.g. "eager" for bit-exact Gemma activations. None =
    #     transformers' default (sdpa).
    enable_thinking: bool | None = field(default_factory=_default_enable_thinking)
    attn_implementation: str | None = field(default_factory=_default_attn_impl)

    # ---- Which layer's residual stream we analyse (Issue #4) ---------------
    # The paper centres on ~2/3 depth, where "planned/operative" emotion lives.
    # Storing a fraction (not an index) keeps this correct for any model depth.
    layer_fraction: float = 2.0 / 3.0

    # ---- Paper constants ----------------------------------------------------
    token_skip: int = 50
    """Average the residual stream over token positions >= this index
    (the paper starts at the 50th token, by which point emotion is apparent)."""

    neutral_variance_threshold: float = 0.50
    """PCA confound removal: project out the top principal components of NEUTRAL
    text that explain at least this fraction of variance (paper: 50%)."""

    expression_orthogonalization_threshold: float = 0.99
    """Deflection vectors are additionally orthogonalised against the
    story-emotion space, removing PCs up to this cumulative variance (paper: 99%)."""

    # ---- Dataset sizes ------------------------------------------------------
    stories_per_topic: int = 12          # paper: 12 stories per (emotion, topic)
    displayed_per_target: int = 14       # deflection: masking emotions per target
    max_emotions: int | None = None
    """Optional cap on the number of emotions generated (None = all 171). Used to
    run a smaller, cheaper corpus while keeping the full methodology."""
    max_topics: int | None = None
    """Optional cap on the number of topics per emotion (None = all 100). Lets a
    reduced run keep all emotions but fewer (emotion, topic) story samples."""
    dissimilar_fraction: float = 0.25
    """Deflection pairings are drawn from the most-DISSIMILAR `dissimilar_fraction`
    of emotions (so a hidden emotion is masked by a genuinely different one).
    NOTE (Issue #5): the original ``select_pairs.py`` set this to 0.25 but its
    comments said "top 50% / top half". The value is 0.25 (top quartile); the
    name and docs here are made to agree."""

    # ---- Generation (Issues #2, #6) ----------------------------------------
    api_generator_model: str = "anthropic:claude-sonnet-4-5"
    """External API model used when ``self_generate`` is False (pydantic-ai model
    string; reads ANTHROPIC_API_KEY from the environment)."""
    api_generator_model_fast: str = "anthropic:claude-sonnet-4-5"
    """Model for the very large deflection dataset (kept on Sonnet by request; can
    drop to a cheaper model to cut cost on that stage)."""
    self_generate: bool = False
    """Issue #2: if True, the *probed* open model writes its own stories, so the
    emotion vectors reflect what THAT model associates with each emotion (the
    paper's design — it used the probed model, Sonnet 4.5, as the author)."""
    one_story_per_call: bool = True
    """Issue #6: generate one story per API call instead of 12-in-one-call, to
    avoid intra-call homogeneity / boost diversity."""

    # ---- Extraction (forward passes) ---------------------------------------
    # Both env-overridable ($EMOTION_PROBES_BATCH_SIZE / _DEFLECTION_BATCH_SIZE)
    # so batch can be tuned for a big model without code edits.
    batch_size: int = field(default_factory=_default_batch_size)
    deflection_batch_size: int = field(default_factory=_default_deflection_batch_size)
    max_length: int = 512
    deflection_max_length: int = 2048
    checkpoint_every: int = 500          # batches between checkpoints (12h-notice cluster policy)

    # ---- Steering -----------------------------------------------------------
    preference_steering_strength: float = 0.50
    """Steering strength for the preference experiment, as a FRACTION of the
    average residual-stream norm at that layer (paper footnote 3)."""

    # ---- Visualiser (the Flask token-heatmap server) -----------------------
    visualiser_port: int = 8080
    visualiser_max_length: int = 4096
    """Max tokens the visualiser will render for one input (the original server
    truncated at 4096; kept here so long transcripts still display)."""

    # ---- Paths (relocate with $EMOTION_PROBES_DATA) ------------------------
    data_dir: Path = field(default_factory=_default_data_dir)

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    def analysis_layer(self, num_layers: int) -> int:
        """Turn ``layer_fraction`` into a concrete 0-based layer index for a
        model with ``num_layers`` decoder layers.

        Example: 42 layers, fraction 2/3 -> index 27 (≈ the paper's "layer 28
        of 42", which is index 27 when 0-based)."""
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        idx = round(self.layer_fraction * (num_layers - 1))
        return max(0, min(num_layers - 1, idx))

    def with_(self, **overrides) -> "Config":
        """Return a copy with some fields changed, e.g. ``cfg.with_(model_id=...)``."""
        return replace(self, **overrides)

    # --- generated-dataset directories (raw text) ---
    @property
    def stories_dir(self) -> Path:
        return self.data_dir / "stories"

    @property
    def neutral_stories_dir(self) -> Path:
        return self.data_dir / "neutral_stories"

    @property
    def neutral_dialogues_dir(self) -> Path:
        return self.data_dir / "neutral_dialogues"

    @property
    def deflection_dialogues_dir(self) -> Path:
        return self.data_dir / "deflection_dialogues"

    # --- consolidated CSV/JSON datasets ---
    @property
    def stories_csv(self) -> Path:
        return self.data_dir / "stories.csv"

    @property
    def neutral_stories_csv(self) -> Path:
        return self.data_dir / "neutral_stories.csv"

    @property
    def neutral_dialogues_csv(self) -> Path:
        return self.data_dir / "neutral_dialogues.csv"

    @property
    def deflection_dialogues_csv(self) -> Path:
        return self.data_dir / "deflection_dialogues.csv"

    @property
    def deflection_pairs_json(self) -> Path:
        return self.data_dir / "deflection_pairs.json"

    # --- extracted activations ---
    @property
    def activations_dir(self) -> Path:
        return self.data_dir / "activations" / "stories"

    @property
    def neutral_story_activations(self) -> Path:
        return self.data_dir / "activations" / "neutral_stories.npz"

    @property
    def neutral_dialogue_activations(self) -> Path:
        return self.data_dir / "activations" / "neutral_dialogues.npz"

    @property
    def deflection_activations(self) -> Path:
        return self.data_dir / "activations" / "deflection.npz"

    @property
    def deflection_checkpoint(self) -> Path:
        return self.data_dir / "activations" / "deflection_checkpoint.npz"

    # --- computed vectors ---
    @property
    def emotion_vectors_path(self) -> Path:
        return self.data_dir / "vectors" / "emotion_vectors.npz"

    @property
    def deflection_vectors_path(self) -> Path:
        return self.data_dir / "vectors" / "deflection_vectors.npz"

    # --- analysis outputs ---
    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"


def default_config() -> Config:
    """The standard configuration (reads ``$EMOTION_PROBES_DATA`` for paths)."""
    return Config()
