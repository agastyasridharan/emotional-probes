"""
The model **suite** — the fixed roster of open-weights models we replicate on.

This is the single source of truth for "which models, and how each one must be
loaded". Everything else (the cluster runner ``scripts/run_suite.sh``, the
``scripts/preflight_suite.py`` checker, the ``docs/MODELS.md`` table) reads from
the :data:`SUITE` list below, so a model id, a batch size, or a per-model quirk
is recorded in exactly ONE place.

Why a manifest and not just ``EMOTION_PROBES_MODEL_ID=...`` per run
------------------------------------------------------------------
The pipeline is already model-agnostic (:class:`emotion_probes.config.Config` is
env-switchable and :class:`~emotion_probes.models.language_model.ProbedModel`
auto-detects layers/width). But a *14-model* run needs more than an id: each
model has its own device-map (single card vs sharded), batch size, gating
status, and a couple of load-time quirks (Qwen3's thinking mode, Gemma's
attention impl, the 235B's FP8 packaging). Pinning all of that per-model, here,
is what makes "run the whole suite" a loop instead of fourteen bespoke scripts.

Every field below was verified against the live HF model cards / ``config.json``
in 2026-06 (params, layer count, hidden size, architecture class, gating). The
code still reads the *live* config at load time, so a stale number here only
misinforms planning — it never silently corrupts a run (and the
``ProbeBank.check_against`` guard refuses a mismatched bank).

Run it directly:

    python -m emotion_probes.models.suite list          # the planning table
    python -m emotion_probes.models.suite keys           # one key per line
    python -m emotion_probes.models.suite env qwen3-32b  # export lines for the runner
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from emotion_probes.config import Config

# Per-card VRAM on the target box (Athena = 4x H100 NVL, ~94 GB each).
CARD_GB = 94
NUM_CARDS = 4


@dataclass(frozen=True)
class SuiteModel:
    """Everything needed to load and schedule one model in the suite."""

    key: str                       # stable short slug (also the per-model data dir name)
    model_id: str                  # canonical HF repo id (the variant we actually load)
    family: str                    # qwen3 | llama | gemma3 | olmo2 | mistral
    params_b: float                # total params in billions (MoE: total, not active)
    num_layers: int                # decoder layers (== analysis-layer denominator)
    hidden_size: int               # residual-stream width
    arch: str                      # HF architecture class (informational)

    # ---- loading ----
    device_map: str = "cuda"       # "cuda" = one visible card; "auto" = shard across all
    dtype: str = "bfloat16"        # compute dtype; FP8 checkpoints stay FP8 via their own config
    batch_size: int = 32           # story/neutral forward batch (@512 tok)
    deflection_batch_size: int = 8  # deflection forward batch (@2048 tok)

    # ---- per-model quirks (None = use the model's own default) ----
    enable_thinking: bool | None = None   # Qwen3 dense: False, to extract a non-reasoning state
    attn_implementation: str | None = None  # Gemma3: "eager" for bit-exact activations
    fp8: bool = False              # checkpoint ships pre-quantized FP8 (235B) — load as-is
    multimodal: bool = False       # decoder nested under a VLM wrapper (layers auto-resolved)
    gated: bool = False            # needs accepted HF terms + a token to download

    notes: str = ""

    # -- derived --
    def analysis_layer(self) -> int:
        """The 0-based layer index at ``layer_fraction`` (2/3) depth for this model."""
        return Config().analysis_layer(self.num_layers)

    def weight_gb(self) -> float:
        """Approximate resident weight size (bf16 = 2 B/param, FP8 = 1 B/param)."""
        return self.params_b * (1.0 if self.fp8 else 2.0)

    def fits_one_card(self) -> bool:
        """True if weights + a working margin fit a single card (so we can run
        one-model-per-GPU data-parallel rather than sharding)."""
        return self.weight_gb() < 0.8 * CARD_GB


# --------------------------------------------------------------------------- #
# The roster. Ordered cheapest-first within each family.
# --------------------------------------------------------------------------- #
SUITE: list[SuiteModel] = [
    # ---- Qwen3 (Apache-2.0, ungated). Dense 1.7-32B + the 235B MoE. ----
    # The four dense Qwen3 are the April-2025 *hybrid* checkpoints: thinking is ON
    # by default, so we force it OFF for a faithful non-reasoning residual state.
    SuiteModel("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen3", 1.7, 28, 2048,
               "Qwen3ForCausalLM", batch_size=64, deflection_batch_size=16,
               enable_thinking=False, notes="tie_word_embeddings (logit-lens only)"),
    SuiteModel("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", 8.2, 36, 4096,
               "Qwen3ForCausalLM", batch_size=48, deflection_batch_size=12,
               enable_thinking=False),
    SuiteModel("qwen3-14b", "Qwen/Qwen3-14B", "qwen3", 14.8, 40, 5120,
               "Qwen3ForCausalLM", batch_size=48, deflection_batch_size=12,
               enable_thinking=False),
    SuiteModel("qwen3-32b", "Qwen/Qwen3-32B", "qwen3", 32.8, 64, 5120,
               "Qwen3ForCausalLM", batch_size=32, deflection_batch_size=8,
               enable_thinking=False,
               notes="headline dense workhorse; Assistant-Axis vectors exist @ L32"),
    # 235B-A22B-Instruct-2507 is non-thinking-only (enable_thinking is a no-op).
    # bf16 (~470 GB) does NOT fit; the official FP8 (~235 GB) does, across 4 cards.
    # FP8 leaves lm_head / layernorms / MoE gate in bf16, so the residual stream we
    # read is unquantized. quant_method="fp8" is transformers' native fine-grained
    # FP8 -> loads via torch float8 + triton (both already in the cluster .venv);
    # no compressed-tensors/fbgemm needed (those are the other two FP8 schemes).
    SuiteModel("qwen3-235b", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "qwen3", 235.0, 94, 4096,
               "Qwen3MoeForCausalLM", device_map="auto", fp8=True,
               batch_size=16, deflection_batch_size=4,
               notes="MoE 235B/22B-active; bf16 id Qwen/Qwen3-235B-A22B-Instruct-2507 does NOT fit"),

    # ---- Llama (gated: Llama Community License). 3.2 for 1B/3B, 3.1 for 8B. ----
    SuiteModel("llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "llama", 1.2, 16, 2048,
               "LlamaForCausalLM", batch_size=64, deflection_batch_size=16, gated=True,
               notes="tie_word_embeddings"),
    SuiteModel("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "llama", 3.2, 28, 3072,
               "LlamaForCausalLM", batch_size=64, deflection_batch_size=16, gated=True,
               notes="tie_word_embeddings"),
    SuiteModel("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama", 8.0, 32, 4096,
               "LlamaForCausalLM", batch_size=48, deflection_batch_size=12, gated=True),

    # ---- Gemma 3 (gated: Gemma Terms). 1B is text-only; 27B is a VLM wrapper. ----
    # Gemma 3 dropped Gemma 2's logit soft-capping (uses QK-norm), so eager is not
    # strictly required — but it gives the most deterministic/bit-exact activations.
    SuiteModel("gemma3-1b", "google/gemma-3-1b-it", "gemma3", 1.0, 26, 1152,
               "Gemma3ForCausalLM", batch_size=64, deflection_batch_size=16,
               attn_implementation="eager", gated=True),
    SuiteModel("gemma3-27b", "google/gemma-3-27b-it", "gemma3", 27.0, 62, 5376,
               "Gemma3ForConditionalGeneration", batch_size=32, deflection_batch_size=8,
               attn_implementation="eager", multimodal=True, gated=True,
               notes="layers nest at model.language_model.layers (auto-resolved); SigLIP vision tower ignored"),

    # ---- OLMo 2 (Apache-2.0, ungated, fully-open training data). ----
    # Post-norm + QK-norm placement differs from Llama/Qwen; residual stream is
    # still standard additive, so hooks are unaffected (note it for cross-family
    # layer-by-layer comparisons).
    SuiteModel("olmo2-7b", "allenai/OLMo-2-1124-7B-Instruct", "olmo2", 7.3, 32, 4096,
               "Olmo2ForCausalLM", batch_size=48, deflection_batch_size=12,
               notes="full MHA (kv=32); post-norm"),
    SuiteModel("olmo2-32b", "allenai/OLMo-2-0325-32B-Instruct", "olmo2", 32.0, 64, 5120,
               "Olmo2ForCausalLM", batch_size=32, deflection_batch_size=8,
               notes="post-norm"),

    # ---- Mistral. Small-24B-2501 is the TEXT-ONLY variant (flat model.layers); ----
    # we deliberately avoid the 3.1/Pixtral VLM wrapper. Large-2411 is gated
    # (Mistral Research License, non-commercial).
    SuiteModel("mistral-small-24b", "mistralai/Mistral-Small-24B-Instruct-2501", "mistral", 24.0, 40, 5120,
               "MistralForCausalLM", batch_size=32, deflection_batch_size=8,
               notes="text-only 2501; 3.1-2503 is multimodal if vision ever needed"),
    SuiteModel("mistral-large-123b", "mistralai/Mistral-Large-Instruct-2411", "mistral", 123.0, 88, 12288,
               "MistralForCausalLM", device_map="auto", batch_size=16, deflection_batch_size=4,
               gated=True, notes="bf16 ~245 GB -> shard across all 4 cards; vocab only 32768"),
]

_BY_KEY = {m.key: m for m in SUITE}


def by_key(key: str) -> SuiteModel:
    """Look up one model by its slug, with a helpful error listing valid keys."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise SystemExit(
            f"unknown suite key {key!r}. valid keys:\n  " + "\n  ".join(_BY_KEY)
        )


def keys() -> list[str]:
    return [m.key for m in SUITE]


def single_card_lpt() -> list[str]:
    """Single-card keys in longest-processing-time-first order (proxy: params
    desc, then layers desc). Priming a 4-GPU work-queue in this order minimises
    the makespan tail -- the big models start first, the small ones backfill."""
    return [
        m.key for m in sorted(
            (m for m in SUITE if m.fits_one_card()),
            key=lambda m: (m.params_b, m.num_layers), reverse=True,
        )
    ]


def suite_root() -> str:
    """Root dir under which each model gets an isolated ``<root>/<key>`` data dir.
    Override with ``$EMOTION_PROBES_SUITE_ROOT`` (cluster default is on /data)."""
    return os.environ.get(
        "EMOTION_PROBES_SUITE_ROOT", "/data/agastyas/emotion-probes/suite"
    )


def env_exports(m: SuiteModel) -> list[str]:
    """The shell ``export`` lines that pin :class:`Config` to this model.

    ``scripts/run_suite.sh`` evals these. Only non-default quirks are emitted, so
    the environment stays minimal and readable.
    """
    lines = [
        f"export EMOTION_PROBES_MODEL_ID={m.model_id}",
        f"export EMOTION_PROBES_DEVICE_MAP={m.device_map}",
        f"export EMOTION_PROBES_BATCH_SIZE={m.batch_size}",
        f"export EMOTION_PROBES_DEFLECTION_BATCH_SIZE={m.deflection_batch_size}",
        f"export EMOTION_PROBES_DATA={suite_root()}/{m.key}",
    ]
    if m.enable_thinking is not None:
        lines.append(f"export EMOTION_PROBES_ENABLE_THINKING={'1' if m.enable_thinking else '0'}")
    if m.attn_implementation:
        lines.append(f"export EMOTION_PROBES_ATTN_IMPL={m.attn_implementation}")
    # Hints the runner reads (not Config fields): gating + FP8 backend need.
    lines.append(f"export SUITE_GATED={'1' if m.gated else '0'}")
    lines.append(f"export SUITE_FP8={'1' if m.fp8 else '0'}")
    lines.append(f"export SUITE_KEY={m.key}")
    return lines


def _print_table() -> None:
    hdr = (
        f"{'key':<18} {'hf id':<46} {'params':>7} {'L':>3} {'L*':>3} "
        f"{'hidden':>6} {'dev':<4} {'wt GB':>6} {'fit1':>4} {'gate':>4} {'fp8':>3}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in SUITE:
        print(
            f"{m.key:<18} {m.model_id:<46} {m.params_b:>6.1f}B {m.num_layers:>3} "
            f"{m.analysis_layer():>3} {m.hidden_size:>6} {m.device_map:<4} "
            f"{m.weight_gb():>5.0f}G {'Y' if m.fits_one_card() else 'N':>4} "
            f"{'Y' if m.gated else '-':>4} {'Y' if m.fp8 else '-':>3}"
        )
    single = [m.key for m in SUITE if m.fits_one_card()]
    sharded = [m.key for m in SUITE if not m.fits_one_card()]
    print(f"\nL* = analysis layer (2/3 depth). box = {NUM_CARDS}x{CARD_GB}GB.")
    print(f"single-card ({len(single)}): {', '.join(single)}")
    print(f"sharded (device_map=auto): {', '.join(sharded)}")


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        _print_table()
    elif cmd == "keys":
        which = argv[1] if len(argv) > 1 else "all"
        if which == "single":
            print("\n".join(m.key for m in SUITE if m.fits_one_card()))
        elif which == "single-lpt":
            print("\n".join(single_card_lpt()))
        elif which == "sharded":
            print("\n".join(m.key for m in SUITE if not m.fits_one_card()))
        elif which == "all":
            print("\n".join(keys()))
        else:
            raise SystemExit("usage: ... keys [all|single|sharded]")
    elif cmd == "env":
        if len(argv) < 2:
            raise SystemExit("usage: python -m emotion_probes.models.suite env <key>")
        print("\n".join(env_exports(by_key(argv[1]))))
    else:
        raise SystemExit(f"unknown command {cmd!r}; use: list | keys | env <key>")


if __name__ == "__main__":
    main()
