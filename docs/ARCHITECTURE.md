# Architecture

This document explains how the `emotion_probes` package is laid out, *why* it is
laid out that way, and exactly which paper experiment each module replicates. Read
it alongside [`PAPER_DEEP_READ.md`](PAPER_DEEP_READ.md) (the conceptual companion).

## The one idea that shapes everything

> **All maths is pure NumPy and CPU-testable; all GPU/torch work is isolated behind one class.**

The paper's pipeline needs a big GPU. But most of what can go *wrong* — the
vector recipe, the PCA confound removal, the Elo, the ranking, the
character→token span alignment — is ordinary linear algebra. So we split the
codebase along that line:

- `core/` and everything that builds on it is **pure NumPy**. It has unit tests
  (`scripts/selfcheck.py`, 45 tests) that run on a laptop in under a second.
- `models/language_model.py` is the **only** module that imports torch, and it
  does so **lazily** (inside methods), so the entire package imports on a machine
  with no torch installed. Every analysis/alignment module keeps torch out of its
  top level too (matplotlib / sklearn / scipy are imported lazily as well).

The practical payoff: you can develop, test, and review the whole thing locally,
and only the final `run()` calls need the cluster.

## The dependency layering

```
                 config.py            ← single source of truth (model, layer, paths)
                    │
        ┌───────────┼───────────────────────────┐
        │           │                            │
     data/      core/  (pure NumPy)        models/language_model.py
   (171 emotions,  linalg  probes               ProbedModel
    100 topics,    vectors elo                 (the ONLY torch;
    scenarios,     ranking spans                lazy import)
    human norms)      │                            │
        │             │                            │
        └──────┬──────┴──────────────┬─────────────┘
               │                     │
         generation/           extraction/        pipeline.py
        (stories, dialogues,  (forward passes →   (activations →
         self-gen or API)      activations .npz)   probe vectors)
               │                     │                 │
               └─────────────────────┴────────┬────────┘
                                               │
                          ┌────────────────────┼─────────────────────┐
                          │                     │                     │
                     analysis/             alignment/          visualisation/
                  (Parts 1 & 2:          (Part 3: causal      (Flask token-heatmap
                   validate +             alignment evals)     web UI)
                   characterise)
```

Arrows point *downward* = "imports from". Nothing in `core/` imports torch; nothing
above `models/` reaches into the model except through `ProbedModel`'s public API.

## Module-by-module, mapped to the paper

### `config.py` — the single source of truth
A frozen dataclass. Change the model, the analysis layer, or a path in **one**
place. The analysis layer is stored as `layer_fraction` (default `2/3`) and turned
into a concrete index per-model by `analysis_layer(num_layers)`, so it is correct
whether the model has 28 layers or 80 (paper's "~2/3 depth, where operative
emotion lives"). Set `EMOTION_PROBES_DATA` to relocate all artifacts off `$HOME`.

### `data/` — verbatim paper stimuli
`emotions.py` (171 emotions, p. 82), `topics.py` (100 topics, pp. 82–85),
`scenarios.py` (implicit scenarios, user-vs-assistant prompts, intensity
templates, the 64 activities, sycophancy prompts, the SummitBridge blackmail
context), and `human_norms.py` (a blank Russell & Mehrabian valence/arousal
template — *no fabricated values*).

### `core/` — the pure-NumPy kernels (fully unit-tested)
| Module | What it is | Paper |
|--------|-----------|-------|
| `linalg.py` | PCA-for-variance, project-out, unit-normalise, cosine matrix | §2.1 recipe |
| `vectors.py` | `EmotionVectorComputer`, `DeflectionVectorComputer`, `ProbeBank`/`DeflectionVectors` containers | §2.1, deflection appendix |
| `probes.py` | `ProbeBank` — vectors `(L,E,H)` + `project()` | the probe object |
| `elo.py` | sequential Elo + games-from-preferences | Fig 4 |
| `ranking.py` | RRF of (mean score, autocorrelation) | the visualiser ranking |
| `spans.py` | **exact** char→token mask via `offset_mapping` | fixes Issue #7 |

### `models/language_model.py` — `ProbedModel` (the only torch)
Auto-detects the decoder-layer list across architectures (Gemma/Llama/Qwen/
Mistral/GPT/OPT…), the hidden size, and the analysis layer. Exposes: `extract_means`,
`iter_token_activations` (with offsets), `activation_at_last_token`,
`last_token_logits`, `logit_lens`, `average_residual_norm`, `steer` (context
manager), and `generate`. This is the contract every experiment builds on.

### `generation/` — building the datasets
`backends.py` chooses between an external API (`ApiBackend`) and **self-generation
by the probed model** (`LocalModelBackend`, fixes Issue #2). `generators.py`
writes emotional stories, neutral baselines, and deflection dialogues, with a
`one_story_per_call` mode (fixes Issue #6). `pair_selection.py` picks the
most-dissimilar emotion pairs for deflection (fixes Issue #5 — one documented
`dissimilar_fraction`).

### `extraction/` + `pipeline.py` — activations → vectors
`extractor.py` runs forward passes and writes activation `.npz` files (deflection
uses exact offset spans + checkpointing). `pipeline.py` turns those into the
probe banks via the `core` computers.

### `analysis/` — Parts 1 & 2 (validate + characterise)
| Module | Class | Paper |
|--------|-------|-------|
| `logit_lens.py` | `LogitLensAnalysis` | Table 1 |
| `context_activation.py` | `ContextActivationAnalysis` | Fig 1 |
| `implicit_scenarios.py` | `ImplicitScenarioAnalysis` | Fig 2 |
| `intensity_templates.py` | `IntensityTemplateAnalysis` | Fig 3 |
| `classification.py` | `EmotionClassification` | held-out probe sanity check (was missing — Issue #3) |
| `preferences.py` | `PreferenceExperiment` | Fig 4 (Elo + steering) |
| `geometry.py` | `GeometryAnalysis` | Figs 5–9 (clusters, PCA=valence/arousal, RSA) |
| `human_alignment.py` | `HumanAlignmentAnalysis` | Figs 8/58 |
| `layerwise.py` | `LayerwiseAnalysis` | Figs 12–14 |
| `speaker.py` | `SpeakerAnalysis` | Figs 10–11 (17–19 sketched) |
| `chronic_state.py` | `ChronicStateProbe` | Table 5 / Fig 16 |
| `steering.py` | `SteeringEngine` | the shared steering helper (footnote 3) |

### `alignment/` — Part 3 (causal alignment behaviours)
| Module | Class | Paper |
|--------|-------|-------|
| `behavior.py` | `BehaviorSteeringExperiment` | the shared "steer → measure a behaviour rate" loop |
| `reward_hacking.py` | `RewardHackingExperiment` | Figs 30–31 |
| `sycophancy.py` | `SycophancyExperiment` | Figs 32–35 |
| `blackmail.py` | `BlackmailExperiment` | Figs 26–29 (research-safety harness) |
| `post_training.py` | `PostTrainingComparison` | Figs 36–39 / Table 16 |

### `visualisation/` — the web UI (refactored, UI preserved byte-for-byte)
`clusters.py` (spider-chart clusters + mode labels), `template.py` (the verbatim
HTML/JS single-page app — only model-agnostic parameterisations changed),
`scoring.py` (`TokenActivationScorer` — one forward pass → per-(token, emotion)
scores for both probe spaces + RRF ranking + clusters), `server.py`
(`VisualiserServer` — loads model + banks and serves the page).

## What needs a GPU vs what doesn't

| Runs on a laptop (CPU) | Needs the cluster (GPU) |
|------------------------|--------------------------|
| `scripts/selfcheck.py` (all 45 tests) | extraction (forward passes) |
| importing the whole package | generation when `self_generate` |
| `GeometryAnalysis` (works from a saved bank) | every `analysis`/`alignment` `run()` |
| the visualiser's `TokenActivationScorer` logic | the visualiser server (loads the model) |

## Model compatibility — can I pass an arbitrary open model?

**Yes for the mainstream family** of autoregressive Transformer decoder LMs **with a fast (Rust) tokenizer**: Llama, Qwen2/3, Mistral, Mixtral, Gemma 2/3, Phi-3, GPT-2/J, GPT-NeoX/Pythia, OPT, Falcon, BLOOM, StableLM, Command-R, MPT, DBRX, and MoE variants. Set `Config.model_id` and the pipeline + experiments run. The fraction-of-depth analysis layer, the layer auto-detection, and the capture/steering hooks are genuinely model-agnostic and were adversarially audited.

**What is and isn't supported:**

| Works | Notes |
|---|---|
| Standard decoder-only Transformers + MoE | layer container auto-detected via `_LAYER_PATHS` |
| Any depth (6 → 80 layers) | analysis layer is a *fraction*, not a fixed index |
| Different dtypes | `Config.dtype` validated against `torch` with a clear error |

| Limitation | Behaviour |
|---|---|
| Architecture not in `_LAYER_PATHS` | **fails loudly at load** with an actionable message; add one path string |
| State-space models (Mamba, RWKV, RecurrentGemma) | paths included, but a "layer" is not an attention residual — interpret with care |
| **Slow-tokenizer-only models** | the per-token experiments (deflection extraction, visualiser, layerwise/speaker/context) raise a clear error — they need character offsets, which only fast tokenizers provide. The bulk story/neutral extraction and preferences still work. |
| Base (non-chat) models | the behaviour evals (reward-hacking/sycophancy/blackmail) need an instruction-tuned model; `generate(chat=True)` falls back to raw text on a base model |
| Non-English / non-Python outputs | the alignment behaviour *detectors* are English/Python regex heuristics |

**The one foot-gun, now guarded.** Emotion vectors are only meaningful for the *exact* model they were built from (a vector lives in that model's residual basis; layer `L` means *that* model's layer `L`). To prevent silently-wrong results from reusing a bank with a different model:

- Saved vectors are **stamped with `source_model_id`** (the building model).
- `ProbeBank.check_against(model)` **raises** on a layer-count or hidden-size mismatch and **warns** on a `source_model_id` mismatch. It runs automatically wherever a bank meets a model — `SteeringEngine`, the visualiser, and every analysis/alignment experiment's constructor.

So: rebuild the vectors when you change `Config.model_id` (`python -m emotion_probes.pipeline emotion`). If you forget, you get a loud error (different shape) or an explicit warning (same shape, different model) — never silent nonsense.

**Which models to actually run.** This section answers *"can it take an arbitrary model?"*. For *"which specific open models should I run, in what order, and will they fit my GPUs?"* — an ordered ladder (7B → 32B → 9B → 70B → 235B-FP8), the 4×H100 NVL capacity math, the tooling-reuse rationale (Persona Vectors / Assistant Axis / Gemma Scope), and per-model extraction settings — see [`MODELS.md`](MODELS.md).

**Tokenizer-sensitive conventions** (robust on common tokenizers, documented where not): the `"Assistant:"`-colon probe position and the `(A)`/`(B)` preference tokens are resolved in-context (`token_ids("(A")[-1]`) to handle SentencePiece/BPE leading-space merges; the behaviour evals expose a `chat=` flag (default `True`, consistent across all three) to apply the model's chat template or use the paper's raw `Human:/Assistant:` scaffold.

## Conventions (the "house style")

- `from __future__ import annotations` at the top of every module.
- One class per experiment; a `run()` returning a JSON-serialisable dict and
  saving under `config.analysis_dir`; an argparse `main()` where natural.
- Numbers and paths come from `config`, never hard-coded.
- Heavy/optional imports (torch, sklearn, scipy, matplotlib, flask) are **lazy**.
- **Nothing fabricates results** — modules implement the *measurement*; you supply
  the GPU (and, where noted, the eval data) at run time.
