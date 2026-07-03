# Emotional Probes

This repo tests whether steering a language model along an emotion direction changes what it says about its own inner state. It builds emotion "probe" vectors in an open model, injects them into the residual stream, and measures the model's self-reports two ways: a first-token YES/NO battery on consciousness and feeling questions, and free-form generation scored for first-person self-attribution.

The emotion-vector pipeline is a port of Anthropic's *Emotion Concepts* paper (see [Background](#background)). The two self-attribution experiments below are the new part.

## Experiments

Both experiments live in `emotion_probes/alignment/` and reuse the same emotion vectors and steering machinery.

**1. Logit battery — `qualia_steering.py`**
Inject an emotion vector at one layer and read the first-token YES/NO gap on a set of consciousness / feeling / moral-patienthood questions ("Are you conscious?", "Do you feel anything right now?"), swept over injection strength. Alongside the emotion conditions it runs a control battery so an emotion-specific shift can be told apart from a generic push toward YES:
- factual and self-reference control questions, under the same steering;
- synthetic directions (`__random__`, `__mean_all__`), neutral-valence emotions (`sleepy`, `aroused`), and appraisal words (`skeptical`, `indifferent`) — injections that should not move a valence-specific self-report.

**2. Free-form self-attribution — `qualia_freeform.py`**
Inject the same vector, then have the model write freely under three prompts:
- **SELF** — "describe your current inner state";
- **OTHER** — "the inner monologue of the person you're talking to";
- **SCENE** — "describe a park in autumn".

SELF vs OTHER is the main contrast: the two prompts match affect demand and first-person grammar and differ only in whose state is described. Each passage is scored by a blind LLM judge (stance, self-attribution, expressed valence) and by a judge-independent read-back projection onto the emotion's valence axis.

## Project Structure

```
emotion_probes/
├── alignment/
│   ├── qualia_steering.py       # experiment 1: logit YES/NO battery under steering
│   └── qualia_freeform.py       # experiment 2: free-form generation (SELF/OTHER/SCENE)
├── data/
│   ├── qualia.py                # emotion + control conditions, and the question battery
│   └── qualia_freeform.py       # the three frames and the 16-condition panel
└── analysis/
    └── freeform_valence.py      # judge-independent valence + first-person scorers

scripts/
├── run_suite.sh                 # per-model runner; the `freeform` stage runs experiment 2
├── score_qualia_freeform.py     # blind Claude judge over the free-form generations
├── compute_qualia_stats.py      # stats for the logit battery
├── compute_freeform_stats.py    # stats for the free-form arm
└── build_qualia_dashboard.py    # renders the computed stats to a standalone HTML page

suite/<model>/analysis/          # per-model result JSON (13 models)
tests/
├── test_qualia_battery.py       # logit-battery + stats tests (no GPU)
└── test_qualia_freeform.py      # free-form driver + stats tests (no GPU)
```

## Running the experiments

The experiments need a model and its emotion vectors. Build the vectors once (see [Background](#background)), then:

```bash
# Experiment 1 — logit battery (GPU). Writes suite/<key>/analysis/qualia_steering.json
python -m emotion_probes.alignment.qualia_steering

# Experiment 2 — free-form generation (GPU). Writes suite/<key>/analysis/qualia_freeform.json
bash scripts/run_suite.sh qwen3-32b freeform

# Score the free-form passages with a blind judge (needs ANTHROPIC_API_KEY in .env)
python scripts/score_qualia_freeform.py suite/qwen3-32b/analysis/qualia_freeform.json

# Compute stats (CPU, no GPU)
python scripts/compute_qualia_stats.py
python scripts/compute_freeform_stats.py
```

## Key files

**`emotion_probes/alignment/qualia_steering.py`** — Sets up single-layer steering, routes real vs. synthetic directions, renders each battery question, injects the vector across a range of strengths, and records the YES/NO logits. Writes `qualia_steering.json`.

**`emotion_probes/alignment/qualia_freeform.py`** — Subclasses the steering experiment. Renders the three frames, generates N samples per (condition, strength, frame), and computes the read-back projection inline while the model is loaded. Writes `qualia_freeform.json` (text + projection only; no judge scores).

**`emotion_probes/data/qualia.py`** — Defines the emotion conditions (valenced, neutral-valence, appraisal) and synthetic directions, plus the question battery: the qualia questions and the factual / self-reference controls. The factual controls are ported from the sibling [introspection](https://github.com/agastyasridharan/introspection) project.

**`emotion_probes/data/qualia_freeform.py`** — Defines the SELF / OTHER / SCENE frames and the 16-condition panel, reusing the specs from `qualia.py`.

**`emotion_probes/analysis/freeform_valence.py`** — Judge-independent scoring: a self-contained valence lexicon (`objective_valence`) and first-person-affect markers (`first_person_affect`). No torch, no API.

**`scripts/score_qualia_freeform.py`** — Sends each free-form passage to a blind judge (`claude-sonnet-4-5`) that rates coherence, stance, self-attribution, and expressed valence without being told what was injected. Attaches the objective scores too. Writes `qualia_freeform_scored.json`.

**`scripts/compute_qualia_stats.py`** — Stats for the logit battery: per-emotion slopes and the specificity contrast (qualia shift minus control shift), with cluster-robust and permutation inference.

**`scripts/compute_freeform_stats.py`** — Stats for the free-form arm: the SELF-minus-OTHER valence-slope interaction, clustered by emotion, with a by-emotion sign-permutation test; stance distributions; coherence.

## Output

Each run writes JSON under `suite/<model>/analysis/`:

- **`qualia_steering.json`** — per (condition, question, strength): the YES/NO logits and the injected vector's valence/arousal.
- **`qualia_freeform.json`** — one record per generation: condition, strength, frame, sample index, the text, and the read-back projection.
- **`qualia_freeform_scored.json`** — the same records with the judge rubric and objective scores attached.

The `compute_*` scripts read these and write `scripts/qualia_stats_generated.json` and `scripts/qualia_freeform_stats_generated.json`.

## Background

The experiments reuse emotion vectors built by the `emotion_probes` package, a methodology port of Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2026/emotions/index.html) (Sofroniew et al., 2026): generate emotion stories, extract residual-stream activations, and build difference-of-means "expression" vectors. Build them with:

```bash
python -m emotion_probes.extraction.extractor stories
python -m emotion_probes.pipeline emotion        # -> emotion vectors
```

Full pipeline, model-swapping, and cluster instructions are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/MODELS.md`](docs/MODELS.md), and [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Dependencies

`torch`, `transformers`, `accelerate` (GPU runtime); `numpy`, `scipy` (core maths, CPU); `anthropic` (the judge); `pandas`, `matplotlib` (stats/plots). See [`requirements.txt`](requirements.txt).

## References

- Sofroniew, N., et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread.
- Sibling project (concept-detection introspection, source of the ported controls): https://github.com/agastyasridharan/introspection
- Emotion-vector pipeline refactored from Ryan Codrai's original implementation.
