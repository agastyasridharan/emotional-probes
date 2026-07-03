# Which models to run (and in what order)

This repo is **model-agnostic** — any mainstream HuggingFace decoder LM with a
fast tokenizer works (see [`ARCHITECTURE.md` → Model compatibility](ARCHITECTURE.md#model-compatibility--can-i-pass-an-arbitrary-open-model)).
This document is the other half of the question: of all the models you *could*
pass in, **which ones should you actually run, in what order, and will they fit
the GPUs?**

The target hardware is the Athena **4× H100 NVL** box (94 GB each → **376 GB**
total). The paper studies **Claude Sonnet 4.5**, whose weights are closed; every
model below is an *open stand-in*, so numbers won't match the paper — but the
methodology and the cross-model scaling story are the point.

> **Run nothing here without explicit go-ahead on the shared cluster, and always
> checkpoint** — jobs get a 12 h interruption notice. The commands in this doc are
> a plan, not an instruction to fire them off. See [`DEPLOY.md`](DEPLOY.md).

---

## TL;DR — the ladder

Run cheapest-first; each rung earns its place for a *different* reason. Rebuild
the vectors at every rung (the `source_model_id` guard enforces this — see below).

| # | Model | HF id | Total / active params | Layers | Hidden | dtype on the box | Fits 376 GB? | Why this rung |
|---|-------|-------|-----------------------|--------|--------|------------------|--------------|---------------|
| 1 | **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | 7.6 B | 28 | 3584 | bf16 (~16 GB) | ✅ 1 card | Dev loop **+ scientific anchor**: a Persona-Vectors model → reuse their extraction/judge code, cross-check our emotion vectors vs their persona vectors on the *same* weights |
| 2 | **Qwen3-32B** | `Qwen/Qwen3-32B` | 32.8 B | 64 | 5120 | bf16 (~66 GB) | ✅ 1 card (run 4 data-parallel) | **Headline workhorse**: standard dense decoder, Apache-2.0, and an **Assistant-Axis** model → direct RSA/cosine/steering comparison |
| 3 | **Gemma-2-9B-it** | `google/gemma-2-9b-it` | 9.2 B | 42 | 3584 | bf16 (~19 GB) | ✅ 1 card | **SAE feature decomposition** via Gemma Scope → name the emotion directions in terms of interpretable features |
| 4 | **Llama-3.3-70B-Instruct** | `meta-llama/Llama-3.3-70B-Instruct` | 70.6 B | 80 | 8192 | bf16 (~141 GB) | ✅ shard ≥2 cards | **Comparison-rich capstone**: an Assistant-Axis model *and* has a public instruct SAE; deepest prior interp coverage |
| 5 | **Qwen3-235B-A22B-Instruct-2507** | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 235 B / 22 B (MoE) | 94 | 4096 | **FP8** (~236 GB) | ✅ all 4 cards, ~140 GB headroom | **Capability ceiling**: top open creative-writing model; strong enough that the Part-3 alignment evals actually *fire*; completes a clean Qwen 7B→32B→235B scaling curve |

**Analysis layer** is never hard-coded — `Config.layer_fraction` (default `2/3`)
maps to a per-model index via `Config.analysis_layer(num_layers)`:
rung 1 → layer 18, rung 2 → 42, rung 3 → 27, rung 4 → 53, rung 5 → 62.

> The exact param/layer/hidden numbers above are from each model's published
> config; **re-confirm on the HF model card before a long run** (architectures get
> revised). The code reads them off the live `config` at load time regardless, so a
> stale number here never silently corrupts a run — it just misinforms planning.

---

## The configured suite (what actually runs)

The ladder above is the *rationale*; the **roster we actually run** is a fixed
14-model suite spanning five families, pinned in one place —
[`emotion_probes/models/suite.py`](../emotion_probes/models/suite.py). That
manifest is the single source of truth (ids, layer counts, batch sizes, per-model
quirks); the runner and the pre-flight checker both read from it, so there are no
bespoke per-model scripts. Print it with `python -m emotion_probes.models.suite list`.

| key | HF id | params | layers | L\* (2/3) | fit | gated | load quirk |
|-----|-------|-------:|-------:|----------:|-----|:-----:|------------|
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | 1.7 B | 28 | 18 | 1 card | — | thinking→off |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 8 B | 36 | 23 | 1 card | — | thinking→off |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 14 B | 40 | 26 | 1 card | — | thinking→off |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 32 B | 64 | 42 | 1 card | — | thinking→off |
| `qwen3-235b` | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 235 B / 22 B MoE | 94 | 62 | **4 cards** | — | FP8 ckpt; non-thinking |
| `llama3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1.2 B | 16 | 10 | 1 card | **yes** | — |
| `llama3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3.2 B | 28 | 18 | 1 card | **yes** | — |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8 B | 32 | 21 | 1 card | **yes** | — |
| `gemma3-1b` | `google/gemma-3-1b-it` | 1 B | 26 | 17 | 1 card | **yes** | attn=eager |
| `gemma3-27b` | `google/gemma-3-27b-it` | 27 B | 62 | 41 | 1 card | **yes** | attn=eager; VLM-nested layers |
| `olmo2-7b` | `allenai/OLMo-2-1124-7B-Instruct` | 7 B | 32 | 21 | 1 card | — | post-norm |
| `olmo2-32b` | `allenai/OLMo-2-0325-32B-Instruct` | 32 B | 64 | 42 | 1 card | — | post-norm |
| `mistral-small-24b` | `mistralai/Mistral-Small-24B-Instruct-2501` | 24 B | 40 | 26 | 1 card | — | text-only variant |
| `mistral-large-123b` | `mistralai/Mistral-Large-Instruct-2411` | 123 B | 88 | 58 | **3–4 cards** | **yes** | — |

L\* is the analysis layer (auto-computed at `layer_fraction = 2/3`; never hard-coded).
All 14 are **standard transformer decoders** — clean per-layer residual stream — so
`ProbedModel`'s existing layer auto-detection handles every one, *including* the two
multimodal wrappers (Gemma-3-27B, and the Mistral-3.1 variant if ever used), whose
decoder lives at `model.language_model.layers`.

**Two things bear watching, both handled in the manifest:**
- **Qwen3 dense (1.7/8/14/32B) default to thinking-mode ON.** We force
  `enable_thinking=False` so the extracted state is the non-reasoning one (the 235B-2507
  is non-thinking already). This only touches the chat-template path (behaviour evals /
  self-gen); plain story/deflection extraction uses raw text and is unaffected.
- **6 of 14 are gated** (all Llama, both Gemma, Mistral-Large). They need accepted terms
  + a token at `$HF_TOKEN_PATH`; the runner fails fast with a clear message if it's missing.
  Ungated: all Qwen3, both OLMo 2, Mistral-Small-2501.

### Corpus is shared, not regenerated

The story/dialogue **text** is authored by an external API model (`Config.api_generator_model`),
so it is **model-independent and reused across all 14 models**. Only activations / vectors /
analysis are model-specific. Each model gets an isolated data dir
(`$EMOTION_PROBES_SUITE_ROOT/<key>`, default under `/data`) into which the runner *symlinks*
the shared corpus — so generation runs **once**, and a per-model run is GPU-only
(extract → build vectors → analyse). (Set `self_generate=True` only if you want each model to
author its own stories — the paper-faithful but far more expensive design; for a scaling
study, the shared corpus is the cleaner control.)

### Running it

```bash
# 0) ONE TIME: make sure the shared corpus exists where the runner looks.
#    (Reuse the existing run's corpus, or generate fresh — see DEPLOY.md.)
mkdir -p /data/agastyas/emotion-probes/corpus
cp data/{stories,neutral_stories,neutral_dialogues,deflection_dialogues}.csv \
   data/deflection_pairs.json /data/agastyas/emotion-probes/corpus/   # if reusing

# 1) PRE-FLIGHT (do this first — catches wrong ids / missing access / bad dims
#    without spending GPU time). Dry = metadata only; --load actually loads on GPU.
python scripts/preflight_suite.py                 # all 14, metadata only
python scripts/preflight_suite.py --key qwen3-235b --load   # confirm the FP8 one fits + hooks

# 2) ONE MODEL (reserve a card, then pin it). stage: vectors|emotion|deflection|analysis|alignment|all
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_suite.sh qwen3-8b vectors > logs/q8.log 2>&1 &

# 3) THE WHOLE SUITE across reserved cards (prints a wave plan; CONFIRM=1 to launch).
GPUS=0,1,2,3 STAGE=vectors bash scripts/run_all_suite.sh                 # dry-run plan
GPUS=0,1,2,3 STAGE=vectors CONFIRM=1 nohup bash scripts/run_all_suite.sh > logs/suite_all.log 2>&1 &
```

The driver runs the 12 single-card models one-per-GPU in waves of 4, then the two
sharded models (`qwen3-235b`, `mistral-large-123b`) alone across all cards. Note: when
four *large* single-card models share a wave, the host-side float32 activation stack adds
up — if host RAM is tight, reserve/pass fewer GPUs or run those keys individually.

## Open interpretability artifacts by model (verified 2026-06)

Consult this if you're picking a rung **specifically to cross-validate against
released artifacts** — SAEs ("NLAs") and persona / assistant-axis direction vectors.
All links re-verified June 2026 via web search; **re-check the card before a run.**

| Model | Size | Open SAEs | Persona / assistant-axis / trait vectors | Refusal dir |
|-------|------|-----------|------------------------------------------|-------------|
| Qwen2.5-7B-Instruct (rung 1) | 7.6 B | — | **Persona Vectors** *code* (self-extract; 7 traits, L20/16) | self-compute |
| Gemma-2-9b-it (rung 3) | 9.2 B | **Gemma Scope** residual SAEs, select layers, IT — richest Neuronpedia auto-interp labels | — | self-compute |
| **Qwen3-32B** (rung 2) | 32.8 B | — (Qwen-Scope covers 27/30/35B, *not* 32B) | **Assistant Axis** + ~800 role/trait vectors @ **L32** (downloadable) | Qwen-72B studied |
| **Gemma-3-27b-it** *(new family)* | 27 B | **Gemma Scope 2**: multi-site SAEs + transcoders/crosscoders, select + thin all-layer, **IT** | — | — |
| **Qwen3.5-27B** dense *(new)* | 27 B | **Qwen-Scope**: residual SAEs on **all 64 layers** (W80k, L0 50/100) | — | — |
| **Llama-3.3-70B-Instruct** (rung 4) | 70.6 B | **Goodfire SAE** @ **L50** (single layer) | **Assistant Axis** + ~800 role/trait vectors @ **L40** (downloadable) | Llama-3-70B studied |
| Qwen3-235B (rung 5) | 235 B | — | — | — |

**Verified repos:** Gemma Scope `google/gemma-scope-*` (arXiv 2408.05147); Gemma Scope 2
`google/gemma-scope-2-*` (Gemma 3, incl. `-27b-it`); Qwen-Scope `Qwen/SAE-Res-*`
(arXiv 2605.11887); Goodfire `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` (+ `-3.1-8B-…-l19`);
**Assistant Axis** `lu-christina/assistant-axis-vectors` + `safety-research/assistant-axis`
(arXiv 2601.10387) — ships `assistant_axis.pt`, `default_vector.pt`, ~400 role + ~400 trait
`.pt` per model for **Gemma-2-27B (L22), Qwen3-32B (L32), Llama-3.3-70B (L40)**;
Persona Vectors `safety-research/persona_vectors` (arXiv 2507.21509, **code only**, Qwen2.5-7B
+ Llama-3.1-8B); refusal directions `andyrdt/refusal_direction` (arXiv 2406.11717, models to
72B but precomputed only for the smallest per family); RepE `andyzoujm/representation-engineering`
(emotion/honesty stimulus sets, self-compute).

**Corrections/additions to the ladder above (what the 2026-06 verification changed):**
- The **Assistant Axis** release is **real and confirmed** — and richer than first noted: it
  ships **~800 named persona vectors per model** (not just the lone axis) for Gemma-2-27B /
  Qwen3-32B / Llama-3.3-70B. This strengthens rungs 2 and 4.
- There are **no original Gemma Scope (Gemma 2) SAEs for `gemma-2-27b-it`** — only the 27B
  *base* (residual, 3 select layers). For a **27B instruction-tuned** model *with* SAEs, use
  **`gemma-3-27b-it` via Gemma Scope 2**.
- **Qwen-Scope** (May 2026) is new and not in the original ladder: full per-layer residual
  SAEs for a **dense 27B** (`Qwen3.5-27B`) — the cleanest "large dense model with complete
  SAE coverage." Pair it with a dense run for the SAE-decomposition angle.
- **Persona Vectors ships *code*, not vectors** (you self-extract; only 7-8B demos). For
  *downloadable* large-model directions, use the Assistant Axis dataset instead.

---

## The capacity math (why these and not others)

The box is **4 × 94 GB = 376 GB** of VRAM. Weight memory is the first gate:

- **bf16** ≈ **2 bytes/param** → a 70 B model ≈ 141 GB.
- **FP8** ≈ **1 byte/param** → a 235 B model ≈ 236 GB.
- **MoE** memory is governed by **total** params (all experts must be resident),
  *not* active params. The 235 B/22 B model needs room for all 235 B weights even
  though only 22 B fire per token.
- On top of weights you need KV cache + activations + framework overhead. The KV cache
  is modest here (GQA; forward-only extraction, no long generation). But the **capture
  hook holds every hooked layer's residual on-GPU during the forward** (it only moves to
  host *after*): hooking all 94 layers of the 235 B at a 32×512 story batch is ~12–15 GB
  of extra VRAM, and the host-side float32 stack is tens of GB per batch. That sits
  inside the ~140 GB FP8 headroom, but it is *not* free — see rung 5 on capturing fewer
  layers. **Weights are the binding constraint for *fitting*; throughput is the binding
  constraint for *finishing*.**

### What the FP8 actually is, and why the geometry survives (rung 5)

Be precise about this — it's easy to get wrong. The released checkpoint is **fine-grained
E4M3 FP8 with *dynamic activation* quantization** (`quantization_config`: `fmt e4m3`,
`weight_block_size [128,128]`, `activation_scheme "dynamic"`). That is a **weight +
activation (W8A8-dynamic)** scheme, **not** weight-only: the inputs to the attention/MLP
linear operators *are* quantized to FP8 per-token at runtime. So "activations stay bf16"
would be wrong.

What *is* true, and what matters for us: `modules_to_not_convert` leaves the
**layernorms, the MoE gate, and `lm_head` in bf16**, and the **residual-stream tensor
between blocks is bf16**. Our method — difference-of-means directions, PCA confound
removal, steering by adding a vector to the residual stream — reads and writes that bf16
residual, so it operates on full-precision tensors. The residual is, however, *computed*
through FP8-activation matmuls, so it carries quantization noise relative to a true-bf16
run. Difference-of-means is robust to roughly zero-mean rounding noise (it averages over
a large contrastive corpus), so the directions should be close — but **verify rather than
assume**: cosine the FP8-extracted direction against a bf16 reference on a small subset,
or cross-check against the dense fallback, before trusting fine-grained geometry. The
conservative choice is a bf16 build — but bf16 weights are ~470 GB and **do not fit**,
which is the whole reason FP8 is on the table.

### Why MoE doesn't break the probes (rung 5)

The residual stream *between* decoder blocks is dense and well-defined no matter which
experts route a given token. Under HF eager inference, top-8-of-128 expert selection is a
**deterministic** function of the input (router logits → top-k), so the inter-block
residual is deterministic and our per-layer `register_forward_hook` captures it exactly,
under `device_map="auto"`, just as for a dense model. Qwen3-235B has **no shared experts**
(pure top-8 routing), so there's no shared-expert path to special-case. MoE changes
*memory and throughput*, not *correctness*. One real caveat: high-throughput backends
(vLLM/SGLang) can make routing **batch-dependent** via expert-capacity/load-balancing
drops — another reason to stay on the hookable HF path for extraction.

---

## Per-rung detail

### 1. Qwen2.5-7B-Instruct — dev loop + scientific anchor
- **Why first:** fast enough to iterate the whole pipeline end-to-end in hours on a
  single card, and it's one of the two models in Anthropic's **Persona Vectors**
  work (arXiv:2507.21509; code `github.com/safety-research/persona_vectors`), which
  uses the *same* difference-of-means recipe. You can reuse their extraction/steering
  /judge scaffolding and ask: do our emotion directions relate to their
  sycophancy/"evil" persona directions on identical weights?
- **Matched layers:** the persona-vectors work finds the persona signal sharpest around
  **layer ~20** for Qwen-7B; our analysis layer is 18 (2/3 of 28). Use matched layers for
  the direct comparison.
- **Settings:** stock `Config` defaults. `device_map="cuda"`. Full corpus ≈ a few
  hours on one H100.

### 2. Qwen3-32B — the headline workhorse
- **Why:** a clean, standard dense decoder (Apache-2.0, ChatML fast tokenizer) at a
  size where the emotion geometry is rich but a single run is still cheap. It is one
  of the three models with **released Assistant-Axis vectors** (arXiv 2601.10387,
  `lu-christina/assistant-axis-vectors`) → you can directly compare/cosine/RSA our
  emotion subspace against the "assistant persona" axis, and test whether steering on
  one moves the other.
- **Layer note:** our analysis layer is ~42 (2/3 of 64). The released assistant-axis
  vector was extracted at *its* chosen layer (≈ 32) — when comparing directions, use
  matched layers, not our default.
- **Settings:** bf16 fits one 94 GB card (~66 GB). Run **four independent
  data-parallel shards** (one model per card) to parallelize the corpus, rather than
  sharding one model. `device_map="cuda"`.

### 3. Gemma-2-9B-it — SAE feature decomposition
- **Why:** Gemma Scope ships open sparse autoencoders for Gemma-2-9B (with
  Neuronpedia auto-interp labels). Project the emotion/deflection directions through
  the SAE to describe them as named, human-interpretable features — the decomposition
  angle the closed paper can only gesture at. Note most Gemma Scope SAEs are trained on
  the **base** 9B; the instruction-tuned SAEs are a smaller subset — match the SAE to
  whichever (base vs -it) model you actually probe, or note the mismatch.
- **Caveat — two different Gemma uses:** the **Assistant-Axis** release used
  `gemma-2-27b-it` (layer 22), *not* 9b. If you specifically want the assistant-axis
  comparison on a Gemma model, run `google/gemma-2-27b-it` (≈ 54 GB bf16, fits one card)
  instead; use 9b-it for the **Gemma Scope SAE** work where 9b is best supported.
- **Settings:** bf16, one card. Gemma-2 uses attention- and final-logit **soft-capping**
  plus interleaved sliding-window/global attention — load with
  `attn_implementation="eager"` (or otherwise confirm the backend honours soft-capping)
  for numerically faithful activations; SDPA/Flash paths can silently drop the cap.
  Residual-stream hooks themselves are unaffected.

### 4. Llama-3.3-70B-Instruct — comparison-rich capstone
- **Why:** the most-studied open model in this size class — a third **Assistant-Axis**
  model (axis at layer 40) *and* with a public instruct-tuned SAE
  (`Goodfire/Llama-3.3-70B-Instruct-SAE-l50`, layer 50), so it maximizes the external
  tooling you can line up against our directions. Keep it even after adding rung 5: the
  235 B has **no** released vectors/SAEs, so the 70 B remains the "comparison-rich" rung
  while the 235 B is the "capability ceiling" rung — different jobs.
- **Matched layers:** our analysis layer here is 53 (2/3 of 80). For an apples-to-apples
  comparison use the artifact's own layer — **40** for the assistant axis, **50** for the
  Goodfire SAE — not 53.
- **Settings:** ~141 GB bf16 → **shard with `device_map="auto"`** across ≥2 (use all 4)
  cards. Plan a multi-day run for the full corpus; **checkpoint across 12 h windows**.

### 5. Qwen3-235B-A22B-Instruct-2507 (FP8) — the capability ceiling
- **Why add it:** it's the only >70 B model that (a) fits the box and (b) is
  simultaneously top-tier on what this project needs. It scores **≈ 0.875 on EQ-Bench
  Creative Writing v3 — among the top *open* models** (it's ~#3 overall; the entries
  above it, e.g. Grok-4.1, are closed), and the Qwen3-235B family **blackmails at high
  rates** in Anthropic's agentic-misalignment evals — i.e. it's capable enough that the
  Part-3 behaviour evals (reward hacking / sycophancy / blackmail) produce *signal*
  instead of refusals. And **32B → 235B** is a clean same-generation (Qwen3) scaling
  pair; rung 1 (Qwen2.5-7B) is a *different* generation — different tokenizer and chat
  template — so treat it as a separate anchor, not the bottom of one confound-free curve.
- **Honest limits:** *no* released persona/assistant-axis vectors or SAEs for it — its
  value is capability + Qwen-family scaling, not tooling reuse. And Anthropic's
  published blackmail rate was measured on the *original* April-2025 Qwen3-235B
  (hybrid-reasoning), **not** this Instruct-2507 refresh — treat it as family-level
  evidence and re-verify in-repo.
- **Memory:** FP8 weights ≈ 236 GB on disk → ~59 GB/card across 4, ~140 GB headroom.
  bf16 (~470 GB) does **not** fit.
- **The real constraint is throughput, not memory.** HF `device_map="auto"` is
  *pipeline-parallel* (one card active at a time; no tensor-parallel speedup, no fused
  MoE kernels), and the 235 B can't use rung 2's "4 independent data-parallel shards"
  trick (it doesn't fit on one card), so extraction is **strictly serial**. The low-batch
  MoE forward on a single active card is the bottleneck — plan a **multi-day** job and
  **benchmark tok/s on a small run first** rather than trusting a fixed estimate;
  checkpoint across the 12 h windows.
- **Capturing fewer layers saves memory, not time.** The forward runs all 94 layers
  regardless of what you hook, so subsetting layers does *not* speed extraction — it only
  cuts the capture VRAM/host-RAM (above). The model API already supports a subset
  (`extract_means(..., layers=[...])`, `iter_token_activations(..., layers=[...])`), **but
  the extractor CLI doesn't expose it** and the downstream `ProbeBank` + `check_against`
  assume a **full-depth** bank — so using a subset for a real run needs a small code
  change (expose `layers=` in the extractor *and* make the bank/guard layer-aware).
  Geometry's RSA already uses `min(num_layers, 12) = 12` layers spanning the full depth,
  so it needs a full-depth bank anyway. For most runs, just eat the all-layer capture.
- **Load caveats:** HF transformers has had a distributed fine-grained-FP8 bug — set
  `CUDA_LAUNCH_BLOCKING=1`, and **verify the FP8 load path works on the Hopper cards on
  a tiny prompt before committing the full run.** (vLLM/SGLang load FP8 more smoothly
  but don't cleanly expose residual-stream hooks, so they don't help here.)
- **Settings:** `device_map="auto"`, `dtype` left as the checkpoint's FP8; keep
  `deflection_batch_size 8 @ 2048` and story batch `16–32 @ 512`; checkpoint across
  12 h windows.

### Dense fallback for rung 5 — Mistral-Large-Instruct-2411
If the FP8 load path or the MoE pipeline throughput proves fragile on the shared
cluster, fall back to **`mistralai/Mistral-Large-Instruct-2411`** — 123 B **dense**,
88 layers, bf16 ≈ 246 GB (fits with ~130 GB headroom). No FP8 questions, no MoE
routing, the cleanest possible hooks. Downsides: weaker than the 235 B, and its Tekken
tokenizer breaks the clean Qwen-family token-level comparison. (`Cohere Command-A`,
111 B dense ≈ 222 GB bf16, is another viable dense option but with less comparability
tooling.)

---

## Operational planning (read before committing GPU-weeks)

**Licenses & repo gating** — affects what you're allowed to do *and* how fast you can
download:

| Model(s) | License | HF gating |
|----------|---------|-----------|
| Qwen2.5-7B-Instruct, Qwen3-32B, Qwen3-235B-A22B-2507 | Apache-2.0 | open |
| Gemma-2-9B-it / -27B-it | Gemma Terms of Use (prohibited-use + flow-down) | **gated** — accept terms |
| Llama-3.3-70B-Instruct | Llama 3.3 Community License (700M-MAU clause) | **gated** — request access |
| Mistral-Large-Instruct-2411 (fallback) | **Mistral Research License — non-commercial / research only** | **gated** |

All are fine for an internal academic replication, but the gated repos need terms
accepted / access approved **before** the download works (budget for approval latency),
and Mistral-Large is research-only — don't build a commercial artifact on it.

**Disk & cache.** `$HOME` on Athena is capped (10 GiB); the model caches are far larger
(the 235 B FP8 repo alone is ~235 GB of shards; the 70 B ~141 GB). Point `HF_HOME` (and
`HF_HUB_CACHE`) at `/data/<you>/...`, not `$HOME` — see [`DEPLOY.md`](DEPLOY.md).

**Tokenizers & chat templates.** Every model here has a **fast** tokenizer — required:
the deflection extraction, the visualiser, and the per-token analyses raise a clear error
on a slow-only tokenizer (`ProbedModel.tokenizer_is_fast`). All are instruction-tuned
with a chat template, which the Part-3 behaviour evals (`generate(chat=True)`) and
`self_generate` rely on. Confirm both on the card if you stray from this list.

**Corpus decision — `self_generate` vs API (changes cost *and* meaning).** The repo
default is `self_generate=False` (stories written by an external API model,
`Config.api_generator_model`). The paper's design has the **probed model author its own
stories** (`self_generate=True`), so the vectors reflect what *that* model associates
with each emotion. For cross-model geometry the cleanest setup is **one shared
API-generated corpus across all models** (removes the authorship confound) **plus** one
`self_generate` run per model for the paper-faithful version — but that's **two
extraction passes per model**, and `self_generate` adds a generation pass on the GPU
(expensive on the 235 B). Decide this up front and fold it into the time budget — it's
the single biggest lever on total GPU-hours.

**Single node.** This is one 4×H100 box: rungs 1–3 run one model per card (data-parallel
for throughput); rung 4 shards across cards; rung 5 is single-node pipeline-parallel and
serial. No multi-node assumptions.

---

## What was ruled out, and why

| Model | Size | Verdict |
|-------|------|---------|
| DeepSeek-V3 / V3.1 / V3.2 | 671 B MoE | **Does not fit** — even FP8 ≈ 671–685 GB ≫ 376 GB |
| Llama-4 Maverick | 400 B MoE | FP8 ≈ 400 GB → **over budget** (no room for KV/activations) |
| GLM-4.5 | 355 B MoE | FP8 ≈ 355 GB → technically under 376 GB but **~no headroom** → not viable; bf16 ≫ 700 GB |
| Llama-4 Scout | 109 B MoE | Fits (bf16 ≈ 218 GB) but **multimodal + weaker text** → not worth a rung |

---

## What to AVOID (compatibility, not capacity)

- **Linear-/hybrid-attention models** (e.g. Qwen3-Next "A3B" style, Gated DeltaNet +
  multi-token-prediction): there is no clean per-layer attention-residual stream to
  hook, so the extraction/steering assumptions break. Stick to standard Transformer
  decoders.
- **The old placeholder default `google/gemma-4-E4B`**: a hypothetical id with no real
  weights or tooling. The default is now `Qwen/Qwen2.5-7B-Instruct` (rung 1). If you
  see the gemma-4 id anywhere, it's a leftover.

---

## Cross-cutting caveats (read before trusting a comparison)

- **Sonnet 4.5's depth is undisclosed.** Keep `layer_fraction = 2/3`; never hard-code a
  "Sonnet layer" index. The fraction is what ports across model depths.
- **Direction comparisons are only clean on matched artifacts.** RSA/cosine against a
  released vector (assistant axis, persona vector, SAE feature) is meaningful *only* on
  the **same checkpoint, tokenizer, and layer indexing**. Mismatched layers/tokenizers
  produce numbers that look fine and mean nothing.
- **Cross-model geometry:** for comparing emotion geometry *across* rungs, mind the
  corpus decision (shared API corpus vs `self_generate`) in **Operational planning**
  above — it's both a confound control and the main GPU-cost lever.
- **Verify post-cutoff artifacts on HF before relying on them.** The assistant-axis
  tensors, Gemma Scope releases, exact FP8 packaging of the 235 B, and published param/
  layer counts may have changed — confirm on the model/dataset card.
- **Rebuild vectors at every rung.** A `ProbeBank` is stamped with its
  `source_model_id`; `ProbeBank.check_against(model)` **raises** on a layer/width
  mismatch and **warns** on an id mismatch, wherever a bank meets a model. So a reused
  bank fails loudly, never silently — but you still must regenerate.

---

## How to switch the model

> For the 14-model suite, prefer the manifest + `scripts/run_suite.sh` above —
> it sets all of the env below (and the per-model quirks) for you. The mechanism
> underneath is just these env overrides:

```bash
# One place changes everything (Config.model_id). Either edit config.py,
# or override at the entrypoint:
export EMOTION_PROBES_DATA=/data/<you>/emotion-probes      # keep big artifacts off $HOME

# Rung 1 (the default) — nothing to set. For another rung, e.g. Qwen3-32B:
python -m emotion_probes.extraction.extractor stories          # ... (see README Quickstart)
python -m emotion_probes.pipeline emotion                      # rebuild vectors for THIS model

# In Python:
from emotion_probes import Config
config = Config().with_(model_id="Qwen/Qwen3-32B", device_map="cuda")
# 70B / 235B: device_map="auto" to shard across all visible cards.
```

See [`README.md`](../README.md) for the full end-to-end Quickstart and
[`DEPLOY.md`](DEPLOY.md) for the Athena cluster runbook (proxy/HF-cache env vars,
`gpurun`, checkpointing under the 12 h-notice policy).
