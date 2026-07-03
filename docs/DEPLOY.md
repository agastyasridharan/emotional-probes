# Deploying to a GPU cluster (Athena H100s)

This guide is for running the GPU stages (extraction, generation-by-self,
analysis, alignment, visualiser) on the **Athena** cluster — 4× NVIDIA H100 NVL,
shared. The maths and tests run locally; only forward passes need this.

> The author has Athena access. **Nothing in this repo runs anything on the
> cluster automatically** — every command below is for *you* to run.

## Why not just `gpurun script.py`?

`gpurun` uploads **only the single `.py`** you point it at — helper modules and
data next to it are *not* sent. This project is a multi-module package, so
`gpurun` alone can't run it. Instead: put the **whole package** on `/data` once,
then run modules with the cluster's Python. (`gpurun` is still handy for a quick
one-file probe, and it conveniently sets all the env vars for you.)

## 1. Get the code onto `/data` (not `$HOME`)

`$HOME` is capped at **10 GiB**; `/data/agastyas` is a **14 TB** scratch volume.
All big artifacts must live on `/data`.

```bash
# from the Mac, in the repo root
rsync -av --exclude '.git' --exclude 'data' --exclude '__pycache__' \
  ./ agastyas@athena01.ig32s.ruhr-uni-bochum.de:/data/agastyas/emotion-probes/
```

## 2. The environment block (required for direct SSH / detached runs)

A non-interactive shell does **not** source the login profile, so without these,
HuggingFace downloads hang forever (the cluster has no direct internet — only the
proxy) and the cache/token break. Put this at the top of every detached session:

```bash
# internet egress — cluster reaches the web only through this proxy
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
# shared HF cache (many models already downloaded)
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # only needed for gated models

# this project's settings
export EMOTION_PROBES_DATA=/data/agastyas/emotion-probes/data   # keep artifacts on /data
export PYTHONPATH=/data/agastyas/emotion-probes                 # find the package
CLUSTER_PY=/data/agastyas/Miniconda3/bin/python                 # conda base: torch 2.12, transformers 5.12
```

## 3. GPU selection

- **Never** put `os.environ["CUDA_VISIBLE_DEVICES"]=...` in code — this repo
  doesn't, and you shouldn't add it. The launcher chooses the GPU.
- For a single GPU, leave `Config.device_map = "cuda"` and let
  `CUDA_VISIBLE_DEVICES` (set by you / `gpurun`) pick the device.
- To shard a large model across all 4 H100s, run with
  `Config().with_(device_map="auto")` — e.g. pass `--model` and edit, or set it
  in a tiny launcher script.
- The cluster `nvidia-smi`-checks and refuses a GPU another user is on; reserve
  per the cluster guide for long jobs.

## 4. Run the pipeline (detached + checkpointed)

Jobs must survive interruption on **12h notice**, and `gpurun`/SSH is synchronous
(drops if the connection dies), so run long stages under `tmux`:

```bash
ssh agastyas@athena01.ig32s.ruhr-uni-bochum.de
tmux new -s probes
# paste the environment block from §2, then:

cd /data/agastyas/emotion-probes

# Stage 1 — emotion vectors
$CLUSTER_PY -m emotion_probes.generation.generators emotion_stories
$CLUSTER_PY -m emotion_probes.generation.generators neutral_stories
$CLUSTER_PY -m emotion_probes.extraction.extractor stories
$CLUSTER_PY -m emotion_probes.extraction.extractor neutral_stories
$CLUSTER_PY -m emotion_probes.pipeline emotion

# Stage 2 — deflection vectors (needs the emotion vectors from Stage 1)
$CLUSTER_PY -m emotion_probes.generation.pair_selection
$CLUSTER_PY -m emotion_probes.generation.generators neutral_dialogues
$CLUSTER_PY -m emotion_probes.generation.generators deflection
$CLUSTER_PY -m emotion_probes.extraction.extractor neutral_dialogues
$CLUSTER_PY -m emotion_probes.extraction.extractor deflection
$CLUSTER_PY -m emotion_probes.pipeline deflection
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t probes`. The deflection
extractor checkpoints every `config.checkpoint_every` batches, so an interrupted
run resumes from its last checkpoint under `EMOTION_PROBES_DATA/activations/`.

> **Self-generation (Issue #2):** to have the *probed* model write its own
> stories (the paper's design), set `EMOTION_PROBES_SELF_GENERATE=1` before the
> generation steps. Otherwise generation uses the external API (`pydantic-ai`),
> which also needs the proxy from §2 and your API key.

## 5. Run the experiments

Each experiment loads the model + a probe bank and writes JSON/plots under
`EMOTION_PROBES_DATA/analysis/`:

```bash
$CLUSTER_PY -m emotion_probes.analysis.logit_lens
$CLUSTER_PY -m emotion_probes.analysis.preferences
$CLUSTER_PY -m emotion_probes.alignment.reward_hacking
# ...etc; see docs/ARCHITECTURE.md for the full list
```

`GeometryAnalysis` needs no GPU — you can run it from the saved bank on the Mac.

## 6. The visualiser (port-forwarded)

```bash
# on the cluster (inside tmux, env block loaded):
$CLUSTER_PY visualise.py            # serves on config.visualiser_port (8080)

# on the Mac:
ssh -L 8080:localhost:8080 agastyas@athena01.ig32s.ruhr-uni-bochum.de
# then open http://localhost:8080
```

## 7. Copy results back

Big files stay on `/data`; pull the small outputs (vectors, analysis JSON/PNGs):

```bash
# from the Mac
scp -r agastyas@athena01.ig32s.ruhr-uni-bochum.de:/data/agastyas/emotion-probes/data/analysis ./data/
scp -r agastyas@athena01.ig32s.ruhr-uni-bochum.de:/data/agastyas/emotion-probes/data/vectors  ./data/
```

## Checklist

- [ ] Code on `/data`, **not** `$HOME` (10 GiB cap).
- [ ] `EMOTION_PROBES_DATA` points to `/data/...` so artifacts go there.
- [ ] Env block (proxy + HF cache) exported in the session.
- [ ] No hard-coded `CUDA_VISIBLE_DEVICES`; `device_map="auto"` only when sharding.
- [ ] Long jobs under `tmux`; rely on the deflection checkpointing for 12h interrupts.
