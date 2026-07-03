#!/bin/bash
# Generic per-model runner for the 14-model suite (docs/MODELS.md).
#
# Everything model-specific comes from the manifest
# (emotion_probes/models/suite.py) via `suite env <key>`, so there is nothing
# bespoke per model here -- this one script replaces the old run_extract_llama.sh.
#
# Usage (run DETACHED -- an ssh drop must not kill a multi-hour job):
#   nohup bash scripts/run_suite.sh <key> [stage] > suite_<key>_<stage>.log 2>&1 &
#
#   <key>   one of: qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b qwen3-235b
#                   llama3.2-1b llama3.2-3b llama3.1-8b gemma3-1b gemma3-27b
#                   olmo2-7b olmo2-32b mistral-small-24b mistral-large-123b
#   [stage] vectors (default) | emotion | deflection | analysis | alignment | freeform | all
#           (freeform reads extra flags from $FREEFORM_ARGS, e.g.
#            FREEFORM_ARGS="--quick" or FREEFORM_ARGS="--strengths 0.06 --frames self other")
#
# GPU selection is NOT hardcoded (Athena rule): reserve cards per the cluster
# guide and `export CUDA_VISIBLE_DEVICES=...` before calling. A single-card model
# wants ONE card; the two sharded models (qwen3-235b, mistral-large-123b) want
# all the cards you reserved (they load with device_map=auto).
set -uo pipefail

REPO="${SUITE_REPO:-/data/agastyas/emotion-probes}"
PY="${SUITE_PY:-$REPO/.venv/bin/python}"   # the .venv has pandas+torch+transformers; conda base lacks pandas

# --- proxy + shared HF cache (a non-interactive shell sources no profile) ---
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # only needed for gated models
# Variable-length 2048-tok dialogues fragment the CUDA caching allocator (a 1.7B
# model was seen reserving ~84GB); expandable segments keep peak reservation near
# the true footprint so the 24-32B models fit one card at the configured batch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

KEY="${1:-}"
STAGE="${2:-vectors}"
if [ -z "$KEY" ]; then echo "usage: $0 <key> [vectors|emotion|deflection|analysis|alignment|all]"; exit 2; fi

cd "$REPO"
export PYTHONPATH="$REPO"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ -x "$PY" ] || { log "ERROR: python not found/executable at $PY (set \$SUITE_PY)"; exit 4; }

# --- pull this model's settings from the manifest (the single source of truth) ---
ENV_LINES="$($PY -m emotion_probes.models.suite env "$KEY")" || exit $?
eval "$ENV_LINES"
# exported now: EMOTION_PROBES_{MODEL_ID,DEVICE_MAP,BATCH_SIZE,DEFLECTION_BATCH_SIZE,DATA},
# optionally _{ENABLE_THINKING,ATTN_IMPL}, plus SUITE_{GATED,FP8,KEY}.

# --- gated-model token guard (fail early, before a long job spins up) ---
if [ "${SUITE_GATED:-0}" = "1" ] && [ ! -s "$HF_TOKEN_PATH" ]; then
  log "ERROR: $EMOTION_PROBES_MODEL_ID is GATED but no HF token at $HF_TOKEN_PATH."
  log "       Accept the model's terms on its HF page, then run: huggingface-cli login"
  exit 3
fi

# --- FP8 checkpoints: the transformers distributed-FP8 path wants synchronous launch ---
if [ "${SUITE_FP8:-0}" = "1" ]; then export CUDA_LAUNCH_BLOCKING=1; log "FP8: CUDA_LAUNCH_BLOCKING=1"; fi

# --- isolated per-model data dir + symlink the SHARED, model-independent corpus ---
# Story/dialogue TEXT is authored by an external API model, so it is reused across
# the whole suite; only activations/vectors/analysis are model-specific and land
# in this model's own dir. (Set SUITE_CORPUS_DIR to point elsewhere.)
CORPUS="${SUITE_CORPUS_DIR:-$REPO/corpus}"
if [ ! -e "$CORPUS/stories.csv" ] && [ -e "$REPO/data/stories.csv" ]; then
  CORPUS="$REPO/data"; log "corpus not at \$SUITE_CORPUS_DIR; using existing $CORPUS"
fi
mkdir -p "$EMOTION_PROBES_DATA"
for f in stories.csv neutral_stories.csv neutral_dialogues.csv deflection_dialogues.csv deflection_pairs.json; do
  if [ ! -e "$EMOTION_PROBES_DATA/$f" ]; then
    if [ -e "$CORPUS/$f" ]; then ln -s "$CORPUS/$f" "$EMOTION_PROBES_DATA/$f"
    else log "WARN: shared corpus file missing: $CORPUS/$f (generate it, or set \$SUITE_CORPUS_DIR)"; fi
  fi
done

# sanity: sharded models need >1 visible card; single-card models want exactly one
NVIS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{print ($1==""?0:NF)}')
if [ "$EMOTION_PROBES_DEVICE_MAP" = "auto" ] && [ "$NVIS" = "1" ]; then
  log "WARN: device_map=auto but only 1 card visible -- $KEY may not fit. Reserve more cards."
fi

log "BEGIN key=$KEY stage=$STAGE model=$EMOTION_PROBES_MODEL_ID dev=$EMOTION_PROBES_DEVICE_MAP"
log "      data=$EMOTION_PROBES_DATA batch=$EMOTION_PROBES_BATCH_SIZE/$EMOTION_PROBES_DEFLECTION_BATCH_SIZE gpus=${CUDA_VISIBLE_DEVICES:-<all-visible>}"

run_emotion() {
  log "emotion 1/3: story activations (171 emotions, all layers)"; $PY -m emotion_probes.extraction.extractor stories
  log "emotion 2/3: neutral-story activations (PCA confound baseline)"; $PY -m emotion_probes.extraction.extractor neutral_stories
  log "emotion 3/3: build emotion vectors -> vectors/emotion_vectors.npz"; $PY -m emotion_probes.pipeline emotion
}
run_deflection() {
  log "deflection 1/3: neutral-dialogue activations"; $PY -m emotion_probes.extraction.extractor neutral_dialogues
  log "deflection 2/3: deflection activations (checkpointed, resumes on restart)"; $PY -m emotion_probes.extraction.extractor deflection
  log "deflection 3/3: build deflection vectors -> vectors/deflection_vectors.npz"; $PY -m emotion_probes.pipeline deflection
}
# Parts 1-2 (need the emotion/deflection vectors) and Part 3 (behaviour evals).
# Run with `|| WARN` so one fragile module (e.g. the classification smoke test, or
# chronic_state which wants a bespoke bundle) does not abort the whole stage.
ANALYSIS_MODULES="geometry intensity_templates implicit_scenarios layerwise speaker context_activation logit_lens classification human_alignment chronic_state preferences"
ALIGNMENT_MODULES="reward_hacking sycophancy blackmail post_training"
run_set() { local pkg="$1"; shift; for m in "$@"; do log "$pkg: $m"; $PY -m emotion_probes.$pkg.$m || log "WARN: $pkg.$m exited $?"; done; }
# Free-form self-attribution generation (reuses this model's emotion bank). Extra
# flags come from $FREEFORM_ARGS (word-split intentionally): --quick for the smoke,
# or e.g. --strengths 0.06 --frames self other for the 235B confirmatory subset.
run_freeform() { log "freeform: generation (args: ${FREEFORM_ARGS:-<defaults>})"; $PY -m emotion_probes.alignment.qualia_freeform ${FREEFORM_ARGS:-}; }

case "$STAGE" in
  emotion)    run_emotion ;;
  deflection) run_deflection ;;
  vectors)    run_emotion; run_deflection ;;
  analysis)   run_set analysis $ANALYSIS_MODULES ;;
  alignment)  run_set alignment $ALIGNMENT_MODULES ;;
  freeform)   run_freeform ;;
  all)        run_emotion; run_deflection; run_set analysis $ANALYSIS_MODULES; run_set alignment $ALIGNMENT_MODULES ;;
  *) echo "unknown stage $STAGE; use vectors|emotion|deflection|analysis|alignment|freeform|all"; exit 2 ;;
esac
log "DONE key=$KEY stage=$STAGE"
