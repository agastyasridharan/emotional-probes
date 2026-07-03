#!/bin/bash
# Full Llama-3.3-70B extraction on Athena. GPUs 0,2,3 ONLY (never GPU 1).
# Sharded with device_map="auto"; writes to the isolated data-llama70b dir so the
# completed Qwen run is untouched. Checkpointed (deflection resumes from its .npz).
#
# Usage (run DETACHED so an ssh drop can't kill a multi-day job):
#   nohup bash run_extract_llama.sh emotion    > extract_llama_emotion.log    2>&1 &
#   nohup bash run_extract_llama.sh deflection > extract_llama_deflection.log 2>&1 &
#   nohup bash run_extract_llama.sh all        > extract_llama_all.log        2>&1 &
set -uo pipefail

# internet egress (proxy) + shared HF cache + gated token
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token

# pipeline knobs (isolated data dir; shard across the visible GPUs)
export EMOTION_PROBES_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
export EMOTION_PROBES_DEVICE_MAP=auto
export EMOTION_PROBES_DATA=/data/agastyas/emotion-probes/data-llama70b

# *** GPU policy: only 0, 2, 3 are allowed; GPU 1 must never be touched ***
export CUDA_VISIBLE_DEVICES=0,2,3

cd /data/agastyas/emotion-probes
PY=/data/agastyas/emotion-probes/.venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_emotion() {
  log "STAGE1 a/3: story activations (171 emotions, all layers)"
  $PY -m emotion_probes.extraction.extractor stories
  log "STAGE1 b/3: neutral-story activations (PCA confound baseline)"
  $PY -m emotion_probes.extraction.extractor neutral_stories
  log "STAGE1 c/3: build emotion vectors -> vectors/emotion_vectors.npz"
  $PY -m emotion_probes.pipeline emotion
  log "STAGE1_DONE"
}

run_deflection() {
  log "STAGE2 a/3: neutral-dialogue activations"
  $PY -m emotion_probes.extraction.extractor neutral_dialogues
  log "STAGE2 b/3: deflection activations (checkpointed every 4000 dialogues)"
  $PY -m emotion_probes.extraction.extractor deflection
  log "STAGE2 c/3: build deflection vectors -> vectors/deflection_vectors.npz"
  $PY -m emotion_probes.pipeline deflection
  log "STAGE2_DONE"
}

STAGE="${1:-all}"
log "BEGIN stage=$STAGE model=$EMOTION_PROBES_MODEL_ID data=$EMOTION_PROBES_DATA gpus=$CUDA_VISIBLE_DEVICES"
case "$STAGE" in
  emotion)    run_emotion ;;
  deflection) run_deflection ;;
  all)        run_emotion; run_deflection ;;
  *) echo "usage: $0 {emotion|deflection|all}"; exit 2 ;;
esac
log "EXTRACT_COMPLETE stage=$STAGE"
