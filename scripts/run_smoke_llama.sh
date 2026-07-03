#!/bin/bash
# Athena runner for the Llama-3.3-70B smoke test. Pins GPUs 0,2,3 ONLY (never 1).
# CUDA_VISIBLE_DEVICES is set HERE (the shell), never in the .py — per Athena rules.
set -euo pipefail

# internet egress (proxy) + shared HF cache + gated token
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token

# pipeline knobs for the 70B run (isolated data dir; sharded across visible GPUs)
export EMOTION_PROBES_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
export EMOTION_PROBES_DEVICE_MAP=auto
export EMOTION_PROBES_DATA=/data/agastyas/emotion-probes/data-llama70b

# *** GPU policy: only 0, 2, 3 are allowed; GPU 1 must never be touched ***
export CUDA_VISIBLE_DEVICES=0,2,3

cd /data/agastyas/emotion-probes
exec /data/agastyas/emotion-probes/.venv/bin/python gpu_smoke_llama.py
