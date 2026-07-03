#!/bin/bash
# Athena runner for the dtype diagnostic. GPUs 0,2,3 ONLY (never 1).
set -euo pipefail
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80
export https_proxy=$http_proxy
export no_proxy=.rub.de,.ruhr-uni-bochum.de,localhost,127.0.0.1,::1
export HF_HOME=/data/hf_cache
export HF_HUB_CACHE=/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export EMOTION_PROBES_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
export EMOTION_PROBES_DEVICE_MAP=auto
export EMOTION_PROBES_DATA=/data/agastyas/emotion-probes/data-llama70b
export CUDA_VISIBLE_DEVICES=0,2,3
cd /data/agastyas/emotion-probes
exec /data/agastyas/emotion-probes/.venv/bin/python gpu_dtype_check.py
