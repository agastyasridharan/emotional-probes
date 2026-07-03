#!/bin/bash
# Guarded, detached launch of the in-load gc-fix verification. 0,2,3 only.
set -uo pipefail
cd /data/agastyas/emotion-probes
echo "===PRELAUNCH_SMI==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for g in 0 2 3; do
  u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${u:-0}" -gt 3000 ]; then echo "ABORT_GPU_${g}_BUSY ${u}MiB"; exit 1; fi
done
echo "GPUS_0_2_3_FREE_OK"
rm -f verify_fix.log
cat > /data/agastyas/emotion-probes/run_verify.sh <<'EOF'
#!/bin/bash
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
exec /data/agastyas/emotion-probes/.venv/bin/python verify_load_fix.py
EOF
nohup bash run_verify.sh > verify_fix.log 2>&1 &
echo "LAUNCHED_PID $!"
