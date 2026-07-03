#!/bin/bash
# Guarded, detached launch of the 70B smoke test on Athena.
# Re-verifies GPUs 0,2,3 are free right before launch (caution policy).
# GPU 1 is intentionally NOT inspected and NOT used.
set -uo pipefail
cd /data/agastyas/emotion-probes

echo "===PRELAUNCH_SMI==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

for g in 0 2 3; do
  u=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${u:-0}" -gt 3000 ]; then
    echo "ABORT_GPU_${g}_BUSY ${u}MiB"
    exit 1
  fi
done
echo "GPUS_0_2_3_FREE_OK"

rm -f smoke_llama.log smoke_smi.log
nohup bash run_smoke_llama.sh > smoke_llama.log 2>&1 &
echo "LAUNCHED_PID $!"
