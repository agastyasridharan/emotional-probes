#!/bin/bash
# One-glance status of the suite run. Usage: bash scripts/suite_status.sh
REPO="${SUITE_REPO:-/data/agastyas/emotion-probes}"
echo "=== auto-scaler ==="
pgrep -f run_suite_auto.sh >/dev/null && echo "running (pid $(pgrep -f run_suite_auto.sh))" || echo "NOT running"
echo "=== recent pool events ==="
grep -E -- "-> dispatch|<- |WARNING|=> sharded|PHASE|ALL SUITE MODELS DONE" "$REPO/logs/pool.log" 2>/dev/null | tail -10
echo "=== per-model progress (stories /171, emotion vec, deflection ckpt/extract/vec) ==="
for d in "$REPO"/suite/*/; do
  [ -d "$d" ] || continue; k=$(basename "$d")
  s=$(ls "$d"activations/stories/*.npy 2>/dev/null | wc -l)
  ev=$([ -f "$d"vectors/emotion_vectors.npz ] && echo Y || echo -)
  dc=$(ls -lh "$d"activations/deflection_checkpoint.npz 2>/dev/null | awk '{print $5}')
  df=$(ls -lh "$d"activations/deflection.npz 2>/dev/null | awk '{print $5}')
  dv=$([ -f "$d"vectors/deflection_vectors.npz ] && echo Y || echo -)
  printf "  %-18s stories=%3s/171  emoVec=%s  deflCkpt=%-5s  deflExtract=%-5s  deflVec=%s\n" \
    "$k" "$s" "$ev" "${dc:-none}" "${df:-no}" "$dv"
done
echo "=== GPUs (util, used, free) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader
echo "=== GPU owners ==="
nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | while IFS=, read -r bus pid; do
  idx=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F, -v b="$(echo "$bus"|tr -d ' ')" '{gsub(/ /,"",$2)} $2==b{print $1}')
  echo "GPU$idx=$(ps -o user= -p "$(echo "$pid"|tr -d ' ')" 2>/dev/null)"
done | sort -u
