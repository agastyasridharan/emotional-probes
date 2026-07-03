#!/bin/bash
# Non-blocking watcher: wait for Qwen3-32B union vectors -> launch steer2-32b ->
# wait for results -> pull to local. SSH-failure tolerant; bounded runtime.
set -u
HOST="agastyas@athena01.ig32s.ruhr-uni-bochum.de"
SSH="ssh -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o BatchMode=yes"
REMOTE="/data/agastyas/research"
VEC="$REMOTE/data-studyA-union-qwen32b/vectors/emotion_vectors.npz"
OUT="$REMOTE/studyA_steer2_32b_results.json"
LOG="$REMOTE/studyA_steer2_32b.log"
LOCAL="/Users/agastyasridharan/emotional-probes/research/results/studyA_steer2_32b_results.json"

rexec() { $SSH "$HOST" "$1" 2>/dev/null; }

echo "WATCH_START $(date +%H:%M:%S)"

# Phase 1: wait for vectors (<= 45 min)
for i in $(seq 1 90); do
  if rexec "test -f $VEC && echo yes" | grep -q yes; then echo "VECTORS_READY $(date +%H:%M:%S)"; break; fi
  sleep 30
done

# Phase 2: launch steer2-32b if needed
if rexec "test -f $OUT && echo yes" | grep -q yes; then
  echo "RESULTS_ALREADY_EXIST"
elif rexec "test -f $VEC && echo yes" | grep -q yes; then
  RUNNING=$(rexec "pgrep -f 'studyA_steer2.py' | head -1")
  if [ -z "$RUNNING" ]; then
    GPU=$(rexec "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '\$2<1000{print \$1; exit}'")
    GPU=${GPU:-2}
    echo "LAUNCHING_STEER2_32B gpu=$GPU $(date +%H:%M:%S)"
    rexec "cd /data/agastyas/emotion-probes && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/data/agastyas/emotion-probes PYTHONUNBUFFERED=1 nohup .venv/bin/python $REMOTE/gpu/studyA_steer2.py --model Qwen/Qwen3-32B --bank $VEC --out $OUT > $LOG 2>&1 & echo launched_pid \$!"
  else
    echo "STEER2_ALREADY_RUNNING pid=$RUNNING"
  fi
else
  echo "VECTORS_NEVER_APPEARED — extraction may have stalled; exiting"
  echo "WATCHER_DONE"
  exit 0
fi

# Phase 3: wait for results (<= 60 min)
for i in $(seq 1 120); do
  if rexec "test -f $OUT && echo yes" | grep -q yes; then echo "RESULTS_READY $(date +%H:%M:%S)"; break; fi
  # surface a crash early
  if rexec "tail -n 3 $LOG 2>/dev/null" | grep -qiE "traceback|error|oom|killed"; then
    echo "STEER2_ERROR_DETECTED:"; rexec "tail -n 8 $LOG"; echo "WATCHER_DONE"; exit 0
  fi
  sleep 30
done

# Phase 4: pull
if rexec "test -f $OUT && echo yes" | grep -q yes; then
  scp -o ConnectTimeout=12 -o BatchMode=yes "$HOST:$OUT" "$LOCAL" 2>/dev/null && echo "PULLED_TO_LOCAL $LOCAL" || echo "PULL_FAILED"
else
  echo "RESULTS_NEVER_APPEARED within window"
fi
echo "WATCHER_DONE $(date +%H:%M:%S)"
