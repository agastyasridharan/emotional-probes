#!/bin/bash
# Self-healing runner for the Qwen3-32B cross-scale steer2 check.
# VPN to Athena is flaky; this tolerates drops, polls sparsely (90s), relaunches
# via nohup only if nothing is running and no results yet, and pulls on completion.
set -u
HOST="agastyas@athena01.ig32s.ruhr-uni-bochum.de"
SSH="ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o BatchMode=yes"
R="/data/agastyas/research"
VEC="$R/data-studyA-union-qwen32b/vectors/emotion_vectors.npz"
OUT="$R/studyA_steer2_32b_results.json"
LOG="$R/studyA_steer2_32b.log"
LOCAL="/Users/agastyasridharan/emotional-probes/research/results/studyA_steer2_32b_results.json"
rx() { $SSH "$HOST" "$1" 2>/dev/null; }

LAUNCH="cd /data/agastyas/emotion-probes && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/agastyas/emotion-probes PYTHONUNBUFFERED=1 nohup .venv/bin/python $R/gpu/studyA_steer2.py --model Qwen/Qwen3-32B --bank $VEC --out $OUT > $LOG 2>&1 & sleep 15; echo held"

echo "RUN_START $(date +%H:%M:%S)"
for i in $(seq 1 40); do      # ~40 * 90s = 60 min cap
  # done?
  if rx "test -f $OUT && echo yes" | grep -q yes; then
    echo "RESULTS_READY iter=$i $(date +%H:%M:%S)"
    scp -o ConnectTimeout=15 -o BatchMode=yes "$HOST:$OUT" "$LOCAL" 2>/dev/null && { echo "PULLED $LOCAL"; echo RUN_DONE; exit 0; }
    echo "pull failed, will retry"; sleep 30; continue
  fi
  # alive?
  ALIVE=$(rx "pgrep -f 'studyA_steer2.py.*Qwen3-32B' | grep -v pgrep | head -1")
  if [ -n "$ALIVE" ]; then
    echo "alive iter=$i pid=$ALIVE $(rx "tail -n1 $LOG | cut -c1-70")"
  else
    # VPN reachable? only relaunch if we can actually talk to the host
    if rx "echo ping" | grep -q ping; then
      echo "RELAUNCH iter=$i $(date +%H:%M:%S)"
      rx "pkill -f 'studyA_steer2.py.*Qwen3-32B' 2>/dev/null; sleep 2; $LAUNCH" >/dev/null 2>&1
    else
      echo "vpn-unreachable iter=$i — backing off $(date +%H:%M:%S)"
    fi
  fi
  sleep 90
done
echo "RUN_TIMEOUT after cap"
rx "test -f $OUT && echo yes" | grep -q yes && scp -o ConnectTimeout=15 -o BatchMode=yes "$HOST:$OUT" "$LOCAL" 2>/dev/null && echo "PULLED_LATE $LOCAL"
echo RUN_DONE
