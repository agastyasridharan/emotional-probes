#!/bin/bash
# Opportunistic, GENTLE runner for the 32B cross-scale check. The Athena VPN is
# dropping connections in ~1-2s. This does NOT hammer it: one short SSH every 4
# minutes. When it catches a window, it launches the job in a tmux session
# (daemonized — survives SSH drops once started) and pulls results when ready.
set -u
HOST="agastyas@athena01.ig32s.ruhr-uni-bochum.de"
SSH="ssh -o ConnectTimeout=12 -o ServerAliveInterval=8 -o ServerAliveCountMax=2 -o BatchMode=yes"
R="/data/agastyas/research"
VEC="$R/data-studyA-union-qwen32b/vectors/emotion_vectors.npz"
OUT="$R/studyA_steer2_32b_results.json"
LOG="$R/studyA_steer2_32b.log"
LOCAL="/Users/agastyasridharan/emotional-probes/research/results/studyA_steer2_32b_results.json"

JOB="cd /data/agastyas/emotion-probes && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/agastyas/emotion-probes PYTHONUNBUFFERED=1 .venv/bin/python $R/gpu/studyA_steer2.py --model Qwen/Qwen3-32B --bank $VEC --out $OUT > $LOG 2>&1"

echo "OPP_START $(date +%H:%M:%S)"
for i in $(seq 1 14); do   # 14 * 240s = ~56 min
  # Single compact SSH: report state, and launch via tmux iff needed.
  STATE=$($SSH "$HOST" "
    if [ -f $OUT ]; then echo DONE; exit 0; fi
    if tmux has-session -t s32 2>/dev/null; then echo RUNNING; exit 0; fi
    pkill -9 -f studyA_steer2.py 2>/dev/null; sleep 1
    tmux new-session -d -s s32 \"$JOB\" 2>/dev/null && echo LAUNCHED || echo LAUNCHFAIL
  " 2>/dev/null)
  RC=$?
  if [ -z "$STATE" ]; then echo "iter=$i vpn-down (no response) $(date +%H:%M:%S)"; sleep 240; continue; fi
  echo "iter=$i state=$STATE $(date +%H:%M:%S)"
  if [ "$STATE" = "DONE" ]; then
    scp -o ConnectTimeout=12 -o BatchMode=yes "$HOST:$OUT" "$LOCAL" 2>/dev/null && { echo "PULLED $LOCAL"; echo OPP_DONE; exit 0; }
    echo "pull failed; retry next iter"
  fi
  sleep 240
done
echo "OPP_TIMEOUT $(date +%H:%M:%S)"
# last-chance pull
$SSH "$HOST" "test -f $OUT && echo yes" 2>/dev/null | grep -q yes && scp -o ConnectTimeout=12 -o BatchMode=yes "$HOST:$OUT" "$LOCAL" 2>/dev/null && echo "PULLED_LATE $LOCAL"
echo OPP_DONE
