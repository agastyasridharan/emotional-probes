#!/bin/bash
# Maximally-efficient whole-suite runner: a 4-GPU WORK-QUEUE (no wave barriers).
#
# Unlike run_all_suite.sh (which waits for each wave's slowest model), this keeps
# every GPU saturated: single-card models are dispatched longest-first and the
# moment any GPU frees, the next queued model starts on it. After the single-card
# queue drains, the two sharded models (qwen3-235b, mistral-large-123b) run on ALL
# cards, one after the other (they can't share — each needs >=3 cards).
#
# Run it UNDER NOHUP -- this orchestrator lives for the whole multi-day run, and
# every child model run is itself nohup'd + checkpointed, so an ssh drop or a 12h
# interruption is survivable (restart this script; finished models are skipped via
# SKIP=, and the deflection stage resumes from its checkpoint).
#
#   GPUS=0,1,2,3 STAGE=vectors SKIP="qwen3-1.7b" \
#     nohup bash scripts/run_suite_pool.sh > logs/pool.log 2>&1 &
#
# Env: GPUS (default 0,1,2,3), STAGE (default vectors), SKIP (space-separated keys
#      to omit, e.g. ones already finished).
set -uo pipefail

REPO="${SUITE_REPO:-/data/agastyas/emotion-probes}"
PY="${SUITE_PY:-$REPO/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
STAGE="${STAGE:-vectors}"
SKIP=" ${SKIP:-} "   # padded so substring match is word-safe
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p logs
IFS=',' read -ra CARDS <<< "$GPUS"
log() { echo "[$(date -u +%H:%M:%S)] POOL: $*"; }

# --- single-card queue (longest-job-first), minus SKIP ---
QUEUE=()
for k in $($PY -m emotion_probes.models.suite keys single-lpt); do
  if [[ "$SKIP" == *" $k "* ]]; then log "skip $k (in SKIP)"; continue; fi
  QUEUE+=("$k")
done
SHARDED=$($PY -m emotion_probes.models.suite keys sharded)
log "GPUs=[${CARDS[*]}] stage=$STAGE"
log "single-card queue (LPT): ${QUEUE[*]:-<none>}"
log "sharded (sequential, all cards): $SHARDED"

declare -A PID KEY
dispatch() {  # $1=gpu  $2=key
  local g=$1 k=$2
  CUDA_VISIBLE_DEVICES=$g nohup bash scripts/run_suite.sh "$k" "$STAGE" > "logs/suite_${k}.log" 2>&1 &
  PID[$g]=$!; KEY[$g]=$k
  log "-> dispatch $k on GPU $g (pid ${PID[$g]})"
}

qi=0
for g in "${CARDS[@]}"; do                       # prime every card
  [ $qi -lt ${#QUEUE[@]} ] || break
  dispatch "$g" "${QUEUE[$qi]}"; qi=$((qi+1))
done

while :; do                                      # refill as slots free
  alive=0
  for g in "${CARDS[@]}"; do
    p=${PID[$g]:-}
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then alive=1; continue; fi
    if [ -n "${KEY[$g]:-}" ]; then log "<- ${KEY[$g]} finished on GPU $g"; unset 'KEY[$g]'; PID[$g]=""; fi
    if [ $qi -lt ${#QUEUE[@]} ]; then dispatch "$g" "${QUEUE[$qi]}"; qi=$((qi+1)); alive=1; fi
  done
  [ $alive -eq 0 ] && [ $qi -ge ${#QUEUE[@]} ] && break
  sleep 20
done
log "single-card phase COMPLETE"

for k in $SHARDED; do                            # sharded models, all cards, serial
  if [[ "$SKIP" == *" $k "* ]]; then log "skip $k (in SKIP)"; continue; fi
  log "=> sharded $k across [$GPUS]"
  CUDA_VISIBLE_DEVICES=$GPUS bash scripts/run_suite.sh "$k" "$STAGE" > "logs/suite_${k}.log" 2>&1
  log "<= sharded $k done"
done
log "ALL SUITE MODELS DONE stage=$STAGE"
