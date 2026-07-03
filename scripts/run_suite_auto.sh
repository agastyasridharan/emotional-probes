#!/bin/bash
# AUTO-SCALING suite runner for a SHARED box (some GPUs may be other users').
#
# Phase 1 (single-card models): a work-queue that dispatches each model, longest
#   first, to any GPU that is currently EMPTY of other users -- and expands
#   automatically as cards free up (so it starts on whatever is free now and
#   grabs more later, never landing on someone else's GPU).
# Phase 2 (sharded models): waits until >=3 cards are free, then runs each across
#   all currently-free cards (device_map=auto), one at a time.
#
# "Empty" = a card reporting >= MIN_FREE_MIB free; any real job uses far more than
# the slack, so this reliably distinguishes an idle card from one in use. Between
# the free-check and dispatch another user could grab a card (non-reserved
# cluster) -- the worst case is one model OOMing, which is checkpointed and re-runs.
#
# Run UNDER NOHUP -- it can live for days; every child model is itself nohup'd +
# checkpointed, so ssh drops / 12h interruptions are survivable (just restart it
# with SKIP= for finished models).
#
#   STAGE=vectors nohup bash scripts/run_suite_auto.sh > logs/pool.log 2>&1 &
#
# Env: STAGE (vectors), MIN_FREE_MIB (85000), POLL secs (60), SKIP "k1 k2".
set -uo pipefail

REPO="${SUITE_REPO:-/data/agastyas/emotion-probes}"
PY="${SUITE_PY:-$REPO/.venv/bin/python}"
STAGE="${STAGE:-vectors}"
MIN_FREE_MIB="${MIN_FREE_MIB:-85000}"
POLL="${POLL:-60}"
SKIP=" ${SKIP:-} "
ALL=(0 1 2 3)
SHARDED_NEED=3
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p logs
log() { echo "[$(date -u +%H:%M:%S)] AUTO: $*"; }

mapfile -t _SINGLE < <($PY -m emotion_probes.models.suite keys single-lpt)
mapfile -t _SHARDED < <($PY -m emotion_probes.models.suite keys sharded)
Q=();  for k in "${_SINGLE[@]}";  do [[ "$SKIP" == *" $k "* ]] && { log "skip $k"; continue; }; Q+=("$k"); done
SH=(); for k in "${_SHARDED[@]}"; do [[ "$SKIP" == *" $k "* ]] && { log "skip $k"; continue; }; SH+=("$k"); done
log "single-card queue (LPT): ${Q[*]:-<none>}"
log "sharded queue (need >=$SHARDED_NEED free cards each): ${SH[*]:-<none>}"
log "MIN_FREE_MIB=$MIN_FREE_MIB POLL=${POLL}s"

declare -A PID KEY
card_free()  { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
myjob_on()   { local g=$1; [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; }
dispatch()   { local g=$1 k=$2
  CUDA_VISIBLE_DEVICES=$g nohup bash scripts/run_suite.sh "$k" "$STAGE" >"logs/suite_${k}.log" 2>&1 &
  PID[$g]=$!; KEY[$g]=$k; log "-> dispatch $k on GPU $g (pid ${PID[$g]})"; }
reap() {
  for g in "${ALL[@]}"; do
    local p=${PID[$g]:-}; [ -z "$p" ] && continue
    kill -0 "$p" 2>/dev/null && continue
    local k=${KEY[$g]}
    if grep -q "DONE key=$k" "logs/suite_${k}.log" 2>/dev/null; then log "<- $k finished OK on GPU $g"
    else log "<- WARNING: $k on GPU $g exited without DONE (see logs/suite_${k}.log)"; fi
    unset 'PID[$g]' 'KEY[$g]'
  done
}

# ---------------- phase 1: single-card, auto-scaling ----------------
qi=0
while :; do
  reap
  for g in "${ALL[@]}"; do
    [ $qi -lt ${#Q[@]} ] || break
    myjob_on "$g" && continue                                   # already running my job
    f=$(card_free "$g"); [ "$f" -ge "$MIN_FREE_MIB" ] || continue  # busy with another user
    dispatch "$g" "${Q[$qi]}"; qi=$((qi+1))
  done
  live=0; for g in "${ALL[@]}"; do myjob_on "$g" && live=1; done
  [ $qi -ge ${#Q[@]} ] && [ $live -eq 0 ] && break
  fl=""; for g in "${ALL[@]}"; do fl+="g$g:$(card_free "$g")M$(myjob_on "$g" && echo '*') "; done
  log "phase1 ${qi}/${#Q[@]} dispatched, live=$live | $fl"
  sleep "$POLL"
done
log "PHASE 1 COMPLETE -- all single-card models finished"

# ---------------- phase 2: sharded, wait for enough free cards ----------------
for k in "${SH[@]}"; do
  log "sharded $k: need >=$SHARDED_NEED empty cards"
  while :; do
    free=(); for g in "${ALL[@]}"; do [ "$(card_free "$g")" -ge "$MIN_FREE_MIB" ] && free+=("$g"); done
    [ ${#free[@]} -ge $SHARDED_NEED ] && break
    log "  ${#free[@]}/$SHARDED_NEED free (${free[*]:-none}); waiting for a card to free up..."
    sleep "$POLL"
  done
  use=$(IFS=,; echo "${free[*]}")
  log "=> sharded $k across GPUs [$use]"
  CUDA_VISIBLE_DEVICES=$use bash scripts/run_suite.sh "$k" "$STAGE" >"logs/suite_${k}.log" 2>&1
  grep -q "DONE key=$k" "logs/suite_${k}.log" 2>/dev/null && log "<= $k done OK" || log "<= WARNING: $k exited without DONE"
done
log "ALL SUITE MODELS DONE stage=$STAGE"
