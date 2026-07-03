#!/bin/bash
# Orchestrate the WHOLE 14-model suite across reserved GPUs.
#
# Scheduling (from the manifest's own single-card/sharded partition):
#   * 12 single-card models run one-per-GPU, in waves of N = #reserved cards.
#   * 2 sharded models (qwen3-235b, mistral-large-123b) run alone, device_map=auto
#     across ALL reserved cards, one after the other.
#
# This is a convenience over scripts/run_suite.sh -- for full control you can run
# individual `run_suite.sh <key>` calls in your own tmux panes instead.
#
# SAFETY: prints the plan and EXITS unless you pass CONFIRM=1. It launches many
# GPU-days of work on a SHARED cluster, so RESERVE the cards first (per the
# cluster guide) and pass exactly those ids in GPUS.
#
#   GPUS=0,1,2,3 STAGE=vectors bash scripts/run_all_suite.sh            # dry-run: print plan
#   GPUS=0,1,2,3 STAGE=vectors CONFIRM=1 nohup bash scripts/run_all_suite.sh > logs/suite_all.log 2>&1 &
#
# Env: GPUS (default 0,1,2,3), STAGE (default vectors; see run_suite.sh), CONFIRM.
set -uo pipefail

REPO="${SUITE_REPO:-/data/agastyas/emotion-probes}"
PY="${SUITE_PY:-$REPO/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
STAGE="${STAGE:-vectors}"
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p logs
IFS=',' read -ra CARDS <<< "$GPUS"; N=${#CARDS[@]}

SINGLE=$($PY -m emotion_probes.models.suite keys single)
SHARDED=$($PY -m emotion_probes.models.suite keys sharded)

echo "Suite plan: stage=$STAGE  reserved GPUs=[$GPUS] (N=$N)"
echo "Single-card (one per GPU, waves of $N):"
i=0; for k in $SINGLE; do echo "    wave $((i / N + 1))  GPU ${CARDS[$((i % N))]}  $k"; i=$((i+1)); done
echo "Sharded (all $N cards, sequential):"
for k in $SHARDED; do echo "    GPUs $GPUS  $k"; done

if [ "${CONFIRM:-0}" != "1" ]; then
  echo; echo "DRY-RUN. Re-run with CONFIRM=1 to launch. (Reserve the cards first!)"; exit 0
fi

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- single-card models: bounded parallelism, one card each, barrier per wave ---
i=0
for key in $SINGLE; do
  card=${CARDS[$((i % N))]}
  log "launch $key on GPU $card (stage=$STAGE)"
  CUDA_VISIBLE_DEVICES=$card nohup bash scripts/run_suite.sh "$key" "$STAGE" > "logs/suite_${key}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % N)) -eq 0 ]; then log "wave full ($N jobs) -- waiting"; wait; fi
done
wait  # drain any final partial wave
log "single-card models DONE"

# --- sharded models: all cards, one at a time ---
for key in $SHARDED; do
  log "launch $key sharded across [$GPUS] (stage=$STAGE)"
  CUDA_VISIBLE_DEVICES=$GPUS bash scripts/run_suite.sh "$key" "$STAGE" > "logs/suite_${key}.log" 2>&1
done
log "ALL SUITE MODELS DONE stage=$STAGE"
