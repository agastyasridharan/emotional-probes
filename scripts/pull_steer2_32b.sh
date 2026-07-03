#!/usr/bin/env bash
# Minimal VPN-tolerant fetch: the 32B job is DONE and its results JSON is complete
# and stable on the remote. Just retry scp every 300s until it lands, then exit.
# No process-checking (avoids the pgrep self-match false-positive that stalled v1).
set -u
HOST=agastyas@athena01.ig32s.ruhr-uni-bochum.de
REMOTE=/data/agastyas/research/studyA_steer2_32b_results.json
LOCAL=/Users/agastyasridharan/emotional-probes/research/results/studyA_steer2_32b_results.json
log(){ echo "$(date +%H:%M:%S) $*"; }
log "FETCH_START (waiting for VPN to pull a finished 189KB file)"
for i in $(seq 1 30); do
  if scp -o ConnectTimeout=15 -o BatchMode=yes "$HOST:$REMOTE" "$LOCAL" 2>/dev/null; then
    SZ=$(stat -f %z "$LOCAL" 2>/dev/null)
    log "FETCHED iter=$i -> $LOCAL ($SZ bytes)"
    # sanity: must be valid JSON with records
    if /usr/bin/python3 -c "import json,sys; d=json.load(open('$LOCAL')); print('records=%d baseline=%d model=%s'%(len(d.get('records',[])), len(d.get('baseline',[])), d.get('model','?')))" 2>/dev/null; then
      log "FETCH_OK"
      exit 0
    else
      log "fetched but JSON invalid/incomplete; retrying"
    fi
  else
    log "iter=$i vpn-down/no-response"
  fi
  sleep 300
done
log "FETCH_TIMEOUT"
