#!/bin/bash
# ---------------------------------------------------------------------------
# publish_qualia_dashboard.sh
#
# Publishes qualia_dashboard.html to the `gh-pages` branch so it goes live at
# https://agastyasridharan.github.io/emotional-probes/
#
# Triggered automatically by the launchd WatchPaths agent
#   ~/Library/LaunchAgents/com.agastya.qualia-dashboard-publish.plist
# whenever qualia_dashboard.html changes. Can also be run by hand.
#
# Design notes:
#   * Publishing happens in a DEDICATED gh-pages worktree so the messy `main`
#     working tree is never touched.
#   * Pushes over SSH (git@github.com) so it works non-interactively under
#     launchd without depending on the macOS keychain / gh credential helper.
#   * Idempotent: commits + pushes only when index.html actually changed.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="/Users/agastyasridharan/emotional-probes"
WT="/Users/agastyasridharan/emotional-probes-ghpages"
SRC="$REPO/qualia_dashboard.html"
SRC_ALL="$REPO/qualia_dashboard_all_models.html"   # full 13-model build, linked from the flagship
REMOTE_URL="git@github.com:agastyasridharan/emotional-probes.git"
BRANCH="gh-pages"
GIT="/usr/bin/git"
LOG="$HOME/Library/Logs/qualia-dashboard-publish.log"

mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG"; }

# Let a burst of writes (a rebuild) settle before copying.
sleep 3

[ -f "$SRC" ] || { log "source missing: $SRC"; exit 0; }
[ -e "$WT/.git" ] || { log "gh-pages worktree missing: $WT (run setup)"; exit 1; }

cp "$SRC" "$WT/index.html"
"$GIT" -C "$WT" add index.html
# The flagship links to the full all-models dashboard; publish it alongside (same
# filename) so that link resolves at .../emotional-probes/qualia_dashboard_all_models.html
if [ -f "$SRC_ALL" ]; then
    cp "$SRC_ALL" "$WT/qualia_dashboard_all_models.html"
    "$GIT" -C "$WT" add qualia_dashboard_all_models.html
fi

if "$GIT" -C "$WT" diff --cached --quiet; then
    log "no change in index.html; nothing to publish"
    exit 0
fi

sha=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
"$GIT" -C "$WT" \
    -c user.name="Agastya Sridharan" \
    -c user.email="agastya.sridharan@gmail.com" \
    commit -q -m "Auto-publish dashboard ($(date '+%Y-%m-%d %H:%M:%S'), sha ${sha:0:12})"

if "$GIT" -C "$WT" push "$REMOTE_URL" "$BRANCH" >>"$LOG" 2>&1; then
    log "published $("$GIT" -C "$WT" rev-parse --short HEAD) -> $BRANCH"
else
    log "PUSH FAILED for $("$GIT" -C "$WT" rev-parse --short HEAD) (see git output above)"
    exit 1
fi
