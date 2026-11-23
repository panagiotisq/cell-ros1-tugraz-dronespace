#!/usr/bin/env bash
# Stops everything started by launch/start_all.sh (or any subset of the individual
# start_*.sh scripts) -- and only that; the external driver stack's own nodes are left untouched.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for n in cell rviz bridge master; do
    f="$CLR1_RUN/$n.pid"
    [[ -f "$f" ]] || continue
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        echo "stopped $n (PID $pid)"
    fi
    rm -f "$f"
done
