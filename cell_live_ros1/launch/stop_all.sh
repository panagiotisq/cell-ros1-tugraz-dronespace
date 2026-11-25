#!/usr/bin/env bash
# Stops everything started by launch/start_all.sh (or any subset of the individual start_*.sh
# scripts) -- and ONLY that: each piece is tracked by its run/<name>.pid, so a master or drivers
# that were already running before start_all.sh (no pid file) are left untouched.
#
# Stops in reverse start order and escalates SIGINT -> SIGTERM -> SIGKILL to the whole process
# group, waiting after each (roscore/roslaunch shut their nodes down cleanly on SIGINT).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

stop_one() {   # stop_one NAME
    local name=$1 f="$CLR1_RUN/$name.pid" pid
    [[ -f "$f" ]] || return 0
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
        for sig in INT TERM KILL; do
            kill -"$sig" "-$pid" 2>/dev/null || kill -"$sig" "$pid" 2>/dev/null || true
            for _ in $(seq 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done   # up to 8s
            kill -0 "$pid" 2>/dev/null || break
        done
        if kill -0 "$pid" 2>/dev/null; then echo "WARNING: $name (PID $pid) would not stop" >&2
        else echo "stopped $name (PID $pid)"; fi
    fi
    rm -f "$f"
}

any=0
for n in cell rviz bridge camera mocap master; do
    [[ -f "$CLR1_RUN/$n.pid" ]] && any=1
    stop_one "$n"
done
[[ $any -eq 0 ]] && echo "nothing to stop (no run/*.pid files)"
