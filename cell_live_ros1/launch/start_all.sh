#!/usr/bin/env bash
# Start the bridge, RViz and the CELL process in the background (in that order). Assumes
# the external mocap + Prophesee driver stack is ALREADY RUNNING against the master this uses
# (default: port 11311, the lab's normal master -- see common.sh / README.md).
#
#   launch/start_all.sh [-- <extra start_cell.sh arguments>]
#     e.g. launch/start_all.sh -- --interval-ms 500 --iters 16
#
# Logs: run/logs/<program>.log    Stop everything: launch/stop_all.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$CLR1_ROOT"

CELL_ARGS=()
if [[ "${1:-}" == "--" ]]; then shift; CELL_ARGS=("$@"); fi
LOGS="$CLR1_RUN/logs"
mkdir -p "$LOGS"

fail() { echo "ERROR: $*" >&2; echo "Already started programs are still running; stop them with launch/stop_all.sh" >&2; exit 1; }
wait_for() {   # wait_for SECONDS DESCRIPTION COMMAND...
    local t=$1 what=$2; shift 2
    for _ in $(seq $((t * 5))); do "$@" && return 0; sleep 0.2; done
    fail "$what did not come up within ${t}s"
}
alive() { clr1_pid_alive "$1"; }
start() {
    local name=$1; shift
    setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
    echo "  started $name (log: run/logs/$name.log)"
}

clr1_require_master
for p in "$CLR1_RESULT_PORT" "$CLR1_GUESS_PORT" "$CLR1_EVENTS_PORT"; do
    clr1_port_busy "$p" && { echo "port $p is in use (already running? use launch/stop_all.sh)" >&2; exit 1; }
done
if ! rosnode list 2>/dev/null | grep -qi mocap; then
    echo "WARNING: no mocap node seen on the master -- /cell/guess/* will stay empty until it starts" >&2
fi

echo "starting (master 127.0.0.1:$CLR1_MASTER_PORT)"
start bridge launch/start_bridge.sh
wait_for 15 "bridge" clr1_port_busy "$CLR1_RESULT_PORT"   # the bridge only LISTENS on RESULT_PORT; it
                                                          # connects OUT to the CELL process for guess/events,
                                                          # so GUESS/EVENTS ports bind only once start_cell.sh runs below
start rviz launch/start_rviz.sh
start cell launch/start_cell.sh "${CELL_ARGS[@]}"
wait_for 120 "CELL (model load + extensions + first tick)" grep -q -E "waiting for the bridge|Error|Traceback" "$LOGS/cell.log"
grep -q -E "Error|Traceback" "$LOGS/cell.log" && fail "CELL failed (see run/logs/cell.log)"

echo
for n in bridge rviz cell; do
    [[ -f "$CLR1_RUN/$n.pid" ]] || continue
    if alive "$n"; then printf "  %-10s running (PID %s)\n" "$n" "$(cat "$CLR1_RUN/$n.pid")"
    else printf "  %-10s NOT RUNNING (see run/logs/%s.log)\n" "$n" "$n"; fi
done
echo
echo "All started. RViz is open. Stop everything with: launch/stop_all.sh"
