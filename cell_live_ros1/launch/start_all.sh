#!/usr/bin/env bash
# One command to bring up the whole live stack, checking each piece is really up before the
# next:   master -> (camera) -> (mocap) -> bridge -> RViz -> CELL.
# Anything already running on the target master (default :11311) is REUSED, not restarted. On
# any failure, everything THIS run started is torn down again (launch/stop_all.sh); pieces that
# were already running are left alone.
#
#   launch/start_all.sh [--mode A|B] [--no-drivers] [-- <extra run_cell_live.py args>]
#     --mode A | a | mocap       guess from live mocap every tick      (--guess-mode mocap)
#     --mode B | b | selftrack    self-track, mocap re-seed on loss     (--guess-mode selftrack) [default]
#     --no-drivers                do not start the camera/mocap drivers; use already-running ones only
#     -- ...                      passed verbatim to run_cell_live.py (e.g. -- --iters 16);
#                                 --guess-mode here is refused -- use --mode.
#     -h | --help
#
# Logs: run/logs/<program>.log    Stop everything: launch/stop_all.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$CLR1_ROOT"

usage() {
    cat <<'EOF'
Usage: launch/start_all.sh [--mode A|B] [--no-drivers] [-- <extra run_cell_live.py args>]
  --mode A | a | mocap       guess from live mocap every tick    (--guess-mode mocap)
  --mode B | b | selftrack    self-track, mocap re-seed on loss   (--guess-mode selftrack) [default]
  --no-drivers                do not start the camera/mocap drivers; use already-running ones only
  -- ...                      passed verbatim to run_cell_live.py (e.g. -- --iters 16);
                              --guess-mode here is refused -- use --mode.
  -h | --help
Reuses any master/camera/mocap already running; rolls back on failure. Stop: launch/stop_all.sh
EOF
}

MODE=selftrack; NO_DRIVERS=0; CELL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) shift
            case "${1:-}" in
                A|a|1|mocap)     MODE=mocap ;;
                B|b|2|selftrack) MODE=selftrack ;;
                *) echo "invalid --mode '${1:-}' (use A/mocap or B/selftrack)" >&2; exit 2 ;;
            esac ;;
        --no-drivers) NO_DRIVERS=1 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; CELL_ARGS=("$@"); break ;;
        *) echo "unknown option '$1' (see --help)" >&2; exit 2 ;;
    esac
    shift
done
if [[ ${#CELL_ARGS[@]} -gt 0 ]]; then
    for a in "${CELL_ARGS[@]}"; do
        [[ "$a" == "--guess-mode" ]] && { echo "pass the guess mode via --mode, not after --" >&2; exit 2; }
    done
fi

LOGS="$CLR1_RUN/logs"; mkdir -p "$LOGS"
STARTED=()   # names THIS run started (reused pieces are not listed)
cleanup_on_fail() {
    echo "ERROR: $*" >&2
    echo "rolling back what this run started..." >&2
    "$CLR1_ROOT/launch/stop_all.sh" || true
    exit 1
}
wait_for() {   # wait_for SECONDS DESCRIPTION COMMAND...
    local t=$1 what=$2; shift 2
    for _ in $(seq $((t * 5))); do "$@" && return 0; sleep 0.2; done
    cleanup_on_fail "$what did not come up within ${t}s"
}
start() {      # start NAME SCRIPT...  (backgrounded; waits until its pid file exists -- else a
    local name=$1; shift                 # liveness check could race ahead of the pid file and
    setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &   # leave the process running untracked)
    for _ in $(seq 25); do [[ -f "$CLR1_RUN/$name.pid" ]] && break; sleep 0.2; done
    STARTED+=("$name"); echo "  started $name (log: run/logs/$name.log)"
}

# 1. ROS master -- reuse whatever already listens (the lab's own master), else start our own.
if clr1_port_busy "$CLR1_MASTER_PORT"; then
    echo "reusing ROS master on 127.0.0.1:$CLR1_MASTER_PORT"
else
    start master launch/start_master.sh
    wait_for 10 "ROS master" clr1_port_busy "$CLR1_MASTER_PORT"
fi

# 2. event camera (this repo's own EVK4 launch) -- reuse / skip / start.
if [[ $NO_DRIVERS -eq 0 ]]; then
    if clr1_ros_has_node /prophesee_ros_publisher; then
        echo "reusing camera driver (/prophesee_ros_publisher)"
    else
        start camera launch/start_camera.sh
        wait_for 20 "event camera (is the EVK4 plugged in?)" clr1_ros_topic_live "$CLR1_CAMERA_TOPIC" 2
    fi
fi

# 3. mocap -- reuse if running; start only if a mocap launch is configured (vendor-agnostic).
if [[ $NO_DRIVERS -eq 0 ]]; then
    if clr1_ros_mocap_present; then
        echo "reusing mocap driver"
    elif [[ -n "$CLR1_MOCAP_LAUNCH" ]]; then
        start mocap launch/start_mocap.sh
    fi
fi
# mocap liveness is a WARNING, not a hard requirement: CELL seeds from mocap on the first tick
# and on every recovery, but it simply skips ticks until poses arrive rather than failing.
if ! clr1_ros_topic_live "$CLR1_MOCAP_TOPIC" 3; then
    echo "WARNING: no pose on $CLR1_MOCAP_TOPIC yet (mocap not streaming / rigid body not visible)." >&2
    echo "         CELL will skip ticks with 'no fresh mocap guess' until poses arrive." >&2
fi

# 4. our own IPC ports must be free before the bridge/CELL bind them.
for p in "$CLR1_RESULT_PORT" "$CLR1_GUESS_PORT" "$CLR1_EVENTS_PORT"; do
    clr1_port_busy "$p" && cleanup_on_fail "port $p is in use (already running? use launch/stop_all.sh)"
done

# 5. bridge -- only LISTENS on RESULT_PORT; it connects OUT to CELL for guess/events, so the
#    GUESS/EVENTS ports bind only once start_cell.sh runs below.
start bridge launch/start_bridge.sh
wait_for 15 "bridge" clr1_port_busy "$CLR1_RESULT_PORT"

# 6. RViz, then 7. CELL (guess mode from --mode, plus any extra args after --).
start rviz launch/start_rviz.sh
start cell launch/start_cell.sh --guess-mode "$MODE" "${CELL_ARGS[@]}"
wait_for 120 "CELL (model load + extensions + first tick)" \
    grep -q -E "waiting for the bridge|Error|Traceback" "$LOGS/cell.log"
grep -q -E "Error|Traceback" "$LOGS/cell.log" && cleanup_on_fail "CELL failed (see run/logs/cell.log)"

echo
echo "status (mode $( [[ $MODE == mocap ]] && echo "A / mocap" || echo "B / selftrack" )):"
for n in master camera mocap bridge rviz cell; do
    [[ -f "$CLR1_RUN/$n.pid" ]] || continue
    if clr1_pid_alive "$n"; then printf "  %-8s running (PID %s)\n" "$n" "$(cat "$CLR1_RUN/$n.pid")"
    else printf "  %-8s NOT RUNNING (see run/logs/%s.log)\n" "$n" "$n"; fi
done
echo
echo "All started. RViz is open. Stop everything with: launch/stop_all.sh"
