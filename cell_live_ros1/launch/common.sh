# shellcheck shell=bash
# Sourced by the launch scripts. Builds clean, *process-local* environments with `env -i`:
# nothing persistent is changed (no ~/.bashrc edits; the caller's shell is not modified).
# Mirrors glace_live_ros1/launch/common.sh (proven on this PC) with CELL's own env/ports.
#
# Dropped for every process: PYTHONPATH, LD_LIBRARY_PATH, ROS_* and CMAKE_PREFIX_PATH from
# ~/.bashrc, CUDA_HOME, conda and VS Code/snap variables.
#   ROS processes:   /opt/ros/noetic/setup.bash only (no catkin overlay unless the lab's own Prophesee+mocap ROS1 workspace is
#                    sourced too -- mocap_optitrack / prophesee_ros_driver live there), its own
#                    dedicated master on 127.0.0.1:$CLR1_MASTER_PORT, ROS logs in run/ros_home.
#   CELL process:    only the isolated conda env's python; no ROS, no user site.

CLR1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLR1_MASTER_PORT="${CLR1_MASTER_PORT:-11311}"            # default: the lab's own master (the external
                                                          # mocap + Prophesee driver stack must already publish there)
CLR1_RESULT_PORT="${CLR1_RESULT_PORT:-48811}"
CLR1_GUESS_PORT="${CLR1_GUESS_PORT:-48812}"
CLR1_EVENTS_PORT="${CLR1_EVENTS_PORT:-48813}"
CLR1_RUN="$CLR1_ROOT/run"
CLR1_EV_WS="${CLR1_EV_WS:-$HOME/workspaces/ros/prophesee_mocap_ws}"   # your own ROS1 driver workspace (mocap + Prophesee), if it has one
if [[ -z "${CLR1_CONDA_ENV:-}" ]]; then
    CLR1_CONDA_ENV="$HOME/anaconda3/envs/cell_live"
    for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
        [[ -x "$base/envs/cell_live/bin/python" ]] && { CLR1_CONDA_ENV="$base/envs/cell_live"; break; }
    done
fi
mkdir -p "$CLR1_RUN/ros_home"

_clr1_base_env() {
    CLR1_BASE_ENV=(HOME="$HOME" USER="$(id -un)" LOGNAME="$(id -un)" LANG="${LANG:-C.UTF-8}"
                   TERM="${TERM:-xterm-256color}" SHELL=/bin/bash PATH=/usr/bin:/bin
                   CLR1_RESULT_PORT="$CLR1_RESULT_PORT" CLR1_GUESS_PORT="$CLR1_GUESS_PORT"
                   CLR1_EVENTS_PORT="$CLR1_EVENTS_PORT" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
                   PYTHONUNBUFFERED=1)
    [[ -n "${DISPLAY:-}" ]] && CLR1_BASE_ENV+=(DISPLAY="$DISPLAY")
    [[ -n "${XAUTHORITY:-}" ]] && CLR1_BASE_ENV+=(XAUTHORITY="$XAUTHORITY")
    [[ -n "${XDG_RUNTIME_DIR:-}" ]] && CLR1_BASE_ENV+=(XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR")
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && CLR1_BASE_ENV+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
    return 0
}

# clr1_ros_exec CMD [ARGS...]: replace this shell by CMD in a ROS-Noetic-only environment,
# with the external driver workspace's devel overlay also sourced (mocap_optitrack, prophesee_ros_driver msgs).
clr1_ros_exec() {
    _clr1_base_env
    exec env -i "${CLR1_BASE_ENV[@]}" \
        ROS_MASTER_URI="http://127.0.0.1:$CLR1_MASTER_PORT" ROS_IP=127.0.0.1 ROS_HOME="$CLR1_RUN/ros_home" \
        bash --noprofile --norc -c '
            source /opt/ros/noetic/setup.bash
            [[ -f "'"$CLR1_EV_WS"'/devel/setup.bash" ]] && source "'"$CLR1_EV_WS"'/devel/setup.bash"
            exec "$@"' clr1-ros "$@"
}

# clr1_cell_exec ARGS...: the isolated CELL conda env's python, nothing else.
clr1_cell_exec() {
    _clr1_base_env
    local base=("${CLR1_BASE_ENV[@]}")
    base=("${base[@]/#PATH=*/PATH=$CLR1_CONDA_ENV/bin:/usr/bin:/bin}")
    exec env -i "${base[@]}" CONDA_PREFIX="$CLR1_CONDA_ENV" LD_LIBRARY_PATH="$CLR1_CONDA_ENV/lib" \
        "$CLR1_CONDA_ENV/bin/python" "$@"
}

clr1_port_busy() {   # 0 if something listens on TCP port $1
    ss -Hltn "sport = :$1" 2>/dev/null | grep -q .
}

clr1_pid_alive() {   # clr1_pid_alive NAME -> 0 if run/NAME.pid names a live process
    local f="$CLR1_RUN/$1.pid"
    [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null
}

clr1_write_pid() {   # clr1_write_pid NAME: record this shell's PID (kept through exec)
    if clr1_pid_alive "$1"; then
        echo "$1 already running (PID $(cat "$CLR1_RUN/$1.pid")); stop it with launch/stop_all.sh" >&2
        exit 1
    fi
    echo $$ > "$CLR1_RUN/$1.pid"
}

clr1_require_master() {   # any master listening on 127.0.0.1:$CLR1_MASTER_PORT (ours or the lab's)
    if ! clr1_port_busy "$CLR1_MASTER_PORT"; then
        echo "no ROS master on port $CLR1_MASTER_PORT." >&2
        echo "Either the lab's own master is already up on this port (normal case: the external" >&2
        echo "mocap + Prophesee driver stack should already be running against it), or start one with" >&2
        echo "launch/start_master.sh (set CLR1_MASTER_PORT to match whichever master has mocap+events)." >&2
        exit 1
    fi
}
