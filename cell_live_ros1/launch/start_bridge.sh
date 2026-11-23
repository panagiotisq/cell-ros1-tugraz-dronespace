#!/usr/bin/env bash
# ROS 1 bridge (system ROS Noetic python3): mocap -> guess + events -> CELL process (IPC
# out), CELL results (IPC in) -> /cell/* topics. Needs the external driver stack's mocap_optitrack +
# prophesee_ros_driver ALREADY PUBLISHING on the same master (normal case: the lab's
# default master, port 11311 -- that is why --allow-default-master is passed below; this
# is the intended shared-master case, not an accident). Extra arguments go to
# scripts/run_ros1_bridge.py. Ctrl-C stops it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clr1_require_master
clr1_write_pid bridge
clr1_ros_exec python3 "$CLR1_ROOT/scripts/run_ros1_bridge.py" \
    --result-port "$CLR1_RESULT_PORT" --guess-port "$CLR1_GUESS_PORT" --events-port "$CLR1_EVENTS_PORT" \
    --allow-default-master "$@"
