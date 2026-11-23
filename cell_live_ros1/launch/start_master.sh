#!/usr/bin/env bash
# Optional: a dedicated ROS master on 127.0.0.1:$CLR1_MASTER_PORT, fully isolated from any
# other roscore. NOT needed for the normal case (attaching to the lab's already-running
# master where the external mocap + Prophesee driver stack publishes, default port 11311) -- only
# use this if you also start those driver nodes under the SAME dedicated master yourself.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if clr1_port_busy "$CLR1_MASTER_PORT"; then
    echo "port $CLR1_MASTER_PORT is already in use (another master already running?)" >&2
    exit 1
fi
clr1_write_pid master
clr1_ros_exec roscore -p "$CLR1_MASTER_PORT"
