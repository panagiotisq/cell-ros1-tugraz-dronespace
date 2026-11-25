#!/usr/bin/env bash
# EVK4 event-camera driver, using THIS repo's own launch file (event_delta_t 10 ms,
# configs/gs_bright.bias -> /prophesee/camera/cd_events_buffer). Needs a master (reused, or
# launch/start_master.sh) and the Prophesee ROS1 driver + Metavision SDK installed (see
# README, "What you need to bring yourself"). Readiness = messages on $CLR1_CAMERA_TOPIC
# (the driver only publishes camera_info once the camera is actually opened). Writes
# run/camera.pid. Extra args go to roslaunch, e.g. bias_file:=/path/to/your.bias. Ctrl-C stops it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clr1_require_master
clr1_write_pid camera
clr1_ros_exec roslaunch "$CLR1_ROOT/launch/prophesee_for_cell_live.launch" "$@"
