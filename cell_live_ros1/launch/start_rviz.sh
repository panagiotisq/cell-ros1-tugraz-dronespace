#!/usr/bin/env bash
# RViz 1 showing the CELL vs mocap-guess comparison. Ctrl-C stops it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clr1_require_master
clr1_write_pid rviz
clr1_ros_exec rosrun rviz rviz -d "$CLR1_ROOT/configs/cell_live.rviz"
