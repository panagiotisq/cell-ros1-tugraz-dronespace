#!/usr/bin/env bash
# OPTIONAL mocap-driver launcher -- deployment-specific, and OFF unless you configure it.
# CELL only needs geometry_msgs/PoseStamped on $CLR1_MOCAP_TOPIC from ANY mocap system
# (OptiTrack mocap_optitrack, Vicon vicon_bridge, ...), so this repo ships no mocap launch
# file. Point it at the package + launch in your own driver workspace ($CLR1_EV_WS):
#
#   CLR1_MOCAP_PKG=mocap_optitrack  CLR1_MOCAP_LAUNCH=evk4.launch  launch/start_mocap.sh
#
# start_all.sh starts mocap automatically only when CLR1_MOCAP_LAUNCH is set; otherwise it
# just reuses an already-running mocap node (or warns). Writes run/mocap.pid. Extra args go
# to roslaunch. Ctrl-C stops it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${CLR1_MOCAP_LAUNCH:-}" ]]; then
    echo "no mocap launch configured. Set CLR1_MOCAP_PKG and CLR1_MOCAP_LAUNCH, or start your" >&2
    echo "mocap driver yourself so it publishes PoseStamped on $CLR1_MOCAP_TOPIC." >&2
    exit 2
fi
clr1_require_master
clr1_write_pid mocap
clr1_ros_exec roslaunch "$CLR1_MOCAP_PKG" "$CLR1_MOCAP_LAUNCH" "$@"
