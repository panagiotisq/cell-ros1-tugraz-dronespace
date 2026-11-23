#!/usr/bin/env bash
# CELL process (isolated conda env $CLR1_CONDA_ENV, default ~/anaconda3/envs/cell_live):
# EVK4 events (forwarded by the bridge over IPC) + live mocap guess -> CELL -> pose,
# sent to the ROS 1 bridge over IPC. Never touches ROS or the camera directly. Arguments
# go to scripts/run_cell_live.py, e.g.:
#   launch/start_cell.sh --interval-ms 500 --iters 16   # lighter load
# Ctrl-C stops it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ ! -x "$CLR1_CONDA_ENV/bin/python" ]]; then
    echo "CELL environment not found at $CLR1_CONDA_ENV." >&2
    echo "See README.md 'Environment setup' (mirrors glace_live_ros1's isolated-conda-env approach)." >&2
    exit 2
fi
clr1_write_pid cell
cd "$CLR1_ROOT"
clr1_cell_exec "$CLR1_ROOT/scripts/run_cell_live.py" \
    --result-port "$CLR1_RESULT_PORT" --guess-port "$CLR1_GUESS_PORT" --events-port "$CLR1_EVENTS_PORT" "$@"
