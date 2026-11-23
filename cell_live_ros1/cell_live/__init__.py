"""ROS 1 (Noetic) live deployment of CELL on the drone-space EVK4 rig.

Two processes, joined by a localhost socket (cell_live.ipc), same split as
glace_live_ros1 / glace_ros1:

    CELL process (isolated conda env `cell_live`)    ROS 1 bridge process (system ROS Noetic)
    EVK4 -> event frame -> CELL -> PnP  <--IPC-->     rospy publishers -> RViz 1
                                         --IPC-->      (mocap pose -> guess, forwarded in)

The CELL side imports the deployment bundle's `code/` (CELL repo + checkpoint) read-only;
nothing there is copied or changed here. The bridge side imports only rospy, tf2_ros and
message modules.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The unpacked deployment bundle (read-only reference), nested inside this package since
# it is only ever used by this deployment. Override with CELL_LIVE_DEPLOY.
DEPLOY_ROOT = Path(os.environ.get("CELL_LIVE_DEPLOY", PROJECT_ROOT / "cell_dronespace_bundle")).resolve()
DEPLOY_CODE = DEPLOY_ROOT / "code"
CHECKPOINT = DEPLOY_ROOT / "checkpoint" / "best_model.pth"


def use_deploy_package():
    """Make the packaged CELL `code/` importable without writing into it (bytecode writing
    disabled, inherited by any spawned child)."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not (DEPLOY_CODE / "eval_full.py").is_file():
        raise SystemExit(f"deployment bundle not found at {DEPLOY_CODE} (set CELL_LIVE_DEPLOY)")
    for p in (str(DEPLOY_CODE), str(DEPLOY_CODE / "core"), str(DEPLOY_CODE / "core" / "correlation_package")):
        if p not in sys.path:
            sys.path.insert(0, p)
