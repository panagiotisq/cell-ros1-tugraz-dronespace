"""One-shot, latched publish of the aligned drone-space map as a coloured PointCloud2, for
RViz reference only (CELL itself loads the PLY directly in the CELL process -- this is a
purely visual convenience on the bridge side). Needs numpy, which the bridge side otherwise
avoids (see ros1_bridge.py's docstring) -- fine here since this runs in the system ROS
python3, not shared with the CELL process; numpy is a system dist-packages dependency on
this PC (confirmed importable under the same sanitized env launch/common.sh uses)."""

from __future__ import annotations

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

from .map_loader import VERTEX_DTYPE


def _load_xyzrgb(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = next((int(h.split()[-1]) for h in header if h.startswith("element vertex")), None)
        if n is None:
            raise ValueError(f"{path}: no 'element vertex' line in header")
        data = np.fromfile(f, dtype=VERTEX_DTYPE, count=n)
    return data


_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def build_cloud_msg(path, frame_id):
    """PLY (x,y,z,nx,ny,nz,r,g,b per vertex) -> sensor_msgs/PointCloud2 (x,y,z,rgb), built
    directly with numpy (not sensor_msgs.point_cloud2.create_cloud's per-point python loop,
    which would take too long for ~1.2M points)."""
    data = _load_xyzrgb(path)
    n = data.shape[0]
    rgb_u32 = (data["r"].astype(np.uint32) << 16) | (data["g"].astype(np.uint32) << 8) | data["b"].astype(np.uint32)
    rgb_f32 = rgb_u32.view(np.float32)             # reinterpret the bit pattern (standard PCL/RViz packing)

    out = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out["x"], out["y"], out["z"] = data["x"].astype(np.float32), data["y"].astype(np.float32), data["z"].astype(np.float32)
    out["rgb"] = rgb_f32

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.fields = _FIELDS
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = out.tobytes()
    return msg


def publish_static_map(path, frame_id, topic="/cell/map", log=rospy.loginfo):
    """Publish once, latched (queue_size=1, latch=True): late RViz subscribers still get it."""
    log(f"loading map for RViz from {path} ...")
    msg = build_cloud_msg(path, frame_id)
    pub = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    log(f"published {msg.width} points to {topic} (latched)")
    return pub        # keep a reference alive for the process lifetime
