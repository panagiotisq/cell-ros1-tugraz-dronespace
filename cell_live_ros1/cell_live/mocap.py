"""Mocap pose handling for the visualization adapter. Standard library only (no numpy), so it
runs in the system ROS Noetic python and is unit-testable without ROS.

Frames (see README.md, "Mocap"):
    M  mocap world `world` published by the lab's mocap_optitrack node (Motive Y-up already
       converted to ROS Z-up by that node; metres)
    B  tracked rigid body (e.g. OptiTrack body 33 -> child frame `evk4/base_link`)
    C  EVK4 optical frame (x right, y down, z forward), as GLACE uses
    D  GLACE training world `dronespace_world`

    T_M<-C = T_M<-B . T_B<-C     only with an explicitly supplied extrinsic T_B<-C
A pose of B is NOT a pose of C; nothing here relates M and D.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

from . import geometry


def matrix_from_pose(position, orientation):
    """(x, y, z), (qx, qy, qz, qw) -> 4x4 rows. The quaternion is normalized; a zero
    quaternion raises ValueError."""
    qx, qy, qz, qw = (float(v) for v in orientation)
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("invalid quaternion")
    R = geometry.rotation_from_quaternion((qx / n, qy / n, qz / n, qw / n))
    x, y, z = (float(v) for v in position)
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError("non-finite position")
    return [R[0] + [x], R[1] + [y], R[2] + [z], [0.0, 0.0, 0.0, 1.0]]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def load_extrinsic(path):
    """T_B<-C from a text file: 16 numbers (4x4, row-major; '#' comments allowed). Must be a
    rigid transform. Returns (T, note), note = the file's comment lines (provenance)."""
    numbers, notes = [], []
    for raw in Path(path).read_text().splitlines():
        line, _, comment = raw.partition("#")
        if comment.strip():
            notes.append(comment.strip())
        numbers += [float(v) for v in line.replace(",", " ").replace("[", " ").replace("]", " ").split()]
    if len(numbers) != 16:
        raise ValueError(f"{path}: expected 16 numbers (4x4 T_B<-C), found {len(numbers)}")
    T = geometry.as_rows(numbers)
    # Hand-eye results are usually printed with ~6 digits: allow that rounding, then project
    # the rotation onto SO(3) (via its quaternion) so that every published pose is exactly rigid
    # and the TF / PoseStamped of T_M<-B . T_B<-C equal the matrix product.
    ok, reason = geometry.rigid_check(T, tol=1e-4)
    if not ok:
        raise ValueError(f"{path}: not a rigid transform ({reason})")
    R = geometry.rotation_from_quaternion(geometry.quaternion_from_rotation([r[:3] for r in T[:3]]))
    T = [R[0] + [T[0][3]], R[1] + [T[1][3]], R[2] + [T[2][3]], [0.0, 0.0, 0.0, 1.0]]
    return T, " | ".join(notes)


class TrackingMonitor:
    """Rate and freshness of a pose stream. mocap_optitrack publishes only frames in which
    the body was tracked, so a lost body shows up as a gap, not as an invalid message."""

    def __init__(self, lost_after_s=0.2, window=200):
        self.lost_after_s = lost_after_s
        self._t = collections.deque(maxlen=window)
        self.count = 0
        self.gaps = 0
        self.longest_gap_s = 0.0

    def update(self, t_s):
        if self._t:
            gap = t_s - self._t[-1]
            if gap > self.lost_after_s:
                self.gaps += 1
                self.longest_gap_s = max(self.longest_gap_s, gap)
        self._t.append(t_s)
        self.count += 1

    def rate_hz(self):
        if len(self._t) < 2 or self._t[-1] <= self._t[0]:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])

    def age_s(self, now_s):
        return float("inf") if not self._t else now_s - self._t[-1]

    def tracking(self, now_s):
        return self.age_s(now_s) <= self.lost_after_s


class Decimator:
    """Keep a path point only when the pose moved (or after max_interval_s), so a 100 Hz
    stream gives a readable, bounded RViz path."""

    def __init__(self, min_step_m=0.005, max_interval_s=0.5):
        self.min_step_m, self.max_interval_s = min_step_m, max_interval_s
        self._last = None

    def accept(self, t_s, xyz):
        if self._last is None:
            self._last = (t_s, xyz)
            return True
        t0, p0 = self._last
        if math.dist(p0, xyz) >= self.min_step_m or t_s - t0 >= self.max_interval_s:
            self._last = (t_s, xyz)
            return True
        return False
