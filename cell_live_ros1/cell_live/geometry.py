"""Pose helpers without numpy (the ROS 1 bridge must not depend on numpy).

Convention, unchanged from glace_live: a pose is the 4x4 matrix T_W<-C (row-major, 16
floats) mapping camera coordinates to world coordinates, i.e. the camera pose in the
world, exactly as DSAC* returns it. The camera frame is the EVK4 optical frame (OpenCV:
x right, y down, z forward); the world is the DroneSpace training world (z up).
"""

import math


def as_rows(flat16):
    """16 numbers (row-major) -> 4x4 list of float rows."""
    v = [float(x) for x in flat16]
    if len(v) != 16:
        raise ValueError("pose must have 16 elements")
    return [v[0:4], v[4:8], v[8:12], v[12:16]]


def rigid_check(T, tol=1e-3):
    """Same test as glace_live.localizer.pose_is_valid (finite, orthonormal rotation with
    det +1, last row 0 0 0 1). Returns (ok, reason)."""
    if len(T) != 4 or any(len(r) != 4 for r in T):
        return False, "pose is not 4x4"
    if not all(math.isfinite(x) for r in T for x in r):
        return False, "non-finite pose"
    R = [r[:3] for r in T[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))        # (R^T R)_ij
            if abs(dot - (1.0 if i == j else 0.0)) > tol:
                return False, "rotation not orthonormal"
    if abs(det3(R) - 1.0) > tol:
        return False, "rotation not orthonormal"
    if any(abs(a - b) > 1e-8 for a, b in zip(T[3], (0.0, 0.0, 0.0, 1.0))):
        return False, "invalid last row"
    return True, ""


def det3(R):
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
            - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))


def quaternion_from_rotation(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method, numerically
    stable for all angles incl. 180 deg); w >= 0, explicitly normalized."""
    m00, m01, m02 = R[0][0], R[0][1], R[0][2]
    m10, m11, m12 = R[1][0], R[1][1], R[1][2]
    m20, m21, m22 = R[2][0], R[2][1], R[2][2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return x, y, z, w


def rotation_from_quaternion(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def transform_point(T, p):
    return [sum(T[i][k] * p[k] for k in range(3)) + T[i][3] for i in range(3)]


def frustum_segments(T, size=0.3, aspect=16 / 9):
    """Line segments (world coordinates) of a camera frustum drawn like the original ROS 2
    bridge: apex at the optical centre, opening along camera +z."""
    hw, hh = size * 0.5 * aspect / 1.5, size * 0.5 / 1.5
    corners = [transform_point(T, c) for c in ([-hw, -hh, size], [hw, -hh, size], [hw, hh, size], [-hw, hh, size])]
    apex = [T[0][3], T[1][3], T[2][3]]
    return [(apex, c) for c in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """T_W<-C (4x4 rows) of an OpenCV camera at `position` looking at `target` (same as the
    original publish_synthetic_poses.look_at)."""
    z = [t - p for t, p in zip(target, position)]
    z = _unit(z)
    x = _unit(_cross([-u for u in up], z))
    y = _cross(z, x)
    return [[x[0], y[0], z[0], position[0]],
            [x[1], y[1], z[1], position[1]],
            [x[2], y[2], z[2], position[2]],
            [0.0, 0.0, 0.0, 1.0]]


def inverse_rigid(T):
    """Inverse of a rigid 4x4 (rows): [R^T, -R^T t]."""
    Rt = [[T[j][i] for j in range(3)] for i in range(3)]
    t = [-sum(Rt[i][k] * T[k][3] for k in range(3)) for i in range(3)]
    return [Rt[0] + [t[0]], Rt[1] + [t[1]], Rt[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def rotation_angle_deg(A, B):
    """Angle of the relative rotation between the rotation parts of two 4x4 poses."""
    tr = sum(A[k][i] * B[k][i] for i in range(3) for k in range(3))     # trace(A_R^T B_R)
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def translation_distance(A, B):
    return math.sqrt(sum((A[i][3] - B[i][3]) ** 2 for i in range(3)))


def flatten(T):
    return [float(v) for r in T for v in r]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]
