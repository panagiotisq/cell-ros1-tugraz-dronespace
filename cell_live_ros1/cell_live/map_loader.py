"""Load the aligned drone-space map once, crop a local submap around the guess camera
each tick (DEPLOY_CELL_DRONESPACE.md sec 6: "crop the 1.22M-point map to the guess camera
frustum before push_fuse"). Pure numpy -- no open3d dependency, the PLY is a fixed binary
layout (see drone_space/map/fused_clean_mocap.ply: Open3D-written, x/y/z/nx/ny/nz double +
r/g/b uchar per vertex)."""

from __future__ import annotations

import numpy as np

VERTEX_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                         ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
                         ("r", "u1"), ("g", "u1"), ("b", "u1")])


def load_xyz(path):
    """Binary little-endian PLY (x,y,z,nx,ny,nz,r,g,b per vertex) -> (N, 3) float64 world
    points. Raises ValueError if the header doesn't match the expected layout."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path}: expected 'format binary_little_endian 1.0'")
        n = next((int(h.split()[-1]) for h in header if h.startswith("element vertex")), None)
        if n is None:
            raise ValueError(f"{path}: no 'element vertex' line in header")
        data = np.fromfile(f, dtype=VERTEX_DTYPE, count=n)
    if data.shape[0] != n:
        raise ValueError(f"{path}: expected {n} vertices, read {data.shape[0]} (truncated file?)")
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


class CroppedMap:
    """Holds the full map once; `crop_homogeneous(cam_pos_world, radius_m)` returns a
    (4, N') float32 homogeneous point array (x,y,z,1 per column, CameraModel's expected
    layout) of the points within `radius_m` of the given world position."""

    def __init__(self, xyz):
        self.xyz = xyz                           # (N, 3) float64

    @classmethod
    def load(cls, path):
        return cls(load_xyz(path))

    def crop_homogeneous(self, cam_pos_world, radius_m=15.0):
        d = np.linalg.norm(self.xyz - np.asarray(cam_pos_world, dtype=np.float64)[None, :], axis=1)
        pts = self.xyz[d < radius_m]
        return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T.astype(np.float32)   # (4, N')
