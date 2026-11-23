"""
Visualise a CELL tracking run (tools/tracking/track_cell.py output).

Produces, from <run>/trajectory.npz:
  - summary.png : top-down (XY) + side (XZ) trajectories (GT vs tracked) and the
                  APE-over-time curve, with GT-reset markers.
  - track.mp4   : the camera flying through the room's LiDAR/depth map, GT frustum
                  (white) vs tracked frustum (green), trails growing over time, with
                  a live frame / drift HUD. (the "demo_sequence" look, self-contained)

The room backdrop is accumulated from the per-frame local maps projected into the
world frame by GT (X_world = inv(G_k) @ pc_k), exactly as the tracker's geometry
defines them — no external global map needed.

Run (repo root, LEAR env):
  python tools/tracking/viz_trajectory.py --run experiments/tracking_cell/tugraz_seq3/floor0_position \
     --data_path data/tugraz [--no_video] [--stride 1]
"""
import os, sys, argparse, subprocess, shutil
sys.path.insert(0, os.getcwd())
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

GT_C = (0.92, 0.92, 0.92); EST_C = (0.15, 0.85, 0.35); INIT_C = (0.95, 0.30, 0.25); BG = (0.06, 0.07, 0.09)


def load_cloud(data_path, seq, file_idx, G_gt, every=2, voxel=0.03, cap=300000):
    """Accumulate per-frame local maps into the world frame via GT, voxel-downsample."""
    md = os.path.join(data_path, seq, "local_maps")
    pts = []
    for k in range(0, len(file_idx), every):
        f = os.path.join(md, f"point_cloud_{file_idx[k] + 1:05d}.h5")
        if not os.path.exists(f):
            continue
        with h5py.File(f, "r") as h:
            pc = h["PC"][:].astype(np.float64)
        X = pc[:3].T if pc.shape[0] in (3, 4) else pc[:, :3]
        X = X[np.isfinite(X).all(1)][::3]
        G = G_gt[k]; W = (X - G[:3, 3]) @ G[:3, :3]          # inv(G) applied: world points
        pts.append(W)
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts, 0)
    key = np.round(P / voxel).astype(np.int64)
    _, uniq = np.unique(key, axis=0, return_index=True)
    P = P[uniq]
    if len(P) > cap:
        P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]
    return P


def load_ply(path, voxel=0.02, cap=300000):
    """Load a colored (world-frame) reconstruction PLY -> (xyz Nx3 float64, rgb Nx3 in 0..1).
    Supports the Open3D binary_little_endian layout (double xyz [+ normals] + uchar rgb)."""
    with open(path, "rb") as f:
        lines = []
        while True:
            ln = f.readline()
            lines.append(ln)
            if b"end_header" in ln:
                break
        hdr = [l.decode("ascii", "replace").strip() for l in lines]
        assert "binary_little_endian" in hdr[1], "only binary_little_endian PLY supported"
        tmap = {"double": "<f8", "float": "<f4", "uchar": "u1", "uint8": "u1", "int": "<i4"}
        fields = [(p.split()[2], tmap[p.split()[1]]) for p in hdr if p.startswith("property")]
        a = np.fromfile(f, dtype=np.dtype(fields))
    xyz = np.stack([a["x"].astype(np.float64), a["y"].astype(np.float64), a["z"].astype(np.float64)], 1)
    have_rgb = all(c in a.dtype.names for c in ("red", "green", "blue"))
    rgb = (np.stack([a["red"], a["green"], a["blue"]], 1).astype(np.float64) / 255.0
           if have_rgb else np.full((len(xyz), 3), 0.7))
    m = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[m], rgb[m]
    key = np.round(xyz / voxel).astype(np.int64)                 # voxel downsample (keep first)
    _, uniq = np.unique(key, axis=0, return_index=True)
    xyz, rgb = xyz[uniq], rgb[uniq]
    if len(xyz) > cap:
        sel = np.random.default_rng(0).choice(len(xyz), cap, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def frustum(P, dep, aspect=0.5625):
    hw = dep * 0.6; hh = hw * aspect
    cc = np.array([[0, 0, 0], [-hw, -hh, dep], [hw, -hh, dep], [hw, hh, dep], [-hw, hh, dep]])
    R, t = P[:3, :3], P[:3, 3]
    return (cc - t) @ R


def set_axes(ax, lo, hi):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.set_facecolor(BG); ax.set_axis_off()


def summary_png(d, out):
    ce, cg, ape = d["centre_est"], d["centre_gt"], d["ape"]
    reset = d["reset"].astype(bool)
    fig = plt.figure(figsize=(16, 5.2), facecolor="white")
    for n, (ix, iy, lbl) in enumerate([(0, 1, "top-down  X-Y (m)"), (0, 2, "side  X-Z (m)")]):
        ax = fig.add_subplot(1, 3, n + 1)
        ax.plot(cg[:, ix], cg[:, iy], color="0.45", lw=2.4, label="ground truth")
        ax.plot(ce[:, ix], ce[:, iy], color=(0.05, 0.6, 0.2), lw=1.5, label="CELL tracked")
        ax.scatter(cg[0, ix], cg[0, iy], c="k", s=60, marker="o", zorder=5, label="start (GT)")
        if reset.any():
            ax.scatter(ce[reset, ix], ce[reset, iy], c="red", s=18, marker="x", zorder=6, label="GT reset")
        ax.set_title(lbl); ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
        if n == 0:
            ax.legend(fontsize=8, loc="best")
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(ape, color=(0.05, 0.6, 0.2), lw=1.3)
    ax.axhline(np.median(ape), color="0.5", ls="--", lw=1, label=f"median {np.median(ape):.1f} cm")
    ax.axhline(25, color="orange", ls=":", lw=1, label="25 cm")
    if reset.any():
        for r in np.where(reset)[0]:
            ax.axvline(r, color="red", alpha=0.25, lw=0.8)
    ax.set_xlabel("frame"); ax.set_ylabel("APE (cm)"); ax.set_title("absolute trajectory error")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"CELL tracking — {str(d['seq'])}  |  sel={str(d['selection'])} motion={str(d['motion'])}  "
                 f"|  APE med {np.median(ape):.1f} cm, max {ape.max():.1f} cm, final {ape[-1]:.1f} cm",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  saved {out}")


def render_video(d, cloud, out, stride=1, fps=30, elev=22, azim=-60,
                 follow=True, view_half=1.5, orbit=20.0, cloud_rgb=None, mark_reanchor=False):
    He, Gg, ape = d["H_est"], d["G_gt"], d["ape"]
    ce, cg = d["centre_est"], d["centre_gt"]
    reset = d["reset"].astype(bool)
    # full-room static bbox (used when follow is off, and to colour the cloud by height)
    traj = np.concatenate([cg, ce], 0)
    margin = np.array([0.4, 0.4, 0.3])
    if len(cloud):                                        # robust bbox: trim only stray outliers
        clo = np.percentile(cloud, 0.5, axis=0); chi = np.percentile(cloud, 99.5, axis=0)
        lo = np.minimum(traj.min(0), clo) - margin
        hi = np.maximum(traj.max(0), chi) + margin
    else:
        lo = traj.min(0) - margin; hi = traj.max(0) + margin
    # moving "follow" camera: a fixed-size cube window that pans along a SMOOTHED camera
    # path, so the view stays close to the action and glides with the trajectory while the
    # whole room scrolls past (matplotlib clips the cloud to each frame's window).
    foc = cg.copy()
    if follow and len(foc) > 3:
        k = max(3, int(round(fps * 0.6)) | 1)             # ~0.6 s odd-length moving average
        pad = np.pad(foc, ((k // 2, k // 2), (0, 0)), mode="edge")
        foc = np.stack([np.convolve(pad[:, c], np.ones(k) / k, "valid") for c in range(3)], 1)
    span = (2 * view_half) if follow else float(np.median(hi - lo)); dep = 0.11 * span
    tmp = out + "_frames"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    rng = np.random.default_rng(0)
    if len(cloud) > 180000:
        keep = rng.choice(len(cloud), 180000, replace=False)
        cloud = cloud[keep]
        if cloud_rgb is not None:
            cloud_rgb = cloud_rgb[keep]
    if cloud_rgb is not None and len(cloud):                      # true reconstruction colours
        cc = np.concatenate([np.clip(cloud_rgb, 0, 1), np.full((len(cloud), 1), 0.95)], 1)
    elif len(cloud):                                              # fallback: shade by height
        h = cloud[:, 2]; hl, hh = np.percentile(h, [3, 97])
        cc = plt.colormaps["bone"](0.25 + 0.65 * np.clip((h - hl) / (hh - hl + 1e-9), 0, 1)); cc[:, 3] = 0.35
    idxs = list(range(0, len(He), stride))
    for n, i in enumerate(idxs):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        if follow:
            c = foc[i]; flo = c - view_half; fhi = c + view_half   # cube window centred on the camera
            set_axes(ax, flo, fhi)
        else:
            set_axes(ax, lo, hi)
        az = azim + orbit * (n / max(1, len(idxs) - 1))            # slow cinematic orbit
        ax.view_init(elev=elev, azim=az)
        ax.computed_zorder = False                                 # honour explicit zorder (draw order), not depth
        if len(cloud):
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=3.0, c=cc, linewidths=0, depthshade=False, zorder=0)
        for key, col, lw, al in [(cg, GT_C, 2.2, 0.85), (ce, EST_C, 2.0, 1.0)]:   # trajectories ON TOP of the cloud
            T = key[:i + 1]
            if len(T) > 1:
                ax.plot(T[:, 0], T[:, 1], T[:, 2], color=col, lw=lw, alpha=al, zorder=5)
        for P, col, lw in [(Gg[i], GT_C, 1.6), (He[i], EST_C, 2.2)]:
            fr = frustum(P, dep)
            segs = [[fr[0], fr[k]] for k in (1, 2, 3, 4)] + [[fr[k], fr[k % 4 + 1]] for k in (1, 2, 3, 4)]
            lc = Line3DCollection(segs, colors=[col], linewidths=lw, alpha=0.98); lc.set_zorder(10)
            pc = Poly3DCollection([fr[1:5]], facecolors=[col], alpha=0.14, linewidths=0); pc.set_zorder(9)
            ax.add_collection3d(lc); ax.add_collection3d(pc)
        fig.text(0.03, 0.95, f"CELL pose tracking — {str(d['seq'])}", color="w", fontsize=15, weight="bold", va="top")
        fig.text(0.03, 0.90, "white = ground truth      green = CELL (GT-init frame 0, then self-tracked)",
                 color="0.8", fontsize=10, va="top")
        fig.text(0.03, 0.055, f"frame {int(d['file_idx'][i])}", color="0.8", fontsize=11)
        fig.text(0.03, 0.025, f"drift {ape[i]:.1f} cm", color=EST_C, fontsize=13, weight="bold")
        fig.text(0.97, 0.025, f"median drift so far {np.median(ape[:i+1]):.1f} cm", color="0.8", fontsize=10, ha="right")
        if reset[i] and mark_reanchor:
            fig.text(0.5, 0.52, "RE-ANCHOR", color=(1.0, 0.75, 0.2), fontsize=20, ha="center", va="center", alpha=0.8)
        fig.savefig(os.path.join(tmp, f"f{n:05d}.png"), dpi=96, facecolor=BG); plt.close(fig)
        if n % 50 == 0:
            print(f"  rendered {n}/{len(idxs)}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # force even dims (yuv420p requirement)
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    shutil.rmtree(tmp)
    print(f"  saved {out}  ({len(idxs)} frames, {len(idxs)/fps:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a track_cell.py output dir (with trajectory.npz)")
    ap.add_argument("--data_path", default="data/tugraz")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--static", action="store_true", help="whole-room fixed view (disable the moving follow camera)")
    ap.add_argument("--view_half", type=float, default=1.5, help="follow-camera half-window (m); smaller = closer")
    ap.add_argument("--orbit", type=float, default=20.0, help="total azimuth orbit over the clip (deg)")
    ap.add_argument("--cloud_ply", default="/path/to/fused_clean_mocap.ply",
                    help="colored world-frame reconstruction PLY to use as backdrop (world frame = GT frame)")
    ap.add_argument("--no_ply", action="store_true", help="ignore the reconstruction PLY; shade accumulated local maps by height")
    ap.add_argument("--mark_reanchor", action="store_true", help="flash the 'RE-ANCHOR' overlay on re-anchor frames (off by default)")
    a = ap.parse_args()
    d = dict(np.load(os.path.join(a.run, "trajectory.npz"), allow_pickle=True))
    seq = str(d["seq"])
    summary_png(d, os.path.join(a.run, "summary.png"))
    if not a.no_video:
        cloud_rgb = None
        if not a.no_ply and a.cloud_ply and os.path.exists(a.cloud_ply):
            cloud, cloud_rgb = load_ply(a.cloud_ply)
            print(f"  reconstruction cloud (RGB): {len(cloud)} pts  <- {a.cloud_ply}")
        else:
            cloud = load_cloud(a.data_path, seq, d["file_idx"], d["G_gt"])
            print(f"  room cloud (height-shaded): {len(cloud)} pts")
        render_video(d, cloud, os.path.join(a.run, "track.mp4"), stride=a.stride, fps=a.fps,
                     follow=not a.static, view_half=a.view_half, orbit=a.orbit, cloud_rgb=cloud_rgb,
                     mark_reanchor=a.mark_reanchor)


if __name__ == "__main__":
    main()
