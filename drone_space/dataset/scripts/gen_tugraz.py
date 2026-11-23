#!/usr/bin/env python3
"""
Generate CELL-format sequences from the drone-space tugraz rosbags.

Per sequence, at 10 Hz cadence (one frame per 100 ms) aligned to the GT mocap pose:
  - event_frames_ours_50ms_half/left/event_frame_XXXXX.npy : (360,640,2) float32
        2-ch exp-decay time-surface, 50 ms window ending at the pose ts, TAU=15 ms,
        ch0=ON ch1=OFF, half-res (coords//2), value exp(-age/TAU), latest event per pixel.
  - local_maps/point_cloud_XXXXX.h5 : PC [4,M] float16, corrected map in the GT CAMERA frame
        (T_cam_world @ pc_world), kept to z in (0.1, DEPTH_KEEP].
  - <seq>_pose_gt.h5 : dataset 'Ln_T_L0' [N,4,4] float64 = world->cam per kept frame
        (= inv(T_world_cam)); this is the GT pose file the CELL dataloader reads.
  - meta.npz : per-kept-frame ts, T_world_cam, bridged flag, + calib/params.

Frames with < MIN_EVENTS (static lead-in/out) are dropped; kept frames are renumbered
contiguously (event frame & local map indices stay in sync). Poses = glitch-bridged npz.
Events parsed straight from the ROS1 wire bytes with numpy; streamed in one pass (bounded RAM).
Writes to ext4 (DATA_ROOT), never the NTFS Windata drive.
"""
import os, sys, argparse, time, struct
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R
from rosbags.rosbag1 import Reader
from pathlib import Path

# Placeholders, overridden by the CLI flags added at the bottom of this file
# (--bagdir/--posedir/--ply/--ext/--data-root). PLY/EXT fall back to this repo's own
# drone_space/map, drone_space/calib when not given explicitly (see __main__ below) --
# see drone_space/dataset/DATASET.md for where the hosted bags/poses/map actually live.
BAGDIR = Path("/path/to/bags")
POSEDIR = BAGDIR / "poses_fixed"
PLY = None
EXT = None
DATA_ROOT = Path("/path/to/output")
SCENE_PREFIX = "tugraz_"      # output scene name = SCENE_PREFIX + input bag stem (seq1 -> tugraz_seq1)
REP_DIR = "event_frames_ours_50ms_half"
CALIB = [1687.4514251875887, 1687.4514251875887, 653.35913097450623, 351.42057452312019]

CADENCE = 0.100; WIN = 0.050; TAU = 0.015; PAD = 0.050
K = 2; H, W = 720 // K, 1280 // K
DEPTH_KEEP = 12.0
MIN_EVENTS = 5000
EVDT = np.dtype([('x', '<u2'), ('y', '<u2'), ('sec', '<u4'), ('nsec', '<u4'), ('pol', 'u1')])


def read_ply_xyz(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        le = b"little" in f.readline().strip(); props = []; n = 0
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"): n = int(ln.split()[-1])
            elif ln.startswith(b"property"): props.append((ln.split()[2].decode(), ln.split()[1].decode()))
            elif ln == b"end_header": break
        tm = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1",
              "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
        dt = np.dtype({"names": [p[0] for p in props],
                       "formats": [("<" if le else ">") + tm[p[1]] for p in props]})
        a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
        return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def parse_buffer(raw):
    fidlen = struct.unpack_from('<I', raw, 12)[0]; off = 16 + fidlen
    alen = struct.unpack_from('<I', raw, off + 8)[0]; off += 12
    if alen == 0: return None
    ev = np.frombuffer(raw, dtype=EVDT, count=alen, offset=off)
    t = ev['sec'].astype(np.float64) + ev['nsec'] * 1e-9
    return (ev['x'].astype(np.int32), ev['y'].astype(np.int32), t, ev['pol'].astype(np.int64))


def build_frame(x, y, t, p, t_end):
    img = np.zeros((H, W, 2), np.float32)
    if len(x) == 0: return img
    o = np.argsort(t, kind="stable")                 # ascending -> latest wins on scatter
    yy = (y[o] // K); xx = (x[o] // K); dec = np.exp(-(t_end - t[o]) / TAU).astype(np.float32)
    img[yy, xx, p[o]] = dec
    return img


def process(seq, max_ticks=None):
    scene = f"{SCENE_PREFIX}{seq}"            # input files use the bag stem `seq`; the generated
                                             # sequence dir uses the CELL scene name (tugraz_seqN)
    bag = BAGDIR / f"{seq}_trim.bag"
    npz = np.load(POSEDIR / f"{seq}_trim_poses_fixed.npz")
    pts, pos, quat, bridged = npz["ts"], npz["pos"], npz["quat"], npz["bridged"]
    T_bc = np.loadtxt(EXT, comments="#")
    Pw = read_ply_xyz(PLY); Pwh = np.concatenate([Pw, np.ones((len(Pw), 1))], 1).T  # [4,N]

    t0, t1 = pts[0], pts[-1]
    ticks = np.arange(t0 + WIN, t1, CADENCE)
    if max_ticks: ticks = ticks[:max_ticks]
    pidx = np.clip(np.searchsorted(pts, ticks), 0, len(pts) - 1)
    frame_ts = pts[pidx]; nf = len(frame_ts)

    out = DATA_ROOT / scene
    (out / REP_DIR / "left").mkdir(parents=True, exist_ok=True)
    (out / "local_maps").mkdir(parents=True, exist_ok=True)
    print(f"[{seq}] {nf} frames; streaming events ...", flush=True)
    te = time.time()

    chunks = []; next_f = 0; j = 0; skipped = 0
    kept_ts = []; kept_Twc = []; kept_br = []

    def emit(fi):
        nonlocal j, skipped
        ti = frame_ts[fi]
        if chunks:
            X = np.concatenate([c[0] for c in chunks]); Y = np.concatenate([c[1] for c in chunks])
            T = np.concatenate([c[2] for c in chunks]); P = np.concatenate([c[3] for c in chunks])
            m = (T >= ti - WIN) & (T < ti)
        else:
            m = np.array([], bool); X = Y = T = P = m
        if int(m.sum()) < MIN_EVENTS:
            skipped += 1; return
        img = build_frame(X[m], Y[m], T[m], P[m], ti)
        np.save(out / REP_DIR / "left" / f"event_frame_{j:05d}.npy", img)
        pp = pidx[fi]
        T_wb = np.eye(4); T_wb[:3, :3] = R.from_quat(quat[pp]).as_matrix(); T_wb[:3, 3] = pos[pp]
        T_wc = T_wb @ T_bc
        Pc = (np.linalg.inv(T_wc) @ Pwh)[:3]
        keep = (Pc[2] > 0.1) & (Pc[2] <= DEPTH_KEEP)
        PC = np.concatenate([Pc[:, keep], np.ones((1, int(keep.sum())))], 0).astype(np.float16)
        with h5py.File(out / "local_maps" / f"point_cloud_{j:05d}.h5", "w") as hf:
            hf.create_dataset("PC", data=PC, compression="lzf", shuffle=True)
        kept_ts.append(ti); kept_Twc.append(T_wc); kept_br.append(bool(bridged[pp]))
        if j % 50 == 0:
            print(f"  [{seq}] out={j} in={fi} ev={int(m.sum())} map_pts={int(keep.sum())} "
                  f"({time.time()-te:.0f}s)", flush=True)
        j += 1

    with Reader(bag) as r:
        ec = [c for c in r.connections if 'cd_events' in c.topic]
        for conn, tbns, raw in r.messages(connections=ec):
            if next_f >= nf: break
            tb = tbns / 1e9
            pb = parse_buffer(raw)
            if pb is not None: chunks.append(pb)
            while next_f < nf and frame_ts[next_f] + PAD <= tb:
                emit(next_f); next_f += 1
                if next_f < nf:
                    need = frame_ts[next_f] - WIN
                    chunks = [c for c in chunks if c[2][-1] >= need]
    while next_f < nf:                                  # flush any remaining
        emit(next_f); next_f += 1

    np.savez(out / "meta.npz", ts=np.array(kept_ts), T_world_cam=np.array(kept_Twc, np.float32),
             bridged=np.array(kept_br), window=WIN, tau=TAU, cadence=CADENCE, calib=CALIB, half=True)
    # GT pose file the CELL dataloader (core/datasets_m3ed.py) reads: world->cam per kept
    # frame = inv(T_world_cam), stored as the single dataset 'Ln_T_L0' [N,4,4]. A sequence
    # without this file is silently skipped by the loader, so it must be written here.
    with h5py.File(out / f"{scene}_pose_gt.h5", "w") as hf:
        hf.create_dataset("Ln_T_L0", data=np.linalg.inv(np.asarray(kept_Twc, dtype=np.float64)))
    print(f"[{scene}] DONE -> {out}  kept {j}, dropped {skipped}  ({time.time()-te:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["seq1", "seq2", "seq3", "seq4"])
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--bagdir", type=Path, default=BAGDIR,
                    help="dir holding <seq>_trim.bag (the hosted bags/ folder, see DATASET.md)")
    ap.add_argument("--posedir", type=Path, default=None,
                    help="dir holding <seq>_trim_poses_fixed.npz (default: drone_space/dataset/poses/, "
                         "resolved relative to this script)")
    ap.add_argument("--ply", default=None,
                    help="aligned room cloud path (default: drone_space/map/fused_clean_mocap.ply, "
                         "resolved relative to this script)")
    ap.add_argument("--ext", default=None,
                    help="T_body_camera.txt path (default: drone_space/calib/T_body_camera.txt, "
                         "resolved relative to this script)")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="output dir for generated sequences (an ext4 path -- the format's "
                         "f16/scatter writes are slow over NTFS/exFAT)")
    ap.add_argument("--scene-prefix", default=SCENE_PREFIX,
                    help="prefix for the output scene dir name (default 'tugraz_': bag stem "
                         "seq1 -> sequence tugraz_seq1, matching core/scene_config.py). Set to "
                         "'' if your bag stems already carry the scene name.")
    a = ap.parse_args()

    _dataset_dir = Path(__file__).resolve().parent.parent      # drone_space/dataset/
    _drone_space = _dataset_dir.parent                         # drone_space/
    BAGDIR = a.bagdir
    POSEDIR = a.posedir or (_dataset_dir / "poses")
    PLY = a.ply or str(_drone_space / "map" / "fused_clean_mocap.ply")
    EXT = a.ext or str(_drone_space / "calib" / "T_body_camera.txt")
    DATA_ROOT = a.data_root
    SCENE_PREFIX = a.scene_prefix

    for s in a.seqs:
        process(s, a.max_ticks)
