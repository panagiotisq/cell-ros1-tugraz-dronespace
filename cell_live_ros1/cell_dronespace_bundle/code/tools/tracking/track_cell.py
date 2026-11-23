"""
CELL continuous pose-tracking on the tugraz drone-space sequences.

Demo use-case (advisor's spec): we have NO global relocalization, so the track
starts from the GROUND-TRUTH pose of the first frame. From there CELL processes
the frames sequentially; each new frame is initialised from the PREVIOUS frame's
estimate (zero-velocity / "position" motion model), the local map is projected by
that guess, CELL predicts the flow, PnP recovers the correction, and the estimate
is updated. The resulting estimates should trace a smooth, continuous trajectory.

Geometry (verified, world->cam convention; G_k = Ln_T_L0[idx] = GT world->cam):
  - local_maps/point_cloud_{idx+1}.h5 is the frame's map IN ITS GT CAMERA FRAME,
    i.e. pc_k = G_k @ X_world  (same pairing as the registration DatasetM3ED).
  - Feeding push_fuse the perturbation RT = H_guess @ inv(G_k) makes the depth
    input exactly  pc_rotated = RT @ pc_k = H_guess @ X_world  — i.e. the map as
    seen from the guess pose, with NO ground truth used at inference. (Identical to
    the world-map tracker, expressed through the per-frame cam-frame map.)
  - CELL + PnP estimate RT (eval convention: err_Pose compares the solved pose to
    RT), so the corrected absolute pose is  H_est = inv(RT_pred) @ H_guess.
  - Proven: ||centre(H_est) - centre(G_k)|| == err_Pose translation error exactly,
    so a guess==GT run reproduces the eval_full per-frame numbers (--motion gt).

Robustness guards (advisor: "if the tracking is bad, add constraints"):
  --min_inlier_rate : reject (HOLD previous estimate) a PnP update with too few inliers.
  --max_step_cm     : reject a correction that jumps the camera centre more than this
                      vs the guess (a PnP blunder; real inter-frame motion is <~10 cm).
  --reset_cm        : if the running estimate drifts more than this from GT, SNAP back
                      to GT for that frame and continue (recovers a diverged stretch).
                      --reset_cm 0 disables (honest free-running drift).

Outputs (per run, under --out_dir): trajectory.npz (per-frame poses + diagnostics),
trajectory.csv, and report.txt. Visualise with tools/tracking/viz_trajectory.py.

Run (repo root, LEAR env):
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -u tools/tracking/track_cell.py \
     --seq tugraz_seq3 --ckpt <best_model.pth> --selection floor0 --reset_cm 50
"""
import os, sys, csv, json, time, argparse
sys.path.insert(0, os.getcwd()); sys.path.append("core")
import numpy as np, torch, torch.nn.functional as F
import h5py

import eval_full as EF                                   # get_corr / solve / sel_* (module globals set below)
from core.datasets_m3ed import DatasetM3ED
from core.data_preprocess import Data_preprocess
from core.scene_config import scene_geometry
from core.flow2pose import err_Pose
from core.utils_point import quat2mat, tvector2mat, quaternion_from_matrix
from track import build_model
# post-pipeline edge-matching pose refinement (same operator as tools/pose_refine/eval_refine.py)
from tools.pose_refine.edge_extract import skeletonize
from tools.pose_refine.refine import distance_transform, deproject, project, exp_so3, refine_pose, sample_dt


# --- se(3) <-> Flow2Pose quaternion (copied verbatim from eval_refine.py so the convention matches) ---
def flip_quat(q):
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def so3_log(R):
    cos = ((R[0, 0] + R[1, 1] + R[2, 2] - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = torch.sin(theta)
    scale = torch.where(s.abs() < 1e-6, torch.full_like(theta, 0.5), theta / (2 * s))
    return scale * w


def lear_to_xi(R_pred, T_pred):
    q_raw = flip_quat(R_pred); R_raw = quat2mat(q_raw)[:3, :3]; t_raw = -T_pred
    return torch.cat([so3_log(R_raw), t_raw])


def xi_to_lear(xi):
    R = exp_so3(xi[:3]); q = quaternion_from_matrix(R)
    return flip_quat(q), -xi[3:]


def edge_refine(depth2edge, depth_input, event_input, calib, o, dev, geom, MAXD,
                lock_rot, anchor_w, refine_iters, conf_map=None, edge_conf="none",
                edge_thresh=0.5, min_pts=30):
    """Refine the PnP pose `o=(R,T)` (Flow2Pose convention) by chamfer-aligning the model's
    predicted depth-edges (3D, deprojected through the guess-pose depth) to the observed event
    edges. Returns a refined (R,T) in the SAME convention, or `o` unchanged if too few edge points."""
    CROP_H, CROP_W, CROP_X, CROP_Y = geom
    ep = depth2edge[-1] if isinstance(depth2edge, (list, tuple)) else depth2edge
    ep = ep[0, 0].float()
    if ep.min() < 0 or ep.max() > 1:
        ep = torch.sigmoid(ep)
    ep_np = ep.detach().cpu().numpy()
    edge_bin = (ep > edge_thresh).cpu().numpy().astype(np.float32)
    depth_m = depth_input[0, 0].cpu().numpy() * MAXD
    rc = np.argwhere(edge_bin > 0)
    if len(rc):
        z = depth_m[rc[:, 0], rc[:, 1]]; keep = z > 0.1; rc = rc[keep]; z = z[keep]
    if len(rc) < min_pts:
        return o, float("nan")
    fx, fy, cx, cy = [float(v) for v in calib]
    cxc = cx - CROP_Y; cyc = cy - CROP_X                   # principal point in the cropped frame
    uv = torch.tensor(np.stack([rc[:, 1], rc[:, 0]], 1), dtype=torch.float32, device=dev)
    X = deproject(uv, torch.tensor(z, dtype=torch.float32, device=dev), fx, fy, cxc, cyc)
    weights = None
    if edge_conf == "both" and conf_map is not None:       # edgeprob * flow-confidence weighted chamfer
        epv = ep_np[rc[:, 0], rc[:, 1]].astype(np.float64)
        cfv = conf_map[rc[:, 0], rc[:, 1]].astype(np.float64)
        weights = torch.tensor(epv * cfv, dtype=torch.float32, device=dev)
    act = event_input[0].abs().sum(0).cpu().numpy(); occ = (act > 0).astype(np.float32)
    dt = distance_transform(skeletonize(occ)).to(dev)
    xi_init = lear_to_xi(o[0], o[1]).to(dev).float()
    xi_opt, hist = refine_pose(X, dt, (fx, fy, cxc, cyc), init_xi=xi_init, iters=refine_iters,
                               huber_delta=3.0, z_reg=1e-3, anchor_w=anchor_w, lock_rot=lock_rot,
                               weights=weights)
    R_ref, T_ref = xi_to_lear(xi_opt)
    return (R_ref.cpu(), T_ref.cpu()), (hist[-1] if hist else float("nan"))


def centre(P):
    """Camera centre in world, from a world->cam 4x4 pose."""
    return -P[:3, :3].T @ P[:3, 3]


def mat_from_qt(q, t):
    """4x4 (float64) from quaternion (w,x,y,z) and translation, Flow2Pose convention
    RT = T(t) @ R(q) — the SAME construction err_Pose uses for the predicted pose."""
    R = quat2mat(torch.as_tensor(q, dtype=torch.float32)).double()
    T = tvector2mat(torch.as_tensor(t, dtype=torch.float32)).double()
    return (T @ R).numpy()


def ortho(P):
    """Re-orthonormalise the rotation block to kill float drift over long tracks."""
    U, _, Vt = np.linalg.svd(P[:3, :3]); R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    Q = P.copy(); Q[:3, :3] = R; return Q


def pose_gap(A, B):
    """Translation (cm) and rotation (deg) of world->cam pose A relative to B."""
    t = float(np.linalg.norm(centre(A) - centre(B)) * 100.0)
    c = (np.trace(A[:3, :3] @ B[:3, :3].T) - 1.0) * 0.5
    r = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return t, r


def mid_pose(A, B):
    """Midpoint of two world->cam poses: camera-centre midpoint + SVD-averaged rotation."""
    cM = 0.5 * (centre(A) + centre(B))
    M = ortho(0.5 * (A + B))                       # orthonormalised rotation average
    M[:3, 3] = -M[:3, :3] @ cM                      # enforce the midpoint camera centre exactly
    M[3] = (0.0, 0.0, 0.0, 1.0)
    return M


def guided_init(prev, G, tcm, rdeg, cap=40):
    """If the previous estimate sits outside the (tcm, rdeg) perturbation margin of the
    current GT, pull it halfway toward GT repeatedly until the init re-enters the margin
    (each pass halves the gap). Returns (init_pose, was_pulled)."""
    H = prev.copy(); pulled = False
    for _ in range(cap):
        t, r = pose_gap(H, G)
        if t <= tcm and r <= rdeg:
            break
        H = mid_pose(H, G); pulled = True
    return ortho(H), pulled


def cv_predict(Hm2, Hm1, gain=1.0):
    """Constant-velocity init: extrapolate the world->cam pose for the next frame from the
    last two estimates (Hm2 at k-2, Hm1 at k-1). In cam->world coordinates C=inv(H), the
    constant-velocity prediction is C_pred = C_{k-1} @ inv(C_{k-2}) @ C_{k-1} (body- and
    world-frame increments coincide for this two-fold composition). `gain` scales the SE(3)
    increment (1 = full predicted step; <1 damps it toward the previous pose)."""
    Cm2 = np.linalg.inv(Hm2); Cm1 = np.linalg.inv(Hm1)
    rel = np.linalg.inv(Cm2) @ Cm1                              # body-frame increment (SE3)
    if abs(gain - 1.0) > 1e-9:                                  # scale the increment via so3 log/exp
        Rr = torch.as_tensor(rel[:3, :3], dtype=torch.float32)
        R_s = exp_so3(gain * so3_log(Rr)).numpy().astype(np.float64)
        G = np.eye(4); G[:3, :3] = R_s; G[:3, 3] = gain * rel[:3, 3]
        rel = G
    C_pred = Cm1 @ rel
    return ortho(np.linalg.inv(C_pred))                        # back to world->cam


def main():
    ap = argparse.ArgumentParser(description="CELL continuous pose tracking (tugraz)")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_path", default="data/tugraz")
    ap.add_argument("--ev_input", default="ours_50ms_half")
    ap.add_argument("--selection", default="floor0", choices=["floor0", "floor0.5", "all", "median"],
                    help="correspondence selection fed to PnP (floor0 = CELL deployed default)")
    ap.add_argument("--motion", default="position", choices=["position", "velocity", "gt", "gt_prev"],
                    help="position = init from previous estimate (zero-velocity); velocity = constant-velocity "
                         "extrapolation from the last two estimates (smoother); gt / gt_prev are oracles")
    ap.add_argument("--vel_gain", type=float, default=1.0,
                    help="scale on the constant-velocity increment (1 = full predicted step; <1 damps it)")
    ap.add_argument("--iters", type=int, default=24, help="flow refinement iterations at inference")
    ap.add_argument("--min_inlier_rate", type=float, default=0.30, help="hold the estimate if PnP inlier rate < this")
    ap.add_argument("--max_step_cm", type=float, default=40.0, help="reject a correction jumping the centre more than this vs the guess")
    ap.add_argument("--reset_cm", type=float, default=0.0, help="snap the estimate back to GT if APE exceeds this (0 = disabled)")
    ap.add_argument("--guided_init", action="store_true",
                    help="soft re-anchor: if the previous estimate is outside the (margin_cm, margin_deg) "
                         "perturbation envelope of the current GT, average it toward GT until the init re-enters "
                         "the envelope (replaces the hard --reset_cm snap)")
    ap.add_argument("--margin_cm", type=float, default=25.0, help="guided-init translation margin (cm) = training perturbation envelope")
    ap.add_argument("--margin_deg", type=float, default=2.5, help="guided-init rotation margin (deg) = training perturbation envelope")
    ap.add_argument("--pweight_seed", type=int, default=0)
    # post-pipeline edge-matching refinement (tools/pose_refine) applied per frame after PnP
    ap.add_argument("--refine", action="store_true", help="refine each frame's PnP pose by edge-matching (chamfer/DT) before the update")
    ap.add_argument("--refine_full", action="store_true", help="refine all 6 DOF (default: rotation-locked, translation-only = the proven indoor operator)")
    ap.add_argument("--refine_anchor_w", type=float, default=30.0, help="trust-region anchor weight on ||xi-init||^2 (guards chamfer local minima)")
    ap.add_argument("--refine_iters", type=int, default=80, help="LBFGS iterations for the refinement")
    ap.add_argument("--refine_edge_conf", default="none", choices=["none", "both"], help="'both' = edgeprob*flow-conf weighted chamfer; 'none' = uniform")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    EF.dev = dev
    g = scene_geometry(a.seq, "m3ed")
    EF.CH, EF.CW, EF.CX, EF.CY = g["crop"]; EF.MAXD = g["max_depth"]
    CH, CW, MAXD = EF.CH, EF.CW, EF.MAXD

    model = build_model("edge", True, a.ckpt, dev, [a.gpu])

    # registration dataset = the per-frame cam-frame local maps + events, in temporal order.
    # A test_RT is required by the loader but UNUSED here (we supply our own perturbation per frame).
    rt = os.path.join(a.data_path, f"test_RT_seq_{a.seq}_5.00_0.50.csv")
    kw = {"test_RT_path": rt} if os.path.exists(rt) else {}
    ds = DatasetM3ED(a.data_path, event_representation=a.ev_input, max_r=5, max_t=0.5,
                     split="test", test_sequence=a.seq, **kw)
    N = len(ds)

    # GT poses straight from the pose file, indexed by each file's own frame idx (no quaternion round-trip)
    pose_path = os.path.join(a.data_path, a.seq, f"{a.seq}_pose_gt.h5")
    Ln = h5py.File(pose_path, "r")["Ln_T_L0"][:].astype(np.float64)   # [Nframes,4,4] world->cam
    file_idx = [int(ds.all_files[i].split("/")[-1]) for i in range(N)]

    rng = np.random.default_rng(a.pweight_seed)
    def select(conf, gre):
        if a.selection == "all":     return np.arange(len(conf))
        if a.selection == "median":  return EF.sel_global(conf, 0.5)
        if a.selection == "floor0.5": return EF.sel_pweight(conf, 0.5, rng)
        return EF.sel_pweight(conf, 0.0, rng)                          # floor0 (deployed)

    guard = (f"guided_init(margin {a.margin_cm:g}cm/{a.margin_deg:g}deg)" if a.guided_init
             else f"reset={a.reset_cm}cm")
    print(f"[track] seq={a.seq} N={N} sel={a.selection} motion={a.motion} iters={a.iters} "
          f"guards(inlier>={a.min_inlier_rate}, step<={a.max_step_cm}cm, {guard})", flush=True)

    H_est = None
    rows = []                              # per-frame diagnostics
    H_est_hist, G_hist = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            s = ds[i]; ev = s["event_frame"].unsqueeze(0); pc = s["point_cloud"]; calib = s["calib"]
            G = Ln[file_idx[i]]                                        # GT world->cam for this frame

            reanchor = False
            if a.motion == "gt":            H_guess = G.copy()
            elif a.motion == "gt_prev":     H_guess = (G_hist[-1].copy() if G_hist else G.copy())
            elif i == 0 or H_est is None:   H_guess = G.copy()         # demo: frame 0 starts from GT
            else:
                if a.motion == "velocity" and len(H_est_hist) >= 2:    # constant-velocity extrapolation
                    H_pred = cv_predict(H_est_hist[-2], H_est_hist[-1], a.vel_gain)
                else:                                                  # position (or velocity cold-start at frame 1)
                    H_pred = H_est.copy()
                if a.guided_init:                                      # soft re-anchor into the training envelope
                    H_guess, reanchor = guided_init(H_pred, G, a.margin_cm, a.margin_deg)
                else:
                    H_guess = H_pred

            # perturbation relative to this frame's GT that realises the guess view
            RT = ortho(H_guess @ np.linalg.inv(G))
            R_err = quaternion_from_matrix(torch.tensor(RT)).float()   # (w,x,y,z)
            T_err = torch.tensor(RT[:3, 3]).float()

            dp = Data_preprocess(calib.unsqueeze(0), 3, 5, partial_fill="partial_light")
            ei, di, _, _ = dp.push_fuse(ev, [pc], T_err.unsqueeze(0), R_err.unsqueeze(0), dev,
                                        MAX_DEPTH=MAXD, split="test", h=CH, w=CW)
            di = di[:, 1:2, :, :]                                      # partial_light completion (ch1) = CELL input
            _, flow_up, depth2edge, cf = model(di, ei, iters=a.iters, test_mode=True, output_conf=True, idx=i)
            cm = F.softplus(cf)[0, 0].cpu().numpy()
            p2, p3, idxc, cp = EF.get_corr(flow_up, di, calib.unsqueeze(0))

            guess_err = float(np.linalg.norm(centre(H_guess) - centre(G)) * 100.0)
            held, reset, inl, step = True, reanchor, 0.0, 0.0        # reset flag doubles as the re-anchor marker
            sresid_t, sresid_r = float("nan"), float("nan")
            if p3.shape[0] >= 20:
                r = idxc[:, 0].astype(int); c = idxc[:, 1].astype(int); conf = cm[r, c]
                gre = EF.gt_reproj(p3.astype(np.float64), p2.astype(np.float64), R_err.numpy(), T_err.numpy(), cp)
                sel = select(conf, gre)
                if len(sel) >= 8:
                    o = EF.solve(p2[sel], p3[sel], cp)
                    if o is not None:
                        if a.refine:                                   # post-pipeline edge-matching refinement
                            o, _ = edge_refine(depth2edge, di, ei, calib, o, dev,
                                               (EF.CH, EF.CW, EF.CX, EF.CY), MAXD,
                                               lock_rot=not a.refine_full, anchor_w=a.refine_anchor_w,
                                               refine_iters=a.refine_iters, conf_map=cm,
                                               edge_conf=a.refine_edge_conf)
                        rr, tt = err_Pose(o[0], o[1], R_err, T_err)
                        sresid_t, sresid_r = float(tt), float(rr)      # how well THIS step's guess->true was corrected
                        RT_pred = mat_from_qt(o[0], o[1])
                        H_corr = ortho(np.linalg.inv(RT_pred) @ H_guess)
                        step = float(np.linalg.norm(centre(H_corr) - centre(H_guess)) * 100.0)
                        inl = len(sel) / max(1, int((di > 0).sum().cpu()))   # kept-point rate (proxy)
                        if step <= a.max_step_cm:
                            H_est = H_corr; held = False

            if held:                                                  # PnP failed / rejected -> hold the guess
                H_est = H_guess.copy()

            ape = float(np.linalg.norm(centre(H_est) - centre(G)) * 100.0)
            if not a.guided_init and a.reset_cm > 0 and ape > a.reset_cm:   # hard divergence reset (legacy)
                H_est = G.copy(); reset = True; ape = 0.0

            H_est_hist.append(H_est.copy()); G_hist.append(G.copy())
            rows.append((file_idx[i], guess_err, ape, sresid_t, sresid_r, step, inl, int(held), int(reset)))
            if i < 6 or i % 100 == 0:
                print(f"  [{i:04d}] guess_err={guess_err:6.1f}  APE={ape:6.1f}cm  "
                      f"step_resid={sresid_t:5.1f}cm/{sresid_r:4.2f}deg  {'HELD' if held else ''}{'RESET' if reset else ''}", flush=True)

    dt = time.time() - t0
    A = np.array([r[1:] for r in rows], dtype=float)
    guess_err, ape, sres_t, sres_r, stepv, inlv, heldv, resetv = [A[:, k] for k in range(8)]
    rmse = float(np.sqrt(np.mean(ape ** 2)))
    report = (
        f"=== CELL tracking: {a.seq} | sel={a.selection} | motion={a.motion} | iters={a.iters} ===\n"
        f"  frames            : {N}\n"
        f"  guess error (cm)  : med {np.median(guess_err):.1f}  p90 {np.percentile(guess_err,90):.1f}  max {guess_err.max():.1f}\n"
        f"  APE   (cm)        : med {np.median(ape):.1f}  mean {ape.mean():.1f}  rmse {rmse:.1f}  max {ape.max():.1f}\n"
        f"  final drift (cm)  : {ape[-1]:.1f}\n"
        f"  step-resid (cm)   : med {np.nanmedian(sres_t):.1f}  (per-frame guess->true correction quality)\n"
        f"  step-resid (deg)  : med {np.nanmedian(sres_r):.2f}\n"
        f"  held frames       : {int(heldv.sum())} ({100*heldv.mean():.1f}%)\n"
        f"  {'re-anchors        ' if a.guided_init else 'GT resets         '}: {int(resetv.sum())} ({100*resetv.mean():.1f}%)\n"
        f"  APE<10cm / <25cm  : {100*np.mean(ape<10):.1f}% / {100*np.mean(ape<25):.1f}%\n"
        f"  wall              : {dt:.1f}s ({1000*dt/max(1,N):.0f} ms/frame)\n"
    )
    print("\n" + report)

    tag = (f"_guided{int(a.margin_cm)}_{a.margin_deg:g}" if a.guided_init
           else (f"_reset{int(a.reset_cm)}" if a.reset_cm else ""))
    out_dir = a.out_dir or os.path.join("experiments/tracking_cell", a.seq, f"{a.selection}_{a.motion}" + tag)
    os.makedirs(out_dir, exist_ok=True)
    He = np.array(H_est_hist); Gg = np.array(G_hist)
    np.savez(os.path.join(out_dir, "trajectory.npz"),
             seq=a.seq, H_est=He, G_gt=Gg, file_idx=np.array(file_idx),
             guess_err=guess_err, ape=ape, step_resid_t=sres_t, step_resid_r=sres_r,
             step=stepv, inlier=inlv, held=heldv, reset=resetv,
             centre_est=np.array([centre(P) for P in He]),
             centre_gt=np.array([centre(P) for P in Gg]),
             selection=a.selection, motion=a.motion, iters=a.iters)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "ex", "ey", "ez", "gx", "gy", "gz", "ape_cm", "held", "reset"])
        for k in range(N):
            ce = centre(He[k]); cg = centre(Gg[k])
            w.writerow([file_idx[k], *ce, *cg, ape[k], int(heldv[k]), int(resetv[k])])
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
