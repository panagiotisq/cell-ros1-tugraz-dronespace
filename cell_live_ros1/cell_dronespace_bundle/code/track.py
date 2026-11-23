"""
LEAR pose *tracking* (Option C: world-frame maps).

Turns the per-frame LEAR registration model into a stateful open-loop tracker:
for each frame the initial guess is derived from the PREVIOUS estimate via a
motion model, the world-frame local map is projected by that guess, the model
predicts the flow, PnP recovers the correction, and the estimate is updated:

    P_guess = motion_model(P_est history)      # world -> event-cam
    depth   = project(P_guess @ pc_world)      # honest: no GT used
    flow    = model(depth, event_frame)
    RT_pred = Flow2Pose(flow, depth, calib)    # guess -> true correction
    P_est   = inv(RT_pred) @ P_guess           # (consistent with err_Pose)

Init: frame 0 starts from the exact GT pose. On PnP failure the estimate falls
back to the motion guess (never skips, or the track desyncs).

Nothing here modifies the registration pipeline; it reuses Data_preprocess,
Backbone_Edge_FF and Flow2Pose unchanged.
"""
import os
import csv
import json
import time
import argparse
import numpy as np
import torch

from core.data_preprocess import Data_preprocess
from core.flow2pose import Flow2Pose
from core.backbone import Backbone_Edge_FF, Backbone_Edge
from core.utils_point import to_rotation_matrix, quaternion_from_matrix
from tracking.track_dataset import TrackDataset
from tracking.metrics import compute_metrics, format_report

# M3ED half-resolution crop (same as core/main.py test path)
CROP_H, CROP_W, CROP_X, CROP_Y = 288, 512, 36, 64


class _Args:
    """Minimal args object supporting attribute access AND `in` (the backbone
    checks `'dropout' not in self.args`)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __contains__(self, k):
        return k in self.__dict__


def build_model(backbone, use_ff, ckpt, device, gpus):
    margs = _Args(mixed_precision=False)
    # infer the conf-head kernel from the checkpoint (if it has one) so shapes match on load
    sd = torch.load(ckpt, map_location="cpu")
    if "module.conf_cfg" in sd:
        # newer checkpoints save the exact head config: [conf_nets, use_net, use_cnet, use_motion, kernel]
        cfg = sd["module.conf_cfg"].tolist()
        margs.conf_nets = int(cfg[0]); margs.conf_kernel = int(cfg[4])
        feats = [n for n, u in zip(["net", "cnet", "motion"], cfg[1:4]) if int(u)]
        margs.conf_feats = ",".join(feats) if feats else "net"
    elif "module.conf_head.conv1.weight" in sd:
        # older net-only checkpoints: infer conf_nets from the input channels
        w = sd["module.conf_head.conv1.weight"]
        margs.conf_kernel = w.shape[-1]
        margs.conf_nets = max(1, w.shape[1] // 128)   # in_channels = 128 * conf_nets
        margs.conf_feats = "net"
    if backbone == "edge" and use_ff:
        net = Backbone_Edge_FF(margs)
    elif backbone == "edge":
        net = Backbone_Edge(margs)
    else:
        raise ValueError("tracking supports --backbone edge (optionally --use_feature_fusion)")
    model = torch.nn.DataParallel(net, device_ids=gpus)
    # strict=False: baseline checkpoints lack conf_head (stays fresh, unused); conf-head
    # checkpoints load it. Any OTHER missing/unexpected key is a real error -> report it.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "conf_head" not in k and "conf_cfg" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={real_missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def motion_guess(P_hist, i, motion):
    """Return the world->cam guess for frame i from the estimate history.
    P_hist[k] is the estimate for frame k (numpy 4x4)."""
    if i == 0:
        raise RuntimeError("frame 0 must be initialised from GT, not the motion model")
    if motion == "position" or i == 1:
        return P_hist[i - 1].copy()
    # constant velocity: extrapolate the last relative motion
    rel = P_hist[i - 1] @ np.linalg.inv(P_hist[i - 2])
    return rel @ P_hist[i - 1]


def track(args):
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, args.use_feature_fusion, args.load_checkpoints, device, args.gpus)

    ds = TrackDataset(args.data_path, args.test_sequence, args.ev_input)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers,
                                         collate_fn=lambda b: b[0], pin_memory=False)
    print(f"Tracking {args.test_sequence} | {args.ev_input} | motion={args.motion} | "
          f"{len(ds)} frames | iters={args.test_iters}")

    P_est_hist, P_gt_hist, P_guess_hist = [], [], []
    failed_hist, inlier_hist = [], []
    R_id = torch.tensor([[1., 0., 0., 0.]])
    T_id = torch.zeros(1, 3)
    warm = None            # 1/8-res flow carried from the previous frame (exp 2)
    lost_streak = 0
    model_time = 0.0       # accumulated model-forward time (isolates IFR-iters cost from I/O)
    def _cam_c(P):
        return np.linalg.inv(P)[:3, 3]
    t0 = time.time()

    with torch.no_grad():
        for i, s in enumerate(loader):
            if args.max_frames and i >= args.max_frames:
                break
            P_gt = s["P_gt"].numpy().astype(np.float64)
            calib = s["calib"]                                   # [4]
            event_frame = s["event_frame"].unsqueeze(0).to(device)   # [1,2,H,W]
            pc_world = s["pc_world"].to(device)                  # [4,M]

            # --- initial guess from motion model (frame 0 = exact GT) ---
            if args.motion == "gt":
                P_guess = P_gt.copy()                    # oracle: guess = THIS frame's GT
            elif args.motion == "gt_prev":
                # guess = PREVIOUS frame's GT pose: isolates the model's ability to
                # correct one inter-frame step, with NO error accumulation (the guess
                # never depends on the drifting estimate).
                P_guess = P_gt_hist[-1].copy() if P_gt_hist else P_gt.copy()
            elif i == 0:
                P_guess = P_gt.copy()
            else:
                P_guess = motion_guess(P_est_hist, i, args.motion)

            # --- project world map by the guess pose (honest: no GT) ---
            Pg = torch.tensor(P_guess, dtype=torch.float32, device=device)
            pc_guess = Pg @ pc_world                             # [4,M]

            # --- build depth_input via the SAME preprocessing as registration ---
            dp = Data_preprocess(calib.unsqueeze(0), args.occlusion_threshold, args.occlusion_kernel)
            event_input, depth_input, _, _ = dp.push_fuse(
                event_frame, [pc_guess], T_id, R_id, device,
                MAX_DEPTH=args.max_depth, split="test", h=CROP_H, w=CROP_W)
            depth_input = depth_input[:, 0, :, :].unsqueeze(1)

            # --- model + pose (exp 2: warm-start flow from previous frame) ---
            flow_init = warm if args.warm_start else None
            if device.type == "cuda":
                torch.cuda.synchronize()
            mt0 = time.time()
            _, flow_up, _ = model(depth_input, event_input, iters=args.test_iters,
                                  flow_init=flow_init, test_mode=True, idx=i)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_time += time.time() - mt0
            if args.warm_start:
                # carry this frame's flow to 1/8 res for the next frame's init
                warm = model.module.downsample_flow(flow_up.detach(), 0.125)
            R_pred, T_pred, inliers, pnp_failed = Flow2Pose(
                flow_up, depth_input, calib.unsqueeze(0), MAX_DEPTH=args.max_depth,
                x=CROP_X, y=CROP_Y, h=CROP_H, w=CROP_W)

            # Robust update: accept the correction only if PnP succeeded AND the
            # inlier rate is high enough to trust it. Otherwise HOLD the last
            # estimate (never extrapolate) so one bad frame can't feed a runaway
            # guess into the next step.
            inl = 0.0
            held = True
            if not pnp_failed:
                RT_pred = to_rotation_matrix(R_pred, T_pred).cpu().numpy().astype(np.float64)
                P_corr = np.linalg.inv(RT_pred) @ P_guess
                denom = float((depth_input > 0).sum().cpu())
                inl = len(inliers) / denom if denom > 0 else 0.0
                if inl >= args.min_inlier_rate:
                    held = False
            if held:
                P_est = P_est_hist[-1].copy() if P_est_hist else P_guess.copy()
            else:
                P_est = P_corr

            P_est_hist.append(P_est)
            P_gt_hist.append(P_gt)
            P_guess_hist.append(P_guess)
            failed_hist.append(bool(held))          # "held" = update rejected (fail or low-inlier)
            inlier_hist.append(inl)

            # --- optional early stop once clearly & persistently lost (saves time
            #     on diverging sweep configs; lifetime/in-lock are already decided) ---
            if args.stop_after_lost:
                ape_now = np.linalg.norm(_cam_c(P_est) - _cam_c(P_gt)) * 100.0
                lost_streak = lost_streak + 1 if ape_now > 100.0 else 0
                if lost_streak >= args.stop_after_lost:
                    print(f"  early stop at frame {i} ({args.stop_after_lost} consecutive frames >1 m)")
                    break

            if args.verbose and (i < 20 or i % 100 == 0):
                gt_c = np.linalg.inv(P_gt)[:3, 3]
                es_c = np.linalg.inv(P_est)[:3, 3]
                gs_c = np.linalg.inv(P_guess)[:3, 3]
                print(f"  [{i:05d}] guess_err={np.linalg.norm(gs_c-gt_c)*100:6.1f}cm  "
                      f"est_err={np.linalg.norm(es_c-gt_c)*100:6.1f}cm  inl={inl:.2f}  "
                      f"{'HELD' if held else ''}")

    dt = time.time() - t0
    m = compute_metrics(P_est_hist, P_gt_hist, P_guess=P_guess_hist,
                        failed=failed_hist, inlier=inlier_hist)
    nproc = len(P_est_hist)
    m["seconds_total"] = dt
    m["seconds_per_frame"] = dt / max(1, nproc)
    m["model_ms_per_frame"] = (model_time / max(1, nproc)) * 1000.0   # IFR-iters cost, I/O-free
    m["frames_processed"] = nproc
    m["config"] = {"test_sequence": args.test_sequence, "ev_input": args.ev_input,
                   "motion": args.motion, "test_iters": args.test_iters,
                   "warm_start": bool(args.warm_start), "checkpoint": args.load_checkpoints}

    # --- save outputs to a NEW folder (run_tag separates exp-2 sweep configs) ---
    sub = args.motion if not args.run_tag else os.path.join(args.motion, args.run_tag)
    out_dir = os.path.join(args.output_dir, args.test_sequence, args.ev_input, sub)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    report = format_report(m, title=f"=== Tracking: {args.test_sequence} | {args.ev_input} | "
                                     f"motion={args.motion} | iters={args.test_iters} ===")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report + "\n")
    # trajectory CSV: idx, est t+quat, gt t+quat
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ex", "ey", "ez", "eqw", "eqx", "eqy", "eqz",
                    "gx", "gy", "gz", "gqw", "gqx", "gqy", "gqz"])
        for k, (Pe, Pg) in enumerate(zip(P_est_hist, P_gt_hist)):
            ce = np.linalg.inv(Pe)[:3, 3]; cg = np.linalg.inv(Pg)[:3, 3]
            qe = np.asarray(quaternion_from_matrix(torch.tensor(Pe)))
            qg = np.asarray(quaternion_from_matrix(torch.tensor(Pg)))
            w.writerow([ds.idxs[k], *ce, *qe, *cg, *qg])

    print("\n" + report)
    print(f"\n  wall: {dt:.1f}s ({m['seconds_per_frame']*1000:.0f} ms/frame)")
    print(f"  saved -> {out_dir}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LEAR pose tracking (world-frame maps)")
    ap.add_argument("--data_path", default="data/m3ed")
    ap.add_argument("--test_sequence", required=True)
    ap.add_argument("--ev_input", default="iwe_evflownet")
    ap.add_argument("--load_checkpoints", required=True)
    ap.add_argument("--backbone", default="edge")
    ap.add_argument("--use_feature_fusion", action="store_true")
    ap.add_argument("--motion", choices=["position", "velocity", "gt", "gt_prev"], default="position")
    ap.add_argument("--test_iters", type=int, default=12)
    ap.add_argument("--max_depth", type=float, default=10.0)
    ap.add_argument("--occlusion_kernel", type=int, default=5)
    ap.add_argument("--occlusion_threshold", type=int, default=3)
    ap.add_argument("--min_inlier_rate", type=float, default=0.30,
                    help="reject (hold) a PnP update whose inlier rate is below this")
    ap.add_argument("--warm_start", action="store_true",
                    help="exp 2: initialise each frame's flow from the previous frame's flow")
    ap.add_argument("--stop_after_lost", type=int, default=0,
                    help="stop a run after N consecutive frames with APE>1 m (0=disabled)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--output_dir", default="experiments/tracking")
    ap.add_argument("--run_tag", default="", help="subfolder tag for a run (e.g. warm_it4)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all; else stop early (debug)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    track(args)
