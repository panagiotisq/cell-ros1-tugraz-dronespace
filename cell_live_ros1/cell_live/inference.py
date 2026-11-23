"""Live CELL inference wrapper: one tick = one call to .localize(). Ports
DEPLOY_CELL_DRONESPACE.md sec 6 (single-frame recipe) using the SAME helper functions the
offline tools use (eval_full.get_corr/solve/sel_pweight, Data_preprocess.push_fuse,
tools.tracking.track_cell.{ortho,mat_from_qt,centre,cv_predict}) -- nothing re-derived.

Convention used throughout THIS module and the IPC schema (note: the OPPOSITE of
DEPLOY_CELL_DRONESPACE.md's own "H_guess" notation, chosen to match ROS/RViz's T_W<-C
convention, same as glace_live_ros1):
    *_c2w   camera pose IN THE WORLD (T_world<-cam), 4x4, what you'd publish as PoseStamped
    *_cv    the CV/CELL extrinsic (T_cam<-world = inv(*_c2w)), what push_fuse/get_corr/solve
            actually consume ("H_guess"/"RT" in the deploy doc)
"""

from __future__ import annotations

import time

import numpy as np

#   track_cell.py's own defaults: min_inlier_rate 0.30, max_step_cm 40 (translation only, no
# rotation guard at all) -- tuned for OFFLINE SELF-TRACKING, where the guess is the tracker's
# own previous (possibly already-drifted) estimate, so a larger legitimate correction is
# expected. In live Mode A the guess is the live mocap pose (near-ground-truth every tick),
# so an accepted correction should be small; a large "valid" jump away from an already-accurate
# guess is a sign of a bad PnP solve, not a real correction -- confirmed on-site (a 40cm-class
# "valid" result was visibly far from the live mocap frustum in RViz). max_step_cm: tightened
# to 15cm, then relaxed to 25cm after confirming on-site that most ticks were being held.
# min_inlier_rate: confirmed on-site that 115/124 held results were gated by this (not by
# max_step_cm/deg, which bound 9 and 0 respectively) -- loosened from 0.30 to 0.15. A rotation
# guard is new here (track_cell.py never had one): the same "guess is already accurate"
# argument applies equally to rotation; relaxed from 12 to 15deg, though note it has not
# actually been the binding constraint on any held result yet.
MIN_INLIER_RATE = 0.15
MAX_STEP_CM = 25.0
MAX_STEP_DEG = 15.0
MIN_EVENTS = 500           # skip inference on a near-empty window


class CellLiveLocalizer:
    def __init__(self, checkpoint, device, crop=(288, 512, 36, 64), max_depth=10.0, iters=24,
                 min_inlier_rate=MIN_INLIER_RATE, max_step_cm=MAX_STEP_CM, max_step_deg=MAX_STEP_DEG,
                 min_events=MIN_EVENTS):
        import torch
        import torch.nn.functional as F
        import eval_full as EF
        from track import build_model
        from core.data_preprocess import Data_preprocess
        from tools.tracking.track_cell import mat_from_qt, ortho, pose_gap

        self.torch, self.F, self.EF = torch, F, EF
        self.Data_preprocess = Data_preprocess
        self.ortho, self.mat_from_qt, self.pose_gap = ortho, mat_from_qt, pose_gap

        self.dev = torch.device(device)
        self.CH, self.CW, self.CX, self.CY = [int(v) for v in crop]
        self.MAXD = float(max_depth)
        self.iters = int(iters)
        self.min_inlier_rate, self.min_events = min_inlier_rate, min_events
        self.max_step_cm, self.max_step_deg = max_step_cm, max_step_deg
        self.rng = np.random.default_rng(0)

        EF.dev = self.dev
        EF.CH, EF.CW, EF.CX, EF.CY, EF.MAXD = self.CH, self.CW, self.CX, self.CY, self.MAXD

        self.model = build_model("edge", True, str(checkpoint), self.dev, [0])

    def warmup(self):
        """One dummy forward pass to pay CUDA/cuDNN init cost before the first live tick."""
        torch = self.torch
        di = torch.zeros(1, 1, self.CH, self.CW, device=self.dev)
        ei = torch.zeros(1, 2, self.CH, self.CW, device=self.dev)
        with torch.no_grad():
            self.model(di, ei, iters=self.iters, test_mode=True, output_conf=True, idx=0)
        torch.cuda.synchronize()

    def localize(self, H_guess_c2w, ev_frame_2ch, pc_world_h, calib4, n_events):
        """
        H_guess_c2w  : (4,4) float64, guess camera pose in world (T_world<-cam)
        ev_frame_2ch : (360, 640, 2) float32 [0,1], cell_live.event_repr.build_frame output
        pc_world_h   : (4, N) float32, map points already cropped near the guess (homogeneous)
        calib4       : (4,) float32 [fx, fy, cx, cy] at the HALF-RES (640x360) resolution
        n_events     : int, events actually in the window (for the skip gate + diagnostics)

        Returns a dict: status in {"valid","held","skipped"}, success, H_est_c2w (or None),
        inliers, step_cm, reason, timings_ms.
        """
        torch, F, EF = self.torch, self.F, self.EF
        t_all = time.monotonic()
        if n_events < self.min_events:
            return self._result("skipped", None, H_guess_c2w, 0, None, None,
                                f"only {n_events} events (< {self.min_events})", {})

        H_guess_cv = self.ortho(np.linalg.inv(H_guess_c2w))
        from core.utils_point import quaternion_from_matrix
        R_err = torch.as_tensor(quaternion_from_matrix(torch.tensor(H_guess_cv))).float()   # (w,x,y,z)
        T_err = torch.as_tensor(H_guess_cv[:3, 3]).float()

        ev_t = torch.from_numpy(ev_frame_2ch).permute(2, 0, 1).float().to(self.dev)
        pc_t = torch.from_numpy(pc_world_h).to(self.dev)
        calib_t = torch.as_tensor(calib4, dtype=torch.float32)

        timings = {}
        t0 = time.monotonic()
        dp = self.Data_preprocess(calib_t.unsqueeze(0), 3, 5, partial_fill="partial_light")
        ei, di, _, _ = dp.push_fuse(ev_t.unsqueeze(0), [pc_t], T_err.unsqueeze(0), R_err.unsqueeze(0), self.dev,
                                    MAX_DEPTH=self.MAXD, split="test", h=self.CH, w=self.CW)
        di = di[:, 1:2, :, :]                        # ch1 = partial_light completion (sec 5)
        timings["preproc_ms"] = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        with torch.no_grad():
            _, flow_up, _, cf = self.model(di, ei, iters=self.iters, test_mode=True, output_conf=True, idx=0)
        torch.cuda.synchronize()
        timings["model_ms"] = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        cm = F.softplus(cf)[0, 0].cpu().numpy()
        p2, p3, idxc, cp = EF.get_corr(flow_up, di, calib_t.unsqueeze(0))
        if p3.shape[0] < 20:
            timings["pnp_ms"] = (time.monotonic() - t0) * 1000
            return self._result("held", None, H_guess_c2w, 0, None, None,
                                f"too few correspondences ({p3.shape[0]} < 20)", timings, t_all)
        r, c = idxc[:, 0].astype(int), idxc[:, 1].astype(int)
        conf = cm[r, c]
        sel = EF.sel_pweight(conf, 0.0, self.rng)                 # floor0, the deployed default
        if len(sel) < 8:
            timings["pnp_ms"] = (time.monotonic() - t0) * 1000
            return self._result("held", None, H_guess_c2w, 0, None, None,
                                f"too few selected points ({len(sel)} < 8)", timings, t_all)
        o = EF.solve(p2[sel], p3[sel], cp)
        timings["pnp_ms"] = (time.monotonic() - t0) * 1000
        if o is None:
            return self._result("held", None, H_guess_c2w, 0, None, None, "PnP solve failed", timings, t_all)

        inlier_rate = len(sel) / max(1, int((di > 0).sum().cpu()))
        RT_pred = self.mat_from_qt(o[0], o[1])
        H_est_cv = self.ortho(np.linalg.inv(RT_pred) @ H_guess_cv)
        step_cm, step_deg = self.pose_gap(H_est_cv, H_guess_cv)

        if inlier_rate < self.min_inlier_rate:
            return self._result("held", None, H_guess_c2w, len(sel), step_cm, step_deg,
                                f"inlier rate {inlier_rate:.2f} < {self.min_inlier_rate}", timings, t_all)
        if step_cm > self.max_step_cm:
            return self._result("held", None, H_guess_c2w, len(sel), step_cm, step_deg,
                                f"step {step_cm:.1f}cm > {self.max_step_cm}cm (PnP blunder guard)", timings, t_all)
        if step_deg > self.max_step_deg:
            return self._result("held", None, H_guess_c2w, len(sel), step_cm, step_deg,
                                f"step {step_deg:.1f}deg > {self.max_step_deg}deg (PnP blunder guard)", timings, t_all)

        H_est_c2w = self.ortho(np.linalg.inv(H_est_cv))
        return self._result("valid", H_est_c2w, H_guess_c2w, len(sel), step_cm, step_deg, "", timings, t_all)

    @staticmethod
    def _result(status, H_est_c2w, H_guess_c2w, inliers, step_cm, step_deg, reason, timings, t_all=None):
        timings = dict(timings)
        if t_all is not None:
            timings["total_ms"] = (time.monotonic() - t_all) * 1000
        return {"status": status, "success": status == "valid", "H_est_c2w": H_est_c2w,
                "H_guess_c2w": H_guess_c2w, "inliers": int(inliers), "step_cm": step_cm, "step_deg": step_deg,
                "reason": reason, "timings_ms": timings}
