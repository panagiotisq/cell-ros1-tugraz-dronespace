#!/usr/bin/env python3
"""CELL live localization process for the drone-space ROS1 deployment. Runs in the
isolated `cell_live` conda environment (launch/start_cell.sh) and talks to the ROS1
bridge purely over localhost IPC (cell_live.ipc); it never imports rospy or any ROS
message type, and it never opens the camera itself (see cell_live/ipc.py's "events" type
docstring for why: Metavision SDK python bindings on this PC are cp38-only, cell_live is
python 3.10). Supports both guess modes from DEPLOY_CELL_DRONESPACE.md sec 7:
Mode B (--guess-mode selftrack, the default): H_guess is CELL's own last accepted estimate
(optionally propagated by a constant-velocity model), with mocap re-seeding whenever
tracking is lost (--selftrack-lost-after consecutive non-valid ticks). Mode A
(--guess-mode mocap): H_guess is the live mocap pose every tick.

Two processes (like glace_live_ros1, same reason): CELL's model forward + poselib PnP
hold Python's GIL for the whole ~150-400ms tick, which would stall a same-process
"live tap" thread just as badly as it stalls everything else -- confirmed on-site (the
fast event-frame tap didn't actually look any faster than the per-result image while both
ran as threads in one process). The acquisition process (events + mocap guess receivers,
rolling buffer, the live-tap image) never imports torch, so its own GIL is never blocked
by CELL's compute; it hands the inference process the latest snapshot through a
"newest wins" multiprocessing.Queue.

    python scripts/run_cell_live.py
    python scripts/run_cell_live.py --interval-ms 500 --iters 16   # lighter load
"""

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cell_live import CHECKPOINT, PROJECT_ROOT, event_repr, use_deploy_package  # noqa: E402

use_deploy_package()

import numpy as np  # noqa: E402

from cell_live import map_loader  # noqa: E402
from cell_live.inference import CellLiveLocalizer  # noqa: E402
from tools.tracking.track_cell import cv_predict  # noqa: E402 -- pure numpy, no torch; see Mode B below

DEFAULT_MAP = PROJECT_ROOT.parent / "drone_space" / "map" / "fused_clean_mocap.ply"
DEFAULT_INTRINSICS = PROJECT_ROOT / "configs" / "intrinsics.json"


def check_isolation():
    """The CELL (inference) process must not see ROS Noetic or user-site packages. Unlike
    glace_live_ros1 (whose deployment bundle sits outside /workspaces/ros entirely), our
    bundle legitimately lives under /workspaces/ros -- so we only reject genuine
    contamination (the ROS system install, user site-packages), not our own paths."""
    bad = [p for p in sys.path if p.startswith("/opt/ros") or "/.local/lib/" in p]
    if bad:
        raise SystemExit(f"unsanitized sys.path entries {bad}: start with launch/start_cell.sh")
    import cv2
    import torch
    info = {m.__name__: (getattr(m, "__version__", "?"), m.__file__) for m in (np, cv2, torch)}
    prefix = sys.prefix
    for name, (_, path) in info.items():
        if not path.startswith(prefix):
            raise SystemExit(f"{name} resolved to {path}, outside this environment ({prefix})")
    return {**{k: v[0] for k, v in info.items()}, "python": sys.version.split()[0], "prefix": prefix,
            "cuda_available": torch.cuda.is_available()}


def load_half_res_calib(intrinsics_path):
    calib = json.loads(Path(intrinsics_path).read_text())
    cm = calib["camera_matrix"]["data"]
    fx, fy, cx, cy = cm[0], cm[4], cm[2], cm[5]
    full_w, full_h = calib["image_size"]
    return np.array([fx / 2.0, fy / 2.0, cx / 2.0, cy / 2.0], dtype=np.float32), (full_h // 2, full_w // 2)


def _install_stop_signals():
    import signal

    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _stop)


# --------------------------------------------------------------------------- acquisition process
# No torch import anywhere in this function or what it calls: a separate GIL from the
# inference process is the entire point.

def _acquisition_main(events_port, guess_port, result_port, half_h, half_w, guess_max_age_s,
                      guess_sync_tol_s, live_tap_period_s, window_s, repr_mode, repr_tau,
                      snapshot_queue, stop_event):
    import signal as _signal

    from cell_live import cell_sink, event_repr

    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)     # the parent handles Ctrl-C and sets stop_event

    buffer = event_repr.RollingEventBuffer(keep_s=0.15)
    events_rx = cell_sink.EventsReceiver(buffer, port=events_port, log=lambda *_: None)
    guess_rx = cell_sink.GuessReceiver(port=guess_port, max_age_s=guess_max_age_s, log=lambda *_: None)
    live_sink = cell_sink.ResultSink(port=result_port, K=None, send_vis=False, log=lambda *_: None)

    next_tick = time.monotonic()
    try:
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, min(live_tap_period_s, next_tick - now)))
                continue
            next_tick += live_tap_period_s
            if time.monotonic() > next_tick + live_tap_period_s:
                next_tick = time.monotonic() + live_tap_period_s

            snap = buffer.snapshot(window_s=window_s)
            ev_frame, n_events, w_end = None, 0, None
            if snap is not None:
                t, x, y, p, w_start, w_end = snap
                n_events = len(t)
                ev_frame = event_repr.build_frame(t, x, y, p, w_start, w_end, half_h, half_w,
                                                  mode=repr_mode, tau=repr_tau)
                if live_tap_period_s:
                    try:
                        live_sink.publish_live(ev_frame, time.time_ns())
                    except Exception:
                        pass    # best-effort display feed; never take the acquisition process down
            # Time-matched pairing, reinstated: now that EventForwarder's raw decode mode has
            # cut event staleness from 3000-4000ms down to ~50-140ms (see its class docstring),
            # a mocap sample within tol_s of the event window's end should almost always exist
            # (mocap runs at 100Hz), so this should rarely produce a dead tick the way it did
            # against the old multi-second backlog. Falls back to latest() only when there are
            # no events yet this tick (nothing to pair against).
            if w_end is not None:
                guess = guess_rx.at(w_end, tol_s=guess_sync_tol_s)
            else:
                guess = guess_rx.latest()      # (H_guess_c2w, age_s, source) or None

            item = (ev_frame, n_events, guess, w_end, time.time_ns())
            try:                            # "newest wins": drop whatever the inference process hasn't taken yet
                snapshot_queue.get_nowait()
            except queue_mod.Empty:
                pass
            try:
                snapshot_queue.put_nowait(item)
            except queue_mod.Full:
                pass
    finally:
        live_sink.close()
        guess_rx.close()
        events_rx.close()


def main():
    _install_stop_signals()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS))
    p.add_argument("--crop", type=int, nargs=4, default=[288, 512, 36, 64], metavar=("H", "W", "X", "Y"))
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--max-step-cm", type=float, default=None,
                   help="reject (hold) a correction whose translation jump from the guess exceeds "
                        "this many cm (default: inference.MAX_STEP_CM)")
    p.add_argument("--max-step-deg", type=float, default=None,
                   help="reject (hold) a correction whose rotation jump from the guess exceeds "
                        "this many degrees (default: inference.MAX_STEP_DEG)")
    p.add_argument("--min-inlier-rate", type=float, default=None,
                   help="reject (hold) a correction below this selected-point density "
                        "(default: inference.MIN_INLIER_RATE)")
    p.add_argument("--iters", type=int, default=12,
                   help="flow refinement iterations (default 12; measured live on this rig: iters=24 + "
                        "15m map crop -> 300-650ms/tick, iters=12 + 6m crop -> 220-400ms/tick)")
    p.add_argument("--interval-ms", type=float, default=150.0, help="fixed tick interval")
    p.add_argument("--map-crop-radius-m", type=float, default=6.0)
    p.add_argument("--guess-max-age-s", type=float, default=0.5,
                   help="skip a tick if the newest mocap guess is older than this (stale/disconnected)")
    p.add_argument("--guess-sync-tol-s", type=float, default=0.2,
                   help="pair each event window with the mocap sample closest to the window's own "
                        "end time, skipping the tick if none is within this tolerance -- tighter "
                        "than --guess-max-age-s, which only checks mocap is alive at all, not that "
                        "this specific guess/event pair are time-aligned. Mocap runs at 100Hz, so "
                        "0.2s should almost always find a match now that EventForwarder's raw "
                        "decode mode keeps event staleness around 50-140ms in practice (was 3-4s "
                        "before that fix, which is why this was loosened to 0.4s / briefly "
                        "disabled earlier -- see git history / README). The step_cm/inlier-rate "
                        "guards in inference.py still reject a bad correction regardless, so this "
                        "never reintroduces the original far-from-mocap 'valid' bug.")
    p.add_argument("--result-port", type=int, default=None)
    p.add_argument("--guess-port", type=int, default=None)
    p.add_argument("--events-port", type=int, default=None)
    p.add_argument("--no-vis", action="store_true", help="do not send the per-result event-frame image")
    p.add_argument("--live-tap-ms", type=float, default=60.0,
                   help="period of the fast display-only event-frame image, sent from the separate "
                        "acquisition process so it is never stalled by CELL's GIL-holding compute "
                        "(0 = disable)")
    p.add_argument("--guess-mode", choices=["mocap", "selftrack"], default="selftrack",
                   help="Mode B (selftrack, default): H_guess is CELL's own last accepted "
                        "estimate, with mocap re-seeding whenever tracking is lost (DEPLOY sec "
                        "7). Mode A (mocap): H_guess is the live mocap pose every tick -- note "
                        "that even in selftrack mode, mocap always seeds the very first estimate "
                        "and every recovery after a lost streak, so mocap must still be live.")
    p.add_argument("--selftrack-motion", choices=["position", "velocity"], default="position",
                   help="Mode B only: 'position' reuses the last accepted estimate unchanged "
                        "(zero-velocity, the more robust default); 'velocity' extrapolates with a "
                        "constant-velocity model (cv_predict) from the last two estimates -- only "
                        "used when the implied step stays within the step_cm/step_deg guards, "
                        "else falls back to 'position' for that tick (DEPLOY sec 7's robustification)")
    p.add_argument("--selftrack-vel-gain", type=float, default=1.0,
                   help="Mode B velocity motion: damping on the constant-velocity extrapolation "
                        "(1 = full predicted step, <1 damps it toward the last estimate)")
    p.add_argument("--selftrack-lost-after", type=int, default=5,
                   help="Mode B: after this many consecutive non-valid ticks, re-seed H_guess from "
                        "mocap for one tick instead of continuing to self-track")
    p.add_argument("--event-repr-mode", choices=["exp", "linear"], default=event_repr.DEFAULT_MODE,
                   help="per-pixel value formula for the event time-surface. 'exp' (default) "
                        "matches best_model.pth's actual training data (verified empirically on "
                        "the real training frames: min nonzero value = exp(-window/tau) exactly, "
                        "a hard floor -- see DEPLOY_CELL_DRONESPACE.md sec 4). 'linear' is the "
                        "earlier (incorrect, train/deploy-mismatched) formula, kept only in case a "
                        "future checkpoint is trained that way")
    p.add_argument("--event-repr-tau", type=float, default=event_repr.TAU_S,
                   help="decay constant (s) for --event-repr-mode=exp")
    args = p.parse_args()

    versions = check_isolation()
    print(f"environment: {versions}", flush=True)

    from cell_live import ipc
    result_port = args.result_port or int(os.environ.get("CLR1_RESULT_PORT", ipc.DEFAULT_RESULT_PORT))
    guess_port = args.guess_port or int(os.environ.get("CLR1_GUESS_PORT", ipc.DEFAULT_GUESS_PORT))
    events_port = args.events_port or int(os.environ.get("CLR1_EVENTS_PORT", ipc.DEFAULT_EVENTS_PORT))

    calib4, (half_h, half_w) = load_half_res_calib(args.intrinsics)
    print(f"half-res calib: fx={calib4[0]:.2f} fy={calib4[1]:.2f} cx={calib4[2]:.2f} cy={calib4[3]:.2f} "
          f"frame={half_h}x{half_w}", flush=True)

    print(f"loading map from {args.map} ...", flush=True)
    cmap = map_loader.CroppedMap.load(args.map)
    print(f"map: {cmap.xyz.shape[0]} points", flush=True)

    print("loading CELL model ...", flush=True)
    loc_kwargs = {}
    if args.max_step_cm is not None:
        loc_kwargs["max_step_cm"] = args.max_step_cm
    if args.max_step_deg is not None:
        loc_kwargs["max_step_deg"] = args.max_step_deg
    if args.min_inlier_rate is not None:
        loc_kwargs["min_inlier_rate"] = args.min_inlier_rate
    loc = CellLiveLocalizer(args.checkpoint, "cuda:0", crop=args.crop, max_depth=args.max_depth, iters=args.iters,
                           **loc_kwargs)
    loc.warmup()
    print("model ready (warmed up)", flush=True)

    from cell_live.cell_sink import ResultSink

    ctx = mp.get_context("spawn")
    snapshot_queue = ctx.Queue(maxsize=1)
    stop_event = ctx.Event()
    acq = ctx.Process(target=_acquisition_main,
                      args=(events_port, guess_port, result_port, half_h, half_w, args.guess_max_age_s,
                           args.guess_sync_tol_s, args.live_tap_ms / 1000.0, event_repr.WINDOW_S,
                           args.event_repr_mode, args.event_repr_tau, snapshot_queue, stop_event),
                      name="cell-acquisition", daemon=True)
    print(f"event representation: mode={args.event_repr_mode} tau={args.event_repr_tau}s "
         f"window={event_repr.WINDOW_S}s", flush=True)
    acq.start()
    sink = ResultSink(port=result_port, K=calib4, send_vis=not args.no_vis)

    print(f"waiting for the bridge on ports (results {result_port} <-, guess {guess_port} ->, "
          f"events {events_port} ->) ...", flush=True)

    frame_index = 0
    next_tick = time.monotonic()
    interval_s = args.interval_ms / 1000.0
    latest = None
    # Mode B (selftrack) state -- lives here, not in inference.py, since it's per-tick history
    # across calls, the same way track_cell.py's own tracking loop keeps it at the top level
    # rather than inside the per-frame model wrapper.
    track_hist_cv = []          # up to the last 2 (H_est_cv, w_end) ACCEPTED estimates (cam<-world)
    lost_streak = 0
    forcing_recovery = False    # True: use mocap this tick regardless (re-seeding after a lost streak)
    try:
        while True:
            if not acq.is_alive():
                raise SystemExit(f"acquisition process exited (code {acq.exitcode})")
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick += interval_s
            if time.monotonic() > next_tick + interval_s:   # fell far behind: resync instead of free-running
                next_tick = time.monotonic() + interval_s

            wall_end_ns = time.time_ns()
            while True:                      # drain the queue: take only the newest snapshot
                try:
                    latest = snapshot_queue.get_nowait()
                except queue_mod.Empty:
                    break
            if latest is None:
                frame_index += 1
                continue
            ev_frame, n_events, guess, w_end, _snap_ns = latest
            event_age_s = None if w_end is None else time.time() - w_end

            if guess is None:
                result = {"status": "skipped", "success": False, "H_guess_c2w": None, "H_est_c2w": None,
                         "inliers": 0, "step_cm": None, "step_deg": None, "reason": "no fresh mocap guess",
                         "timings_ms": {}}
                sink.publish(result, frame_index, None, 0, "none", wall_end_ns)
                print(f"[{frame_index:06d}] {result['status']:7s} reason={result['reason']}", flush=True)
                frame_index += 1
                continue
            H_mocap_c2w, guess_age_s, _ = guess       # the live mocap pose this tick, always available

            # ---- decide which guess to actually use this tick (Mode A vs Mode B) ----
            track_guess_source = "mocap"
            H_guess_c2w = H_mocap_c2w
            if args.guess_mode == "selftrack" and not forcing_recovery and len(track_hist_cv) >= 1:
                H1, t1 = track_hist_cv[-1]
                H_pred_cv = H1                            # position (zero-velocity) default/fallback
                if args.selftrack_motion == "velocity" and len(track_hist_cv) >= 2 and w_end is not None:
                    H2, t2 = track_hist_cv[-2]
                    observed_dt, target_dt = t1 - t2, w_end - t1
                    # Time-aware extrapolation: cv_predict's gain scales "one observed step"
                    # (t2->t1) to whatever horizon we actually need (t1->this tick's event
                    # window, w_end), instead of assuming uniform tick timing. Clamped so a
                    # freak tiny observed_dt (two accepted estimates landing close together)
                    # can't blow up into a wild extrapolation.
                    if observed_dt > 1e-3 and target_dt > 0:
                        eff_gain = args.selftrack_vel_gain * min(target_dt / observed_dt, 3.0)
                        H_vel_cv = cv_predict(H2, H1, eff_gain)
                        step_cm, step_deg = loc.pose_gap(H_vel_cv, H1)
                        if step_cm <= loc.max_step_cm and step_deg <= loc.max_step_deg:
                            H_pred_cv = H_vel_cv
                        # else: DEPLOY sec 7's robustification -- don't let the motion model
                        # extrapolate past a physical step bound; keep the zero-velocity default.
                H_guess_c2w = loc.ortho(np.linalg.inv(H_pred_cv))
                track_guess_source = "selftrack"

            if ev_frame is None:
                result = {"status": "skipped", "success": False, "H_guess_c2w": H_guess_c2w, "H_est_c2w": None,
                         "inliers": 0, "step_cm": None, "step_deg": None, "reason": "no events buffered yet",
                         "timings_ms": {}}
            else:
                cam_pos_world = H_guess_c2w[:3, 3]
                pc_h = cmap.crop_homogeneous(cam_pos_world, radius_m=args.map_crop_radius_m)
                result = loc.localize(H_guess_c2w, ev_frame, pc_h, calib4, n_events)

            # ---- update Mode B state from this tick's outcome ----
            if args.guess_mode == "selftrack":
                if result["success"] and w_end is not None:
                    H_est_cv = loc.ortho(np.linalg.inv(result["H_est_c2w"]))
                    # Timestamp this estimate by the event window it was actually computed
                    # from (w_end), not wall-clock publish time, so the velocity model's
                    # observed_dt/target_dt above reflect real elapsed capture time, not
                    # however long this tick's compute happened to take.
                    track_hist_cv = (track_hist_cv + [(H_est_cv, w_end)])[-2:]
                    lost_streak, forcing_recovery = 0, False
                else:
                    lost_streak += 1
                    if lost_streak >= args.selftrack_lost_after:
                        forcing_recovery = True    # re-seed from mocap next tick(s) until valid again

            sink.publish(result, frame_index, ev_frame, n_events, track_guess_source, wall_end_ns)
            lat = result.get("timings_ms", {}).get("total_ms")
            step_deg = result.get("step_deg")
            # age_ev/age_guess answer "which is stale, events or the mocap guess" directly:
            # age_ev = how old the event window's own latest timestamp already was at tick time
            # (camera->ROS->IPC->buffer latency); age_guess = same for the mocap sample used.
            # lat = pure compute time (preproc+model+pnp) on top of that staleness.
            age_ev = f"{event_age_s*1000:.0f}ms" if event_age_s is not None else "-"
            age_guess = f"{guess_age_s*1000:.0f}ms" if guess_age_s is not None else "-"
            print(f"[{frame_index:06d}] {result['status']:7s} guess={track_guess_source:9s} "
                  f"inliers={result.get('inliers', 0):4d} events={n_events:6d} "
                  f"step={result.get('step_cm')}cm/{step_deg}deg "
                  f"age_ev={age_ev:>6s} age_guess={age_guess:>6s} "
                  f"lat={f'{lat:.0f}ms' if lat else '-'} reason={result.get('reason', '')}", flush=True)
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        acq.join(timeout=3)
        sink.close()
        print(f"stopped. result sink stats: {sink.stats}", flush=True)


if __name__ == "__main__":
    main()
