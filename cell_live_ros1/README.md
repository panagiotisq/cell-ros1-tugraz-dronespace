# cell_live_ros1 — CELL live localization + mocap comparison in RViz (ROS 1 Noetic)

Live deployment of CELL (map-relative event-camera localization) on the drone-space rig,
against the aligned mocap-world map (`drone_space/map/fused_clean_mocap.ply`). Built
following `DEPLOY_CELL_DRONESPACE.md` (`cell_dronespace_bundle/`), using the same
two-process pattern as other live GLACE-family deployments on this rig: an isolated conda
process does the GPU work and talks to a ROS 1 bridge process over a localhost socket, so
the two never share binaries.

## Architecture

```
ROS1 Prophesee driver (external)          ROS 1 bridge (system python3, rospy)         CELL process (conda env `cell_live`)
/prophesee/.../cd_events_buffer   ------>  EventForwarder  ---- IPC :48813 (events) --->  RollingEventBuffer
/mocap_node/evk4/pose             ------>  GuessForwarder  ---- IPC :48812 (guess)  --->  (fixed-interval tick loop)
                                                                                            build event frame (sec 4)
                                                                                            crop map, push_fuse, model,
                                                                                            PnP (sec 6) -> result
                                            Ros1Bridge      <--- IPC :48811 (result) ----   ResultSink
                                            -> /cell/pose, /cell/path, TF, markers, ...
```

**Why events are forwarded by the bridge, not read by the CELL process directly** (unlike
glace_live_ros1, which opens the EVK4 itself): the Metavision SDK's python bindings
installed on this PC are built for CPython 3.8 only; `cell_live` is python 3.10 (CELL's own
dependency set — PyTorch 2.1.2, DSAC*-era extensions — doesn't need to match GLACE's env).
Reusing the already-running ROS1 `prophesee_ros_driver` (the same one used for dataset
recording all session) avoids a second camera-open contender and the version mismatch
entirely. The CELL process never imports rospy or any ROS message type.

## What you need to bring yourself (not in this repo)

Everything CELL-specific — the model, the two-process bridge, the launch scripts, the
camera bias profile — is in this repo and runs as-is once the large files (below) are
downloaded. Two things are deliberately **not** included, because they're tied to physical
hardware rather than to CELL:

- **An event-camera ROS1 driver**, publishing `prophesee_event_msgs/EventArray` on some
  topic (default `/prophesee/camera/cd_events_buffer`, override with `--events-topic`).
  For a Prophesee camera (what this rig uses, an EVK4) that's Prophesee's own open-source
  `prophesee_ros_wrapper` plus their Metavision SDK — install those for your camera model
  the normal Prophesee way; nothing CELL-specific is involved. `EventForwarder`
  (`cell_live/ros1_bridge.py`) hand-decodes that message's exact wire format for speed (see
  "Event decoding" below), so a non-Prophesee camera would need either a wrapper that
  publishes the same message type, or a small adaptation of that decoder.
- **A mocap ROS1 driver**, publishing plain `geometry_msgs/PoseStamped` on some topic
  (default `/mocap_node/evk4/pose`, override with `--mocap-topic`). This side is already
  vendor-agnostic — OptiTrack's `mocap_optitrack`, Vicon's `vicon_bridge`, or anything else
  that publishes `PoseStamped` works unmodified, since CELL only ever sees the pose message.

Both run against whatever ROS master `common.sh`/`CLR1_MASTER_PORT` points at (default
`11311`) — point your own camera/mocap drivers at the same master and the bridge picks them
up with no code changes, just the two `--*-topic` flags above if your topic names differ.

Everything else under `drone_space/` (intrinsics, extrinsic, the aligned map, the sim3
alignment) is **this specific room's** calibration/reconstruction data — a new space needs
its own calibration/reconstruction pass (see `drone_space/CALIBRATION.md`), not a copy of
these files.

## Status

Validated live on the physical rig (EVK4 + OptiTrack), not just in bag replay:

- **Environment**: `cell_live` conda env (python 3.10, torch 2.1.2+cu118); both CUDA
  extensions (`visibility`, `correlation_cuda`) build and load; `checkpoint/best_model.pth`
  loads with zero missing/unexpected keys and runs real inference on the GPU.
- **Geometry**: map depth rendered at a mocap-derived pose overlays correctly onto the real
  event frame (depth edges sit on event edges), confirming the mocap->camera composition,
  half-res intrinsics, map alignment and event representation are consistent together. The
  static map itself is also published to RViz as a point cloud (`/cell/map`, latched).
- **Both guess modes** (DEPLOY_CELL_DRONESPACE.md sec 7): Mode A (mocap-seeded `H_guess`
  every tick, `--guess-mode mocap`) and Mode B (CELL's own last accepted estimate with
  automatic mocap re-seeding when tracking is lost, `--guess-mode selftrack`, the default),
  including constant-velocity extrapolation between ticks (`--selftrack-motion velocity`).
- **Event/guess synchronization**: each tick pairs its event window with the mocap (or
  self-track) sample closest to the window's own end time (`GuessReceiver.at()`), not just
  "whatever's freshest" -- see `--guess-sync-tol-s`.
- **Event pipeline latency**: see "Event decoding" below -- staleness down from 3+ seconds
  to 50-150ms during fast motion.
- **Event representation**: see "Event-frame value formula" below -- matches the
  checkpoint's actual training data, confirmed both numerically and via an on-rig A/B
  (exponential decay gives roughly 2.5x the mean inliers and more than double the valid
  rate of the earlier, incorrect linear formula, under comparable motion).

## Known limitations

- **No display filter**: a filter that smooths/holds noisy poses for a less flickery RViz
  view (as in other GLACE-family live deployments) isn't implemented here -- every result
  is shown as-is.
- Timings from a real tick: preproc ~20ms / model ~200-400ms / PnP ~150-300ms depending on
  `--iters` and `--map-crop-radius-m` -- tune both if latency exceeds the `--interval-ms`
  budget.

## Environment setup (once)

**Download the large files first** (excluded from git, hosted on Google Drive --
<https://drive.google.com/drive/folders/1GzT_O74EwicPITg9dRQWXtwPS9glM1O0?usp=sharing>, see
`drone_space/dataset/DATASET.md`):
- `cell_dronespace_bundle/checkpoint/best_model.pth` (107MB) -- on Drive under `cell_checkpoint/`
- `../drone_space/map/fused_clean_mocap.ply` (60MB -- the same aligned room reconstruction
  used by `drone_space/dataset/`'s local-map generation, one upload covers both) -- Drive `map/`

Each file's SHA-256 is in `SHA256SUMS.txt` at the Drive folder root (`sha256sum -c SHA256SUMS.txt`).

```bash
conda create -n cell_live python=3.10 -y && conda activate cell_live
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==1.26.4" scipy opencv-python-headless h5py poselib mathutils==3.3.0 \
            matplotlib tensorboardX pandas
# g++ 9.x here only supports up to c++2a: mathutils>3.3.0 needs c++20 and will fail to build.

cd cell_dronespace_bundle/code/core/visibility_package  && CUDA_HOME=/usr/local/cuda-11.6 python setup.py build_ext --inplace
cd ../correlation_package                                  && CUDA_HOME=/usr/local/cuda-11.6 python setup.py build_ext --inplace
ln -sf visibility_package/visibility.cpython-310-*.so ../visibility.cpython-310-*.so
ln -sf correlation_package/correlation_cuda.cpython-310-*.so ../correlation_cuda.cpython-310-*.so
```

`cell_dronespace_bundle/code/track.py` and `eval_full.py` import a `tracking` package that
wasn't included in the bundle (offline-only dataset/metrics code, never called by the live
path) -- `code/tracking/{__init__,track_dataset,metrics}.py` stub it out; already in place.

## Run

```bash
# the external mocap + Prophesee driver stack must already be publishing on the master this uses
# (default: port 11311, the lab's normal master).
cd cell_live_ros1
launch/start_all.sh                      # bridge + RViz + CELL, in the background
launch/start_all.sh -- --iters 16        # lighter model load if latency is too high
launch/stop_all.sh
```

Or step by step: `start_bridge.sh`, `start_rviz.sh`, `start_cell.sh` (own terminal each).
Logs: `run/logs/*.log`.

## Event decoding: raw vs rospy

rospy's python deserialization of `prophesee_event_msgs/EventArray` (one Python object
built per event) has a hard throughput ceiling (~0.28M events/s) -- below what this EVK4 produces during vigorous
motion, which otherwise leaves the event data several seconds stale by the time a tick
uses it (CELL ends up matching "where the camera is right now" against "what the camera
saw seconds ago"). `EventForwarder` (`cell_live/ros1_bridge.py`) avoids this with
`decode_mode="raw"` (the default): subscribes with `rospy.AnyMsg` and hand-decodes the raw
serialized bytes with one `numpy.frombuffer()` call per message instead. Benchmarked
~450x faster (128M vs 0.28M events/s) and byte-for-byte validated against rospy's own
deserialization on live data (`tools/validate_raw_decode.py`, 60/60 messages matched
exactly, including bursts up to 640 events/message) before being made the default.

The original rospy-based method is kept intact, unmodified, side by side
(`_on_events_rospy`), selected by the same `decode_mode` switch -- a live, on/off config
choice, not a one-way migration:
```bash
launch/start_bridge.sh --event-decode rospy      # or: export CLR1_EVENT_DECODE=rospy
```

## Event-frame value formula: exponential decay, not linear

Verified empirically on the real training frames (`event_frame_XXXXX.npy`): the minimum
nonzero pixel value is `0.03568 = exp(-0.050/0.015)` exactly, a hard floor -- an
**exponential decay** (`drone_space/dataset/scripts/gen_tugraz.py`, `TAU=0.015s`), not a
linear ramp. (The M3ED pipeline's own `event2frame.py`, if you go looking for a reference
implementation elsewhere in the CELL codebase, uses linear normalization instead -- that's
a different pipeline and was never used to generate these training sequences; don't
confuse the two.) See `drone_space/dataset/DATASET.md` for the generator and
`DEPLOY_CELL_DRONESPACE.md` sec 4 for the full spec.

`event_repr.build_frame()` takes `mode="exp"|"linear"` (default `"exp"`, matching
`best_model.pth`), exposed as `--event-repr-mode` / `--event-repr-tau` on
`run_cell_live.py` (defaults `exp`, `0.015`). `linear` is kept, not deleted, in case a
future checkpoint is trained that way -- it's a config choice, not a hardcoded mismatch.
An on-rig A/B between the two (same rig, comparable motion) gave `exp` roughly 2.5x the
mean inliers and more than double the valid rate of `linear`, consistent with `exp` being
what the checkpoint actually saw during training.

## What you'll see in RViz

| Colour | Meaning |
|---|---|
| green frustum | CELL pose (correction accepted: inlier rate + step within bounds) |
| orange frustum | CELL held (guess shown instead; correction rejected this tick) |
| blue frustum | live mocap-derived guess (the pose CELL is refining) |
| green/blue lines | CELL / mocap-guess trajectories |
| image panel | live event-frame visualization |
