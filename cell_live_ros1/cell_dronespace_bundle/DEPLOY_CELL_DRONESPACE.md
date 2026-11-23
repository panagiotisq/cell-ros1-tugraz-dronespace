# CELL — Drone-Space Live Deployment Handoff

This bundle contains everything needed to run **CELL** (map-relative event-camera
localization) live on the drone-space demo PC against the aligned mocap-world map.
It is the source of truth for how the drone-space (tugraz) sequences were trained and
must be handled. Read this top to bottom before wiring ROS.

Prepared on the lab PC (where the code, config, and checkpoint actually live).

---

## 0. TL;DR — the decisions, in one place

| Thing | Decision |
|---|---|
| Event representation | **2-channel**, per-polarity **latest-timestamp time surface**, **50 ms** window, **half-res 640×360**, normalized to [0,1]. NOT 1-channel. NOT STC/Trail. |
| Crop (h,w,x,y) | **(288, 512, 36, 64)** |
| Calibration | **half-res intrinsics** (full-res / 2) |
| max_depth | **10.0 m** (indoor z-ceiling) |
| Depth input channel | **ch1 = partial_light completion** (`di[:,1:2]`), not sparse ch0 |
| Backbone | `"edge"` → `Backbone_Edge_FF` (flow + edge + confidence head) |
| Inference iters | **24** (drop to 12 for frame-rate; safe — conf snapshots are absolute-indexed) |
| Correspondence selection | **floor0** probabilistic (`sel_pweight(conf, 0.0)`) |
| Checkpoint | `checkpoint/best_model.pth` (seq3 edge, decoupled-β1.5, partial_light, **half-pert**) |
| Map | `fused_clean_mocap.ply` (already on demo PC, mocap-world frame) |
| Pose guess (live) | **Mode A: mocap-seeded** (bring-up). **Mode B: self-track + mocap recovery** (research demo). **Never** pure self-track — it diverges. |

---

## 1. Bundle layout

```
DEPLOY_CELL_DRONESPACE.md        <- this file
code/
  core/                          <- CELL inference code (backbone, geometry, PnP, config)
    backbone.py extractor.py corr.py update.py model_builder.py
    data_preprocess.py camera_model.py depth_completion.py
    flow2pose.py scene_config.py utils.py utils_point.py utils_events.py
    quaternion_distances.py datasets_m3ed.py ...
    visibility_package/          <- CUDA ext SOURCE (must recompile on demo PC)
    correlation_package/         <- CUDA ext SOURCE (must recompile on demo PC)
  eval_full.py                   <- get_corr / solve / sel_pweight / gt_reproj (reused live)
  track.py                       <- build_model()
  tools/tracking/track_cell.py   <- the offline sequential tracker (reference for the live loop)
  tools/tracking/viz_trajectory.py
  tools/pose_refine/             <- optional edge-matching refinement (edge_extract.py, refine.py)
checkpoint/
  best_model.pth                 <- the trained drone-space model (112 MB)
```

Note: the extension **build artifacts were intentionally excluded** — the `.so` files
are compiled for this PC's torch/CUDA/cpython-310 and will not load on the demo PC.
The `.cpp/.cu/setup.py` sources are included so you can rebuild in-place (§3).

---

## 2. What the demo PC already has vs. what this bundle supplies

**Already on the demo PC (confirmed by the deployment session):**
- Aligned map: `drone_space/map/fused_clean_mocap.ply` (1.22 M pts, mocap-world frame)
- Sim3 alignment: `drone_space/align/sim3_colmap_to_mocap.json`
- Calibration: `drone_space/calib/intrinsics.json`, `T_body_camera.txt` (camera→body)
- ROS1 Noetic + an external mocap + EVK4 event driver stack, proven live
- GPU: GTX 1080 Ti (11 GB); torch-capable conda env (`glace_live38`)
- The two-process conda-isolation + IPC-bridge pattern from `glace_live_bundle` (copy it)

**Supplied by this bundle (was missing on the demo PC):**
- The CELL code + extension sources (`code/`)
- The trained checkpoint (`checkpoint/best_model.pth`)
- This config/spec document

**Repoint:** `cell_inference.py`'s stub points at a path that won't exist on a new
machine. Point it at wherever you unpack `code/`.

---

## 3. Environment setup on the demo PC

Use a dedicated conda env (do **not** fold CELL into the rospy/system Python — keep the
torch/CUDA process isolated the same way other live GLACE-family deployments on this rig
do, talking to the ROS1 bridge over the localhost IPC socket).

```bash
conda create -n cell_live python=3.10 -y && conda activate cell_live
# torch matching the 1080 Ti / installed CUDA (CUDA 11.x is fine for sm_61):
pip install torch==2.1.* torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy opencv-python h5py poselib mathutils
#   ^ mathutils is pip-installable (we use 3.3.0). poselib provides the PnP solver.

# Recompile the two CUDA extensions IN-PLACE against this env's torch/CUDA:
cd code/core/visibility_package  && python setup.py build_ext --inplace && cd -
cd code/core/correlation_package && python setup.py build_ext --inplace && cd -
#   -> produces visibility*.so and correlation*.so importable from core/
```

Sanity check: `python -c "import torch,visibility,correlation,mathutils,poselib; print('ok')"`

Runtime note (seen on the lab PC): always launch with
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` so torch picks up the env's libs, and set
`PYTHONUNBUFFERED=1` for live logging.

---

## 4. THE EVENT REPRESENTATION (authoritative — matches the trained model)

A stub `event_representation.py` elsewhere builds a 2-ch STC/Trail surface, and the room's
`EVENT_REPRESENTATION.md` specifies a **1-channel** exp-decay surface. **Both are wrong for
this checkpoint.** `code/tools/m3ed/event2frame.py` (the unrelated M3ED pipeline's own
frame generator, if you encounter it elsewhere in this codebase) uses **linear**
normalization (age / window) — also not what this checkpoint was trained on; don't confuse
the two.

Verified empirically on the real `event_frame_XXXXX.npy` files
(`drone_space/dataset/scripts/gen_tugraz.py`, the actual generator for this checkpoint's
training data): the minimum nonzero value is **0.03568 = exp(-0.050/0.015) exactly**, a
hard floor with zero values below it — an **exponential decay**, not linear.

- The event encoder is `BasicEncoder(input_dim=2)` → **exactly 2 channels required.**
- A real training frame
  (`event_frames_ours_50ms_half/left/event_frame_00000.npy`) has
  **shape `(360, 640, 2)`, dtype float32, values in `[0.03568, 1.0]` for nonzero pixels, 0
  for empty ones.**

### Exact spec to reproduce live from the EVK4 stream
For each pose/inference tick, over the **last 50 ms** of events (`WIN = 0.050`):
1. Split events by **polarity** → channel 0 = one polarity, channel 1 = the other.
2. For each pixel, keep the **timestamp of the most recent event** in that channel
   (a latest-timestamp *time surface*, not a count and not an IWE).
3. `value = exp(-(t_end - t_event) / TAU)`, `TAU = 0.015` (15 ms); empty pixels = 0.
4. **Half resolution:** native EVK4 1280×720 → **640×360** (factor-2 downsample), so the
   stored array is `(H=360, W=640, C=2)`.

It is the plain time-surface method with a 50 ms window — **STC/Trail denoising is NOT
applied** (the proprietary Metavision filters were not installed on the lab PC that made
the training frames, so the model never saw them). Don't add them live; it would be a
train/test mismatch.

Reference implementation: `gen_tugraz.py` (the real generator for this checkpoint's
training data; see `dataset/DATASET.md` in the deployment repo). Not
`code/tools/m3ed/event2frame.py` — see above.

Live implementation: `cell_live/event_repr.py`'s `build_frame()`, `mode="exp"` (the
default). A `mode="linear"` path is kept there too, selectable via
`--event-repr-mode linear` on `run_cell_live.py`, in case a future checkpoint is trained
on that formula instead — but it does **not** match `best_model.pth`.

---

## 5. GEOMETRY / MODEL CONFIG (the "dronespace_room.yaml" equivalent)

From `core/scene_config.py`, every tugraz sequence is class **indoor**:

```
crop (h, w, x, y) = (288, 512, 36, 64)     # center crop of the 360x640 frame
calib_div         = 2.0                     # feed HALF-RES intrinsics (full-res / 2)
max_depth         = 10.0                    # metres; used as MAX_DEPTH in push_fuse / get_corr
native frame_hw   = (360, 640)
backbone          = "edge"  -> Backbone_Edge_FF
inference iters   = 24
depth input       = ch1 (partial_light completion), i.e. di[:, 1:2]
```

**Calibration handling:** the event frame is half-res, so the intrinsics fed to CELL must
be **half-res**. If `intrinsics.json` is full-res (1280×720), divide fx, fy, cx, cy by 2
before passing them. The crop offset (x=36, y=64) is handled internally by
`push_fuse`/`get_corr` (it shifts the principal point into the cropped frame) — don't
pre-apply it.

---

## 6. THE LIVE INFERENCE RECIPE (single frame)

CELL is **map-relative**: it refines a coarse guess pose against the map. It does NOT
produce a pose from nothing. One frame:

```python
# inputs this tick:
#   H_guess : 4x4 world->CAMERA guess pose (float64)   -- see §7 for where it comes from
#   ev      : event frame tensor (1, 2, 360, 640), float32 in [0,1]
#   pc      : the LiDAR/recon map points in WORLD (mocap) frame, (N,3) or (N,4) homogeneous
#   calib   : [fx, fy, cx, cy]  HALF-RES

# 1) express the perturbation as the guess pose itself (world-map formulation):
#    feeding pc in WORLD frame with RT = H_guess makes push_fuse render the map
#    exactly as seen from the guess camera (pc_rotated = H_guess @ X_world). NO GT used.
RT    = ortho(H_guess)                         # world->cam guess
R_err = quaternion_from_matrix(torch.tensor(RT)).float()   # (w,x,y,z)
T_err = torch.tensor(RT[:3, 3]).float()

# 2) build the network inputs (depth rendered from the guess + the event image):
dp = Data_preprocess(calib.unsqueeze(0), 3, 5, partial_fill="partial_light")
ei, di, _, _ = dp.push_fuse(ev, [pc], T_err.unsqueeze(0), R_err.unsqueeze(0), dev,
                            MAX_DEPTH=10.0, split="test", h=288, w=512)
di = di[:, 1:2, :, :]                          # partial_light completion = CELL depth input

# 3) forward pass -> flow + per-correspondence confidence:
_, flow_up, depth2edge, cf = model(di, ei, iters=24, test_mode=True, output_conf=True)
cm = F.softplus(cf)[0, 0].cpu().numpy()

# 4) correspondences -> confidence selection (floor0) -> weightless PnP:
p2, p3, idxc, cp = EF.get_corr(flow_up, di, calib.unsqueeze(0))
conf = cm[idxc[:,0].astype(int), idxc[:,1].astype(int)]
sel  = EF.sel_pweight(conf, 0.0, rng)          # floor0 (deployed default)
o    = EF.solve(p2[sel], p3[sel], cp)          # poselib PnP -> RT_pred (quat, transl)

# 5) update the absolute pose:
RT_pred = mat_from_qt(o[0], o[1])
H_est   = ortho(np.linalg.inv(RT_pred) @ H_guess)   # corrected world->cam
```

`ortho`, `mat_from_qt`, `centre`, `cv_predict` (constant-velocity) are in
`tools/tracking/track_cell.py` — lift them. The offline tracker uses per-frame
**cam-frame map crops** (pc_k = G_k @ X_world, RT = H_guess @ inv(G_k)); the live loop uses
the **world-map formulation** above (pc in world, RT = H_guess) — algebraically identical,
and it uses **no ground truth**.

**Speed:** crop the 1.22 M-point map to the guess camera frustum before `push_fuse` (keep
points in front of the camera within max_depth and the FoV). Expect ~120–230 ms/frame on
the 1080 Ti at iters=24; iters=12 roughly halves the flow cost.

**Health signals per frame (use these for Mode B recovery, §7):** PnP inlier count
(`len(sel)` and the poselib inlier set), and the step magnitude
`||centre(H_est) − centre(H_guess)||`. A tiny inlier count or a large step = bad frame.

---

## 7. WHERE THE GUESS COMES FROM — the live architecture decision

CELL refines a guess inside a **±5° / 0.5 m basin**. The guess source each frame is the
one real design choice. **Empirical finding (measured on seq3, GT-free free-running):**

| self-tracking mode (no recovery) | result |
|---|---|
| position init (reuse last estimate) | median APE **154 cm**, 42% frames lost |
| velocity init (constant-velocity) | **runaway to 55 m** (dead-reckoning compounds on dropouts) |

→ **Pure self-tracking is not deployable.** Plan:

### Mode A — mocap-seeded (START HERE, bring-up + robust demo)
Each frame, set `H_guess` = the **live mocap pose** (world→body from mocap, composed with
`T_body_camera` → world→cam). CELL refines it against the map. In *this lab* mocap is a
live sensor, so this is a legitimate live run, guaranteed inside the basin, smooth. It
proves the whole EVK4→time-surface→CELL→PnP→pose stack end-to-end with minimal risk, and
shows CELL agreeing with mocap live. **Do this first.**

### Mode B — self-track + mocap recovery (the research demo)
Seed frame 0 from mocap. Then `H_guess` = CELL's own last estimate propagated by the
**constant-velocity motion model** (`cv_predict`, added to `track_cell.py`). Monitor the
health signals (§6); when tracking is lost (inlier rate low / step too large for N frames),
**re-seed `H_guess` from mocap** for one frame and continue. Mocap-on-lost replaces the
GT re-anchor from the offline demo and stands in for the global relocalizer you'd use
*outside* the lab. This demonstrates autonomous map-relative tracking with mocap only as a
safety net.

Make it a runtime switch (`--guess {mocap, selftrack}`), defaulting to `mocap` for bring-up.

### Important: robustify the motion model before Mode B
The constant-velocity model (`--motion velocity`) must only extrapolate from **confirmed**
frames and must **not** extrapolate off a held/failed frame (that is what caused the 55 m
runaway). When wiring Mode B: update the velocity only from non-held estimates, fall back
to a static hold on a held frame, and clamp the predicted inter-frame step to a physical
max. (Open improvement — not yet in the offline tracker.)

---

## 8. COORDINATE FRAMES — get this right or everything is silently off

- **Map** `fused_clean_mocap.ply` is in the **mocap world frame** (aligned via the sim3).
- **Poses** from mocap are **world→body** (or body→world — check the mocap driver's convention).
- CELL needs **world→CAMERA**: compose the mocap body pose with **`T_body_camera`**
  (camera→body on the demo PC, so invert as needed) to get `H_guess` (world→cam).
- Units are **metres** throughout (map, mocap, max_depth=10). No scale conversion.
- The map, the poses, and the predicted `H_est` all live in the **same mocap world frame**,
  so you can overlay `H_est` against mocap directly to measure live tracking error.

Validate the frames offline first: pick one logged mocap pose, render the map depth from it
with the recipe in §6, and overlay against the live event frame — the depth edges should sit
on the event edges. If they don't, the frame composition (§8) or the half-res calib (§5) is
wrong. Fix that before going live.

---

## 9. Direct answers to the demo-PC session's questions

1. **"Was the 2-ch STC/Trail stub or the 1-ch exp-decay markdown correct?"** → **Neither.**
   It is a **2-channel, 50 ms, per-polarity latest-timestamp time surface at half-res**,
   no STC/Trail. Rebuild `event_representation.py` to §4.
2. **"What channel/resolution was the model trained at?"** → **2 channels, 640×360
   half-res**, cropped to 288×512, calib/2, max_depth 10. (§4–§5.)
3. **"Where does the per-frame coarse pose guess come from?"** → **Both, switchable;
   default mocap-seeded (Mode A) for bring-up; self-track + mocap-recovery (Mode B) for the
   research demo. Never pure self-track — confirmed to diverge.** (§7.)
4. **"Trained model / aligned map — do you have them?"** → Checkpoint is in this bundle;
   the aligned map is already on the demo PC. (§2.)

---

## 10. Gotchas / known issues

- **Recompile the extensions** — do not copy `.so` files from the lab PC (built for a
  different torch/CUDA/cpython). (§3.)
- **Half-res calib** — if `intrinsics.json` is full-res, divide by 2. (§5.)
- **Depth channel** — feed `di[:, 1:2]` (partial_light completion), not ch0. (§5–§6.)
- **No STC/Trail** live — train/test match requires the plain time surface. (§4.)
- **NVIDIA driver auto-update** (seen on the lab PC): an unattended userspace driver update
  mid-session causes `CUDA Error 804` for new processes until a module reload/reboot. If CELL
  suddenly fails to init CUDA, check `nvidia-smi` for a driver/library mismatch.
- **mathutils** is the Blender math lib (pip `mathutils`), used by `data_preprocess`/`flow2pose`
  for the RT construction — it must be importable.
- **Confidentiality** — lab-internal paths/names are kept out of this repo (sanitized for
  public release). The checkpoint and other large binaries are **not committed** here: they're
  gitignored and hosted externally (checkpoint + map via the Drive link in the repo README /
  `drone_space/dataset/DATASET.md`).

---

## 11. Suggested bring-up order

1. Env + recompile extensions; `import` sanity check (§3).
2. Offline frame-composition validation: render map depth from a logged mocap pose, overlay
   on the matching event frame (§8). Must align before anything live.
3. Port the event representation to the EVK4 callback (§4); compare a live frame to a saved
   training frame (same scene) — distributions should match.
4. **Mode A live** (mocap-seeded) end-to-end through the ROS1 IPC bridge. Confirm `H_est`
   tracks mocap.
5. Add the motion model + health monitor + mocap-recovery → **Mode B**.
6. (Optional) edge-matching refinement (`tools/pose_refine`) for a small indoor accuracy
   bump, once the base loop is solid.
