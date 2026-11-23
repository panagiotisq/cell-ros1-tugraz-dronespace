# Drone-space dataset — generation procedure

Four indoor event-camera flight sequences recorded in a motion-capture room, built in the
M3ED indoor-flight layout so they plug straight into the CELL dataloader. **2 sequences are
used for training, 2 for evaluation: train = `tugraz_seq1`, `tugraz_seq2`; eval =
`tugraz_seq3`, `tugraz_seq4`.** (Perturbation CSVs exist only for seq3/seq4; the checkpoint
treats `tugraz_seq3` as the held-out validation sequence — the training recipe trains on
seq1/seq2 and tests on seq3.)

The large inputs (bags, aligned room map) live in external storage (see **Data hosting**
below); this repository holds only the **scripts** that turn those inputs into the
CELL-format sequences, plus the **fixed ground-truth pose tracks** (`poses/`, shipped as
files — see step 2).

## Capture methodology (how the raw inputs were produced)

The raw data was captured on the drone-space rig and reconstructed off-line:

- **Recording.** Each flight was flown in a motion-capture room and recorded as one ROS 1
  rosbag holding the event stream (`prophesee_event_msgs/EventArray`) and the mocap rigidbody
  pose (`geometry_msgs/PoseStamped`), both stamped on a **single ROS clock** (no inter-stream
  offset). The four flights use distinct trajectories/coverage; the two eval sequences are
  deliberately different from the two training ones.
- **Room map.** With no depth sensor available, the room was reconstructed as a dense point
  cloud with **COLMAP** (dense MVS) from a hand-held phone video shot for wide coverage and
  high overlap. This cloud is the prior map depth is rendered from (the LiDAR-map analogue).
- **Metric alignment.** The cloud was cleaned (`scripts/outlier_tuner.py`) and fit into the
  **mocap world frame** with a 7-DOF Umeyama similarity over **≥4–6 control points** measured
  in mocap-world coordinates at varied, non-coplanar heights (so scale and vertical rotation
  are well constrained). The event-camera intrinsics and the rigidbody→camera extrinsic live
  in `../calib/` (documented in `../CALIBRATION.md`).

The processing from these raw inputs to CELL-format sequences is the pipeline below.

---

## What a finished sequence looks like

```
data/<seq>/
├── event_frames_ours_50ms_half/left/event_frame_XXXXX.npy   # (360,640,2) float32
├── local_maps/point_cloud_XXXXX.h5                           # PC [4,M] float16, cam frame
├── <seq>_pose_gt.h5                                          # Ln_T_L0 [N,4,4] world->cam
└── meta.npz                                                  # per-frame ts, T_world_cam, bridged, params
```

- **Event frame**: 2-channel exp-decay time-surface over the 50 ms window ending at the
  frame timestamp, τ = 15 ms, ch0 = ON / ch1 = OFF, half resolution (coords // 2), value
  `exp(-age/τ)` keeping the latest event per pixel. (Authoritative spec; matches the live
  deployment's event representation.)
- **Local map**: the aligned room point cloud transformed into that frame's GT camera frame
  (`T_cam_world @ pc_world`), kept to `z ∈ (0.1, 12] m`.
- **Pose GT** (`Ln_T_L0`): world→cam at each kept frame, `T_world_cam = T_world_body · T_body_cam`.
- **Geometry** (see `scripts/scene_config_tugraz.md`): registered in `core/scene_config.py`
  `_INDOOR` class — native frame 360×640, crop `(288,512,36,64)`, `calib_div = 2.0`,
  `max_depth = 10.0`, with `scene_ev_input` returning `ours_50ms_half`.

---

## Pipeline (raw bag → sequence)

Two preprocessing steps run **before** frame/map generation, then the generator. A separate
one-time map step is shared by all sequences.

### 0. Record (at the rig)
One rosbag per flight: events + mocap rigidbody poses, time-synced on a common clock.

### 1. Trim
Raw bag → `<seq>_trim.bag`: cut the static lead-in / lead-out. Both topics are cropped to a
common `[t0, t1]` window (chosen on event rate — events drop sharply when the rig is static —
with ~0.5 s margins) and re-written as a ROS1 bag with the pure-python
`rosbags.rosbag1.Writer` (raw message bytes copied, no re-encode, so timestamps and sync are
preserved). The trimmed bag is the authoritative input and is hosted externally.

### 2. Fix poses  →  shipped as `poses/<seq>_trim_poses_fixed.npz`  (method: `scripts/fix_pose_glitches.py`)
From the trimmed bag's mocap stream, flag outlier samples (angular rate > 800 °/s or linear
rate > 3 m/s — brief mocap dropouts) and **bridge them by interpolation** (SLERP on the
quaternion, LERP on position) between clean anchor samples either side of each glitch run.
Output: the full-rate pose track `{ts, pos, quat, bridged}` with a per-sample `bridged` flag
marking interpolated poses (typically well under 2 %).

**These `.npz` files are committed, not regenerated.** The outlier/bridge step is
hand-tuned and not bit-reproducible, so the exact pose track the model was trained and
evaluated on is shipped directly. Regenerating risks a different `bridged` set.

### 3. Reconstruct + align the room map (one-time, shared)
Reconstruct the room as a dense point cloud (COLMAP dense MVS), clean stray points
(`scripts/outlier_tuner.py` — interactive ROR/SOR tuner over the dense `.ply`), then align +
scale into the mocap world frame with a 7-DOF Umeyama similarity on hand-measured control
points. Output: the aligned, cleaned room cloud (hosted externally) + the rigidbody→camera
extrinsic `T_body_camera.txt`. CELL renders depth from this cloud at each pose.

### 4. Generate the sequences  →  `scripts/gen_tugraz.py`
Per 100 ms tick aligned to the nearest GT pose: build the event frame (step above), render
the local map in the GT camera frame, drop frames with < 5000 events (residual static
stretches), renumber kept frames contiguously (event-frame and local-map indices stay in
sync), and write `meta.npz`. Events are parsed straight from the ROS1 wire bytes with numpy
and streamed in one pass (bounded RAM); no ROS install required.

```bash
python scripts/gen_tugraz.py --seqs seq1 seq2 seq3 seq4
```

---

## Reproduce: generate → train → eval

Full chain from a fresh clone. Steps 1–3 of the pipeline do **not** need rerunning — the
trimmed bags, fixed poses and aligned map are the shipped inputs; only **step 4 (generation)**
runs, then training/eval use the vendored CELL bundle. Set up its conda env + compiled CUDA
extensions first (see `cell_live_ros1/README.md`). Use **absolute** paths below:
`<DATA_ROOT>` = an ext4 output dir, `<BAGS>`/`<MAP>` = the downloaded `bags/`/`map/`,
`<CKPT>` = the downloaded `cell_checkpoint/best_model.pth`.

**1. Generate the four sequences** (bag stem `seqN` → scene dir `tugraz_seqN`):
```bash
python drone_space/dataset/scripts/gen_tugraz.py \
    --seqs seq1 seq2 seq3 seq4 \
    --bagdir <BAGS> --ply <MAP>/fused_clean_mocap.ply --data-root <DATA_ROOT>
# posedir/ext default to this repo's drone_space/dataset/poses + drone_space/calib.
# -> <DATA_ROOT>/tugraz_seq{1..4}/ with event_frames, local_maps, meta.npz, *_pose_gt.h5
```

Training and eval run from the bundle's `code/` directory (its imports are relative to it):
```bash
cd cell_live_ros1/cell_dronespace_bundle/code
```

**2. Train** — the exact recipe that produced the shipped checkpoint (100 epochs from scratch,
decoupled-β1.5 pose-supervised confidence, partial-light depth completion, half-magnitude
training perturbation):
```bash
python main.py \
    --data_path <DATA_ROOT> --dataset m3ed \
    --train_sequence tugraz_seq1,tugraz_seq2 --test_sequence tugraz_seq3 \
    --ev_input ours_50ms_half --backbone edge \
    --partial_depth partial_light --conf_head \
    --pose_e2e --pose_e2e_decouple --pose_e2e_beta_pred 1.5 \
    --max_t 0.25 --max_r 2.5 --epochs 100 \
    --run_tag scratch100_decplB15_partial_light_halfpert
```

**3. Evaluate** (perturbed-pose refinement; the per-frame perturbation CSV is auto-generated
reproducibly from `--rt_seed` if absent). `--partial_depth` must match training; `--ev_input`
is resolved from `scene_config` (= `ours_50ms_half`) so it can be omitted:
```bash
python eval_full.py --ckpt <CKPT> --tag tugraz_eval --seq tugraz_seq3 \
    --partial_depth partial_light --max_r 5 --max_t 0.5
```

**4. (Optional) sequential tracking demo:**
```bash
python tools/tracking/track_cell.py --seq tugraz_seq3 --ckpt <CKPT> \
    --selection floor0 --motion position --reset_cm 50
```

## Data hosting

Frames and local maps are regenerated, never stored. The reproducible inputs are hosted on
Google Drive: <https://drive.google.com/drive/folders/1GzT_O74EwicPITg9dRQWXtwPS9glM1O0?usp=sharing>

Each file's SHA-256 is listed in `SHA256SUMS.txt` at the folder root -- verify a download with
`sha256sum -c SHA256SUMS.txt`. (The folder also carries `cell_checkpoint/` -- the trained model
for the live deployment, see `cell_live_ros1/README.md`.)

| folder | contents | approx size |
|---|---|---|
| `bags/` | the 4 trimmed bags (`<seq>_trim.bag`) | ~27 GB |
| `poses/` | the 4 fixed pose tracks (also committed in this repo) | ~1.3 MB |
| `map/`  | aligned room cloud + `T_body_camera.txt` + intrinsics (the same file as `drone_space/map/fused_clean_mocap.ply`, used by the live deployment too -- one upload covers both) | ~60 MB |

Anything over GitHub's 100 MB limit (bags, map, generated frames/maps) must use the external
host or Git LFS — only the scripts and the small pose `.npz` belong in the repo tree.
