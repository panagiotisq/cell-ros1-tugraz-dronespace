# cell-ros1-tugraz-dronespace

Live **CELL** (map-relative event-camera localization) on a drone-space rig — ROS 1 Noetic,
Prophesee EVK4 event camera, OptiTrack mocap — together with the TUGraz drone-space dataset
and its generation tooling. The live deployment lives in `cell_live_ros1/`.

## Data & trained model (Google Drive)

Want the drone-space data or the trained CELL model? The large files aren't in the repo —
they're all in one Google Drive folder:

### 📂 https://drive.google.com/drive/folders/1GzT_O74EwicPITg9dRQWXtwPS9glM1O0?usp=sharing

| folder | what it holds |
|---|---|
| `cell_checkpoint/` | the **trained CELL checkpoint** (`best_model.pth`) — for eval and live deployment |
| `bags/` | the 4 trimmed event + mocap rosbags — the raw dataset input (~27 GB) |
| `map/` | the aligned room point cloud (`fused_clean_mocap.ply`, ~60 MB) + calibration |
| `poses/` | the fixed ground-truth pose tracks (also committed in this repo) |
| `SHA256SUMS.txt` | checksums — verify a download with `sha256sum -c SHA256SUMS.txt` |

Where each file goes: `drone_space/DATASET.md` (dataset) and `cell_live_ros1/README.md`
(deployment).

## Layout

```
cell-ros1-tugraz-dronespace/
├── drone_space/        The TUGraz drone-space dataset and the shared physical setup:
│                       DATASET.md (the authoritative generation procedure), CALIBRATION.md,
│                       the calibration (calib/), the mocap<->map alignment (align/), the
│                       aligned room map (map/), the fixed GT pose tracks (poses/*.npz,
│                       committed), and the generation scripts (scripts/). The large inputs --
│                       trimmed bags (~27GB) and the aligned map (~60MB) -- are hosted
│                       externally, not in this repo; see DATASET.md's hosting table.
└── cell_live_ros1/      The live ROS1 deployment itself -- see its own README.md for
                        architecture, setup, and how to run it. Self-contained: includes
                        cell_dronespace_bundle/ (the CELL method code + trained checkpoint)
                        as cell_live_ros1/cell_dronespace_bundle/.
```

Start with `cell_live_ros1/README.md`.
