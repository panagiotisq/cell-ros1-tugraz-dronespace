# cell-ros1-tugraz-dronespace

Live **CELL** (map-relative event-camera localization) on a drone-space rig — ROS 1 Noetic,
Prophesee EVK4 event camera, OptiTrack mocap — together with the TUGraz drone-space dataset
and its generation tooling. The live deployment lives in `cell_live_ros1/`.

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
