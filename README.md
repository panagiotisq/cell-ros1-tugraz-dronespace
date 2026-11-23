# cell_dronespace_ros1

Live CELL (map-relative event-camera localization) deployment on the TU Graz drone-space
rig: ROS 1 Noetic, EVK4 event camera, OptiTrack mocap. Originally scaffolded as a ROS 2
package; that approach was abandoned in favor of the ROS 1 live deployment actually built
and validated on-site, in `cell_live_ros1/`.

## Layout

```
cell_dronespace_ros1/
├── drone_space/        Calibration, mocap<->map alignment, the aligned room map, the
│   │                   calibration reference (CALIBRATION.md), and:
│   └── dataset/        The dataset-generation pipeline (bag -> CELL-format training
│                       sequences): DATASET.md (the authoritative procedure doc), the fixed
│                       ground-truth pose tracks (poses/*.npz, committed as-is), and the
│                       generation scripts (scripts/). The large inputs it consumes --
│                       trimmed bags (~27GB) and the aligned room map (~60MB) -- are hosted
│                       externally, not in this repo; see DATASET.md's hosting table.
└── cell_live_ros1/      The live ROS1 deployment itself -- see its own README.md for
                        architecture, setup, and how to run it. Self-contained: includes
                        cell_dronespace_bundle/ (the CELL method code + trained checkpoint)
                        as cell_live_ros1/cell_dronespace_bundle/.
```

Start with `cell_live_ros1/README.md`.
