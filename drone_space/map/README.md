# Aligned room map

Place the aligned room point cloud here:

```
fused_clean_mocap.ply
```

It is **not committed** (~60 MB, gitignored) — download it from the hosted Google Drive
folder (`map/`) linked in [`../DATASET.md`](../DATASET.md) / `cell_live_ros1/README.md`.

It is the prior map CELL renders depth from, in the **mocap world frame** (aligned via the
sim3 in `../align/`), and is shared by both the live deployment and the dataset's local-map
generation (`../scripts/gen_tugraz.py`, default `--ply drone_space/map/fused_clean_mocap.ply`).
