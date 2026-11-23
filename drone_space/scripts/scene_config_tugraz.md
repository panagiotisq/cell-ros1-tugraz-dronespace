# Registering the drone-space sequences in `core/scene_config.py`

> **Already done in this repo.** The vendored copy
> (`cell_live_ros1/cell_dronespace_bundle/code/core/scene_config.py`) has the four sequences
> registered in `_INDOOR` **and** a `scene_ev_input()` branch returning `ours_50ms_half`, so
> `main.py` (train), `eval_full.py` and `track.py` run on tugraz with no extra flags
> (`main.py` is vendored from public CELL). The notes below are for porting the registration
> into a *different* CELL/LEAR checkout.

The sequences are the **indoor half-res** class. Add the four names to the `_INDOOR` set:

```python
_INDOOR = {
    "falcon_indoor_flight_1", "falcon_indoor_flight_2", "falcon_indoor_flight_3",
    "spot_indoor_building_loop",
    # drone-space custom indoor sequences: half-res 360x640, crop 288x512, calib/2, md 10
    "tugraz_seq1", "tugraz_seq2", "tugraz_seq3", "tugraz_seq4",
}
```

That alone gives them the correct geometry from the existing `indoor` class entry in `_GEOM`:

```
crop = (288, 512, 36, 64)   calib_div = 2.0   max_depth = 10.0   frame_hw = (360, 640)
```

**Event-representation dir:** the tugraz frames live in `event_frames_ours_50ms_half/`, so
`scene_ev_input()` must return `ours_50ms_half` for them. The vendored copy does this via an
explicit `tugraz_seq1..4 -> "ours_50ms_half"` branch. If you port into a checkout that lacks
it, either add that branch or pass `--ev_input ours_50ms_half` on every run.

The `assert_frame_resolution` safety net already covers these: a mis-generated frame that
isn't 360×640 will fail loudly instead of running with the wrong crop/calib.
