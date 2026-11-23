# Calibration — event camera intrinsics + body↔camera extrinsic

Files in `calib/`:
- `intrinsics.json` — event-camera intrinsics (checkerboard calibration, 6×9, 31.79 mm squares, 50 views).
- `T_body_camera.txt` — 4×4 transform, camera frame → mocap rigidbody ("body") frame.

## Intrinsics (what CELL uses on the event frames)
| param | value |
|---|---|
| image size | 1280 × 720 |
| fx = fy | **1687.451** px |
| cx | **653.359** px |
| cy | **351.421** px |
| distortion (OpenCV k1,k2,p1,p2,k3) | −0.0651, 0.3277, −6.44e-5, 9.68e-4, 0.0 |
| calibration RMS reproj. error | **0.408 px** (good) |

- **This is the EVENT camera**, not the phone used for the room reconstruction. The ~1536 px focal
  COLMAP estimates for the phone video is a *different camera* and is irrelevant to CELL — the phone
  only builds the map geometry; depth is rendered and matched against events using **these** intrinsics.

## Extrinsic `T_body_camera`
4×4, metres, row-major. Rotation ≈ **3° about X**; translation ≈ **[−2.1, −2.1, 0.7] cm**.

**Convention: camera→body.** `T_body_camera` maps a point/pose from the
**camera** frame into the **body** (mocap rigidbody) frame — i.e. it is the camera's pose expressed in
the body frame. The camera's world pose per timestamp is therefore:

```
T_world_camera = T_world_body (from mocap) · T_body_camera
```

## Time sync & topics
- **Time sync** — events and mocap poses share **one ROS time base** (`header.stamp`, seconds);
  no separate clock or offset to correct for.
- **Topics** — events on `/prophesee/camera/cd_events_buffer`
  (`prophesee_event_msgs/EventArray`); mocap pose on `/mocap_node/evk4/pose`
  (`geometry_msgs/PoseStamped`).
