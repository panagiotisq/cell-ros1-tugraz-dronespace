"""ROS 1 (Noetic) side of the live CELL system. Two independent jobs, both in this module:

    GuessForwarder   subscribes to the live mocap body pose, composes it with
                     T_body_camera to get H_guess_c2w (camera pose in world), publishes it
                     to RViz (blue, "MOCAP GUESS") for comparison, and forwards it to the
                     CELL process over cell_live.ipc (Mode A, DEPLOY_CELL_DRONESPACE.md sec 7).
    Ros1Bridge       receives each tick's CELL result over cell_live.ipc and publishes
                     /cell/pose (valid only), /cell/path, TF, markers, diagnostics, the
                     event-frame visualization.

Runs in the system ROS Noetic Python: imports only rospy, tf2_ros and message modules (no
OpenCV or torch), mirroring glace_live_ros1.ros1_bridge. One deliberate, documented
exception: EventForwarder's "raw" decode mode uses numpy + struct to hand-decode message
bytes directly (bypassing rospy's slow per-event deserialization) -- see its class
docstring. Numpy here is a system dist-packages install, not shared with the CELL process
(a separate conda env), so this does not reintroduce the binary-sharing concern the "no
numpy" rule otherwise guards against.
"""

from __future__ import annotations

import array
import struct
import time

import numpy as np
import rospy
import tf2_ros
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, ColorRGBA, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from . import geometry, ipc, mocap
from .status import GUESS, HELD, SKIPPED, STATUS_COLORS, STATUS_LABELS, VALID


def to_time(stamp_ns: int) -> rospy.Time:
    return rospy.Time(secs=int(stamp_ns // 1_000_000_000), nsecs=int(stamp_ns % 1_000_000_000))


def pose_msg(T, frame_id, stamp):
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = frame_id, stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = T[0][3], T[1][3], T[2][3]
    qx, qy, qz, qw = geometry.quaternion_from_rotation([r[:3] for r in T[:3]])
    o = msg.pose.orientation
    o.x, o.y, o.z, o.w = qx, qy, qz, qw
    return msg


def transform_msg(pose: PoseStamped, child_frame):
    tf = TransformStamped()
    tf.header.frame_id, tf.header.stamp, tf.child_frame_id = pose.header.frame_id, pose.header.stamp, child_frame
    p = pose.pose.position
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p.x, p.y, p.z
    tf.transform.rotation = pose.pose.orientation
    return tf


def _color(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def camera_markers(T, kind, text, frame_id, stamp, ns, label=None):
    color = _color(STATUS_COLORS.get(kind, (1.0, 1.0, 1.0)))
    frustum = Marker(ns=ns, id=0, type=Marker.LINE_LIST)
    frustum.header.frame_id, frustum.header.stamp = frame_id, stamp
    if T is None:
        frustum.action = Marker.DELETE
        anchor = [0.0, 0.0, 0.0]
    else:
        frustum.action = Marker.ADD
        frustum.pose.orientation.w = 1.0
        frustum.scale.x = 0.01
        frustum.color = color
        frustum.points = [Point(x=q[0], y=q[1], z=q[2]) for seg in geometry.frustum_segments(T) for q in seg]
        anchor = [T[0][3], T[1][3], T[2][3]]
    text_marker = Marker(ns=ns, id=1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, text=text)
    text_marker.header.frame_id, text_marker.header.stamp = frame_id, stamp
    text_marker.pose.position.x, text_marker.pose.position.y, text_marker.pose.position.z = (
        anchor[0], anchor[1], anchor[2] + 0.35)
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.08
    text_marker.color = color
    return MarkerArray(markers=[frustum, text_marker])


def image_msg(desc, blob, frame_id, stamp):
    ch = 1 if desc["encoding"] == "mono8" else 3
    msg = Image(height=desc["height"], width=desc["width"], encoding=desc["encoding"], is_bigendian=0,
                step=desc["width"] * ch)
    msg.header.frame_id, msg.header.stamp = frame_id, stamp
    msg.data = blob[desc["offset"]:desc["offset"] + desc["length"]]
    return msg


class _Path:
    def __init__(self, pub, frame_id, max_len):
        self.pub, self.max_len = pub, max_len
        self.msg = Path()
        self.msg.header.frame_id = frame_id

    def append(self, pose, stamp):
        self.msg.poses.append(pose)
        del self.msg.poses[:-self.max_len]
        self.msg.header.stamp = stamp
        self.pub.publish(self.msg)


# --------------------------------------------------------------------------- guess forwarder (mocap -> CELL)

class GuessForwarder:
    """Subscribes to the live mocap body pose, composes T_world_camera = T_world_body @
    T_body_camera, publishes it to RViz for comparison (blue "MOCAP GUESS" frustum), and
    forwards it to the CELL process's guess port (DEPLOY sec 7, Mode A)."""

    def __init__(self, mocap_topic, extrinsic_path, world_frame, guess_frame, prefix,
                 guess_port=ipc.DEFAULT_GUESS_PORT, max_guess_path=200):
        self.world_frame, self.guess_frame = world_frame, guess_frame
        self.T_body_camera, note = mocap.load_extrinsic(extrinsic_path)
        rospy.loginfo(f"T_body_camera loaded from {extrinsic_path}" + (f" ({note})" if note else ""))
        self.sender = ipc.Sender(port=guess_port, log=rospy.logdebug)
        self.monitor = mocap.TrackingMonitor()
        self.seq = 0
        P = rospy.Publisher
        self.pub_pose = P(f"{prefix}/guess/pose", PoseStamped, queue_size=10)
        self.path = _Path(P(f"{prefix}/guess/path", Path, queue_size=2), world_frame, max_guess_path)
        self.pub_markers = P(f"{prefix}/guess/markers", MarkerArray, queue_size=10)
        self.tf = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(mocap_topic, PoseStamped, self._on_mocap, queue_size=50)

    def _on_mocap(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        try:
            T_world_body = mocap.matrix_from_pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        except ValueError as e:
            rospy.logwarn_throttle(5.0, f"invalid mocap pose: {e}")
            return
        T_world_camera = mocap.matmul(T_world_body, self.T_body_camera)
        stamp_ns = msg.header.stamp.to_nsec() or time.time_ns()
        self.monitor.update(stamp_ns / 1e9)

        flat = [float(v) for row in T_world_camera for v in row]
        self.sender.send({"v": ipc.PROTOCOL_VERSION, "type": "guess", "seq": self.seq, "stamp_ns": stamp_ns,
                          "H_guess_c2w": flat, "source": "mocap"})
        self.seq += 1

        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()
        pmsg = pose_msg(T_world_camera, self.world_frame, stamp)
        self.pub_pose.publish(pmsg)
        self.path.append(pmsg, stamp)
        self.tf.sendTransform(transform_msg(pmsg, self.guess_frame))
        self.pub_markers.publish(camera_markers(T_world_camera, GUESS, "MOCAP GUESS", self.world_frame, stamp,
                                                ns="mocap_guess"))


# --------------------------------------------------------------------------- event forwarder (ROS1 driver -> CELL)

class EventForwarder:
    """Subscribes to the existing ROS1 prophesee driver's raw-event topic (the same one
    The external driver stack / rosbag recording used throughout: prophesee_event_msgs/EventArray on
    /prophesee/camera/cd_events_buffer) and forwards events to the CELL process, flushed
    every `flush_period_s` as one packed "events" IPC message (cell_live.ipc) instead of
    one IPC message per incoming ROS message, to keep the message rate sane.

    Two decode modes (`decode_mode`), kept SIDE BY SIDE so this can be reverted in one
    word if the new path ever misbehaves on-site:

    "rospy" (the original, proven method): rospy deserializes each EventArray message into
      real Event objects; we iterate msg.events in a plain python loop. Confirmed correct,
      but capped at ROS's python deserialization throughput for this message type (~0.28M
      events/s measured) -- well below what this
      EVK4 can produce during vigorous motion, so a multi-second backlog can build up
      (confirmed on-site: event timestamps lagging wall-clock by 3+ seconds during fast
      rig motion, the dominant source of bad corrections, NOT model compute time).

    "raw" (new, default): subscribes with rospy.AnyMsg (skips rospy's own per-field
      deserialization entirely) and decodes the raw serialized bytes directly with one
      numpy.frombuffer() call per message -- no per-event python object construction at
      all. Benchmarked at ~128M events/s vs rospy's ~0.28M events/s (~450x). Byte-for-byte
      validated against rospy's own EventArray.deserialize() on live data before being
      wired in here (seen scratchpad/validate_raw_decode.py): 60/60 messages matched
      exactly, including bursty ones up to 640 events.

      Wire format (ROS1 classic serialization, little-endian, unpadded), from
      prophesee_event_msgs/{EventArray,Event}.msg:
        Header header: uint32 seq | time stamp (uint32 secs, uint32 nsecs)
                       | string frame_id (uint32 length, then that many bytes)
        uint32 height | uint32 width | uint32 events_length (array length prefix)
        Event[] events, each FIXED 13 bytes: uint16 x | uint16 y
                       | time ts (uint32 secs, uint32 nsecs) | uint8 polarity

    To revert: pass decode_mode="rospy" (or set CLR1_EVENT_DECODE=rospy -- see
    run_ros1_bridge.py's --event-decode flag)."""

    _EVENT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("sec", "<u4"), ("nsec", "<u4"), ("p", "u1")])
    assert _EVENT_DT.itemsize == 13, "unexpected padding in the packed Event dtype"

    def __init__(self, events_topic, events_port=ipc.DEFAULT_EVENTS_PORT, flush_period_s=0.015,
                 max_pending=64, decode_mode="raw"):
        import threading

        if decode_mode not in ("raw", "rospy"):
            raise ValueError(f"decode_mode must be 'raw' or 'rospy', got {decode_mode!r}")
        self.decode_mode = decode_mode
        self.sender = ipc.Sender(port=events_port, max_pending=max_pending, max_age_s=0.3, log=rospy.logdebug)
        self._lock = threading.Lock()
        self._xs, self._ys, self._ps, self._ts = array.array("H"), array.array("H"), array.array("B"), array.array("d")
        self.seq = 0
        self.n_received = 0
        self.n_sent = 0
        if decode_mode == "raw":
            rospy.Subscriber(events_topic, rospy.AnyMsg, self._on_events_raw, queue_size=200)
        else:
            from prophesee_event_msgs.msg import EventArray
            rospy.Subscriber(events_topic, EventArray, self._on_events_rospy, queue_size=200)
        self._timer = rospy.Timer(rospy.Duration(flush_period_s), self._flush)

    def _on_events_rospy(self, msg):
        """Original, proven path (rospy's own per-field deserialization). See class
        docstring -- kept intact as the documented fallback for decode_mode='rospy'."""
        with self._lock:
            for ev in msg.events:
                self._xs.append(ev.x)
                self._ys.append(ev.y)
                self._ps.append(1 if ev.polarity else 0)
                self._ts.append(ev.ts.to_sec())
        self.n_received += len(msg.events)

    def _on_events_raw(self, msg):
        """New path: msg is a rospy.AnyMsg (raw bytes in msg._buff); decode by hand with
        numpy instead of letting rospy build one python Event object per event. See the
        class docstring for the wire-format layout and the validation this was checked
        against before being wired in."""
        buf = msg._buff
        off = 4 + 8                                    # seq (4B) + header.stamp (8B)
        strlen, = struct.unpack_from("<I", buf, off); off += 4
        off += strlen                                   # skip header.frame_id bytes
        off += 8                                        # height (4B) + width (4B)
        n, = struct.unpack_from("<I", buf, off); off += 4
        if n == 0:
            return
        arr = np.frombuffer(buf, dtype=self._EVENT_DT, count=n, offset=off)
        ts = arr["sec"].astype(np.float64) + arr["nsec"].astype(np.float64) * 1e-9
        with self._lock:
            self._xs.frombytes(arr["x"].tobytes())
            self._ys.frombytes(arr["y"].tobytes())
            self._ps.frombytes(arr["p"].tobytes())
            self._ts.frombytes(ts.tobytes())
        self.n_received += n

    def _flush(self, _event):
        with self._lock:
            if not self._ts:
                return
            xs, ys, ps, ts = self._xs, self._ys, self._ps, self._ts
            self._xs, self._ys, self._ps, self._ts = (array.array("H"), array.array("H"), array.array("B"),
                                                       array.array("d"))
        n = len(ts)
        blob = xs.tobytes() + ys.tobytes() + ps.tobytes() + ts.tobytes()
        self.sender.send({"v": ipc.PROTOCOL_VERSION, "type": "events", "seq": self.seq,
                          "stamp_ns": time.time_ns(), "n_events": n}, raw_blob=blob)
        self.seq += 1
        self.n_sent += n


# --------------------------------------------------------------------------- CELL result bridge

class Ros1Bridge:
    def __init__(self, world_frame="dronespace_world", camera_frame="cell_camera_frame", prefix="/cell",
                 max_path=5000, max_clock_skew_s=600.0):
        self.world_frame, self.camera_frame = world_frame, camera_frame
        self.max_clock_skew_s = max_clock_skew_s
        P = rospy.Publisher
        self.pub_pose = P(f"{prefix}/pose", PoseStamped, queue_size=10)
        self.path = _Path(P(f"{prefix}/path", Path, queue_size=10), world_frame, max_path)
        self.pub_valid = P(f"{prefix}/tracking_valid", Bool, queue_size=10)
        self.pub_inliers = P(f"{prefix}/inliers", Int32, queue_size=10)
        self.pub_diag = P(f"{prefix}/diagnostics", DiagnosticArray, queue_size=10)
        self.pub_vis = P(f"{prefix}/debug/event_frame_vis", Image, queue_size=2)
        self.pub_live = P(f"{prefix}/debug/event_frame_live", Image, queue_size=1)
        self.pub_status = P(f"{prefix}/debug/status", String, queue_size=10)
        self.pub_markers = P(f"{prefix}/debug/markers", MarkerArray, queue_size=10)
        self.tf = tf2_ros.TransformBroadcaster()
        self._last_wall = None
        self.rate_hz = 0.0
        self.counts = {"results": 0, "valid": 0, "held": 0, "skipped": 0, "rejected": 0}

    def handle(self, meta, blob):
        now_ns = time.time_ns()
        if abs(meta["stamp_ns"] - now_ns) > self.max_clock_skew_s * 1e9:
            self.counts["rejected"] += 1
            return f"stamp_ns {meta['stamp_ns']} is not a plausible Unix-epoch time"
        stamp = to_time(meta["stamp_ns"])
        images = {d["name"]: d for d in meta.get("images", [])}
        if meta["type"] == "live":
            self.pub_live.publish(image_msg(images["event_frame_live"], blob, self.camera_frame, stamp))
            self.counts["live_images"] = self.counts.get("live_images", 0) + 1
            return None
        return self._result(meta, blob, images, stamp)

    def _checked(self, flat, what):
        if flat is None:
            return None, ""
        T = geometry.as_rows(flat)
        ok, reason = geometry.rigid_check(T)
        return (T, "") if ok else (None, f"{what} rejected: {reason}")

    def _result(self, meta, blob, images, stamp):
        guess, err_g = self._checked(meta.get("H_guess_c2w"), "H_guess_c2w")
        est, err_e = self._checked(meta.get("H_est_c2w"), "H_est_c2w")
        if err_g or err_e:
            self.counts["rejected"] += 1
            return err_g or err_e
        now = time.monotonic()
        if self._last_wall is not None:
            inst = 1.0 / max(1e-6, now - self._last_wall)
            self.rate_hz = inst if self.rate_hz == 0 else 0.8 * self.rate_hz + 0.2 * inst
        self._last_wall = now
        self.counts["results"] += 1
        status, success = meta["status"], meta["success"]
        self.counts[status] = self.counts.get(status, 0) + 1

        if success:
            msg = pose_msg(est, self.world_frame, stamp)
            self.pub_pose.publish(msg)
            self.path.append(msg, stamp)
            self.tf.sendTransform(transform_msg(msg, self.camera_frame))
        self.pub_valid.publish(Bool(data=success))
        self.pub_inliers.publish(Int32(data=int(meta.get("inliers", 0))))

        diag = DiagnosticArray()
        diag.header.stamp = stamp
        st = DiagnosticStatus(name="cell_localizer", hardware_id="evk4",
                              level=DiagnosticStatus.OK if success else DiagnosticStatus.WARN,
                              message="valid" if success else (meta.get("reason") or status))
        values = [("status", status), ("guess_source", meta.get("guess_source", "")),
                  ("inliers", meta.get("inliers")), ("events", meta.get("n_events")),
                  ("step_cm", meta.get("step_cm")), ("step_deg", meta.get("step_deg")),
                  ("latency_ms", meta.get("latency_ms")), ("seq", meta.get("seq"))]
        st.values = [KeyValue(key=k, value=str(v)) for k, v in values]
        st.values += [KeyValue(key=f"{k}_ms", value=f"{v:.1f}") for k, v in meta.get("timings_ms", {}).items()]
        diag.status = [st]
        self.pub_diag.publish(diag)

        lines = self._status_lines(meta)
        self.pub_status.publish(String(data=" | ".join(lines)))
        shown = est if success else guess
        kind = VALID if success else (HELD if status == HELD else SKIPPED)
        self.pub_markers.publish(camera_markers(shown, kind, "\n".join(lines[:2]), self.world_frame, stamp,
                                                ns="cell_camera"))
        if "event_frame_vis" in images:
            self.pub_vis.publish(image_msg(images["event_frame_vis"], blob, self.camera_frame, stamp))
        return None

    def _status_lines(self, meta):
        status = meta["status"]
        lat = meta.get("latency_ms")
        perf = f"{self.rate_hz:.1f} Hz" + (f" | latency {lat:.0f} ms" if lat is not None else "")
        t = meta.get("timings_ms", {})
        perf += " | " + " ".join(f"{k.replace('_ms','')} {v:.0f}" for k, v in t.items() if k != "total_ms")
        step, step_deg = meta.get("step_cm"), meta.get("step_deg")
        step_txt = ""
        if step is not None:
            step_txt = f" | step {step:.1f}cm" + (f"/{step_deg:.1f}deg" if step_deg is not None else "")
        lines = [STATUS_LABELS.get(status, status),
                f"inliers {meta.get('inliers', 0)} | events {meta.get('n_events', 0)}" + step_txt, perf]
        if status in (HELD, SKIPPED) and meta.get("reason"):
            lines.append(meta["reason"])
        return lines

    def publish_disconnected(self):
        stamp = rospy.Time.now()
        self.pub_status.publish(String(data="NO INPUT - CELL PROCESS NOT CONNECTED"))
        self.pub_markers.publish(camera_markers(None, SKIPPED, "NO INPUT - CELL PROCESS NOT CONNECTED",
                                                self.world_frame, stamp, ns="cell_camera"))
        self._last_wall, self.rate_hz = None, 0.0


__all__ = ["Ros1Bridge", "GuessForwarder", "pose_msg", "transform_msg", "camera_markers", "to_time"]
