"""Standalone validation of the raw-bytes event decoder used by EventForwarder
(decode_mode="raw" in cell_live/ros1_bridge.py): subscribe to the real
prophesee_event_msgs/EventArray topic two ways on the SAME messages and cross-check they
decode identically:
  (a) normal rospy deserialization (ground truth: msg.events[i].x/.y/.ts/.polarity)
  (b) a hand-written numpy decoder reading the raw serialized bytes via rospy.AnyMsg

Wire format (ROS1 classic serialization, little-endian, no padding), from the .msg files:
  EventArray:
    Header header:
      uint32 seq
      time stamp          -> uint32 secs, uint32 nsecs
      string frame_id     -> uint32 length, then that many bytes (no null terminator)
    uint32 height
    uint32 width
    uint32 events_length  (array length prefix)
    Event[] events:
      uint16 x
      uint16 y
      time ts             -> uint32 secs, uint32 nsecs
      bool polarity       -> uint8
    (Event is fixed-size: 2+2+4+4+1 = 13 bytes, no padding -> safe to numpy.frombuffer in bulk)
"""
import struct
import sys

import numpy as np
import rospy
from prophesee_event_msgs.msg import EventArray

TOPIC = "/prophesee/camera/cd_events_buffer"
_EVENT_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("sec", "<u4"), ("nsec", "<u4"), ("p", "u1")])
assert _EVENT_DT.itemsize == 13, f"unexpected packed size {_EVENT_DT.itemsize} (padding snuck in?)"


def decode_raw(buf: bytes):
    off = 0
    seq, = struct.unpack_from("<I", buf, off); off += 4
    sec, nsec = struct.unpack_from("<II", buf, off); off += 8
    strlen, = struct.unpack_from("<I", buf, off); off += 4
    frame_id = buf[off:off + strlen].decode("utf-8"); off += strlen
    height, width = struct.unpack_from("<II", buf, off); off += 8
    n, = struct.unpack_from("<I", buf, off); off += 4
    arr = np.frombuffer(buf, dtype=_EVENT_DT, count=n, offset=off)
    ts = arr["sec"].astype(np.float64) + arr["nsec"].astype(np.float64) * 1e-9
    return {"seq": seq, "stamp": sec + nsec * 1e-9, "frame_id": frame_id, "height": height,
           "width": width, "n": n, "x": arr["x"].copy(), "y": arr["y"].copy(), "ts": ts,
           "p": arr["p"].copy()}


checked = 0
MAX_CHECK = 40


def on_normal(msg):
    pass  # ground truth is read inside on_raw via the connection, see below


def on_raw(msg):
    global checked
    if checked >= MAX_CHECK:
        return
    buf = msg._buff
    decoded = decode_raw(buf)

    # Ground truth: let rospy itself deserialize the SAME bytes into real Event objects,
    # using the message class's own deserialize() -- independent of our hand parser.
    gt = EventArray().deserialize(buf)
    gt_x = np.array([e.x for e in gt.events], dtype=np.uint16)
    gt_y = np.array([e.y for e in gt.events], dtype=np.uint16)
    gt_ts = np.array([e.ts.to_sec() for e in gt.events], dtype=np.float64)
    gt_p = np.array([1 if e.polarity else 0 for e in gt.events], dtype=np.uint8)

    ok = (decoded["n"] == len(gt.events) and decoded["height"] == gt.height
         and decoded["width"] == gt.width and decoded["frame_id"] == gt.header.frame_id
         and abs(decoded["stamp"] - gt.header.stamp.to_sec()) < 1e-9
         and np.array_equal(decoded["x"], gt_x) and np.array_equal(decoded["y"], gt_y)
         and np.array_equal(decoded["p"], gt_p) and np.allclose(decoded["ts"], gt_ts, atol=1e-9))
    checked += 1
    print(f"[{checked}/{MAX_CHECK}] n_events={decoded['n']:6d} match={'OK' if ok else 'MISMATCH !!!'}",
         flush=True)
    if not ok:
        print("  decoded:", {k: v for k, v in decoded.items() if k not in ("x", "y", "ts", "p")})
        print("  gt     :", gt.height, gt.width, gt.header.frame_id, gt.header.stamp.to_sec())
        if decoded["n"] == len(gt.events) and decoded["n"] > 0:
            dx = np.where(decoded["x"] != gt_x)[0][:5]
            print("  first x mismatches at idx", dx, decoded["x"][dx] if len(dx) else None, gt_x[dx] if len(dx) else None)
    if checked >= MAX_CHECK:
        rospy.signal_shutdown("done")


def main():
    rospy.init_node("validate_raw_decode", anonymous=True)
    rospy.Subscriber(TOPIC, rospy.AnyMsg, on_raw, queue_size=50)
    print(f"subscribed (AnyMsg) to {TOPIC}, waiting for {MAX_CHECK} messages ...", flush=True)
    rospy.spin()
    print("done" if checked >= MAX_CHECK else f"stopped early after {checked} messages", flush=True)


if __name__ == "__main__":
    main()
