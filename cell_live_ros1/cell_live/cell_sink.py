"""CELL-process side of the ROS 1 interface: sends each tick's result to the bridge, and
receives the live mocap-derived pose guess FROM the bridge (Mode A). Needs numpy/cv2 only
(no rospy -- this process must stay ROS-free, see cell_live/__init__.py)."""

from __future__ import annotations

import bisect
import collections
import threading
import time

import numpy as np

from . import event_repr, ipc


def _flat(T):
    return None if T is None else [float(v) for v in np.asarray(T, dtype=np.float64).reshape(16)]


class GuessReceiver:
    """Runs an ipc.Server on DEFAULT_GUESS_PORT; the bridge connects out and pushes
    "guess" messages (the live mocap pose, composed with T_body_camera) at mocap's own
    rate (100 Hz). Keeps a short time-ordered history (not just the single newest sample)
    so a caller can pair a guess with a specific event window by TIME rather than always
    taking "whatever is freshest right now" -- the two arrive on independent schedules
    (mocap every 10ms, event windows end whenever the rolling buffer is sampled), and
    during real motion even a 30-80ms mismatch between "where the guess says the camera
    was" and "when the events were captured" measurably hurts CELL's correspondence
    confidence (confirmed on-site: held results clustered while the rig was moving,
    valid ones while it was still). Thread-safe."""

    def __init__(self, port=ipc.DEFAULT_GUESS_PORT, max_age_s=0.5, history_s=1.0, log=print):
        self.server = ipc.Server(port=port)
        self.max_age_s = max_age_s
        self.history_s = history_s
        self._lock = threading.Lock()
        self._hist_t = collections.deque()    # stamp_ns, ascending (parallel to _hist_item)
        self._hist_item = collections.deque()  # (H_guess_c2w, stamp_ns, source)
        self._log = log
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="clr1-guess-recv", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            for kind, cid, info in self.server.poll(0.1):
                if kind == "message":
                    meta, _blob = info
                    if meta["type"] == "guess":
                        T = np.array(meta["H_guess_c2w"], dtype=np.float64).reshape(4, 4)
                        stamp_ns = meta["stamp_ns"]
                        item = (T, stamp_ns, meta.get("source", "mocap"))
                        with self._lock:
                            self._hist_t.append(stamp_ns)
                            self._hist_item.append(item)
                            cutoff = stamp_ns - int(self.history_s * 1e9)
                            while self._hist_t and self._hist_t[0] < cutoff:
                                self._hist_t.popleft()
                                self._hist_item.popleft()
                elif kind == "connected":
                    self._log(f"[guess] bridge connected from {info}")
                elif kind == "disconnected":
                    self._log(f"[guess] bridge disconnected: {info}")

    def latest(self):
        """Return (H_guess_c2w, age_s, source) for the single newest sample, or None if
        nothing received yet or it is older than max_age_s (stale/disconnected mocap)."""
        with self._lock:
            item = self._hist_item[-1] if self._hist_item else None
        if item is None:
            return None
        T, stamp_ns, source = item
        age_s = (time.time_ns() - stamp_ns) / 1e9
        if age_s > self.max_age_s:
            return None
        return T, age_s, source

    def at(self, target_epoch_s, tol_s=0.1):
        """Return (H_guess_c2w, |dt|_s, source) for the history sample whose OWN
        timestamp is closest to `target_epoch_s` (e.g. an event window's end time), or
        None if the history is empty or the closest sample is further than tol_s away.
        This is the precise guess<->event pairing (see class docstring); prefer it over
        latest() whenever you have a specific instant to pair against."""
        target_ns = int(target_epoch_s * 1e9)
        with self._lock:
            hist_t = list(self._hist_t)
            hist_item = list(self._hist_item)
        if not hist_t:
            return None
        i = bisect.bisect_left(hist_t, target_ns)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(hist_t)]
        j = min(candidates, key=lambda k: abs(hist_t[k] - target_ns))
        dt_s = abs(hist_t[j] - target_ns) / 1e9
        if dt_s > tol_s:
            return None
        T, _stamp_ns, source = hist_item[j]
        return T, dt_s, source

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.server.close()


class EventsReceiver:
    """Runs an ipc.Server on DEFAULT_EVENTS_PORT; the bridge connects out and pushes
    "events" messages (raw events forwarded from the ROS1 prophesee driver, since this
    process has no Metavision SDK access of its own -- see cell_live.ipc). Each batch is
    unpacked and pushed straight into `buffer` (an event_repr.RollingEventBuffer)."""

    def __init__(self, buffer, port=ipc.DEFAULT_EVENTS_PORT, log=print):
        self.buffer = buffer
        self.server = ipc.Server(port=port)
        self._log = log
        self._stop = False
        self.n_events = 0
        self._thread = threading.Thread(target=self._run, name="clr1-events-recv", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            for kind, cid, info in self.server.poll(0.1):
                if kind == "message":
                    meta, blob = info
                    if meta["type"] == "events" and meta["n_events"] > 0:
                        self._ingest(meta["n_events"], blob)
                elif kind == "connected":
                    self._log(f"[events] bridge connected from {info}")
                elif kind == "disconnected":
                    self._log(f"[events] bridge disconnected: {info}")

    def _ingest(self, n, blob):
        x = np.frombuffer(blob, dtype="<u2", count=n, offset=0)
        y = np.frombuffer(blob, dtype="<u2", count=n, offset=2 * n)
        p = np.frombuffer(blob, dtype="u1", count=n, offset=4 * n)
        t = np.frombuffer(blob, dtype="<f8", count=n, offset=5 * n)
        self.buffer.push(t, x.astype(np.int32), y.astype(np.int32), p.astype(np.int8))
        self.n_events += n

    def close(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        self.server.close()


class ResultSink:
    """Sends each localize() result + the event-frame visualization to the bridge."""

    def __init__(self, port=ipc.DEFAULT_RESULT_PORT, K=None, send_vis=True, log=print):
        self.sender = ipc.Sender(port=port, log=log)
        self.K = None if K is None else [float(v) for v in K]
        self.send_vis = send_vis
        self.seq = 0

    @property
    def stats(self):
        return dict(self.sender.stats)

    def publish(self, result, frame_index, ev_frame_2ch, n_events, guess_source, wall_end_ns):
        meta = {
            "v": ipc.PROTOCOL_VERSION, "type": "result", "seq": self.seq, "stamp_ns": int(wall_end_ns),
            "frame_index": int(frame_index), "status": result["status"], "success": bool(result["success"]),
            "guess_source": guess_source,
            "H_guess_c2w": _flat(result.get("H_guess_c2w")), "H_est_c2w": _flat(result.get("H_est_c2w")),
            "inliers": int(result.get("inliers", 0)), "n_events": int(n_events),
            "step_cm": result.get("step_cm"), "step_deg": result.get("step_deg"),
            "reason": str(result.get("reason", ""))[:1024],
            "latency_ms": result.get("timings_ms", {}).get("total_ms"),
            "timings_ms": {k: float(v) for k, v in result.get("timings_ms", {}).items()},
            "K": self.K,
        }
        self.seq += 1
        images = []
        if self.send_vis and ev_frame_2ch is not None:
            vis = event_repr.visualize(ev_frame_2ch)
            images.append(("event_frame_vis", vis.shape[1], vis.shape[0], "mono8", vis.tobytes()))
        self.sender.send(meta, images)

    def publish_live(self, ev_frame_2ch, wall_ns):
        """Display-only event-frame image, independent of the inference tick cadence (see
        cell_live.ipc's "live" message docstring) -- call this from a fast background loop,
        not from the (slow, inference-bound) main tick loop."""
        vis = event_repr.visualize(ev_frame_2ch)
        meta = {"v": ipc.PROTOCOL_VERSION, "type": "live", "seq": self.seq, "stamp_ns": int(wall_ns)}
        self.seq += 1
        self.sender.send(meta, [("event_frame_live", vis.shape[1], vis.shape[0], "mono8", vis.tobytes())])

    def close(self):
        self.sender.close()
