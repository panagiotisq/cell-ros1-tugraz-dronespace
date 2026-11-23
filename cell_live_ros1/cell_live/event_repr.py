"""The CELL event representation, live (DEPLOY_CELL_DRONESPACE.md sec 4).

2-channel, per-polarity LATEST-TIMESTAMP time surface, 50 ms window, half-resolution
(native EVK4 1280x720 -> 640x360, factor-2 downsample). Two value modes:
  mode="exp"    value = exp(-age / tau), tau = 0.015s  (DEFAULT -- matches best_model.pth,
                verified empirically against the real training frames: min nonzero value
                0.03568 = exp(-0.050/0.015) exactly, a hard floor -- see
                drone_space/dataset/scripts/gen_tugraz.py, the real training-data generator)
  mode="linear" value = age / window, clipped to [0, 1]  (does NOT match best_model.pth;
                kept only so this stays usable for a future model trained that way)
No STC/Trail denoising (train/test match).

Geometry/sync validated against logged mocap data: rendering the map depth at a mocap pose
and overlaying it on this representation's visualization, the depth edges sit on the event
edges. The value-formula choice (exp vs. linear) was additionally A/B'd live on the rig --
see cell_live_ros1/README.md's "Event-frame value formula" section.
"""

from __future__ import annotations

import collections

import numpy as np

FULL_H, FULL_W = 720, 1280
HALF_H, HALF_W = 360, 640
WINDOW_S = 0.050
TAU_S = 0.015
DEFAULT_MODE = "exp"

EVENT_DTYPE = np.dtype([("t", "<f8"), ("x", "<i4"), ("y", "<i4"), ("p", "i1")])


def build_frame(ev_t, ev_x, ev_y, ev_p, w_start, w_end, half_h=HALF_H, half_w=HALF_W,
                mode=DEFAULT_MODE, tau=TAU_S):
    """events (already decoded) -> (half_h, half_w, 2) float32 in [0, 1].
    ev_t: float seconds (any epoch, consistent with w_start/w_end). Channel 0 = ON
    (polarity > 0), channel 1 = OFF.

    mode="exp" (default, matches best_model.pth): value = exp(-(w_end - t) / tau).
    mode="linear" (legacy): value = (t - w_start) / (w_end - w_start)."""
    if mode not in ("exp", "linear"):
        raise ValueError(f"mode must be 'exp' or 'linear', got {mode!r}")
    img = np.zeros((half_h, half_w, 2), dtype=np.float32)
    if len(ev_t) == 0:
        return img
    xx = (ev_x // 2).astype(np.int64)
    yy = (ev_y // 2).astype(np.int64)
    ok = (xx >= 0) & (xx < half_w) & (yy >= 0) & (yy < half_h) & (ev_t >= w_start) & (ev_t < w_end)
    if not np.any(ok):
        return img
    xx, yy, tt, pp = xx[ok], yy[ok], ev_t[ok], ev_p[ok]
    ch = np.where(pp > 0, 0, 1)
    order = np.argsort(tt)                       # ascending: scatter-assign keeps the LAST write
    xx, yy, tt, ch = xx[order], yy[order], tt[order], ch[order]
    if mode == "exp":
        age = (w_end - tt).astype(np.float32)
        val = np.exp(-age / tau).astype(np.float32)
    else:
        rel = (tt - w_start).astype(np.float32)
        val = rel / float(w_end - w_start)
    img[yy, xx, ch] = val
    np.clip(img, 0.0, 1.0, out=img)
    return img


def visualize(frame2ch):
    """(H, W, 2) float32 [0,1] -> uint8 mono8 display image (max over the two polarity
    channels; brighter = more recent)."""
    return (frame2ch.max(axis=2) * 255.0).astype(np.uint8)


class RollingEventBuffer:
    """Thread-safe rolling buffer of the last `keep_s` seconds of events, fed by the
    acquisition thread (push) and sampled by the inference loop (snapshot). Bounded by
    time, not count: old chunks are dropped as new ones arrive."""

    def __init__(self, keep_s=0.15):
        import threading
        self.keep_s = keep_s
        self._lock = threading.Lock()
        self._chunks = collections.deque()      # each: EVENT_DTYPE structured array, t ascending
        self._last_t = None

    def push(self, t, x, y, p):
        """t: float seconds (monotonic-ish, must be non-decreasing across calls)."""
        if len(t) == 0:
            return
        chunk = np.empty(len(t), dtype=EVENT_DTYPE)
        chunk["t"], chunk["x"], chunk["y"], chunk["p"] = t, x, y, p
        with self._lock:
            self._chunks.append(chunk)
            self._last_t = float(t[-1])
            cutoff = self._last_t - self.keep_s
            while self._chunks and self._chunks[0]["t"][-1] < cutoff:
                self._chunks.popleft()

    def snapshot(self, window_s=WINDOW_S):
        """Return (t, x, y, p, w_start, w_end) for the last `window_s` seconds currently
        buffered, or None if nothing has arrived yet."""
        with self._lock:
            if self._last_t is None:
                return None
            w_end = self._last_t
            w_start = w_end - window_s
            chunks = list(self._chunks)
        if not chunks:
            return None
        full = np.concatenate(chunks)
        mask = (full["t"] >= w_start) & (full["t"] < w_end)
        full = full[mask]
        return full["t"], full["x"], full["y"], full["p"], w_start, w_end
