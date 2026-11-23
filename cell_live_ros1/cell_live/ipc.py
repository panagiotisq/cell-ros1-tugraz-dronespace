"""Localhost IPC between the CELL process and the ROS 1 bridge. Python standard library
only. Same framing as glace_live_ros1.ipc (proven on this PC), three message families:

    cell process -> bridge   type "result"   one CELL tick's output (+ optional event image)
    bridge -> cell process   type "guess"    the live mocap-derived H_guess_c2w (Mode A)
    bridge -> cell process   type "events"   a batch of raw events forwarded from the ROS 1
                                             prophesee driver (/prophesee/camera/cd_events_buffer)
                                             -- the CELL process has no Metavision SDK access of
                                             its own (SDK python bindings are cp38-only; cell_live
                                             is a python 3.10 env), so it never opens the camera;
                                             the already-running ROS1 driver does, same as every
                                             other consumer on this PC all session.
    cell process -> bridge   type "live"     a display-only event-frame image, sent at a fast
                                             fixed rate independent of the inference tick (which
                                             can take 200-600ms): keeps the RViz image panel
                                             responsive even while a pose solve is in flight. No
                                             pose, no status -- never a localization result.
    either direction         type "bye"      the sender is closing (optional)

Wire format (one message), identical framing to glace_live_ros1.ipc:

    offset 0   4 bytes   magic b"CLR1"
    offset 4   uint32    JSON length J (big endian), 2 <= J <= MAX_JSON_BYTES
    offset 8   uint32    blob length B (big endian), 0 <= B <= MAX_BLOB_BYTES
    offset 12  J bytes   UTF-8 JSON object (the metadata, schema below)
    then       B bytes   raw image bytes, described by meta["images"]

Only JSON is ever decoded (no pickle, no eval). Lengths are checked from the 12-byte
header before any payload is buffered. Partial messages are buffered until complete. A
framing error closes that connection; a well-framed message that fails validation is
dropped and the connection stays open.

"result" metadata (CELL process -> bridge):
    v, seq, stamp_ns          protocol version, increasing seq, Unix-epoch ns of the
                               event window end (wall clock)
    frame_index                int
    status                      "valid" | "held" | "skipped"
    success                     bool, true iff status == "valid"
    guess_source                "mocap" | "selftrack" | "none"
    H_guess_c2w                 16 floats or null: guess camera pose in world (row-major
                                T_W<-C), present whenever a guess was used this tick
    H_est_c2w                   16 floats or null: corrected camera pose in world, present
                                iff success
    inliers, n_events           ints
    step_cm                     float or null: ||centre(H_est) - centre(H_guess)|| in cm
    step_deg                    float or null: rotation angle between H_est and H_guess, degrees
    reason                      str
    latency_ms                  float or null; timings_ms {str: float}
    K                           4 floats or null: fx, fy, cx, cy (crop-resolution, crop-
                                adjusted principal point, i.e. `cp` from eval_full.get_corr)
    images                      list of {name, width, height, encoding, offset, length}

"guess" metadata (bridge -> CELL process):
    v, seq, stamp_ns           the mocap message's own stamp (Unix-epoch ns)
    H_guess_c2w                 16 floats: camera pose in world from mocap + T_body_camera
    source                      "mocap"

"events" metadata (bridge -> CELL process), blob-only payload (no "images"):
    v, seq, stamp_ns           Unix-epoch ns when this batch was flushed (wall clock)
    n_events                    int, number of events in the blob
    blob layout (concatenated, no padding): x u16[n] | y u16[n] | p u8[n] | t_epoch_s f64[n]
    (t_epoch_s: each event's own timestamp, Unix-epoch seconds, matching ev.ts.to_sec())

"live" metadata (CELL process -> bridge):
    v, seq, stamp_ns           Unix-epoch ns when the snapshot was taken (wall clock)
    images                      exactly one: {"event_frame_live", mono8}
"""

from __future__ import annotations

import collections
import ipaddress
import json
import math
import selectors
import socket
import struct
import threading
import time

MAGIC = b"CLR1"
PROTOCOL_VERSION = 1
HEADER = struct.Struct(">4sII")
MAX_JSON_BYTES = 64 * 1024
MAX_BLOB_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
DEFAULT_RESULT_PORT = 48811     # CELL -> bridge (results)
DEFAULT_GUESS_PORT = 48812      # bridge -> CELL (mocap-derived pose guess)
DEFAULT_EVENTS_PORT = 48813     # bridge -> CELL (raw events forwarded from the ROS1 driver)

TYPES = ("result", "guess", "events", "live", "bye")
RESULT_STATUSES = ("valid", "held", "skipped")
GUESS_SOURCES = ("mocap", "selftrack", "none")
IMAGE_NAMES = ("event_frame", "event_frame_vis", "event_frame_live")
IMAGE_TYPES = {"event_frame": "result", "event_frame_vis": "result", "event_frame_live": "live"}
CHANNELS = {"mono8": 1, "bgr8": 3}
EVENT_RECORD_BYTES = 2 + 2 + 1 + 8    # x u16, y u16, p u8, t_epoch_s f64
INT64 = 2 ** 63


class FramingError(ValueError):
    """The byte stream is corrupt; the connection must be closed."""


class MessageError(ValueError):
    """A well-framed message with invalid content; drop it, keep the connection."""


# --------------------------------------------------------------------------- encoding

def encode(meta: dict, images=(), raw_blob: bytes | None = None) -> bytes:
    """images: iterable of (name, width, height, encoding, bytes-like); mutually exclusive
    with raw_blob (used by "events" messages, whose payload is a flat columnar byte layout,
    not an image -- see the module docstring). Returns one framed message. Raises
    MessageError if the message would be rejected by the receiver."""
    meta = dict(meta)
    if raw_blob is not None:
        if images:
            raise MessageError("raw_blob and images are mutually exclusive")
        meta["images"] = []
        blob = bytes(raw_blob)
    else:
        descriptors, blobs, offset = [], [], 0
        for name, width, height, encoding, data in images:
            data = bytes(data)
            descriptors.append({"name": name, "width": int(width), "height": int(height), "encoding": encoding,
                                "offset": offset, "length": len(data)})
            blobs.append(data)
            offset += len(data)
        meta["images"] = descriptors
        blob = b"".join(blobs)
    validate(meta, len(blob))
    try:
        text = json.dumps(meta, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except ValueError as e:
        raise MessageError(f"not JSON-serializable: {e}") from None
    if len(text) > MAX_JSON_BYTES:
        raise MessageError(f"metadata {len(text)} B > {MAX_JSON_BYTES} B")
    return HEADER.pack(MAGIC, len(text), len(blob)) + text + blob


class Decoder:
    """Incremental decoder for one connection. feed() returns a list of (meta, blob) or
    MessageError instances (for well-framed but invalid messages); raises FramingError."""

    def __init__(self):
        self._buf = bytearray()
        self._need = None

    def pending_bytes(self):
        return len(self._buf)

    def feed(self, data):
        self._buf += data
        out = []
        while True:
            if self._need is None:
                if len(self._buf) < HEADER.size:
                    break
                magic, jlen, blen = HEADER.unpack_from(self._buf, 0)
                if magic != MAGIC:
                    raise FramingError("bad magic (not a CLR1 stream)")
                if not 2 <= jlen <= MAX_JSON_BYTES:
                    raise FramingError(f"metadata length {jlen} out of range")
                if blen > MAX_BLOB_BYTES:
                    raise FramingError(f"blob length {blen} > {MAX_BLOB_BYTES}")
                self._need = (jlen, blen)
            jlen, blen = self._need
            total = HEADER.size + jlen + blen
            if len(self._buf) < total:
                break
            text = bytes(self._buf[HEADER.size:HEADER.size + jlen])
            blob = bytes(self._buf[HEADER.size + jlen:total])
            del self._buf[:total]
            self._need = None
            try:
                meta = decode_meta(text)
                validate(meta, len(blob))
                out.append((meta, blob))
            except MessageError as e:
                out.append(e)
        return out


def _reject_constant(name):
    raise MessageError(f"non-finite number {name} in metadata")


def decode_meta(text: bytes) -> dict:
    try:
        meta = json.loads(text.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise MessageError(f"metadata is not valid UTF-8 JSON: {e}") from None
    if not isinstance(meta, dict):
        raise MessageError("metadata is not a JSON object")
    return meta


# --------------------------------------------------------------------------- validation

def _is_int(v, lo=-INT64, hi=INT64 - 1):
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check(cond, msg):
    if not cond:
        raise MessageError(msg)


def _opt(meta, key, pred, what):
    v = meta.get(key)
    _check(v is None or pred(v), f"{key}: expected {what} or null")


def _matrix(v, n):
    return isinstance(v, list) and len(v) == n and all(_is_num(x) for x in v)


def _short_str(v, n=1024):
    return isinstance(v, str) and len(v) <= n


def validate(meta: dict, blob_len: int):
    _check(meta.get("v") == PROTOCOL_VERSION, f"unsupported protocol version {meta.get('v')!r}")
    kind = meta.get("type")
    _check(kind in TYPES, f"unknown message type {kind!r}")
    if kind == "bye":
        return
    _check(_is_int(meta.get("seq"), 0), "seq: expected a non-negative int")
    _check(_is_int(meta.get("stamp_ns"), 0), "stamp_ns: expected a non-negative int (Unix epoch ns)")
    _opt(meta, "H_guess_c2w", lambda v: _matrix(v, 16), "16 finite numbers")

    if kind == "guess":
        _check(meta.get("H_guess_c2w") is not None, "a guess message must carry H_guess_c2w")
        _check(meta.get("source") in GUESS_SOURCES, f"source {meta.get('source')!r} invalid")
        return

    if kind == "events":
        n = meta.get("n_events")
        _check(_is_int(n, 0), "n_events: expected a non-negative int")
        _check(blob_len == n * EVENT_RECORD_BYTES, "events blob length does not match n_events")
        return

    if kind == "live":
        images = meta.get("images", [])
        _check(isinstance(images, list) and len(images) == 1, "a live message must carry exactly one image")
        d = images[0]
        _check(d.get("name") == "event_frame_live", "a live message's image must be named event_frame_live")
        _check(d.get("encoding") in CHANNELS, f"unsupported encoding {d.get('encoding')!r}")
        w, h, off, length = d.get("width"), d.get("height"), d.get("offset"), d.get("length")
        _check(all(_is_int(x, 0) for x in (w, h, off, length)), "image fields must be non-negative ints")
        _check(0 < w <= MAX_IMAGE_SIDE and 0 < h <= MAX_IMAGE_SIDE, "image size out of range")
        _check(length == w * h * CHANNELS[d["encoding"]], "image length does not match width x height x channels")
        _check(off == 0 and length == blob_len, "live image bytes must be the entire blob")
        return

    # kind == "result"
    _check(_is_int(meta.get("frame_index"), 0), "frame_index: expected a non-negative int")
    status = meta.get("status")
    _check(status in RESULT_STATUSES, f"status {status!r} invalid for a result")
    success = meta.get("success")
    _check(isinstance(success, bool) and success == (status == "valid"), "success must equal (status == 'valid')")
    _check(meta.get("guess_source") in GUESS_SOURCES, f"guess_source {meta.get('guess_source')!r} invalid")
    _opt(meta, "H_est_c2w", lambda v: _matrix(v, 16), "16 finite numbers")
    _check((meta.get("H_est_c2w") is not None) == success, "H_est_c2w must be present iff success")
    _check(_is_int(meta.get("inliers", 0), -1, 2 ** 31 - 1), "inliers: expected an int32")
    _check(_is_int(meta.get("n_events", 0), -1), "n_events: expected an int")
    _opt(meta, "step_cm", _is_num, "finite number")
    _opt(meta, "step_deg", _is_num, "finite number")
    _check(_short_str(meta.get("reason", "")), "reason: expected a string <= 1024 chars")
    _opt(meta, "latency_ms", _is_num, "finite number")
    timings = meta.get("timings_ms", {})
    _check(isinstance(timings, dict) and len(timings) <= 16
           and all(_short_str(k, 32) and _is_num(v) for k, v in timings.items()), "timings_ms: invalid")
    _opt(meta, "K", lambda v: _matrix(v, 4), "4 finite numbers (fx,fy,cx,cy)")

    images = meta.get("images", [])
    _check(isinstance(images, list) and len(images) <= len(IMAGE_NAMES), "images: invalid list")
    end = 0
    for d in images:
        _check(isinstance(d, dict), "image descriptor must be an object")
        name, enc = d.get("name"), d.get("encoding")
        _check(name in IMAGE_NAMES, f"unknown image name {name!r}")
        _check(IMAGE_TYPES[name] == "result", f"image {name!r} not allowed in a result message")
        _check(enc in CHANNELS, f"unsupported encoding {enc!r}")
        w, h, off, length = d.get("width"), d.get("height"), d.get("offset"), d.get("length")
        _check(all(_is_int(x, 0) for x in (w, h, off, length)), "image fields must be non-negative ints")
        _check(0 < w <= MAX_IMAGE_SIDE and 0 < h <= MAX_IMAGE_SIDE, "image size out of range")
        _check(length == w * h * CHANNELS[enc], "image length does not match width x height x channels")
        _check(off == end and off + length <= blob_len, "image bytes out of bounds or not contiguous")
        end = off + length
    _check(end == blob_len, "blob has bytes not described by any image")


# --------------------------------------------------------------------------- sender

def _require_loopback(host):
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"IPC is localhost-only; refusing host {host!r} (use 127.0.0.1)")


class Sender:
    """Client side. send() never blocks the caller: messages go into a bounded queue
    (oldest dropped) drained by a daemon thread that (re)connects to the receiver, with
    socket timeouts so a stalled or absent peer only causes dropped messages. Messages
    older than max_age_s are dropped instead of being sent late after a reconnect."""

    def __init__(self, port, host="127.0.0.1", max_pending=2, connect_timeout_s=0.5,
                 send_timeout_s=2.0, retry_interval_s=1.0, max_age_s=1.0, log=print):
        _require_loopback(host)
        self.addr = (host, int(port))
        self.max_age_s, self.retry_interval_s = max_age_s, retry_interval_s
        self.connect_timeout_s, self.send_timeout_s = connect_timeout_s, send_timeout_s
        self._q = collections.deque(maxlen=max_pending)
        self._cv = threading.Condition()
        self._closing = False
        self._sock = None
        self._last_attempt = 0.0
        self._log = log
        self.stats = {"queued": 0, "sent": 0, "dropped_queue": 0, "dropped_stale": 0, "dropped_unconnected": 0,
                      "send_errors": 0, "connects": 0}
        self._thread = threading.Thread(target=self._run, name="clr1-sender", daemon=True)
        self._thread.start()

    @property
    def connected(self):
        return self._sock is not None

    def send(self, meta, images=(), raw_blob=None):
        data = encode(meta, images, raw_blob)
        with self._cv:
            if len(self._q) == self._q.maxlen:
                self.stats["dropped_queue"] += 1
            self._q.append((time.monotonic(), data))
            self.stats["queued"] += 1
            self._cv.notify()

    def close(self, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._q and self._sock is not None and time.monotonic() < deadline:
                self._cv.wait(0.05)
            self._closing = True
            self._cv.notify()
        self._thread.join(max(0.1, deadline - time.monotonic()))
        sock, self._sock = self._sock, None
        if sock is not None and self._thread.is_alive():
            sock.close()
        elif sock is not None:
            try:
                sock.settimeout(0.2)
                sock.sendall(encode({"v": PROTOCOL_VERSION, "type": "bye"}))
            except OSError:
                pass
            sock.close()

    def _connect(self):
        self._last_attempt = time.monotonic()
        try:
            sock = socket.create_connection(self.addr, timeout=self.connect_timeout_s)
        except OSError:
            return False
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.send_timeout_s)
        self._sock = sock
        self.stats["connects"] += 1
        self._log(f"[ipc] connected to {self.addr[0]}:{self.addr[1]}")
        return True

    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._closing:
                    self._cv.wait(0.2)
                if self._closing:
                    return
                t_queued, data = self._q.popleft()
                self._cv.notify_all()
            if time.monotonic() - t_queued > self.max_age_s:
                self.stats["dropped_stale"] += 1
                continue
            if self._sock is None:
                if time.monotonic() - self._last_attempt < self.retry_interval_s or not self._connect():
                    self.stats["dropped_unconnected"] += 1
                    continue
            try:
                self._sock.sendall(data)
                self.stats["sent"] += 1
            except OSError as e:
                self.stats["send_errors"] += 1
                self._log(f"[ipc] send failed ({type(e).__name__}: {e}); will reconnect")
                try:
                    self._sock.close()
                finally:
                    self._sock = None


# --------------------------------------------------------------------------- server

class Server:
    """Listener side: non-blocking listener on a loopback address, several clients, one
    Decoder per client. poll() returns events: ("connected", cid, peer),
    ("message", cid, (meta, blob)), ("rejected", cid, reason), ("disconnected", cid, reason)."""

    def __init__(self, port, host="127.0.0.1", max_clients=4, recv_budget=8 * 1024 * 1024):
        _require_loopback(host)
        self._sel = selectors.DefaultSelector()
        self._lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._lsock.bind((host, int(port)))
        self._lsock.listen(max_clients)
        self._lsock.setblocking(False)
        self._sel.register(self._lsock, selectors.EVENT_READ, None)
        self._clients = {}
        self._next_id = 1
        self.max_clients, self.recv_budget = max_clients, recv_budget
        self.address = self._lsock.getsockname()

    @property
    def n_clients(self):
        return len(self._clients)

    def poll(self, timeout_s=0.1):
        events = []
        for key, _ in self._sel.select(timeout_s):
            if key.data is None:
                self._accept(events)
            else:
                self._read(key.data, events)
        return events

    def _accept(self, events):
        try:
            sock, peer = self._lsock.accept()
        except BlockingIOError:
            return
        if len(self._clients) >= self.max_clients or not ipaddress.ip_address(peer[0]).is_loopback:
            sock.close()
            events.append(("rejected", 0, f"connection from {peer} refused (limit or non-loopback)"))
            return
        sock.setblocking(False)
        cid, self._next_id = self._next_id, self._next_id + 1
        self._clients[cid] = (sock, Decoder())
        self._sel.register(sock, selectors.EVENT_READ, cid)
        events.append(("connected", cid, f"{peer[0]}:{peer[1]}"))

    def _read(self, cid, events):
        sock, dec = self._clients[cid]
        budget = self.recv_budget
        while budget > 0:
            try:
                data = sock.recv(min(1 << 20, budget))
            except BlockingIOError:
                return
            except OSError as e:
                return self.drop(cid, events, f"socket error: {e}")
            if not data:
                reason = "closed by peer" + (f" with {dec.pending_bytes()} B of an incomplete message"
                                             if dec.pending_bytes() else "")
                return self.drop(cid, events, reason)
            budget -= len(data)
            try:
                items = dec.feed(data)
            except FramingError as e:
                return self.drop(cid, events, f"framing error: {e}")
            for item in items:
                if isinstance(item, MessageError):
                    events.append(("rejected", cid, str(item)))
                elif item[0].get("type") == "bye":
                    return self.drop(cid, events, "sender said bye")
                else:
                    events.append(("message", cid, item))

    def drop(self, cid, events, reason):
        sock, _ = self._clients.pop(cid)
        self._sel.unregister(sock)
        sock.close()
        events.append(("disconnected", cid, reason))

    def close(self):
        for cid in list(self._clients):
            self.drop(cid, [], "server closing")
        self._sel.unregister(self._lsock)
        self._lsock.close()
        self._sel.close()
