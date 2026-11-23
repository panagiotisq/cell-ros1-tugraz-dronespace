#!/usr/bin/env python3
"""ROS 1 bridge process: runs the mocap -> guess forwarder, the raw-event forwarder, and
the result receiver that publishes CELL's live output to ROS Noetic. Start it through
launch/start_bridge.sh, which provides a clean environment (ROS Noetic only, no catkin
overlay beyond the external driver workspace, no conda, dedicated ROS master).

    launch/start_bridge.sh [--config <yaml>]

Refuses to run against the default master port 11311 unless --allow-default-master is
given. Ctrl-C or SIGTERM stops it cleanly.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cell_live import ipc  # noqa: E402


def check_master(allow_default):
    uri = os.environ.get("ROS_MASTER_URI", "")
    u = urlparse(uri)
    if u.hostname not in ("127.0.0.1", "localhost") or not u.port:
        raise SystemExit(f"ROS_MASTER_URI={uri!r}: expected a localhost master (use launch/start_bridge.sh)")
    if u.port == 11311 and not allow_default:
        raise SystemExit("ROS_MASTER_URI uses the default port 11311; this bridge uses a dedicated master "
                         "(launch/start_master.sh). Pass --allow-default-master to override.")
    import rosgraph
    if not rosgraph.Master("/cell_ros1_bridge_probe", master_uri=uri).is_online():
        raise SystemExit(f"no ROS master at {uri}: start it with launch/start_master.sh")
    return uri


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result-port", type=int, default=int(os.environ.get("CLR1_RESULT_PORT", ipc.DEFAULT_RESULT_PORT)))
    p.add_argument("--guess-port", type=int, default=int(os.environ.get("CLR1_GUESS_PORT", ipc.DEFAULT_GUESS_PORT)))
    p.add_argument("--events-port", type=int, default=int(os.environ.get("CLR1_EVENTS_PORT", ipc.DEFAULT_EVENTS_PORT)))
    p.add_argument("--mocap-topic", default="/mocap_node/evk4/pose")
    p.add_argument("--events-topic", default="/prophesee/camera/cd_events_buffer")
    p.add_argument("--event-decode", choices=["raw", "rospy"],
                   default=os.environ.get("CLR1_EVENT_DECODE", "raw"),
                   help="'raw' (default): hand-decode message bytes with numpy, ~450x faster "
                        "than rospy's own deserialization for this message type -- see "
                        "EventForwarder's class docstring. 'rospy': the original, slower but "
                        "longer-proven path; pass this (or set CLR1_EVENT_DECODE=rospy) to "
                        "revert instantly if 'raw' ever misbehaves on-site.")
    p.add_argument("--extrinsic", default=str(Path(__file__).resolve().parents[1] / "configs" / "extrinsics"
                                              / "T_body_camera.txt"))
    p.add_argument("--world-frame", default="dronespace_world")
    p.add_argument("--camera-frame", default="cell_camera_frame")
    p.add_argument("--guess-frame", default="mocap_guess_frame")
    p.add_argument("--prefix", default="/cell")
    p.add_argument("--map", default=str(Path(__file__).resolve().parents[2] / "drone_space" / "map"
                                        / "fused_clean_mocap.ply"),
                   help="aligned map PLY to publish once (latched) for RViz reference; '' to skip")
    p.add_argument("--allow-default-master", action="store_true")
    p.add_argument("--stats-every", type=float, default=10.0, help="s between log lines (0 = off)")
    args = p.parse_args()

    uri = check_master(args.allow_default_master)

    import rospy
    for mod in (rospy,):
        if not mod.__file__.startswith(("/opt/ros/noetic/", "/usr/lib/python3/")):
            raise SystemExit(f"{mod.__name__} resolved to {mod.__file__}: environment not sanitized")
    rospy.init_node("cell_ros1_bridge", anonymous=False, disable_signals=False)

    from cell_live.ros1_bridge import EventForwarder, GuessForwarder, Ros1Bridge

    guess_fwd = GuessForwarder(args.mocap_topic, args.extrinsic, args.world_frame, args.guess_frame,
                               args.prefix, guess_port=args.guess_port)
    event_fwd = EventForwarder(args.events_topic, events_port=args.events_port, decode_mode=args.event_decode)
    bridge = Ros1Bridge(world_frame=args.world_frame, camera_frame=args.camera_frame, prefix=args.prefix)
    server = ipc.Server(port=args.result_port)

    map_pub = None
    if args.map:
        from cell_live.map_publisher import publish_static_map
        try:
            map_pub = publish_static_map(args.map, args.world_frame, topic=f"{args.prefix}/map")
        except (OSError, ValueError) as e:
            rospy.logwarn(f"could not publish the map for RViz ({e}); continuing without it")

    rospy.loginfo(f"cell_ros1_bridge: master {uri}")
    rospy.loginfo(f"  mocap guess:  {args.mocap_topic} -> IPC :{args.guess_port} (T_body_camera from {args.extrinsic})")
    rospy.loginfo(f"  raw events:   {args.events_topic} -> IPC :{args.events_port} "
                 f"(decode_mode={args.event_decode})")
    rospy.loginfo(f"  CELL results: IPC :{server.address[1]} -> {args.prefix}/*")

    result_clients = set()
    rejected_log_t = 0.0
    last_stats = time.monotonic()
    try:
        while not rospy.is_shutdown():
            for kind, cid, info in server.poll(0.1):
                if kind == "message":
                    meta, blob = info
                    if meta["type"] == "result":
                        result_clients.add(cid)
                    err = bridge.handle(meta, blob)
                    if err and time.monotonic() - rejected_log_t > 2.0:
                        rospy.logwarn(f"client {cid}: message not published: {err}")
                        rejected_log_t = time.monotonic()
                elif kind == "rejected":
                    bridge.counts["rejected"] += 1
                    if time.monotonic() - rejected_log_t > 2.0:
                        rospy.logwarn(f"client {cid}: invalid message dropped: {info}")
                        rejected_log_t = time.monotonic()
                elif kind == "connected":
                    rospy.loginfo(f"IPC client {cid} connected from {info}")
                elif kind == "disconnected":
                    rospy.loginfo(f"IPC client {cid} disconnected: {info}")
                    if cid in result_clients:
                        result_clients.discard(cid)
                        if not result_clients:
                            bridge.publish_disconnected()
            if args.stats_every and time.monotonic() - last_stats > args.stats_every:
                last_stats = time.monotonic()
                rospy.loginfo(f"clients {server.n_clients} | {bridge.counts} | {bridge.rate_hz:.1f} Hz | "
                              f"events received {event_fwd.n_received} sent {event_fwd.n_sent}")
    finally:
        server.close()
        print(f"cell_ros1_bridge stopped: {bridge.counts}", flush=True)


if __name__ == "__main__":
    main()
