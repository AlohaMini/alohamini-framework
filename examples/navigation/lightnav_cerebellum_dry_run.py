"""Replay recorded camera and wheel events through live LightNav and ROS dry-run."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from lightnav_vln import exchange


NAMESPACE = "/alohamini_cerebellum_dry_run"


def get_plan(ws, seq, jpeg, instruction):
    reply = exchange(ws, "next", {"seq": seq, "instruction": instruction,
                                   "image": base64.b64encode(jpeg).decode("ascii")})
    if type(reply.get("seq")) is not int or reply["seq"] != seq or type(reply.get("stop")) is not bool:
        raise ValueError("invalid LightNav sequence or stop flag")
    if reply["stop"]:
        return None
    nested = reply.get("actions")
    path = nested.get("actions") if isinstance(nested, dict) else None
    if not isinstance(path, list) or not path:
        raise ValueError("missing LightNav path")
    for pose in path:
        if (not isinstance(pose, list) or len(pose) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pose)):
            raise ValueError("expected finite [forward m, left m, yaw rad] waypoint")
    return path


def load_recording(trial):
    actions_path, = trial.rglob("actions.json")
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    log = (trial / "controller.log").read_text(encoding="utf-8")
    captures = dict((int(seq), int(stamp)) for seq, stamp in re.findall(r"seq=(\d+) capture_ns=(\d+)", log))
    frames = [(captures[a["seq"]], actions_path.parent / f'image_{a["step"]:06d}.jpg')
              for a in actions if a["seq"] in captures]
    if not frames or any(b[0] <= a[0] for a, b in zip(frames, frames[1:])):
        raise ValueError("recording lacks ordered, stamped camera frames")
    with (trial / "wheel-odom.csv").open(newline="", encoding="utf-8-sig") as stream:
        wheel = [(int(r["stamp_ns"]), tuple(float(r[key]) for key in ("vx_m_s", "vy_m_s", "w_rad_s")))
                 for r in csv.DictReader(stream)]
    if not wheel or any(b[0] <= a[0] for a, b in zip(wheel, wheel[1:])):
        raise ValueError("recording lacks ordered, stamped wheel feedback")
    first = frames[0][0]
    # Read and validate all inputs before publishing synthetic ROS events.
    return ([(stamp-first, file.read_bytes()) for stamp, file in frames],
            [(stamp-first, velocity) for stamp, velocity in wheel if stamp >= first-100_000_000],
            actions[0]["instruction"])


def run(args):
    import rclpy
    import websocket
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import OccupancyGrid, Path as RosPath
    from std_msgs.msg import Bool, String

    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or not os.environ.get("ROS_DOMAIN_ID"):
        raise RuntimeError("recorded input requires an isolated local ROS domain")
    if urlsplit(args.server).hostname != "127.0.0.1":
        raise ValueError("expected a local LightNav server")
    frames, wheels, instruction = load_recording(args.trial)
    args.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("lightnav_recorded_dry_run")
    path_pub = node.create_publisher(RosPath, NAMESPACE + "/reference", 1)
    stop_pub = node.create_publisher(Bool, NAMESPACE + "/stop", 1)
    wheel_pub = node.create_publisher(TwistStamped, "/alohamini/base_velocity", 10)
    grid_pub = node.create_publisher(OccupancyGrid, NAMESPACE + "/local_grid", 1)
    states, predictions = [], []

    def decision(msg):
        body = json.loads(msg.data)
        if body["dry_run"] is not True:
            raise RuntimeError("expected dry-run diagnostics")
        states.append(body)

    node.create_subscription(String, NAMESPACE + "/decision", decision, 10)
    ws = None
    try:
        ws = websocket.create_connection(args.server, timeout=3., http_no_proxy=["127.0.0.1"])
        exchange(ws, "login", {"clientId": "jjy-cerebellum-recorded"})
        exchange(ws, "reset", {})
        deadline = time.monotonic() + 3
        while not path_pub.get_subscription_count() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
        if not path_pub.get_subscription_count():
            raise RuntimeError("dry-run ROS reference subscriber unavailable")
        zero_steady, zero_wall = time.monotonic_ns(), time.time_ns()
        frame_idx = wheel_idx = 0
        future = None
        next_map_ns = int(args.synthetic_after_s * 1e9) if args.synthetic_after_s is not None else None
        end = frames[-1][0] + 900_000_000

        def stamp(offset_ns):
            sec, nanosec = divmod(zero_wall + offset_ns, 1_000_000_000)
            return Time(sec=sec, nanosec=nanosec)

        with ThreadPoolExecutor(max_workers=1) as worker:
            while (elapsed := time.monotonic_ns() - zero_steady) <= end:
                while wheel_idx < len(wheels) and wheels[wheel_idx][0] <= elapsed:
                    offset, velocity = wheels[wheel_idx]
                    wheel_idx += 1
                    msg = TwistStamped()
                    msg.header.frame_id, msg.header.stamp = "base_link", stamp(offset)
                    msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = velocity
                    wheel_pub.publish(msg)
                if next_map_ns is not None and elapsed >= next_map_ns:
                    grid = OccupancyGrid()
                    grid.header.frame_id, grid.header.stamp = "odom", stamp(elapsed)
                    grid.info.resolution, grid.info.width, grid.info.height = .1, 60, 60
                    grid.info.origin.position.x = grid.info.origin.position.y = -3.
                    grid.info.origin.orientation.w = 1.
                    grid.data = [0] * (grid.info.width * grid.info.height)
                    grid_pub.publish(grid)
                    next_map_ns = elapsed + 200_000_000
                if future is not None and future.done():
                    path = future.result()
                    predictions.append({"seq": active_seq, "capture_ns_original_offset": captured,
                                        "receive_age_ms": (elapsed-captured) / 1e6, "stop": path is None,
                                        "path": path})
                    if path is None:
                        stop_pub.publish(Bool(data=True))
                    else:
                        msg = RosPath()
                        msg.header.frame_id, msg.header.stamp = "base_link", stamp(captured)
                        for x, y, yaw in path:
                            pose = PoseStamped()
                            pose.header = msg.header
                            pose.pose.position.x, pose.pose.position.y = x, y
                            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
                            msg.poses.append(pose)
                        path_pub.publish(msg)
                    future = None
                if future is None and frame_idx < len(frames) and frames[frame_idx][0] <= elapsed:
                    while frame_idx + 1 < len(frames) and frames[frame_idx+1][0] <= elapsed:
                        frame_idx += 1
                    captured, jpeg = frames[frame_idx]
                    active_seq = frame_idx
                    future = worker.submit(get_plan, ws, active_seq, jpeg, instruction)
                    frame_idx += 1
                rclpy.spin_once(node, timeout_sec=.004)
        published = node.get_publisher_names_and_types_by_node("alohamini_cerebellum_dry_run", "/")
        if NAMESPACE + "/decision" not in dict(published) or any(
                set(types) & {"geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped"} for _, types in published):
            raise RuntimeError("unexpected Cerebellum ROS publishers")
        report = {
            "recording": args.trial.name, "replayed_wheel_samples": wheel_idx,
            "predictions": len(predictions), "stop_predictions": sum(p["stop"] for p in predictions),
            "synthetic_grid_after_s": args.synthetic_after_s, "states": Counter(s["state"] for s in states),
            "reasons": Counter(s["reason"] for s in states), "max_jitter_ms": max(
                (s["timing"]["jitter_s"] * 1000 for s in states), default=None),
            "missed_periods": sum(s["timing"]["missed_periods"] for s in states),
            "controller_publishers": published,
        }
        if not predictions or not states:
            raise RuntimeError("missing predictions or dry-run decisions")
        for name, rows in (("lightnav", predictions), ("cerebellum", states)):
            with (args.out / (name + ".jsonl")).open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
        (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if ws is not None:
            ws.close()
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--synthetic-after-s", type=float)
    args = parser.parse_args()
    if args.synthetic_after_s is not None and not (
        math.isfinite(args.synthetic_after_s) and args.synthetic_after_s >= 0
    ):
        parser.error("--synthetic-after-s must be finite and >= 0")
    run(args)


if __name__ == "__main__":
    main()
