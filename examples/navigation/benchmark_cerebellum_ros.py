"""20 Hz closed-loop dry-run benchmark with a near obstacle and a dense far wall."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time


def run(out, cycles, blocked_lookahead=False):
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import OccupancyGrid, Path as RosPath
    from std_msgs.msg import String

    if os.getenv("ROS_LOCALHOST_ONLY") != "1" or not os.getenv("ROS_DOMAIN_ID") or cycles < 200:
        raise ValueError("Select an isolated localhost ROS domain and at least 200 cycles")
    out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node("cerebellum_m1_benchmark")
    ns = "/alohamini_cerebellum_dry_run"
    wheels = node.create_publisher(TwistStamped, "/alohamini/base_velocity", 10)
    maps = node.create_publisher(OccupancyGrid, ns+"/local_grid", 1)
    paths = node.create_publisher(RosPath, ns+"/reference", 1)
    records = []
    def record(msg):
        records.append(json.loads(msg.data))
    node.create_subscription(String, ns+"/decision", record, 100)
    grid = OccupancyGrid()
    grid.header.frame_id = "odom"
    # The second scene puts the 0.8 m lookahead inside the inflated obstacle.
    far_start, block_start, end_x = (72, 34, 2.) if blocked_lookahead else (52, 25, 1.)
    grid.info.width, grid.info.height, grid.info.resolution = far_start + 48, 60, .05
    grid.info.origin.position.x, grid.info.origin.position.y = -1.,-1.5
    grid.info.origin.orientation.w = 1.
    grid.data = [100 if (far_start <= i < far_start+46 and j < 58) or
                 (block_start <= i < block_start+3 and 29 <= j < 31) else 0
                 for j in range(grid.info.height) for i in range(grid.info.width)]
    ref = RosPath()
    ref.header.frame_id = "odom"
    for x in (0., end_x, end_x):
        point = PoseStamped()
        point.pose.position.x, point.pose.orientation.w = x,1.
        ref.poses.append(point)
    wheel = TwistStamped()
    wheel.header.frame_id = "base_link"
    command = [sys.executable,"-m","alohamini.cerebellum.ros_adapter","--ros-args","-p","dry_run:=true"]
    try:
        with (out/"adapter.log").open("w") as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                start = time.monotonic()
                next_wheel = next_map = next_ref = start
                first = None
                while time.monotonic()-start < cycles*.12+12:
                    if process.poll() is not None:
                        raise RuntimeError("adapter exited; see adapter.log")
                    now = time.monotonic()
                    stamp = node.get_clock().now().to_msg()
                    if now >= next_wheel:
                        wheel.header.stamp = stamp
                        if records:
                            v = records[-1]["planned_velocity"]
                            wheel.twist.linear.x = v["x_m_s"]
                            wheel.twist.linear.y = v["y_m_s"]
                            wheel.twist.angular.z = v["yaw_rad_s"]
                        wheels.publish(wheel)
                        next_wheel += .02
                    if now >= next_map:
                        grid.header.stamp = stamp
                        maps.publish(grid)
                        next_map += .20
                    if now >= next_ref:
                        ref.header.stamp = stamp
                        paths.publish(ref)
                        next_ref += .10
                    rclpy.spin_once(node,timeout_sec=.003)
                    if first is None and records and records[-1]["state"] == "TRACKING":
                        first = len(records)-1
                    if first is not None and len(records)-first >= cycles:
                        break
                publishers = dict(node.get_publisher_names_and_types_by_node(ns[1:],"/"))
            finally:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=10)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        with (out/"decisions.jsonl").open("w") as stream:
            for row in records:
                print(json.dumps(row,allow_nan=False),file=stream)
    if first is None or len(records)-first < cycles:
        raise RuntimeError("did not receive the required consecutive dry-run decisions")
    rows = records[first:first+cycles]
    times = sorted(r["tick_compute_s"] for r in rows)
    summary = {
        "host":platform.node(), "python":platform.python_version(), "command":command,
        "domain":os.environ["ROS_DOMAIN_ID"], "localhost_only":True, "dry_run":all(r["dry_run"] for r in rows),
        "occupancy":grid.data.count(100), "map_cells":len(grid.data), "map_resolution_m":grid.info.resolution,
        "scene":"blocked_lookahead" if blocked_lookahead else "m1_near_obstacle",
        "frequency_hz":20, "wheel_hz":50,"map_hz":5,"reference_hz":10,
        "cycles":len(rows),"warmup_rows":first,"elapsed_first_to_last_s":rows[-1]["timing"]["now"]-rows[0]["timing"]["now"],
        "missed_periods":sum(r["timing"]["missed_periods"] for r in rows),
        "expired_during_control_tick":sum(r["reason"]=="expired_during_control_tick" for r in rows),
        "tick_compute_s":{"median":statistics.median(times), "p99":times[math.ceil(.99*len(times))-1], "max":max(times)},
        "max_jitter_ms":1000*max(abs(r["timing"]["jitter_s"]) for r in rows),
        "states":dict(Counter(r["state"] for r in rows)), "reasons":dict(Counter(r["reason"] for r in rows)),
        "final_pose":rows[-1]["pose_odom"],"publishers":publishers,
        "missing_decision_messages":rows[-1]["tick_sequence"]-rows[0]["tick_sequence"]+1-len(rows),
    }
    print(json.dumps(summary,indent=2))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if (not summary["dry_run"] or summary["missing_decision_messages"] or grid.data.count(100)<2500
            or any(t.startswith("geometry_msgs/msg/Twist") for types in publishers.values() for t in types)):
        raise AssertionError("dry-run benchmark output verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cycles",type=int,default=240)
    parser.add_argument("--blocked-lookahead", action="store_true")
    args = parser.parse_args()
    run(args.out, args.cycles, args.blocked_lookahead)

