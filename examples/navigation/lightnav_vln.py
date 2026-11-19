"""Stream the AlohaMini forward camera to a LightNav-0 VLN server over WebSocket."""

import argparse
import base64
import json
import math
import time


def exchange(ws, action, data):
    ws.send(json.dumps({"action": action, "data": data}))
    msg = json.loads(ws.recv())
    if not isinstance(msg, dict) or msg.get("action") != action:
        raise ValueError(f"Unexpected reply to {action}")
    reply = msg.get("data")
    if not isinstance(reply, dict) or type(reply.get("rc")) is not int or reply["rc"] != 0:
        raise ValueError(f"LightNav {action} failed")
    return reply


def predict(ws, seq, jpeg, instruction):
    reply = exchange(ws, "next", {
        "seq": seq, "instruction": instruction, "image": base64.b64encode(jpeg).decode("ascii")
    })
    if type(reply.get("seq")) is not int or reply["seq"] != seq or type(reply.get("stop")) is not bool:
        raise ValueError("Invalid sequence or stop flag")
    if reply["stop"]:
        return None
    chunk = reply.get("actions")
    rows = chunk.get("actions") if isinstance(chunk, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty LightNav trajectory")
    row = rows[0]
    if not isinstance(row, list) or len(row) != 3 or not all(
        type(v) in (int, float) and math.isfinite(v) for v in row
    ):
        raise ValueError("Expected a finite forward, lateral, yaw waypoint")
    return row


def velocity(waypoint):
    """Use LightNav's reference step scales and conservative indoor speed limits."""
    scales = (0.375, 0.25, math.pi / 20)
    limits = (0.15, 0.15, 0.5)  # m/s forward, m/s left, rad/s counterclockwise
    return tuple(max(-1.0, min(1.0, v / scale)) * limit
                 for v, scale, limit in zip(waypoint, scales, limits))


def main():
    import rclpy
    from functools import partial
    from lightnav_views import FrameSet, ViewSessions
    from geometry_msgs.msg import Twist
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--camera", default="/alohamini/cameras/forward/image_raw/compressed")
    parser.add_argument("--chest-camera", metavar="TOPIC", help="check the forward waypoint against a synchronized chest view")
    parser.add_argument("--drive", action="store_true", help="Publish to the robot /cmd_vel")
    args = parser.parse_args()
    if not args.instruction.strip():
        parser.error("provide a navigation instruction")
    if args.chest_camera == args.camera:
        parser.error("provide distinct forward and chest camera topics")
    rclpy.init(args=[])
    node = rclpy.create_node("alohamini_lightnav_vln")
    topic = "/cmd_vel" if args.drive else "/alohamini/lightnav/preview_cmd_vel"
    pub = node.create_publisher(Twist, topic, 1)
    period = 0.25
    max_age = 0.45  # shorter than the default 0.5 s bridge watchdog
    cameras = {"forward": args.camera}
    if args.chest_camera:
        cameras["chest"] = args.chest_camera
    frames = FrameSet(cameras, max_age=max_age)

    def receive(name, msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        frames.put(name, stamp, msg.data)

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    for name, camera_topic in cameras.items():
        node.create_subscription(CompressedImage, camera_topic, partial(receive, name), qos)
    seq = 0
    sessions = None
    try:
        sessions = ViewSessions(args.server, cameras, max_age)
        node.get_logger().info(f"LightNav velocity output: {topic}")
        while rclpy.ok():
            started = time.monotonic()
            pair = None
            while pair is None and time.monotonic() - started < period:
                rclpy.spin_once(node, timeout_sec=0.025)
                pair = frames.take(node.get_clock().now().nanoseconds)
            command = Twist()
            if pair is not None:
                stamp, images = pair
                waypoint, reason = sessions.step(seq, images, args.instruction)
                seq += 1
                if waypoint is None:
                    node.get_logger().info(reason)
                    break
                age = (node.get_clock().now().nanoseconds - stamp) / 1e9
                if 0 <= age < max_age and time.monotonic() - started < max_age:
                    command.linear.x, command.linear.y, command.angular.z = velocity(waypoint)
            pub.publish(command)
            time.sleep(max(0, period - (time.monotonic() - started)))
    finally:
        if rclpy.ok():
            pub.publish(Twist())
        if sessions is not None:
            sessions.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
