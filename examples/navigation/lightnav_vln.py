"""Stream one AlohaMini camera to a LightNav-0 VLN server over WebSocket."""

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


def request_prediction(ws, seq, jpeg, instruction, *, continuous_follow=False):
    # SlowFast stores complete episodes. Rotate visual history before frame 65;
    # the ROS sequence and controller STOP/estop state are independent of reset.
    if continuous_follow and seq > 1 and (seq - 1) % 64 == 0:
        exchange(ws, "reset", {})
    return exchange(ws, "next", {
        "seq": seq, "instruction": instruction, "image": base64.b64encode(jpeg).decode("ascii")
    })


def predict(ws, seq, jpeg, instruction):
    reply = request_prediction(ws, seq, jpeg, instruction)
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


def camera_topic(camera):
    if camera in ("forward", "chest"):
        return f"/alohamini/cameras/{camera}/image_raw/compressed"
    if camera.startswith("/"):
        return camera
    raise ValueError("Select forward, chest, or an absolute compressed-image topic")


def main():
    import rclpy
    import websocket
    from geometry_msgs.msg import Twist
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--camera", default="forward", type=camera_topic,
                        help="forward, chest, or an absolute compressed-image topic")
    parser.add_argument("--drive", action="store_true", help="Publish to the robot /cmd_vel")
    args = parser.parse_args()
    if not args.instruction.strip():
        parser.error("provide a navigation instruction")
    rclpy.init(args=[])
    node = rclpy.create_node("alohamini_lightnav_vln")
    topic = "/cmd_vel" if args.drive else "/alohamini/lightnav/preview_cmd_vel"
    pub = node.create_publisher(Twist, topic, 1)
    latest = None

    def receive(msg):
        nonlocal latest
        latest = msg

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(CompressedImage, args.camera, receive, qos)
    period = 0.25
    max_age = 0.45  # shorter than the default 0.5 s bridge watchdog
    seq = 0
    previous_stamp = None
    ws = None
    try:
        ws = websocket.create_connection(args.server, timeout=2.0)
        exchange(ws, "login", {"clientId": "alohamini"})
        exchange(ws, "reset", {})
        ws.settimeout(max_age)
        node.get_logger().info(f"LightNav velocity output: {topic}")
        while rclpy.ok():
            started = time.monotonic()
            rclpy.spin_once(node, timeout_sec=period)
            msg, latest = latest, None
            command = Twist()
            if msg is not None:
                stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                age = (node.get_clock().now().nanoseconds - stamp) / 1e9
                if stamp > 0 and (previous_stamp is None or stamp > previous_stamp) and 0 <= age < max_age:
                    previous_stamp = stamp
                    waypoint = predict(ws, seq, bytes(msg.data), args.instruction)
                    seq += 1
                    if waypoint is None:
                        node.get_logger().info("LightNav reported goal reached")
                        break
                    age = (node.get_clock().now().nanoseconds - stamp) / 1e9
                    if 0 <= age < max_age and time.monotonic() - started < max_age:
                        command.linear.x, command.linear.y, command.angular.z = velocity(waypoint)
            pub.publish(command)
            time.sleep(max(0, period - (time.monotonic() - started)))
    finally:
        if rclpy.ok():
            pub.publish(Twist())
        if ws is not None:
            ws.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
