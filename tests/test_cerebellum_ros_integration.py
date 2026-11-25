"""Use with ROS_LOCALHOST_ONLY=1 and a dedicated ROS_DOMAIN_ID."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(os.getenv("CEREBELLUM_ROS_SMOKE") == "1" and importlib.util.find_spec("rclpy"),
                     "requires ROS 2 and CEREBELLUM_ROS_SMOKE=1 in an isolated ROS domain")
class RosSmoke(unittest.TestCase):
    def test_synthetic_feedback_path_stop_and_only_diagnostics_output(self):
        self.exercise_stop(1)

    def test_two_stop_frames_with_intervening_reference_reset(self):
        self.exercise_stop(2)

    def exercise_stop(self, confirmations):
        import rclpy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from nav_msgs.msg import OccupancyGrid, Path
        from std_msgs.msg import Bool, String

        self.assertEqual(os.getenv("ROS_LOCALHOST_ONLY"), "1")
        self.assertTrue(os.getenv("ROS_DOMAIN_ID"))
        command = [sys.executable, "-m", "alohamini.cerebellum.ros_adapter"]
        extension = os.getenv("CEREBELLUM_EXTENSION")
        if extension:
            command = [sys.executable, "-c",
                       "import alohamini, os, runpy; alohamini.__path__.insert(0, os.environ['CEREBELLUM_EXTENSION']); "
                       "runpy.run_module('alohamini.cerebellum.ros_adapter', run_name='__main__')"]
        if confirmations != 1:
            command += ["--ros-args", "-p", f"stop_confirm_count:={confirmations}"]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   creationflags=flags)
        rclpy.init()
        node = rclpy.create_node("synthetic_cerebellum_inputs")
        namespace = "/alohamini_cerebellum_dry_run"
        wheels = node.create_publisher(TwistStamped, "/alohamini/base_velocity", 10)
        path = node.create_publisher(Path, namespace + "/reference", 1)
        grid = node.create_publisher(OccupancyGrid, namespace + "/local_grid", 1)
        stop = node.create_publisher(Bool, namespace + "/stop", 1)
        got = []
        node.create_subscription(String, namespace + "/decision", lambda msg: got.append(json.loads(msg.data)), 10)
        try:
            deadline = time.monotonic() + 8.
            requested_stop = False
            stages = 0
            next_action = 0.
            while time.monotonic() < deadline and process.poll() is None:
                stamp = node.get_clock().now().to_msg()
                w = TwistStamped()
                w.header.stamp, w.header.frame_id = stamp, "base_link"
                wheels.publish(w)
                p = Path()
                p.header = w.header
                point = PoseStamped()
                point.header = p.header
                point.pose.position.x = 1.
                point.pose.orientation.w = 1.
                p.poses.append(point)
                if stages == 0 or (stages == 2 and time.monotonic() < next_action - .035):
                    path.publish(p)
                m = OccupancyGrid()
                m.header.stamp, m.header.frame_id = stamp, "odom"
                m.info.resolution, m.info.width, m.info.height = .1, 30, 30
                m.info.origin.position.x = m.info.origin.position.y = -1.5
                m.info.origin.orientation.w = 1.
                m.data = [0] * 900
                grid.publish(m)
                rclpy.spin_once(node, timeout_sec=.025)
                if not requested_stop and got and got[-1]["state"] == "TRACKING" and time.monotonic() >= next_action:
                    if stages in (0, 2, 3):
                        stop.publish(Bool(data=True))
                    stages += 1
                    next_action = time.monotonic() + .10
                    requested_stop = confirmations == 1 or stages == 4
                if not requested_stop and stages:
                    self.assertNotIn(got[-1]["state"], {"STOPPED", "VERIFYING_ARRIVAL", "REOBSERVE"})
                if any(d["state"] == "REOBSERVE" for d in got):
                    break
            self.assertTrue(requested_stop, str(got[-2:]))
            self.assertEqual(got[-1]["state"], "REOBSERVE")
            self.assertTrue(all(d["dry_run"] and d["arrival_status"] != "CONFIRMED" for d in got))
            self.assertTrue(any(d["planned_velocity"]["x_m_s"] > 0 for d in got))
            produced = node.get_publisher_names_and_types_by_node(
                "alohamini_cerebellum_dry_run", "/")
            self.assertIn(namespace + "/decision", dict(produced))
            self.assertTrue(all(set(types).isdisjoint({"geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped"})
                                for _, types in produced))
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
            if process.poll() is None:
                process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
            try:
                log, _ = process.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                log, _ = process.communicate()
            if log:
                print(log)
        self.assertEqual(process.returncode, 0, log)
