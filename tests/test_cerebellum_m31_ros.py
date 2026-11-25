"""Exercise the real actuator adapter on loopback, remapped to a test-only topic."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(os.getenv("CEREBELLUM_ROS_SMOKE")=="1" and importlib.util.find_spec("rclpy"),
                     "requires isolated ROS 2")
class M3RosSmoke(unittest.TestCase):
    def test_motion_hold_estop_and_shutdown_on_mock_topic(self):
        import rclpy
        from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Path as RosPath
        from std_msgs.msg import Bool
        self.assertEqual(os.environ.get("ROS_LOCALHOST_ONLY"),"1")
        self.assertNotIn(os.environ.get("ROS_DOMAIN_ID"),(None,"","0"))
        ns="/m31_test_controller"
        sink="/m31_test_sink/velocity"
        profile=Path(__file__).resolve().parents[1]/"evidence/m3-low-speed.json"
        cmd=[sys.executable,"-m","alohamini.cerebellum.ros_adapter","--ros-args",
             "-r","__node:=m31_test_controller","-r","/cmd_vel:="+sink,
             "-p","dry_run:=false","-p","enable_actuators:=true",
             "-p","grid_source:=measured_metric","-p","config:="+str(profile)]
        rclpy.init()
        n=rclpy.create_node("m31_test_inputs")
        writes=[]
        n.create_subscription(Twist,sink,lambda msg:writes.append((msg.linear.x,msg.linear.y,msg.angular.z)),10)
        wheel=n.create_publisher(TwistStamped,"/alohamini/base_velocity",10)
        path=n.create_publisher(RosPath,ns+"/reference",1)
        maps=n.create_publisher(OccupancyGrid,ns+"/local_grid",1)
        stop=n.create_publisher(Bool,ns+"/estop",10)
        def drive(duration, send_input=True, occupied=False, done=lambda:False):
            end=time.monotonic()+duration
            while time.monotonic()<end and not done():
                if send_input:
                    w=TwistStamped();w.header.frame_id="base_link";w.header.stamp=n.get_clock().now().to_msg()
                    wheel.publish(w)
                    point=PoseStamped();point.header=w.header
                    point.pose.orientation.w=1.;point.pose.position.x=1.2
                    p=RosPath();p.header=w.header;p.poses=[point];path.publish(p)
                    m=OccupancyGrid();m.header.frame_id="odom";m.header.stamp=w.header.stamp
                    m.info.resolution=.1;m.info.width=m.info.height=30
                    m.info.origin.position.x=m.info.origin.position.y=-1.5
                    m.info.origin.orientation.w=1.;m.data=[100 if occupied else 0]*900
                    maps.publish(m)
                rclpy.spin_once(n,timeout_sec=.012)
                time.sleep(.006)
        flags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=="win32" else 0
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=flags)
        zero=(0.,0.,0.)
        try:
            drive(8.,done=lambda:any(v[0]>0 for v in writes))
            self.assertTrue(any(v[0]>0 for v in writes),str(proc.poll()))
            self.assertEqual(n.count_publishers("/cmd_vel"),0)
            drive(.35,occupied=True)
            self.assertEqual(writes[-3:],[zero]*3)
            writes.clear()
            drive(2.,done=lambda:any(v[0]>0 for v in writes))
            self.assertTrue(any(v[0]>0 for v in writes))
            stop.publish(Bool(data=True));drive(.25)
            stop.publish(Bool(data=False));writes.clear();drive(.3)
            self.assertGreaterEqual(len(writes),3)
            self.assertTrue(all(v==zero for v in writes))
            writes.clear()
            if sys.platform=="win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGTERM)
            drive(1.,send_input=False)
            self.assertGreaterEqual(len(writes),3)
            self.assertTrue(all(v==zero for v in writes))
            self.assertEqual(proc.wait(timeout=3),0)
        finally:
            if proc.poll() is None:
                proc.terminate()
            try: output,_=proc.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill();output,_=proc.communicate()
            n.destroy_node();rclpy.try_shutdown()
            if output: print(output)

