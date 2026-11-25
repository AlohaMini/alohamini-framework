"""Exercise the deployed Linux monitor and rosbag against an inert SetBool service."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform.startswith("linux") and os.getenv("CEREBELLUM_ROS_SMOKE")=="1",
                     "requires the Linux supervisor, rosbag2 and an isolated ROS domain")
class FreeDriveSmoke(unittest.TestCase):
    def test_free_stop_chair_gate_and_interrupt(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from geometry_msgs.msg import Twist, TwistStamped
        from nav_msgs.msg import OccupancyGrid, Path as Route
        from std_msgs.msg import String, Bool
        from sensor_msgs.msg import CompressedImage, JointState
        from std_srvs.srv import SetBool

        self.assertEqual(os.getenv("ROS_LOCALHOST_ONLY"),"1")
        self.assertNotIn(os.getenv("ROS_DOMAIN_ID"),(None,"","0"))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).resolve().parents[1]
            rclpy.init()
            ex=SingleThreadedExecutor()
            nodes=[]
            def node(name):
                n=rclpy.create_node(name);nodes.append(n);ex.add_node(n);return n
            fixture=node("free_monitor_fixture")
            def cleanup():
                ex.shutdown()
                for n in nodes:n.destroy_node()
                rclpy.try_shutdown()
            self.addCleanup(cleanup)
            time.sleep(.4)
            self.assertEqual(fixture.count_publishers("/cmd_vel"),0)
            self.assertNotIn("/alohamini_lerobot_bridge/command_enable",
                             dict(fixture.get_service_names_and_types()))
            ctrl=node("alohamini_cerebellum_dry_run")
            brain=node("lightnav_reference_bridge")
            grid_node=node("rgb_grid_node")
            ns="/alohamini_cerebellum_dry_run/"
            velocity=ctrl.create_publisher(Twist,"/cmd_vel",10)
            brain.create_publisher(Route,ns+"reference",1)
            brain.create_publisher(Bool,ns+"stop",10)
            dec=ctrl.create_publisher(String,ns+"decision",10)
            pose=ctrl.create_publisher(String,ns+"capture_pose",qos_profile_sensor_data)
            wheels=fixture.create_publisher(TwistStamped,"/alohamini/base_velocity",qos_profile_sensor_data)
            joints=fixture.create_publisher(JointState,"/joint_states",qos_profile_sensor_data)
            diag=fixture.create_publisher(DiagnosticArray,"/diagnostics",5)
            cameras=[fixture.create_publisher(CompressedImage,f"/alohamini/cameras/{c}/image_raw/compressed",
                                               qos_profile_sensor_data) for c in ("forward","chest")]
            grid=grid_node.create_publisher(OccupancyGrid,ns+"local_grid",1)
            witness=grid_node.create_publisher(String,ns+"target_cell_witnesses",10)
            calls=[]; trips=[]
            fixture.create_subscription(Bool,ns+"estop",lambda m:trips.append(m.data),10)
            def service(req,res):
                calls.append(req.data);res.success=True;return res
            fixture.create_service(SetBool,"/alohamini_lerobot_bridge/command_enable",service)
            def run(case):
                calls.clear();trips.clear()
                out=Path(tmp)/case
                env=dict(os.environ,M3_CONFIG=str(root/"evidence/m3-low-speed-shared.json"),
                         M3_TARGET_ID="purple_toy" if case=="chair" else "")
                cmd=[sys.executable,str(root/"examples/navigation/m3_trial_monitor.py"),str(out),
                     "--dry-run" if case=="dry" else "--operator-confirmed"]
                if case=="chair":cmd.append("--chair")
                proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                begin=time.monotonic(); enabled_at=None; sent=False
                try:
                    while proc.poll() is None and time.monotonic()-begin<18:
                        now=time.monotonic()
                        if True in calls and enabled_at is None:enabled_at=now
                        elapsed=0 if enabled_at is None else now-enabled_at
                        if case=="interrupt" and elapsed>.35 and not sent:
                            proc.send_signal(signal.SIGINT);sent=True
                        stamp=fixture.get_clock().now().to_msg()
                        ns_time=stamp.sec*10**9+stamp.nanosec
                        w=TwistStamped();w.header.stamp=stamp;w.header.frame_id="base_link";wheels.publish(w)
                        j=JointState();j.header.stamp=stamp;j.name=["vertical_move"];j.position=[-.30];joints.publish(j)
                        cam=CompressedImage();cam.header.stamp=stamp;cam.data=[1]
                        for pub in cameras:pub.publish(cam)
                        m=OccupancyGrid();m.header.stamp=stamp;m.header.frame_id="odom"
                        m.info.width=m.info.height=16;m.info.resolution=.25;m.info.origin.orientation.w=1.
                        cells=[0]*256
                        if case=="free" and .05<elapsed<.45:
                            cells[8*16]=100  # underneath the synthetic robot at (0,2)
                        m.data=cells;grid.publish(m)
                        witness.publish(String(data=json.dumps(dict(target_id="purple_toy",frame="odom",epoch=1,
                            stamp_ns=ns_time,origin=[0.,0.],resolution=.25,target_only_cells=[]))))
                        d=DiagnosticStatus();d.name="AlohaMini Host state bridge"
                        d.values=[KeyValue(key="command_enabled",value=str(bool(calls and calls[-1])).lower()),
                                  KeyValue(key="lift_command_ready",value="true")]
                        da=DiagnosticArray();da.header.stamp=stamp;da.status=[d];diag.publish(da)
                        pose.publish(String(data=json.dumps(dict(x=0.,y=2. if elapsed else 0.,yaw=0.,stamp_ns=ns_time,epoch=1))))
                        done=(case=="free" and elapsed>.6) or (case=="dry" and now-begin>5.5)
                        hold=case=="free" and .05<elapsed<.45
                        dec.publish(String(data=json.dumps(dict(state="REOBSERVE" if done else "SAFETY_HOLD" if hold else "TRACKING",
                            reason="arrival_evidence_missing_or_invalid" if done else "best_score",
                            stop_source="model_stop" if done else None,map_source="measured_metric",clock_fault=False,
                            estop_latched=False,planned_velocity={"y_m_s":0.}))))
                        if case!="dry":velocity.publish(Twist())  # All fixture commands are zero.
                        for _ in range(4):ex.spin_once(timeout_sec=.004)
                        time.sleep(.008)
                    self.assertIsNotNone(proc.poll(),case)
                finally:
                    if proc.poll() is None:proc.terminate()
                    output=proc.communicate(timeout=15)[0]
                self.assertTrue((out/"result.json").exists(),output)
                result=json.loads((out/"result.json").read_text())
                if case=="chair":
                    self.assertEqual(calls,[],output)
                    self.assertFalse(result["preflight_passed"])
                elif case=="dry":
                    self.assertEqual(proc.returncode,0,output)
                    self.assertEqual(calls,[])
                    self.assertEqual(result["outcome"],"model_stop")
                else:
                    self.assertEqual(calls,[True,False],output)
                    if case=="free":
                        self.assertEqual(proc.returncode,0,output)
                        self.assertEqual(result["outcome"],"model_stop")
                    else:
                        self.assertIn(True,trips,output)
                        self.assertNotEqual(proc.returncode,0)
            for case in ("chair","free","interrupt","dry"):
                if case=="dry":ctrl.destroy_publisher(velocity)
                with self.subTest(case=case):run(case)
