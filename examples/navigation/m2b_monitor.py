"""Record decisions, actual map frequency, sensor provenance and the scoped ROS graph."""
import argparse
import base64
import zlib
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid,Path as RosPath
from sensor_msgs.msg import CompressedImage,JointState
from std_msgs.msg import String,Bool
from lightnav_vln import camera_topic

CTRL="/alohamini_cerebellum_dry_run/"
def main():
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--seconds",type=float,default=22.)
    p.add_argument("--test-false-stop",action="store_true",help="Inject two labelled test STOP messages at 5 seconds")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=[]);n=rclpy.create_node("m2b_acceptance_monitor")
    counts=Counter();grid_times=[];ticks=[];gpus=[];graph=[];last_grid=None;last_grid_info=None;last_imgs={};q=QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT)
    log=(a.out/"events.jsonl").open("w",buffering=1)
    def record(topic,data):
        counts[topic]+=1
        log.write(json.dumps(dict(topic=topic,receipt=time.time(),data=data))+"\n")
    def dec(msg):
        d=json.loads(msg.data);ticks.append(d);record("decision",d)
    def grid(msg):
        nonlocal last_grid,last_grid_info
        grid_times.append(time.monotonic())
        last_grid=np.asarray(msg.data,dtype=np.int8).reshape(msg.info.height,msg.info.width).copy()
        last_grid_info=dict(sec=msg.header.stamp.sec,ns=msg.header.stamp.nanosec,occupied=int((last_grid==100).sum()),
                            origin=[msg.info.origin.position.x,msg.info.origin.position.y],resolution=msg.info.resolution)
        record("grid",{**last_grid_info,"cells_zlib_b64":base64.b64encode(zlib.compress(last_grid.tobytes(),1)).decode()})
    def img(cam,msg):
        counts[cam]+=1;last_imgs[cam]=bytes(msg.data)
    n.create_subscription(String,CTRL+"decision",dec,10);n.create_subscription(OccupancyGrid,CTRL+"local_grid",grid,5)
    n.create_subscription(String,CTRL+"arrival",lambda m:record("arrival",json.loads(m.data)),5)
    n.create_subscription(RosPath,CTRL+"reference",lambda m:record("reference",dict(poses=len(m.poses),sec=m.header.stamp.sec,ns=m.header.stamp.nanosec,xy=[[p.pose.position.x,p.pose.position.y] for p in m.poses])),5)
    n.create_subscription(Bool,CTRL+"stop",lambda m:record("stop",m.data),10)
    n.create_subscription(TwistStamped,"/alohamini/base_velocity",lambda m:counts.update(["wheel"]),q)
    def joint(m):
        if "lift_axis" in m.name:
            record("joint_states",dict(lift_axis_m=m.position[m.name.index("lift_axis")],sec=m.header.stamp.sec,ns=m.header.stamp.nanosec))
    n.create_subscription(JointState,"/joint_states",joint,q)
    for cam in ("forward","chest"):
        n.create_subscription(CompressedImage,camera_topic(cam),lambda m,c=cam:img(c,m),q)
    test_pub = n.create_publisher(Bool,CTRL+"stop",10) if a.test_false_stop else None
    sent=0
    start=time.monotonic();probe=start+1.
    try:
        while time.monotonic()-start<a.seconds:
            rclpy.spin_once(n,timeout_sec=.025)
            if test_pub and sent<2 and time.monotonic()-start>=5.+.04*sent:
                test_pub.publish(Bool(data=True));sent+=1
                record("test_injected_stop",sent)
            if time.monotonic()>=probe:
                graph.append(n.count_publishers("/cmd_vel"))
                if graph[-1]:raise RuntimeError("unexpected command publisher in dry-run domain")
                gpus.append(int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).splitlines()[0]))
                for cam,b in last_imgs.items():(a.out/(cam+".jpg")).write_bytes(b)
                if last_grid is not None:
                    np.savez_compressed(a.out/"grid.npz",cells=last_grid,**last_grid_info)
                    vis=np.full((*last_grid.shape,3),100,np.uint8);vis[last_grid==0]=[72,155,72];vis[last_grid==100]=[25,35,235]
                    cv2.imwrite(str(a.out/"grid.png"),cv2.resize(np.flipud(vis),(480,480),interpolation=cv2.INTER_NEAREST))
                probe+=2.
        duration=time.monotonic()-start
        metrics=dict(duration_s=duration,counts=dict(counts),grid_hz=len(grid_times)/duration,interval_grid_hz=(len(grid_times)-1)/(grid_times[-1]-grid_times[0]) if len(grid_times)>1 else None,
                     cmd_vel_publisher_counts=graph,gpu_total_used_mib_samples=gpus,
                     decisions=dict(Counter(d["state"]+"/"+d["reason"] for d in ticks)))
        if ticks:
            metrics.update(missed_periods=sum(d["timing"]["missed_periods"] for d in ticks),
                           expired_during_control_tick=sum(d["reason"]=="expired_during_control_tick" for d in ticks),
                           tick_compute_s=dict(zip(("P50","P99","max"),np.percentile([d["tick_compute_s"] for d in ticks],[50,99,100]))))
        (a.out/"summary.json").write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
    finally:log.close();n.destroy_node();rclpy.try_shutdown()


if __name__=="__main__":main()
