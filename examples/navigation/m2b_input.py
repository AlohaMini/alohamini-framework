"""Read-only sensor ingress into an isolated ROS domain. Replay mode explicitly uses stationary fixture twist."""
import argparse
import json
from pathlib import Path
import os
import queue
import struct
import subprocess
import threading
import time

import cv2
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import CompressedImage,JointState

from lightnav_vln import camera_topic

SOURCES=("wheels","forward","chest")
def ros_stamp(header,ns):
    header.stamp.sec,header.stamp.nanosec=divmod(ns,10**9)


class Ingress(Node):
    def __init__(self,args):
        super().__init__("m2b_readonly_ingress")
        self.processes=[];self.q={s:queue.Queue(maxsize=1) for s in SOURCES}
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.img={s:self.create_publisher(CompressedImage,camera_topic(s),qos) for s in SOURCES[1:]}
        self.w=self.create_publisher(TwistStamped,"/alohamini/base_velocity",qos)
        self.offset=None;self.checked=0
        self.known_height=args.verified_lift_height_m
        self.j=self.create_publisher(JointState,"/joint_states",qos)
        if args.fixture:
            frames={cam:cv2.imencode(".jpg",cv2.imread(str(args.fixture/(cam+".png"))))[1].tobytes() for cam in self.img}
            self.create_timer(.04,lambda:self.publish("wheels",time.time_ns(),b"[0,0,0]"))
            def images():
                t=time.time_ns()
                for cam in self.img:self.publish(cam,t,frames[cam])
                j=JointState();ros_stamp(j.header,t);j.name=["lift_axis"];j.position=[0.];self.j.publish(j)
            self.create_timer(.072,images)
        else:
            if not args.ssh_control:raise ValueError("a pinned SSH master is required for live input")
            self.ssh=["ssh","-T","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes",
                      "-o","HostKeyAlias=192.168.2.2","-S",args.ssh_control,args.pi]
            self.source=args.pi_script
            self.clock_proc=self.open("clock")
            self.sync()
            self.synchronizing=threading.Lock()
            self.create_timer(8.,self.resync)
            for s in SOURCES:
                p=self.open(s)
                threading.Thread(target=self.read,args=(s,p),daemon=True).start()
            self.create_timer(.003,self.poll)

    def open(self,source):
        import shlex
        env={k:v for k,v in os.environ.items() if k!="LD_LIBRARY_PATH"}
        if source=="wheels" and self.known_height is not None:
            source += " --verified-lift-height-m "+str(self.known_height)
        p=subprocess.Popen(self.ssh+["~/miniconda3/envs/alohamini_host/bin/python -u "+shlex.quote(self.source)+" "+source],
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
        self.processes.append(p);return p

    def sync(self):
        import select
        best=None
        for _ in range(6):
            start=time.time_ns();self.clock_proc.stdin.write(b"t\n");self.clock_proc.stdin.flush()
            if not select.select([self.clock_proc.stdout],[],[],2.)[0]:raise RuntimeError("clock link timeout")
            pi=int(self.clock_proc.stdout.readline());end=time.time_ns();rtt=end-start
            if best is None or rtt<best[0]:best=rtt,(start+end)//2-pi
        self.offset=best[1] if best[0]<30_000_000 else None
        self.checked=time.monotonic()
        self.get_logger().info(f"Pi to 5090 time offset ns={self.offset}, best RTT ns={best[0]}")

    def resync(self):
        if not self.synchronizing.acquire(False):return
        def refresh():
            try:self.sync()
            except (OSError,ValueError,RuntimeError) as exc:
                self.offset=None
                self.get_logger().warning("clock synchronization failed: "+str(exc))
            finally:self.synchronizing.release()
        threading.Thread(target=refresh,daemon=True).start()

    def read(self,s,p):
        while True:
            hdr=p.stdout.read(12)
            if len(hdr)!=12:return
            ts,n=struct.unpack("!QI",hdr)
            if n>1_000_000:return
            data=p.stdout.read(n)
            if len(data)!=n:return
            q=self.q[s]
            try:q.get_nowait()
            except queue.Empty:pass
            q.put_nowait((ts,data))

    def poll(self):
        if any(p.poll() is not None for p in self.processes):raise RuntimeError("Pi sensor subprocess exited")
        if self.offset is None or time.monotonic()-self.checked>15:return
        for s,q in self.q.items():
            try:ts,data=q.get_nowait()
            except queue.Empty:continue
            t=ts+self.offset
            if 0<=time.time_ns()-t<450_000_000:self.publish(s,t,data)

    def publish(self,s,ts,data):
        if s=="wheels":
            m=TwistStamped();ros_stamp(m.header,ts);m.header.frame_id="base_link"
            sample=json.loads(data)
            m.twist.linear.x,m.twist.linear.y,m.twist.angular.z=map(float,sample[:3]);self.w.publish(m)
            if len(sample)==4 and sample[3] is not None:
                j=JointState();ros_stamp(j.header,ts);j.name=["lift_axis"];j.position=[float(sample[3])];self.j.publish(j)
        else:
            m=CompressedImage();ros_stamp(m.header,ts);m.header.frame_id=s+"_camera_optical"
            m.format="jpeg";m.data=data;self.img[s].publish(m)

    def destroy_node(self):
        for p in self.processes:p.terminate()
        for p in self.processes:p.wait(timeout=5)
        super().destroy_node()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--fixture",type=Path)
    p.add_argument("--ssh-control");p.add_argument("--pi",default="pi5@192.168.2.2");p.add_argument("--pi-script")
    p.add_argument("--verified-lift-height-m",type=float)
    args=p.parse_args()
    rclpy.init(args=[]);n=Ingress(args)
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    finally:n.destroy_node();rclpy.try_shutdown()
