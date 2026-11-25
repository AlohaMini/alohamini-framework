"""M2b isolated perception processes. Every output is data for the dry-run controller."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path as FilePath
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid,Path
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool,String

from alohamini.cerebellum.local_grid import RollingLocalGrid, aligned_points, ground_ray_distance
from alohamini.cerebellum.rgb_navigation import CAMERAS, HSVContact, LiftHistory, PoseHistory, Rectifier, TargetMemory
from alohamini.cerebellum.state import Pose,load_config
from lightnav_vln import camera_topic,exchange

CTRL="/alohamini_cerebellum_dry_run/"
SENSOR=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)


def stamp(msg):
    return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9


def fresh(t,ttl):
    return 0<=time.time()-t<ttl


class CameraNode(Node):
    def __init__(self,name,args):
        super().__init__(name)
        self.frames={}
        self.last_stamp={}
        self.args=args
        self.config=load_config(args.config)
        self.worker=ThreadPoolExecutor(max_workers=1)
        self.pending=None
        self.frame_context=None

    def subscribe_camera(self,cam):
        def receive(msg):
            ts=stamp(msg)
            if fresh(ts,.45) and ts>self.last_stamp.get(cam,0):
                self.frames[cam]=msg
                self.last_stamp[cam]=ts
        self.create_subscription(CompressedImage,camera_topic(cam),receive,SENSOR)

    def destroy_node(self):
        self.worker.shutdown(wait=True,cancel_futures=True)
        super().destroy_node()


class SpatialNode(CameraNode):
    def __init__(self,name,args):
        super().__init__(name,args)
        self.cameras={cam:Rectifier(args.calibration,cam) for cam in CAMERAS}
        self.poses=PoseHistory()
        self.lift=LiftHistory()
        self.wheel=0.
        self.create_subscription(String,CTRL+"capture_pose",self.pose,SENSOR)
        self.create_subscription(TwistStamped,"/alohamini/base_velocity",self.on_wheel,SENSOR)
        self.create_subscription(JointState,"/joint_states",self.joints,SENSOR)
        for cam in CAMERAS:self.subscribe_camera(cam)

    def on_wheel(self,msg):
        if msg.header.frame_id=="base_link" and fresh(stamp(msg),.20):
            self.wheel=stamp(msg)

    def pose(self,msg):
        d=json.loads(msg.data);ts=d["stamp_ns"]*1e-9
        if not fresh(ts,.25):return
        if d["epoch"]!=self.poses.epoch:
            self.frames.clear()
            self.reanchor()
        self.poses.push(ts,Pose(d["x"],d["y"],d["yaw"]),d["epoch"])

    def joints(self,msg):
        t=stamp(msg)
        if not fresh(t,.25):
            return
        before = self.lift.baseline
        self.lift.push_joint_state(t, msg.name, msg.position)
        if before is None and self.lift.baseline is not None:
            self.get_logger().info(f"startup static lift_axis_m_at_capture={self.lift.baseline:.6f} m")

    def geometry(self,cam,t):
        return self.poses.at(t), (0. if cam=="forward" else self.lift.delta(t))


class TargetVoteHistory:
    """Read-only provenance for occupied cells in one odometry epoch."""
    def __init__(self):
        self.previous=set()
        self.owned={}

    def reconcile(self, occupied, target_counts, mixed, minimum, anchor, stamp_ns):
        self.owned={key:proof for key,proof in self.owned.items()
                    if key in occupied and key not in mixed and 0<=stamp_ns-proof[2]<500_000_000}
        if anchor is not None:
            for key,count in target_counts.items():
                if (count>=minimum and key in occupied and key not in mixed
                        and (key not in self.previous or key in self.owned)):
                    self.owned[key]=(anchor.x,anchor.y,stamp_ns)
        self.previous=occupied

    def update(self, grid, mapper, image, depth, camera, delta, fit, pose, detector, stamp_ns, max_range):
        ox,oy=(round(o/grid.resolution) for o in grid.origin)
        occupied={(i+ox,j+oy) for i,j in grid.occupied}
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(detector.hue[0],detector.saturation,detector.value),
                        (detector.hue[1],255,255))
        mask[camera.arrival_excluded]=0
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        anchor=None
        label=0
        if count>1:
            label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            x,y,w,h,area=stats[label]
            if area>=detector.min_area and x>0 and y>0 and x+w<image.shape[1] and y+h<image.shape[0]:
                bottom=int(y+h-1)
                _,xs=np.nonzero(labels[max(0,bottom-2):bottom+1]==label)
                ray=ground_ray_distance(float(np.median(xs)),bottom,camera.K,camera.T,delta)
                if ray and ray[0][0]>0 and ray[1]<=.8:
                    anchor=pose.transform(Pose(*ray[0],0.))
        if anchor is None and not self.owned:
            self.previous=occupied
            return []
        # Use the accepted fit and the very same rectified RGB and depth samples
        # as the occupancy update. A single non-target hit in a cell vetoes it.
        v,u=np.mgrid[:depth.shape[0]:mapper.stride,:depth.shape[1]:mapper.stride]
        valid=np.isfinite(depth[v,u]) & (depth[v,u]>0) & ~camera.excluded[v,u]
        u,v=u[valid],v[valid]
        rel=(np.column_stack((u,v,np.ones_like(u)))@np.linalg.inv(camera.K).T*depth[v,u,None])@camera.T[:3,:3].T
        translation=camera.T[:3,3].copy()
        translation[2]+=delta
        base=aligned_points(rel,translation,fit)
        selected=(base[:,2]>mapper.obstacle_min_z_m)&(base[:,2]<=1.25)
        if max_range is not None:
            selected &= np.hypot(base[:,0],base[:,1])<=max_range
        c,s=math.cos(pose.yaw),math.sin(pose.yaw)
        xy=base[selected,:2]@np.array([[c,s],[-s,c]])+[pose.x,pose.y]
        ij=np.floor(xy/grid.resolution).astype(np.int64)
        is_target=(labels[v[selected],u[selected]]==label) & (anchor is not None)
        keys,counts=np.unique(ij[is_target],axis=0,return_counts=True)
        votes={tuple(k):n for k,n in zip(keys,counts)}
        mixed=set(map(tuple,np.unique(ij[~is_target],axis=0)))
        self.reconcile(occupied,votes,mixed,mapper.min_occupied_points,anchor,stamp_ns)
        return [[int(i-ox),int(j-oy),x,y,t] for (i,j),(x,y,t) in self.owned.items()]


class RGBGrid(SpatialNode):
    def __init__(self,args):
        super().__init__("rgb_grid_node",args)
        grid_config=self.config.get("rgb_grid",{})
        self.chest_max_range_m = grid_config.get("chest_max_range_m")
        self.mapper=RollingLocalGrid(stride=4,min_occupied_points=grid_config.get("min_occupied_points",3),
                                     obstacle_min_z_m=grid_config.get("obstacle_min_z_m",.045))
        self.grid=self.create_publisher(OccupancyGrid,CTRL+"local_grid",1)
        self.witness_pub=self.create_publisher(String,CTRL+"target_cell_witnesses",10)
        self.target_id=os.environ.get("M3_TARGET_ID",args.target_id)
        self.detector=HSVContact(os.environ.get("M3_COLOR",args.color),args.hue,args.min_area,
                                 args.min_saturation,args.min_value)
        self.witnesses=TargetVoteHistory()
        self.last_chest=0.
        self.last_dispatched_stamp=0.
        self.accepted={cam:0 for cam in CAMERAS}
        self.stats_at=time.monotonic()
        self.timer=self.create_timer(.005,self.tick)

    def reanchor(self):
        self.mapper.reanchor()
        self.witnesses=TargetVoteHistory()
        self.last_dispatched_stamp=0.

    def depth(self,cam,data):
        image=self.cameras[cam].decode(data)
        req=urllib.request.Request(f"http://127.0.0.1:{self.args.depth_port}/rectified/{cam}",
                                   data=image.tobytes(),method="POST")
        with urllib.request.urlopen(req,timeout=2) as r:
            return image,np.frombuffer(r.read(640*480*4+1),"<f4").reshape(480,640)

    def tick(self):
        if self.pending is not None:
            if not self.pending.done():return
            cam,msg,delta,epoch=self.frame_context
            completed=self.pending
            self.pending=None
            self.dispatch()
            try:
                image,depth=completed.result();t=stamp(msg);pose=self.poses.at(t)
                if fresh(t,self.config["control"]["map_ttl_s"]) and fresh(self.wheel,.20) and pose is not None and epoch==self.poses.epoch:
                    c=self.cameras[cam]
                    result=self.mapper.update(depth,c.K,c.T,stamp=t,pose_odom=pose,self_mask=c.excluded,
                                              lift_height_m=delta,calibration=c.source,
                                              max_range_m=self.chest_max_range_m if cam == "chest" else None)
                    if result.accepted:
                        grid=self.mapper.export(t)
                        if grid.source=="measured_metric":
                            out=OccupancyGrid();out.header.stamp=msg.header.stamp;out.header.frame_id="odom"
                            out.info.resolution=grid.resolution;out.info.width=out.info.height=self.mapper.width
                            out.info.origin.position.x,out.info.origin.position.y=grid.origin
                            out.info.origin.orientation.w=1.
                            cells=np.full((self.mapper.width,self.mapper.width),-1,np.int8)
                            for (x,y),v in grid.cells.items():cells[y,x]=int(v)
                            from array import array
                            out.data=array("b",cells.tobytes())
                            capture_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
                            try:
                                rows=self.witnesses.update(grid,self.mapper,image,depth,c,delta,result,pose,
                                    self.detector,capture_ns,self.chest_max_range_m if cam=="chest" else None)
                                packet=dict(epoch=epoch,stamp_ns=capture_ns,frame="odom",target_id=self.target_id,
                                            origin=grid.origin,resolution=out.info.resolution,target_only_cells=rows)
                                self.witness_pub.publish(String(data=json.dumps(packet,allow_nan=False)))
                            except (ValueError,ArithmeticError,RuntimeError,cv2.error) as exc:
                                self.witnesses.owned.clear()
                                self.witnesses.previous={(i+round(grid.origin[0]/grid.resolution),
                                                          j+round(grid.origin[1]/grid.resolution)) for i,j in grid.occupied}
                                self.get_logger().warning(f"Target attribution unavailable: {exc}",throttle_duration_sec=2.)
                            finally:
                                self.grid.publish(out)
                            self.accepted[cam]+=1
                            if time.monotonic()-self.stats_at>=3:
                                self.get_logger().info(f"map updates={self.accepted}, lift baseline={self.lift.baseline}")
                                self.stats_at=time.monotonic()
                    else:self.get_logger().warning(cam+" "+result.reason,throttle_duration_sec=2.)
            except (OSError,ValueError,RuntimeError) as e:self.get_logger().warning(str(e),throttle_duration_sec=2.)
        if self.pending is None:self.dispatch()

    def dispatch(self):
        if not fresh(self.wheel,.20) or not self.poses.samples:return
        order=("chest","forward") if time.monotonic()-self.last_chest>.24 else CAMERAS
        for cam in order:
            msg=self.frames.get(cam)
            if msg is None:continue
            t=stamp(msg)
            if not fresh(t,.40) or t<=self.last_dispatched_stamp:
                self.frames.pop(cam);continue
            delta=0. if cam=="forward" else self.lift.delta(t)
            if delta is None:continue
            self.frames.pop(cam)
            if cam=="chest":self.last_chest=time.monotonic()
            self.last_dispatched_stamp=t
            self.frame_context=cam,msg,delta,self.poses.epoch
            self.pending=self.worker.submit(self.depth,cam,bytes(msg.data));break


class ArrivalNode(SpatialNode):
    def __init__(self,args):
        super().__init__("rgb_arrival_node",args)
        self.locator=HSVContact(args.color,args.hue,args.min_area,args.min_saturation,args.min_value)
        self.memory=TargetMemory(args.target_id,self.config["footprint_m"],confirm_pair=True)
        self.awaiting_pose={}
        self.pub=self.create_publisher(String,CTRL+"arrival",1)
        self.timer=self.create_timer(.05,self.tick)

    def reanchor(self):
        self.memory.reset()
        self.awaiting_pose.clear()

    def tick(self):
        if not fresh(self.wheel,.2) or not self.poses.samples:return
        # Keep one exposure per camera until wheel odometry brackets its capture.
        for cam in CAMERAS:
            if cam not in self.awaiting_pose and cam in self.frames:
                self.awaiting_pose[cam]=self.frames.pop(cam)
        for cam,msg in sorted(list(self.awaiting_pose.items()),key=lambda pair:stamp(pair[1])):
            t=stamp(msg)
            if not fresh(t,.4):
                self.awaiting_pose.pop(cam);continue
            pose,delta=self.geometry(cam,t)
            if pose is None or delta is None:continue
            self.awaiting_pose.pop(cam)
            c=self.cameras[cam]
            try:self.memory.observe(t,pose,self.locator(c.decode(bytes(msg.data)),c.arrival_excluded),c,delta)
            except (ValueError,cv2.error) as e:self.get_logger().warning(str(e),throttle_duration_sec=2.)
        ts,pose=self.poses.samples[-1]
        if not fresh(ts,.20):return
        payload=self.memory.message(ts,pose)
        if payload:self.pub.publish(String(data=json.dumps(payload)))


class Brain(CameraNode):
    def __init__(self,args):
        super().__init__("lightnav_reference_bridge",args)
        import websocket
        self.ws=websocket.create_connection(f"ws://127.0.0.1:{args.lightnav_port}",timeout=4,http_no_proxy=["127.0.0.1"])
        exchange(self.ws,"login",{"clientId":"m2b-front-dry-run"})
        exchange(self.ws,"reset",{})
        self.seq=0
        self.ref=self.create_publisher(Path,CTRL+"reference",1)
        self.stop=self.create_publisher(Bool,CTRL+"stop",10)
        self.subscribe_camera("forward")
        self.timer=self.create_timer(.025,self.tick)
        self.dropped=0
        self.started_at=0.

    def predict(self,seq,data):
        return exchange(self.ws,"next",dict(seq=seq,instruction=self.args.instruction,image=base64.b64encode(data).decode()))

    def tick(self):
        if self.pending is not None:
            if not self.pending.done():return
            reply=self.pending.result();self.pending=None
            msg=self.frame_context
            elapsed=time.monotonic()-self.started_at
            valid=(fresh(stamp(msg),self.config["control"]["reference_ttl_s"]) and
                   type(reply.get("seq")) is int and reply["seq"]==self.seq and type(reply.get("stop")) is bool)
            if not valid:
                self.dropped+=1
            self.get_logger().info(f"seq={self.seq} inference_s={elapsed:.4f} capture_age_s={time.time()-stamp(msg):.4f} dropped={self.dropped} accepted={valid}")
            if valid:
                if reply["stop"]:self.stop.publish(Bool(data=True))
                else:
                    rows=np.asarray(reply["actions"]["actions"],float)
                    if rows.shape!=(10,3) or not np.isfinite(rows).all():raise ValueError("invalid LightNav path")
                    path=Path();path.header.stamp=msg.header.stamp;path.header.frame_id="base_link"
                    for x,y,yaw in rows:
                        p=PoseStamped();p.header=path.header;p.pose.position.x=x;p.pose.position.y=y
                        p.pose.orientation.z=math.sin(yaw/2);p.pose.orientation.w=math.cos(yaw/2);path.poses.append(p)
                    self.ref.publish(path)
        msg=self.frames.pop("forward",None)
        if msg is not None and fresh(stamp(msg),.40):
            self.started_at=time.monotonic();self.seq+=1;self.frame_context=msg
            self.pending=self.worker.submit(self.predict,self.seq,bytes(msg.data))

    def destroy_node(self):
        super().destroy_node();self.ws.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("node",choices=["grid","brain","arrival"])
    p.add_argument("--calibration",type=FilePath,default=FilePath.home()/"Alohamini_workspace/calibration/cameras")
    p.add_argument("--config",type=FilePath)
    p.add_argument("--depth-port",type=int,default=18642)
    p.add_argument("--lightnav-port",type=int,default=18643)
    p.add_argument("--color",choices=HSVContact.ranges,default="purple")
    p.add_argument("--hue",nargs=2,type=int,metavar=("LOW","HIGH"))
    p.add_argument("--min-area",type=int,default=24)
    p.add_argument("--min-saturation",type=int,default=65)
    p.add_argument("--min-value",type=int,default=40)
    p.add_argument("--target-id",default="purple_toy")
    p.add_argument("--instruction",default="Approach the purple toy and stop in front of it.")
    a=p.parse_args()
    cv2.setNumThreads(1);rclpy.init(args=[])
    node={"grid":RGBGrid,"arrival":ArrivalNode,"brain":Brain}[a.node](a)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=="__main__":
    main()
