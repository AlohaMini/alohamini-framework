"""Capture-time calibration, floor contact and short-horizon target memory."""
from collections import deque
from pathlib import Path
import math

import numpy as np

from .local_grid import CalibrationSource, ground_ray_distance, _valid_geometry
from .state import Pose

CAMERAS = ("forward", "chest")
STATUS = "accepted_ground_plane_20261124"


def depth_canvas(width):
    """Resized H and (top, bottom, left, right) in the wide Metric3D input."""
    if width not in (384,518):
        raise ValueError("expected 384 or 518")
    height=round(width*480/640)
    canvas_h=math.ceil(height/28)*28
    canvas_w=round(canvas_h*16/9/28)*28
    top,left=(canvas_h-height)//2,(canvas_w-width)//2
    return height,(top,canvas_h-height-top,left,canvas_w-width-left)


def mask_depth_padding(depth,width):
    """Invalidate the upsampled padding halo; these pixels supply no map votes."""
    if width==384:
        depth[:,:24]=depth[:,-24:]=np.nan
        depth[:12,:]=depth[-12:,:]=np.nan
    elif width==518:
        depth[:,:12]=depth[:,-12:]=np.nan
    else:
        raise ValueError("unexpected depth input width")


class Rectifier:
    def __init__(self, folder, camera):
        import cv2
        import yaml
        folder = Path(folder).expanduser()
        def load(part, name):
            return yaml.safe_load((folder/part/(name+".yaml")).read_text())
        raw = load("intrinsics", camera)
        intr = load("intrinsics", camera+"_rectified")
        ext = load("extrinsics", camera)
        rect = intr["rectification"]
        self.K = np.array(rect["K_new"], float)
        self.T = np.array(ext["T_base_from_camera_optical"], float)
        self.source = CalibrationSource(intr, ext)
        self.camera = camera
        K0 = np.array(raw["camera_matrix"]["data"], float).reshape(3,3)
        D = np.array(raw["distortion_coefficients"]["data"], float)
        if (camera not in CAMERAS or any(d["status"] != STATUS or d["camera_name"] != camera
                                          for d in (raw,intr,ext))
                or raw["image_geometry"] != "raw" or raw["distortion_model"] != "plumb_bob"
                or not _valid_geometry(self.K,self.T)
                or not np.array_equal(rect["K_original"],K0) or not np.array_equal(rect["distortion_original"],D)
                or not self.source.matches(self.K,self.T,(480,640))):
            raise ValueError("inconsistent signed raw/rectified calibration for "+camera)
        self.maps = cv2.initUndistortRectifyMap(K0,D,None,self.K,(640,480),cv2.CV_32FC1)
        x,y = self.maps
        self.excluded, self.arrival_excluded = self.masks(x, y, camera)

    @staticmethod
    def masks(x, y, camera):
        # x/y index the RAW image at each rectified output pixel.
        edges = (x < 12) | (x >= 628) | (y < 0) | (y >= 480)
        edges[:, :12] = edges[:, -12:] = True
        depth = edges | (y >= (422 if camera == "forward" else 460))
        if camera == "chest":
            return depth, depth.copy()
        left_loop = (x >= 219) & (x <= 251) & (y >= 463)
        right_loop = (x >= 419) & (x <= 457) & (y >= 451)
        return depth, edges | (y > 475) | left_loop | right_loop

    def decode(self, data):
        import cv2
        raw = cv2.imdecode(np.frombuffer(data,np.uint8), cv2.IMREAD_COLOR)
        if raw is None or raw.shape != (480,640,3):
            raise ValueError("expected raw compressed 640x480 camera image")
        return cv2.remap(raw,*self.maps,cv2.INTER_LINEAR)


class LiftHistory:
    """All values are in the URDF prismatic joint coordinate (metres)."""
    def __init__(self):
        self.samples = deque(maxlen=128)
        self.baseline = None

    def push_joint_state(self, stamp, names, positions):
        # The native bridge and read-only ingress use the same URDF metre coordinate.
        matches = [i for i, name in enumerate(names) if name in ("vertical_move", "lift_axis")]
        if len(matches) == 1 and matches[0] < len(positions):
            self.push(stamp, positions[matches[0]])

    def push(self, stamp, value):
        # The Pro lift can settle one encoder count past its homed lower stop.
        lower = -.30
        tick_m = .131 / 4096
        if not math.isfinite(value) or not lower - tick_m <= value <= .30 or (self.samples and stamp<=self.samples[-1][0]):
            return
        value = max(lower, value)
        self.samples.append((stamp,value))
        if self.baseline is None:
            window = [(t,v) for t,v in self.samples if 0 <= stamp-t <= .6]
            values = [v for _,v in window]
            if (len(values)>=3 and stamp-window[0][0]>=.30 and max(values)-min(values)<=.001
                    and all(b[0]-a[0]<=.20 for a,b in zip(window,window[1:]))):
                self.baseline = float(np.median(values))

    def delta(self, stamp):
        if self.baseline is None:
            return None
        for (a,x),(b,y) in zip(self.samples,list(self.samples)[1:]):
            if a<=stamp<=b and b-a<=.20:
                return x+(y-x)*(stamp-a)/(b-a)-self.baseline
        if self.samples and 0<=stamp-self.samples[-1][0]<=.08:
            a,x=self.samples[-1]
            if len(self.samples)>=2 and abs(x-self.samples[-2][1])<=.0005:
                return x-self.baseline
        return None


class PoseHistory:
    """Interpolate authoritative wheel-derived controller poses in one odometry epoch."""
    def __init__(self):
        self.samples=deque(maxlen=256)
        self.epoch=None

    def push(self, stamp, pose, epoch):
        if epoch!=self.epoch:
            self.samples.clear()
            self.epoch=epoch
        if not self.samples or stamp>self.samples[-1][0]:
            self.samples.append((stamp,pose))

    def at(self, stamp):
        for (a,p),(b,q) in zip(self.samples,list(self.samples)[1:]):
            if a<=stamp<=b and b-a<=.25:
                f=(stamp-a)/(b-a)
                return Pose(p.x+f*(q.x-p.x),p.y+f*(q.y-p.y),p.yaw+f*math.remainder(q.yaw-p.yaw,math.tau))
        return None


class HSVContact:
    # The lab lavender block has H=115..124 on both USB cameras under room light.
    ranges={"purple":(115,165),"blue":(95,124),"green":(35,88)}
    def __init__(self,color, hue=None, min_area=24, min_saturation=65, min_value=40):
        self.hue=self.ranges[color] if hue is None else hue
        if not (0<=self.hue[0]<=self.hue[1]<=179 and min_area>=1
                and 0<=min_saturation<=255 and 0<=min_value<=255):
            raise ValueError("invalid HSV contact range or component size")
        self.min_area=min_area
        self.saturation=min_saturation
        self.value=min_value

    def __call__(self,bgr,excluded=None):
        import cv2
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(self.hue[0],self.saturation,self.value),(self.hue[1],255,255))
        if excluded is not None:
            mask[excluded] = 0
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if n<2:return None
        i=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        x,y,w,h,area=stats[i]
        if area<self.min_area or x<=0 or x+w>=mask.shape[1] or y+h>=mask.shape[0]:return None
        bottom=int(y+h-1)
        ys,xs=np.nonzero(labels[max(0,bottom-2):bottom+1]==i)
        return float(np.median(xs)),bottom


class TargetMemory:
    def __init__(self, target, footprint, memory_s=3.5, confirm_pair=False):
        self.target=target
        self.front=max(x for x,y in footprint)
        self.half_width=max(abs(y) for x,y in footprint)
        self.memory=memory_s
        self.confirm_pair=confirm_pair
        self.reset()

    def reset(self):
        self.last=None
        self.candidates={}

    def observe(self, stamp, pose, pixel, camera, delta):
        if pixel is None or delta is None:return False
        u,v=pixel
        height,width=camera.arrival_excluded.shape
        if not (0<=u<width and 0<=v<height) or camera.arrival_excluded[int(v),int(u)]:return False
        ray=ground_ray_distance(u,v,camera.K,camera.T,delta)
        if ray is None:return False
        (x,y),r=ray
        if x<=0 or (camera.camera=="chest" and not .15<=r<=1.):return False
        if camera.camera=="forward" and (not .20 <= r <= 2.20 or abs(y)>1.):return False
        if self.last and stamp<=self.last[0]:return False
        world=pose.transform(Pose(x,y,0))
        if self.confirm_pair:
            # Two distinct exposures must locate the same contact on the ground.
            previous=self.candidates.get(camera.camera)
            if previous is not None and stamp<=previous[0]:
                return False
            self.candidates[camera.camera]=(stamp,world)
            if (previous is None or stamp-previous[0]>.35
                    or math.hypot(world.x-previous[1].x,world.y-previous[1].y)>.08):
                return False
        self.last=(stamp,world,camera.camera)
        return True

    def message(self, stamp, pose):
        if self.last is None or not 0<=stamp-self.last[0]<self.memory or pose is None:return None
        seen,world,cam=self.last
        dx,dy=world.x-pose.x,world.y-pose.y
        c,s=math.cos(pose.yaw),math.sin(pose.yaw)
        x,y=c*dx+s*dy,-s*dx+c*dy
        if x<=0:return None
        clearance=math.hypot(max(0.,x-self.front),max(0.,abs(y)-self.half_width))
        return dict(target_id=self.target,target_confirmed=True,source="measured_metric",
                    stamp_ns=round(stamp*1e9),observed_ns=round(seen*1e9),camera=cam,
                    distance_m=clearance,bumper_clearance_m=clearance,range_from_base_m=math.hypot(x,y),x_m=x,y_m=y)
