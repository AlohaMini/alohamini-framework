"""Compare capture-time lower-half ground support using one archived raw image."""
import argparse
from pathlib import Path
import urllib.request
import time
import json
import cv2,numpy as np
from alohamini.cerebellum.rgb_navigation import Rectifier
from alohamini.cerebellum.local_grid import RollingLocalGrid
from alohamini.cerebellum.state import Pose

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("image",type=Path);a=p.parse_args()
    c=Rectifier(Path.home()/"Alohamini_workspace/calibration/cameras","forward");b=c.decode(a.image.read_bytes())
    t=time.monotonic();req=urllib.request.Request("http://127.0.0.1:18642/rectified/forward",data=b.tobytes())
    with urllib.request.urlopen(req,timeout=4) as response:z=np.frombuffer(response.read(),"<f4").reshape(480,640)
    print("depth_ms",(time.monotonic()-t)*1000)
    u,v=c.maps
    floor=(v>=240)&(v<422)
    side=(u<226)|(u>414)
    for name,mask in [("lower_half",None),("raw_lower_sides",floor&side),("near_field",floor&(v>=344)),("audited_carpet",floor&(v>=250)&(v<=410)&(((u>=150)&(u<=255))|((u>=385)&(u<=495))))]:
        m=RollingLocalGrid(stride=4)
        fit=m.update(z,c.K,c.T,stamp=0.,pose_odom=Pose(0.,0.,0.),floor_mask=mask,self_mask=c.excluded,calibration=c.source)
        print(name,fit.reason,np.degrees(np.arccos(fit.normal[2])) if fit.normal else None,fit.scale,fit.floor_p95_m)
        if fit.accepted:
            g=m.export(0);print("occupied",len(g.occupied))
