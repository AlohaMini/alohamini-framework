"""Local Metric3D inference. Input and output grids share the signed rectified pixel coordinates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import torch

from alohamini.cerebellum.rgb_navigation import CAMERAS, Rectifier, depth_canvas, mask_depth_padding


class MetricDepth:
    def __init__(self, root, width=384):
        self.width=width
        self.height,self.padding=depth_canvas(width)
        self.top,_,self.left,_=self.padding
        sys.path.append(str(root.parent/"openvo_overlay"))
        sys.path.append(str(root.parent/"envs/anchor3r/lib/python3.11/site-packages"))
        sys.path.insert(0,str(root/"processor/Metric3D"))
        from hubconf import metric3d_vit_large
        model=metric3d_vit_large(pretrain=False)
        state=torch.load(root/"weights/Metric3D/metric_depth_vit_large_800k.pth",map_location="cpu",weights_only=True)
        missing,extra=model.load_state_dict(state["model_state_dict"],strict=False)
        if set(missing)-{"depth_model.encoder.mask_token"} or extra:
            raise ValueError("unexpected Metric3D checkpoint keys")
        # AMP casts these weights to FP16 on every call. Store the same cast once.
        for layer in model.modules():
            if isinstance(layer,(torch.nn.Linear,torch.nn.Conv2d)):
                layer.half()
        self.model=model.cuda().eval()
        self.mean=torch.tensor([123.675,116.28,103.53],device="cuda")[None,:,None,None]
        self.std=torch.tensor([58.395,57.12,57.375],device="cuda")[None,:,None,None]

    @torch.inference_mode()
    def __call__(self,bgr,focal):
        small=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(self.width,self.height),interpolation=cv2.INTER_AREA)
        canvas=cv2.copyMakeBorder(small,*self.padding,cv2.BORDER_CONSTANT,value=(123.675,116.28,103.53))
        x=torch.from_numpy(canvas).permute(2,0,1)[None].cuda().float()
        with torch.autocast("cuda",dtype=torch.float16):
            z=self.model.inference({"input":(x-self.mean)/self.std})[0][:,:,self.top:self.top+self.height,self.left:self.left+self.width]
        # Match the audited 518-wide backend's resize before full-resolution projection.
        depth=cv2.resize(z[0,0].float().cpu().numpy()*(focal*self.width/640/1000),(640,480))
        # At 384, upsampling the model's padding spreads its invalid depth
        # across the first/last 20 columns and 8 rows of the 640x480 image.
        # Keep a one-sample halo for the grid sampler. Missing rays cast no votes.
        mask_depth_padding(depth,self.width)
        return depth.astype("<f4",copy=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width",type=int,choices=(384,518),default=384)
    p.add_argument("--openvo",type=Path,required=True)
    p.add_argument("--calibration",type=Path,default=Path.home()/"Alohamini_workspace/calibration/cameras")
    p.add_argument("--port",type=int,default=18642)
    a=p.parse_args()
    focal={c:Rectifier(a.calibration,c).K[0,0] for c in CAMERAS}
    infer=MetricDepth(a.openvo,a.width)
    guard=threading.Lock()
    worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix="metric3d")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            cam=self.path.removeprefix("/rectified/")
            if self.path!="/rectified/"+cam or cam not in focal or self.headers.get("Content-Length")!=str(640*480*3):
                self.send_error(400);return
            if not guard.acquire(False):
                self.send_error(429);return
            try:
                self.connection.settimeout(2)
                frame=np.frombuffer(self.rfile.read(640*480*3),np.uint8).reshape(480,640,3)
                depth=worker.submit(infer,frame,focal[cam]).result().tobytes()
                self.send_response(200);self.send_header("Content-Length",str(len(depth)));self.end_headers()
                self.wfile.write(depth)
            finally:
                guard.release()
    server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler)
    print("Metric3D ready",a.width,server.server_address,flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.shutdown(wait=True)


if __name__=="__main__":
    main()
