"""Static/live capture adapter. Convert Pi exposure timestamps to the local clock."""
import json
import subprocess
import sys
import threading
import time
import rclpy
from alohamini_camera.camera_node import AlohaMiniCameraNode, ros_stamp_from_unix_ns

def main():
    ctl=sys.argv.pop(1)
    peer=sys.argv.pop(1)
    offset=[None]
    done=threading.Event()
    def probe():
        while not done.is_set():
            try:
                r=subprocess.run([sys.executable,'examples/navigation/m3_clock_check.py','--ssh-control',ctl,peer],capture_output=True,text=True,timeout=7)
                d=json.loads(r.stdout);lo,hi=d['pi_minus_pc_ns']
                if not (0<=hi-lo<=1_000_000 and max(abs(lo),abs(hi))<40_000_000):
                    raise ValueError("uncertain Pi clock")
                offset[0]=(time.monotonic(),hi)
                print("measured Pi-PC ns",lo,hi,flush=True)
            except Exception as exc:
                offset[0]=None;print(exc,flush=True)
            done.wait(3.)
    class Timed(AlohaMiniCameraNode):
        def frame_stamp(self,frame):
            anchor=offset[0]
            if anchor is None or time.monotonic()-anchor[0]>5:
                raise ValueError("missing fresh clock calibration")
            corrected=frame.capture_unix_ns-anchor[1]
            if not 0 <= time.time_ns()-corrected < 250_000_000:
                raise ValueError("camera outside capture deadline")
            return ros_stamp_from_unix_ns(corrected)
    rclpy.init();node=Timed();threading.Thread(target=probe,daemon=True).start()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:done.set();node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
