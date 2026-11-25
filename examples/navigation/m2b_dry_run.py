"""Run ROS nodes in one isolated domain against loopback GPU services; no actuator node."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",type=Path);p.add_argument("--ssh-control");p.add_argument("--pi-script")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seconds",type=float,default=35)
    p.add_argument("--config",type=Path)
    p.add_argument("--verified-lift-height-m",type=float)
    a=p.parse_args()
    if os.getenv("ROS_LOCALHOST_ONLY")!="1" or int(os.getenv("ROS_DOMAIN_ID","0"))==0:
        p.error("select a dedicated localhost-only ROS domain")
    if not a.fixture and not (a.ssh_control and a.pi_script):p.error("supply a fixture or the read-only Pi transport")
    a.out.mkdir(exist_ok=False,parents=True);here=Path(__file__).resolve().parent
    nodes={"controller":["-m","alohamini.cerebellum.ros_adapter","--ros-args","-p","dry_run:=true","-p","publish_capture_pose:=true",
                         "-p","stop_confirm_count:=2","-p","target_id:=purple_toy","-p","grid_source:=measured_metric"]}
    for n in ("grid","arrival","brain"):nodes[n]=[here/"m2b_rgb_nodes.py",n]
    nodes["input"]=[here/"m2b_input.py"]+(["--fixture",a.fixture] if a.fixture else ["--ssh-control",a.ssh_control,"--pi-script",a.pi_script])
    if a.config:
        config=str(a.config.resolve(strict=True))
        nodes["controller"]+=["-p","config:="+config]
        for role in ("grid","arrival","brain"):nodes[role]+=["--config",config]
    if a.verified_lift_height_m is not None and not a.fixture:
        nodes["input"]+=["--verified-lift-height-m",str(a.verified_lift_height_m)]
    children=[];logs=[]
    try:
        for name,argv in nodes.items():
            f=(a.out/(name+".log")).open("wb");logs.append(f)
            proc=subprocess.Popen([sys.executable,"-u",*argv],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((name,proc))
        end=time.monotonic()+a.seconds
        while time.monotonic()<end and not (a.out/"stop").exists():
            for name,proc in children:
                if proc.poll() is not None:raise RuntimeError(f"{name} exited {proc.returncode}")
            time.sleep(.25)
    finally:
        for _,proc in reversed(children):
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for _,proc in children:
            try:proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=4)
        for f in logs:f.close()


if __name__=="__main__":main()
