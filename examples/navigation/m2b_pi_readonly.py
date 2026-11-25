"""Pi-side camera or wheel feedback producer. No Host lifecycle or motor configuration."""
import argparse
from pathlib import Path
import json
import struct
import sys
import time


def emit(ts,data):
    sys.stdout.buffer.write(struct.pack("!QI",ts,len(data))+data)
    sys.stdout.buffer.flush()


def camera(name):
    import cv2
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture("/dev/am_camera_"+name,cv2.CAP_V4L2)
    try:
        cap.set(6,cv2.VideoWriter_fourcc(*"MJPG"));cap.set(3,640);cap.set(4,480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        previous=0.
        while cap.isOpened():
            if not cap.grab():raise IOError("camera grab failed")
            if time.monotonic()-previous<(.055 if name=="forward" else .13):continue
            ok,bgr=cap.retrieve()
            ts=round(cap.get(cv2.CAP_PROP_POS_MSEC)*1e6)
            mono=time.monotonic_ns()
            if not ok or bgr.shape!=(480,640,3):raise IOError("invalid image")
            if not 0<=mono-ts<300_000_000:raise ValueError("invalid V4L2 acquisition timestamp")
            wall=ts+time.time_ns()-mono
            ok,jpeg=cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,78])
            if not ok:raise IOError("JPEG failed")
            emit(wall,jpeg.tobytes());previous=time.monotonic()
    finally:cap.release()


def wheels(known_height=None):
    import serial
    from alohamini.hardware.feetech import FeetechFeedbackReader
    from alohamini.model.loader import load_robot_model
    from alohamini.kinematics.base import OmniBaseKinematics
    from alohamini.runtime.base_control import BaseDrive
    model=load_robot_model(Path.home()/"Alohamini_software/models/alohamini2pro")
    from alohamini.runtime.lift_control import LiftAxisSpec, LiftHeightTracker
    by_name={a.name:a for a in model.actuators}
    tracker=LiftHeightTracker(LiftAxisSpec(by_name["lift_axis"],model.lift_lead_m_per_rev,-1))
    tracker.bind_session("m2b_readonly")
    from alohamini.model.joint_spec import ModelJointSpecification
    lift_joint=ModelJointSpecification.from_model(model).joints["vertical_move"]
    from alohamini.calibration.lift import LiftCalibration
    conversion=LiftCalibration(0.,lift_joint.upper-lift_joint.lower,lift_joint.lower,lift_joint.upper)
    previous_time=None
    first_tick=None
    stable_since=None
    reference_deadline=time.monotonic()+3.
    motors=[by_name["base_"+side+"_wheel"] for side in ("left","back","right")]
    drive=BaseDrive(OmniBaseKinematics(model.wheel_radius_m,model.base_radius_m))
    with serial.Serial("/dev/am_arm_follower_left",1000000,timeout=0,write_timeout=.005,exclusive=True) as port:
        reader=FeetechFeedbackReader(port,motors+[by_name["lift_axis"]],clock_id="m2b_readonly")
        while True:
            batch=reader.read()
            if previous_time is not None and batch.received_s-previous_time>.12:
                if tracker.height_m is not None:
                    print(f"Lift reference lost: feedback gap {batch.received_s-previous_time:.3f} s",file=sys.stderr,flush=True)
                tracker.invalidate()
            previous_time=batch.received_s
            referenced=tracker.height_m is not None
            try:
                tracker.observe(batch)
                if known_height is not None:
                    registers=batch.samples["lift_axis"].registers
                    tick=registers["position_raw"]
                    if first_tick is None:first_tick=tick
                    if abs((tick-first_tick+2048)%4096-2048)>3 or time.monotonic()>reference_deadline:
                        print("Lift startup reference could not be verified; forwarding wheels only",file=sys.stderr,flush=True)
                        known_height=None
                    elif registers["velocity_raw"]==0 and registers["moving"]==0:
                        if stable_since is None:stable_since=batch.received_s
                        if batch.received_s-stable_since>=.20:
                            tracker.establish_reference(known_height)
                            print(f"Physical lift reference {known_height} m, confirmed stationary encoder",file=sys.stderr,flush=True)
                            known_height=None
                    else:stable_since=None
            except (ValueError,RuntimeError) as error:
                if referenced:print("Lift reference lost: "+str(error),file=sys.stderr,flush=True)
                stable_since=None
                if time.monotonic()>reference_deadline:known_height=None
                tracker.invalidate()
            if all(a.name not in batch.failures and a.name in batch.samples and not batch.samples[a.name].packet_error for a in motors):
                speeds=tuple(batch.samples[a.name].registers["velocity_raw"] for a in motors)
                v=drive.measured_velocity(speeds)
                ts=time.time_ns()+round(batch.received_s*1e9)-time.monotonic_ns()
                q=None if tracker.height_m is None else conversion.height_to_position(tracker.height_m)
                emit(ts,json.dumps([v.x_m_s,v.y_m_s,v.yaw_rad_s,q]).encode())
            time.sleep(.02)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",choices=("forward","chest","wheels","clock"))
    p.add_argument("--verified-lift-height-m",type=float,help="Current independently verified metres above the mechanical minimum")
    args=p.parse_args();s=args.source
    if s=="clock":
        for line in sys.stdin.buffer:
            print(time.time_ns(),flush=True)
    elif s=="wheels":wheels(args.verified_lift_height_m)
    else:camera(s)
