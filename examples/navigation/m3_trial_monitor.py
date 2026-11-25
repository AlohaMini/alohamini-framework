"""Monitor an operator-confirmed autonomous run; the controller owns every velocity."""
import argparse,json,math,os,signal,subprocess,time
from pathlib import Path
from collections import OrderedDict
from alohamini.cerebellum.obstacles import occupied_contacts
from alohamini.cerebellum.rgb_navigation import PoseHistory
from alohamini.cerebellum.state import Pose
C='/alohamini_cerebellum_dry_run/'


def confirmed_target_odom(target, history, epoch, now_ns, target_id):
    """Use the pose at stamp_ns. observed_ns is the age of the underlying exposure."""
    if (history.epoch != epoch or target.get('source') != 'measured_metric'
            or target.get('target_id') != target_id or target.get('target_confirmed') is not True):
        return None
    stamp, observed = target.get('stamp_ns'), target.get('observed_ns')
    if not all(type(t) is int and 0 <= now_ns-t < 500_000_000 for t in (observed,stamp)):
        return None
    x, y = target.get('x_m'), target.get('y_m')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y)):
        return None
    pose = history.at(stamp*1e-9)
    return pose.transform(Pose(x,y,0.)) if pose is not None else None


def occupancy_guard(initial, live, footprint, pose, margin, target, witnesses):
    """Witness keys belong to this exact map and identify target-only depth votes."""
    if occupied_contacts(initial, footprint, pose, margin):
        return 'initial occupied_footprint'
    for i,j in occupied_contacts(live, footprint, pose, margin):
        cx = live.origin[0]+(i+.5)*live.resolution
        cy = live.origin[1]+(j+.5)*live.resolution
        if target is None or math.hypot(cx-target.x,cy-target.y) > .12:
            return 'live occupied_footprint'
        if (i,j) not in witnesses:
            return 'unattributed live occupied_footprint'
    return None


def main():
    import rclpy
    from geometry_msgs.msg import Twist,TwistStamped
    from std_msgs.msg import Bool,String
    from diagnostic_msgs.msg import DiagnosticArray
    from sensor_msgs.msg import CompressedImage
    from std_srvs.srv import SetBool
    from rclpy.qos import qos_profile_sensor_data
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--chair',action='store_true')
    p.add_argument('--operator-confirmed',action='store_true');a=p.parse_args()
    if not __debug__ or os.environ.get('ROS_DOMAIN_ID') in (None,'','0') or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('requires normal Python execution in an isolated loopback ROS domain')
    a.out.mkdir(parents=True,exist_ok=False)
    f=(a.out/'events.jsonl').open('w',buffering=1)
    def log(topic,data):f.write(json.dumps(dict(t=time.time_ns(),topic=topic,data=data))+'\n')
    rclpy.init();n=rclpy.create_node('m3_trial_monitor')
    d={};diag={};hw={};images={};wheel=[0,0,0];seen={};vel=[0,0,0]
    recent=[]; history=PoseHistory()
    def decision(msg):
        d.clear();d.update(json.loads(msg.data));seen['decision']=time.monotonic();log('decision',d);recent.append((time.monotonic(),d.get('reason'),d['planned_velocity']['y_m_s']))
    def feedback(msg):
        seen['wheel']=time.monotonic();wheel[:]=[msg.twist.linear.x,msg.twist.linear.y,msg.twist.angular.z]
        log('wheel',[msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,*wheel])
    def cmd(msg):
        vel[:]=[msg.linear.x,msg.linear.y,msg.angular.z];log('cmd',vel)
    def health(msg):
        for s in msg.status:
            vals={v.key:v.value for v in s.values}
            if 'command_enabled' in vals:diag.update(vals);seen['diag']=time.monotonic()
    def pose(msg):
        hw.update(json.loads(msg.data))
        history.push(hw['stamp_ns']*1e-9,Pose(hw['x'],hw['y'],hw['yaw']),hw['epoch'])
    n.create_subscription(String,C+'decision',decision,10);n.create_subscription(String,C+'capture_pose',pose,qos_profile_sensor_data)
    n.create_subscription(TwistStamped,'/alohamini/base_velocity',feedback,qos_profile_sensor_data)
    n.create_subscription(Twist,'/cmd_vel',cmd,10);n.create_subscription(DiagnosticArray,'/diagnostics',health,5)
    for cam in ('forward','chest'):
        n.create_subscription(CompressedImage,'/alohamini/cameras/'+cam+'/image_raw/compressed',lambda m,k=cam:images.update({k:bytes(m.data)}),qos_profile_sensor_data)
    estop=n.create_publisher(Bool,C+'estop',10)
    target={}
    def observe(m):
        target.clear();target.update(json.loads(m.data));log('arrival',target)
    n.create_subscription(String,C+'arrival',observe,10)
    n.create_subscription(Bool,C+'stop',lambda m:log('stop_topic',m.data),10)
    from nav_msgs.msg import OccupancyGrid
    from alohamini.cerebellum.obstacles import Grid
    from alohamini.cerebellum.state import load_config
    import numpy as np
    cfg=load_config(Path(os.environ['M3_CONFIG']))
    maps={}; seen['grid']=0
    witnesses_by_stamp=OrderedDict()
    def witness(m):
        sources=n.get_publishers_info_by_topic(C+'target_cell_witnesses')
        if len(sources)!=1 or sources[0].node_name!='rgb_grid_node':
            witnesses_by_stamp.clear()
            return
        data=json.loads(m.data)
        if data.get('target_id')!=os.environ['M3_TARGET_ID'] or data.get('frame')!='odom':
            return
        # MapMetaData.resolution is a ROS float32; JSON preserves Python's double.
        key=(data['epoch'],data['stamp_ns'],tuple(data['origin']),float(np.float32(data['resolution'])))
        witnesses_by_stamp[key]=data['target_only_cells']
        while len(witnesses_by_stamp)>32:
            witnesses_by_stamp.popitem(last=False)
    n.create_subscription(String,C+'target_cell_witnesses',witness,10)
    def grid(m):
        cells=np.array(m.data).reshape(m.info.height,m.info.width)
        yy,xx=np.where(cells==100)
        origin=(m.info.origin.position.x,m.info.origin.position.y)
        maps['stamp_ns']=m.header.stamp.sec*10**9+m.header.stamp.nanosec
        maps.update(grid=Grid(0,'measured_metric','odom',m.info.resolution,origin,{(int(x),int(y)):100 for x,y in zip(xx,yy)}),
                    xy=np.column_stack((origin[0]+(xx+.5)*m.info.resolution,origin[1]+(yy+.5)*m.info.resolution)))
        seen['grid']=time.monotonic()
    n.create_subscription(OccupancyGrid,C+'local_grid',grid,1)
    service=n.create_client(SetBool,'/alohamini_lerobot_bridge/command_enable')
    def enable(on):
        req=SetBool.Request();req.data=on
        fut=service.call_async(req);end=time.monotonic()+1.2
        while not fut.done() and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.015)
        result=fut.result() if fut.done() else None
        log('bridge',[on, None if not result else result.success,None if not result else result.message])
        if not result or not result.success:raise RuntimeError('bridge enable/disable unacknowledged')
    def spin(dt):
        end=time.monotonic()+dt
        while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.008)
    def trip(msg):
        estop.publish(Bool(data=True));log('estop',msg)
    bag=subprocess.Popen(['ros2','bag','record','-a','-o',str(a.out/'bag')],stdout=(a.out/'bag.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'supervisor received signal {signum}')
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    armed=False;result={}
    try:
        spin(4.5)
        assert diag.get('command_enabled')=='false' and diag.get('lift_command_ready')=='true',diag
        assert [i.node_name for i in n.get_publishers_info_by_topic('/cmd_vel')]==['alohamini_cerebellum_dry_run']
        assert n.get_subscriptions_info_by_topic('/cmd_vel')
        assert not any(i.node_name=='alohamini_lerobot_bridge' for i in n.get_subscriptions_info_by_topic('/cmd_vel'))
        assert len(n.get_publishers_info_by_topic(C+'reference'))==1 and bag.poll() is None
        assert d.get('map_source')=='measured_metric' and not d['estop_latched'] and hw and max(map(abs,wheel))<.05,d
        assert [i.node_name for i in n.get_publishers_info_by_topic(C+'reference')]==['lightnav_reference_bridge']
        assert [i.node_name for i in n.get_publishers_info_by_topic(C+'stop')]==['lightnav_reference_bridge']
        def valid():
            return (target.get('source')=='measured_metric' and target.get('target_id')==os.environ['M3_TARGET_ID']
                    and target.get('target_confirmed') is True and 0<=(time.time_ns()-target['observed_ns'])/1e9<.5)
        x0,y0,w0=hw['x'],hw['y'],hw['yaw'];epoch=hw['epoch'];ca,sa=math.cos(w0),math.sin(w0)
        assert len(images)==2 and maps and time.monotonic()-seen['grid']<.5
        if a.chair:
            live=maps['grid']
            assert any(k[0]==epoch and abs(k[1]-maps['stamp_ns'])<250_000_000
                       and k[3]==live.resolution for k in witnesses_by_stamp), 'missing grid attribution stream'
        assert time.monotonic()-seen['wheel']<.14 and time.monotonic()-seen['decision']<.16
        for k,data in images.items():(a.out/('start-'+k+'.jpg')).write_bytes(data)
        xy=maps['xy']-np.array([x0,y0]);bx=ca*xy[:,0]+sa*xy[:,1];by=-sa*xy[:,0]+ca*xy[:,1]
        chair_points=maps['xy'][(bx>.45)&(bx<1.25)&(abs(by)<.32)]
        right_samples=sum(why=='detour_subgoal' and vy<-.005 for t,why,vy in recent if time.monotonic()-t<=3)
        result.update(start_target=target.copy(),pose_start=[x0,y0,w0],chair_points=chair_points.tolist(),
                      right_samples=right_samples,last_state=d['state'],last_reason=d['reason'])
        in_range=(1.15<=target.get('range_from_base_m',0)<=2.0 and target.get('y_m',0)<.02 if a.chair
                  else .82<=target.get('x_m',0)<=1.30 and abs(target.get('y_m',99))<=.20)
        passes=(valid() and in_range and d['state']=='TRACKING' and not d.get('stop_source') and
                ((len(chair_points)>=3 and right_samples>=30 and vel[1]<-.003) if a.chair else vel[0]>.005))
        print(json.dumps(dict(preflight=passes,**result)),flush=True)
        result['preflight_passed']=passes
        if not a.operator_confirmed:
            if not passes:raise RuntimeError('static gate did not pass')
            return
        if not passes:raise RuntimeError('preflight: bridge stays disabled')
        cap=(2.0 if a.chair else min(.75,target['bumper_clearance_m']-.095))
        initial_grid=maps['grid']
        result['cap_m']=cap
        armed=True;enable(True);start=time.monotonic()
        while True:
            spin(.012);now=time.monotonic()
            x,y=hw['x']-x0,hw['y']-y0
            travel=ca*x+sa*y;side=-sa*x+ca*y
            pose_now=Pose(hw['x'],hw['y'],hw['yaw'])
            dist=target.get('bumper_clearance_m')
            if not isinstance(dist,(float,int)) or not math.isfinite(dist) or dist<.105:
                raise RuntimeError('protective target range floor')
            if a.chair:
                live=maps['grid']
                goal=confirmed_target_odom(target,history,epoch,time.time_ns(),os.environ['M3_TARGET_ID'])
                key=(epoch,maps['stamp_ns'],live.origin,live.resolution)
                witnesses={(i,j) for i,j,x,y,t in witnesses_by_stamp.get(key,())
                           if goal is not None and 0<=time.time_ns()-t<500_000_000
                           and math.hypot(x-goal.x,y-goal.y)<=.12}
                hit=occupancy_guard(initial_grid,live,cfg['footprint_m'],pose_now,
                                    cfg['planner']['margin_m'],goal,witnesses)
                if hit:raise RuntimeError('chair trial occupancy guard: '+hit)
            if (now-seen['wheel']>.14 or now-seen['decision']>.16 or now-seen['grid']>.65
                    or hw['epoch']!=epoch or bag.poll() is not None or d['estop_latched'] or not valid()
                    or travel<-.03 or travel>cap or abs(side)>(1.05 if a.chair else .18)
                    or (a.chair and side>.10)
                    or abs(math.remainder(hw['yaw']-w0,math.tau))>(1.25 if a.chair else .40)
                    or now-start>(65 if a.chair else 24) or (now-start>4 and math.hypot(travel,side)<.015)):
                raise RuntimeError('approach envelope: '+str((travel,side,d['state'],d['reason'],valid())))
            if d['state']=='ARRIVED' and dist<=.20 and max(map(abs,wheel[:2]))<.015 and abs(wheel[2])<.05:
                result.update(arrival_at=target.copy(),elapsed_s=now-start,stop_source=d['stop_source']);break
        enable(False);armed=False;spin(.7)
        result.update(final_pose=[hw['x'],hw['y'],hw['yaw']],final_wheel=wheel.copy(),final_decision=d['state'],final_target=target.copy())
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        log('failure',str(exc));result['failure']=str(exc)
        if armed:
            trip('abort')
        result.update(final_wheel_at_exception=wheel.copy(),last_decision=d.copy(),last_target=target.copy())
        raise
    finally:
        try:
            if armed:
                trip('disable');enable(False);spin(.6)
            for k,data in images.items():(a.out/(k+'.jpg')).write_bytes(data)
            (a.out/'result.json').write_text(json.dumps(result,indent=2))
        finally:
            if bag.poll() is None:os.killpg(bag.pid,signal.SIGINT)
            bag.wait(timeout=12);f.close();n.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':main()
