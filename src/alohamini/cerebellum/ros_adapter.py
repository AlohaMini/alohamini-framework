"""ROS 2 controller adapter with an opt-in, low-speed actuator output."""

from dataclasses import asdict, replace
import json
import math
import signal
import time

from alohamini.schema import BodyVelocity

from .controller import Controller
from .obstacles import Cell, Grid
from .scheduler import Periodic
from .state import Arrival, Feedback, load_config, Pose, Reference, ZERO
from .timeline import clock_kind


class ClockMonitor:
    """Map Unix ROS message stamps into local steady time at receipt."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.offset = None
        self.anchor = None
        self.fault = False

    def observe(self, wall_ns, steady):
        if self.anchor is None:
            self.anchor = (wall_ns, steady)
        offset = wall_ns / 1e9 - steady
        if self.offset is not None and abs(self.offset - offset) > self.cfg["max_wall_steady_step_s"]:
            self.fault = True
        self.offset = offset
        if self.fault:
            raise ValueError("wall_clock_step: restart dry-run after clock synchronization")

    def convert(self, stamp_ns, wall_ns, steady):
        self.observe(wall_ns, steady)
        if clock_kind(stamp_ns) != "unix_like":
            raise ValueError("expected Unix ROS timestamp, not ROS simulated or remote monotonic time")
        age = (wall_ns - stamp_ns) / 1e9
        if not 0 <= age < self.cfg["max_stamp_age_s"]:
            raise ValueError("stamp in future or clock offset/message delay outside accepted window")
        wall_anchor_ns, steady_anchor = self.anchor
        return min(steady, steady_anchor + (stamp_ns - wall_anchor_ns) / 1e9)


def yaw_from_quaternion(q):
    if not all(map(math.isfinite, (q.x, q.y, q.z, q.w))):
        raise ValueError("nonfinite orientation")
    norm = math.hypot(q.x, q.y, q.z, q.w)
    if abs(norm - 1.) > .01 or math.hypot(q.x, q.y) > .01:
        raise ValueError("expected normalized planar quaternion")
    return 2 * math.atan2(q.z, q.w)


def require_dry_run(value):
    if value is not True:
        raise ValueError("V1 requires dry_run=true. No actuator publisher is registered.")


class StopDebounce:
    def __init__(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("stop_confirm_count must be a positive integer")
        self.threshold = count
        self.pending = 0

    def observe(self, stop):
        self.pending = min(self.threshold, self.pending + 1) if stop else 0
        return self.pending == self.threshold


def validate_publish_deadline(controller, decision, now):
    if decision.command != ZERO and (not math.isfinite(decision.valid_until) or now >= decision.valid_until):
        controller.previous = ZERO
        controller.selected = None
        return replace(decision, state="SAFETY_HOLD", reason="expired_during_control_tick",
                       valid_until=now, command=ZERO, selected=None)
    return decision


LOW_SPEED_CAPS = {
    "forward_m_s": .12, "reverse_m_s": .06, "lateral_m_s": .10,
    "linear_m_s": .12, "yaw_rad_s": .25, "brake_m_s2": .45,
}


ACTUATION_CAPS = {
    "alohamini2pro-m3-low-speed-v1": LOW_SPEED_CAPS,
    "alohamini2pro-m4-follow-v1": {
        **LOW_SPEED_CAPS, "forward_m_s": .20, "linear_m_s": .20,
        "lateral_m_s": .14, "yaw_rad_s": .45,
    },
}


def actuator_enabled(dry_run, enable_actuators, grid_source, config_path, config, model):
    if dry_run is True and enable_actuators is False:
        return False
    if dry_run is not False or enable_actuators is not True:
        raise ValueError("actuator output requires both explicit unlocks")
    profile = config.get("actuation_profile")
    if (grid_source != "measured_metric" or model != "alohamini2pro" or not config_path
            or profile not in ACTUATION_CAPS):
        raise ValueError("expected measured_metric and explicit AlohaMini2 Pro low-speed profile")
    if bool(config["control"].get("continuous_follow")) != (profile == "alohamini2pro-m4-follow-v1"):
        raise ValueError("actuation profile does not match continuous_follow")
    for key, cap in ACTUATION_CAPS[profile].items():
        if not 0 < config["limits"][key] <= cap:
            raise ValueError(f"unsafe low-speed limit for {profile}: {key}")
    return True


class BaseOutput:
    """Single executor owns the latch, the deadline check and the volatile Twist writer."""

    def __init__(self, send, limits, encode=lambda velocity: velocity, clock=time.perf_counter):
        self.send = send
        self.encode = encode
        self.clock = clock
        self.limits = limits
        self.estopped = False

    def estop(self, active):
        if active:
            self.estopped = True
            if self.send:
                self.send(self.encode(ZERO))

    def apply(self, controller, decision, now):
        decision = validate_publish_deadline(controller, decision, now)
        if self.estopped:
            decision = replace(decision, state="SAFETY_HOLD", reason="estop_latched",
                               command=ZERO, selected=None, valid_until=now)
        if self.send:
            v = decision.command
            lim = self.limits
            tol = 1e-9  # Ignore numerical round-off at a planned velocity limit.
            if (v != ZERO and (decision.state not in {"TRACKING", "BRAKING"}
                    or not -lim["reverse_m_s"] - tol <= v.x_m_s <= lim["forward_m_s"] + tol
                    or abs(v.y_m_s) > lim["lateral_m_s"] + tol
                    or math.hypot(v.x_m_s, v.y_m_s) > lim["linear_m_s"] + tol
                    or abs(v.yaw_rad_s) > lim["yaw_rad_s"] + tol)):
                decision = replace(decision, state="SAFETY_HOLD", reason="output_interlock",
                                   command=ZERO, selected=None, valid_until=now)
            # Recheck immediately at the writer, after building the ROS message.
            message = self.encode(decision.command)
            decision = validate_publish_deadline(controller, decision, self.clock())
            self.send(message if decision.command != ZERO else self.encode(ZERO))
        if decision.command == ZERO and (self.estopped or decision.reason == "output_interlock"):
            controller.previous = ZERO
            controller.selected = None
        return decision

    def close(self):
        if self.send:
            self.estopped = True
            for i in range(3):
                self.send(self.encode(ZERO))
                if i < 2:
                    time.sleep(.025)


def main():
    import rclpy
    from geometry_msgs.msg import Twist, TwistStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.duration import Duration
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Bool, String

    def ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    class DryRun(Node):
        def __init__(self):
            super().__init__("alohamini_cerebellum_dry_run")
            immutable = ParameterDescriptor(read_only=True)
            self.dry_run = self.declare_parameter("dry_run", True, immutable).value
            enable = self.declare_parameter("enable_actuators", False, immutable).value
            config_path = self.declare_parameter("config", "", immutable).value
            config = load_config(config_path)
            model = self.declare_parameter("robot_model", "alohamini2pro", immutable).value
            self.grid_source = self.declare_parameter("grid_source", "synthetic", immutable).value
            live = actuator_enabled(self.dry_run, enable, self.grid_source, config_path, config, model)
            if live and self.get_parameter("use_sim_time").value:
                raise ValueError("live output requires a system clock")
            self.target_id = self.declare_parameter("target_id", "").value
            if config["control"]["continuous_follow"] and self.target_id:
                raise ValueError("continuous_follow requires an empty metric target ID")
            self.controller = Controller(config, model)
            self.debounce = StopDebounce(self.declare_parameter("stop_confirm_count", 1).value)
            self.loop = Periodic(config["control"]["frequency_hz"])
            self.clock_monitor = ClockMonitor(config["ros_clock"])
            wheel_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(TwistStamped, "/alohamini/base_velocity", self.on_wheel, wheel_qos)
            self.create_subscription(Path, "~/reference", self.on_reference, latest)
            self.create_subscription(OccupancyGrid, "~/local_grid", self.on_grid, latest)
            self.create_subscription(Bool, "~/stop", self.on_stop, 10)
            self.create_subscription(String, "~/arrival", self.on_arrival, latest)
            self.output = self.create_publisher(String, "~/decision", latest)
            if live:
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 lifespan=Duration(seconds=config["control"]["command_ttl_s"]))
                self.velocity_pub = self.create_publisher(Twist, "/cmd_vel", qos)
            self.base = BaseOutput(self.velocity_pub.publish if live else None,
                                   config["limits"], self.velocity_message)
            self.create_subscription(Bool, "~/estop", lambda msg: self.base.estop(msg.data), 10)
            self.capture_pose = None
            self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
            self.odom_epoch = self.system_clock.now().nanoseconds
            if self.declare_parameter("publish_capture_pose", False).value:
                self.capture_pose = self.create_publisher(String, "~/capture_pose", wheel_qos)
            self.tick_sequence = 0
            self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.timer = self.create_timer(self.loop.period, self.on_tick, clock=self.steady_clock)

        @staticmethod
        def velocity_message(velocity):
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = velocity.x_m_s, velocity.y_m_s, velocity.yaw_rad_s
            return msg

        def convert(self, source_ns):
            now = time.perf_counter()
            if self.get_parameter("use_sim_time").value:
                raise ValueError("wall-stamped dry run requires use_sim_time=false")
            return self.clock_monitor.convert(source_ns, self.system_clock.now().nanoseconds, now), now

        def on_wheel(self, msg):
            try:
                source = ns(msg.header.stamp)
                stamp, receipt = self.convert(source)
                velocity = BodyVelocity(msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)
                odom = self.controller.odom
                previous_epoch = odom.reanchor_stamp
                self.controller.feedback(Feedback(stamp, receipt, velocity, msg.header.frame_id, source))
                if self.capture_pose is not None and not odom.fault:
                    if previous_epoch != odom.reanchor_stamp:
                        self.odom_epoch = source
                    pose = odom.history[-1][1]
                    self.capture_pose.publish(String(data=json.dumps(dict(
                        stamp_ns=source, epoch=self.odom_epoch, x=pose.x, y=pose.y, yaw=pose.yaw))))
            except ValueError:
                self.controller.odom.fault = "feedback_invalid_ros_clock_or_velocity"
                self.controller.still_since = None

        def on_reference(self, msg):
            try:
                stamp, receipt = self.convert(ns(msg.header.stamp))
                path = []
                for item in msg.poses:
                    if (item.header.frame_id not in ("", msg.header.frame_id)
                            or ns(item.header.stamp) not in (0, ns(msg.header.stamp))):
                        raise ValueError("mixed pose frame or timestamp")
                    pose = item.pose
                    path.append(Pose(pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation)))
                accepted = self.controller.set_reference(Reference(stamp, receipt, tuple(path), msg.header.frame_id))
                if accepted and receipt - stamp < self.controller.c["reference_ttl_s"]:
                    self.debounce.observe(False)
            except ValueError:
                self.controller.reference_fault("reference_invalid_ros_clock_or_pose")

        def on_grid(self, msg):
            try:
                stamp, _ = self.convert(ns(msg.header.stamp))
                geom = msg.info
                if abs(math.remainder(yaw_from_quaternion(geom.origin.orientation), math.tau)) > 1e-6:
                    raise ValueError("grid origin must be axis-aligned in odom")
                if not (0 < geom.width * geom.height == len(msg.data)):
                    raise ValueError("grid dimensions do not match data")
                origin = (geom.origin.position.x, geom.origin.position.y)
                cells = {(i % geom.width, i // geom.width): Cell(value) for i, value in enumerate(msg.data)
                         if value in (Cell.FREE, Cell.OCCUPIED)}
                self.controller.set_map(Grid(stamp, self.grid_source, msg.header.frame_id, geom.resolution, origin, cells))
            except ValueError:
                self.controller.map_fault("grid_invalid_ros_clock_or_geometry")

        def on_stop(self, msg):
            if self.debounce.observe(msg.data):
                self.controller.stop(received=time.perf_counter())

        def on_arrival(self, msg):
            try:
                data = json.loads(msg.data)
                source_ns = data["stamp_ns"]
                if self.controller.c.get("metric_arrival_autostop") and data["source"] == "measured_metric":
                    # Odometry can refresh the message while its last visual contact ages.
                    observed_ns = data["observed_ns"]
                    if type(observed_ns) is not int or observed_ns > source_ns:
                        raise ValueError("invalid contact exposure timestamp")
                    source_ns = observed_ns
                stamp, _ = self.convert(source_ns)
                confirmed = (bool(self.target_id) and data["target_id"] == self.target_id
                             and data["target_confirmed"] is True)
                target = None
                if (confirmed and self.controller.planner.p.get("metric_target_detour")
                        and data["source"] == "measured_metric"):
                    # x_m/y_m are expressed at the odometry message time, not the exposure.
                    x, y = data.get("x_m"), data.get("y_m")
                    if all(type(v) in (int, float) and math.isfinite(v) for v in (x, y)):
                        pose_stamp, _ = self.convert(data["stamp_ns"])
                        base = self.controller.odom.at_reported_pose(pose_stamp)
                        if base is not None and x > 0 and math.hypot(x, y) <= 2.2:
                            target = base.transform(Pose(x, y, 0.))
                obs = Arrival(stamp, confirmed, data["distance_m"], data["source"], target)
                self.controller.set_arrival(obs)
            except (TypeError, ValueError, KeyError):
                self.controller.arrival = None

        def on_tick(self):
            now = time.perf_counter()
            # rcl has advanced the timer by the time the callback runs. Use its
            # actual steady deadline rather than a second independently phased gate.
            tick = self.loop.timer_tick(now, now + self.timer.time_until_next_call() / 1e9)
            try:
                self.clock_monitor.observe(self.system_clock.now().nanoseconds, now)
            except ValueError:
                self.controller.odom.fault = "wall_clock_step"
            decision = self.controller.tick(tick)
            computed = time.perf_counter()
            if not self.dry_run and self.get_parameter("use_sim_time").value:
                self.base.estop(True)
            decision = self.base.apply(self.controller, decision, computed)
            self.tick_sequence += 1
            payload = {
                "tick_sequence": self.tick_sequence,
                "dry_run": self.dry_run, "estop_latched": self.base.estopped, "state": decision.state, "reason": decision.reason,
                "arrival_status": decision.arrival_status, "clock_fault": self.clock_monitor.fault,
                "planned_velocity": asdict(decision.command), "timing": asdict(tick),
                "tick_compute_s": computed - now,
                "valid_until_local_steady_s": decision.valid_until,
                "pose_odom": asdict(decision.pose) if decision.pose else None,
                "selected_trajectory_odom": [asdict(p) for p in decision.selected.path] if decision.selected else None,
                "score_components": decision.selected.terms if decision.selected else None,
                "swept_clearance_m": decision.selected.clearance_m if decision.selected else None,
                "rejection_counts": decision.rejected,
                "feedback_consumed_stamp_ns": decision.feedback_consumed_stamp_ns,
                **self.controller.diagnostics(now),
            }
            self.output.publish(String(data=json.dumps(payload, allow_nan=False)))

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        signals.append(signal.SIGBREAK)
    old_handlers = {sig: signal.signal(sig, terminate) for sig in signals}
    try:
        node = DryRun()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                try:
                    node.base.close()
                finally:
                    node.destroy_node()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
