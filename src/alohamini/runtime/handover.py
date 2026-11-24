"""Aligned and relative manual takeovers in calibrated Host coordinates."""

import math
from dataclasses import dataclass

from alohamini._validation import finite_number
from alohamini.calibration.encoder import HostPositionUnits

VELOCITIES = ("x.vel", "y.vel", "theta.vel")


def arm_position_units(snapshot):
    return {
        f"{name}.pos": motor["normalization"]
        for name, motor in snapshot.payload["_robot_metadata"]["motors"].items()
        if name.startswith("arm_")
    }


def measured_hold_targets(snapshot, names):
    """Hold the named position channels at feedback and stop all base motion."""
    targets = {name: float(snapshot.payload[name]) for name in names if name not in VELOCITIES}
    targets.update(dict.fromkeys(VELOCITIES, 0.0))
    return targets


def joint_motion_scale(name, snapshot):
    """Physical degrees per Host wire unit, with gripper opening kept in %."""
    motor = snapshot.payload["_robot_metadata"]["motors"][name.removesuffix(".pos")]
    if name.endswith("_gripper.pos"):
        if motor["normalization"] != "range_0_100":
            raise ValueError("Grippers require range_0_100 units")
        return 1.0
    if motor["normalization"] == "degrees":
        return 1.0
    units = HostPositionUnits(
        **{key: motor[key] for key in ("normalization", "range_min", "range_max", "drive_mode")}
    )
    return (units.range_max - units.range_min) * 360 / 4096 / 200


def bounded_leader_targets(requested, previous, snapshot, elapsed_s, max_joint_velocity_deg_s):
    """Limit manual joint travel per command; gripper opening is in percentage points."""
    dt = max(0.0, min(elapsed_s, 0.1))
    targets = {}
    for key, target in requested.items():
        speed = 50.0 if key.endswith("_gripper.pos") else max_joint_velocity_deg_s
        step = speed * dt / joint_motion_scale(key, snapshot)
        old = previous[key]
        if not (math.isfinite(target) and math.isfinite(old)):
            raise ValueError(f"Non-finite manual target: {key}")
        targets[key] = old + max(-step, min(step, target - old))
    return targets


@dataclass(frozen=True)
class AlignmentConfig:
    alignment_tolerance_deg: float = 3.0
    gripper_tolerance: float = 3.0
    settle_time_s: float = 0.3
    hold_output_limit: float = 0.2
    alignment_timeout_s: float = 60.0
    max_joint_velocity_deg_s: float = 60.0

    def __post_init__(self):
        for key in (
            "alignment_tolerance_deg",
            "gripper_tolerance",
            "settle_time_s",
            "hold_output_limit",
            "alignment_timeout_s",
            "max_joint_velocity_deg_s",
        ):
            value = getattr(self, key)
            finite_number(value, key)
            if value <= 0:
                raise ValueError(f"{key} must be positive")
        if self.hold_output_limit > 1:
            raise ValueError("hold_output_limit must be in (0, 1]")
        if self.alignment_tolerance_deg > 10 or self.gripper_tolerance > 10:
            raise ValueError("Alignment tolerances must not exceed 10")


class _ManualMotion:
    def __init__(self, initial, now, max_joint_velocity_deg_s):
        self.previous = dict(initial)
        self.time = now
        self.speed = max_joint_velocity_deg_s

    def step(self, requested, snapshot, now):
        self.previous = bounded_leader_targets(
            requested, self.previous, snapshot, now - self.time, self.speed
        )
        self.time = now
        return dict(self.previous)


class AlignedLeader:
    """Hold aligned joints individually, then follow the released leader in absolute coordinates."""

    def __init__(self, leader, config):
        self.leader = leader
        self.config = config
        self._held = set()
        self._settled = {}
        self._previous_pose = None
        self._previous_time = None
        self._started = None
        self._confirmed = False
        self._motion = None

    def start(self, now):
        self._held.clear()
        self._settled.clear()
        self._previous_pose = self._previous_time = None
        self._started = now
        self._confirmed = False
        self._motion = None

    def _pose(self, snapshot):
        units = arm_position_units(snapshot)
        pose = self.leader.read_handover(units)
        if not pose or pose.keys() != units.keys():
            raise ValueError("Alignment requires every configured leader joint")
        return pose

    def _check_deadline(self, now):
        if self._started is None or self._confirmed:
            raise RuntimeError("Start alignment before checking the leader")
        if now - self._started >= self.config.alignment_timeout_s:
            raise TimeoutError("Leader alignment timed out")

    def _close(self, key, value, snapshot):
        tolerance = (
            self.config.gripper_tolerance
            if key.endswith("_gripper.pos")
            else self.config.alignment_tolerance_deg
        )
        return abs(value - snapshot.payload[key]) * joint_motion_scale(key, snapshot) <= tolerance

    def align(self, snapshot, now):
        """Return the newly held joint (if any), held count, and total joint count."""
        self._check_deadline(now)
        pose = self._pose(snapshot)
        dt = None if self._previous_time is None else now - self._previous_time
        for key, value in pose.items():
            close = self._close(key, value, snapshot)
            if key in self._held:
                if not close:
                    raise RuntimeError("Aligned leader/follower positions diverged")
                continue
            slow = (
                dt is not None
                and dt > 0
                and (
                    abs(value - self._previous_pose[key]) * joint_motion_scale(key, snapshot) / dt
                    <= 5.0
                )
            )
            if close and slow:
                self._settled.setdefault(key, now)
            else:
                self._settled.pop(key, None)
        joint = next(
            (
                key
                for key, since in self._settled.items()
                if now - since >= self.config.settle_time_s
            ),
            None,
        )
        if joint is not None:
            self.leader.hold_joint(joint, output_limit=self.config.hold_output_limit)
            self._held.add(joint)
            del self._settled[joint]
        self._previous_pose, self._previous_time = dict(pose), now
        return joint, len(self._held), len(pose)

    def confirm(self, snapshot, now):
        self._check_deadline(now)
        pose = self._pose(snapshot)
        if self._held != pose.keys() or not all(
            self._close(k, v, snapshot) for k, v in pose.items()
        ):
            return False
        self.leader.release_holds()
        self._confirmed = True
        return True

    def action(self, snapshot, hold, now):
        if not self._confirmed:
            raise RuntimeError("Confirm alignment before following the leader")
        pose = self._pose(snapshot)
        if self._motion is None:
            # Releasing the torque can move the arm. Hold the first command and recheck.
            if not all(self._close(k, v, snapshot) for k, v in pose.items()):
                raise RuntimeError("Leader moved during release; pause and realign")
            self._motion = _ManualMotion(hold, now, self.config.max_joint_velocity_deg_s)
            return dict(hold)
        return {**hold, **self._motion.step(pose, snapshot, now)}

    def release(self):
        self._started = None
        self._confirmed = False
        self._motion = None
        self.leader.release_holds()


class RelativeLeader:
    """Latch both sides on each explicit takeover, in calibrated Host units."""

    def __init__(self, names, bounds, *, max_joint_velocity_deg_s=60.0):
        self.names, self.bounds = tuple(names), bounds
        self.speed = max_joint_velocity_deg_s
        self.release()

    def release(self):
        self.origin = None
        self._motion = None

    def engage(self, leader, snapshot, now):
        self.latch(leader, snapshot.payload)
        self._motion = _ManualMotion(self.origin[1], now, self.speed)

    def action(self, leader, snapshot, now):
        if self._motion is None:
            raise RuntimeError("Relative takeover needs a fresh latch")
        return self._motion.step(self.targets(leader), snapshot, now)

    def latch(self, leader, follower):
        self._motion = None
        self.origin = (
            {n: float(leader[n]) for n in self.names},
            {n: float(follower[n]) for n in self.names},
        )

    def targets(self, leader):
        if self.origin is None:
            raise RuntimeError("Relative takeover needs a fresh latch")
        source, target = self.origin
        result = {}
        for name in self.names:
            value = target[name] + float(leader[name]) - source[name]
            if not math.isfinite(value):
                raise ValueError("Non-finite leader target")
            lower, upper = self.bounds[name]
            result[name] = min(max(value, lower), upper)
        return result
