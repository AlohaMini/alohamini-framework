"""Short-term odometry; x forward, y left, yaw counterclockwise, SI units."""

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

from alohamini.schema import BodyVelocity

ZERO = BodyVelocity(0., 0., 0.)


def load_config(path=None):
    source = Path(path) if path else Path(__file__).with_name("config.json")
    cfg = json.loads(source.read_text(encoding="utf-8"))
    if path:
        # Older saved configurations predate the guidance parameters.
        defaults = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
        cfg["planner"] = {**defaults["planner"], **cfg["planner"]}
    planner = cfg["planner"]
    policy = planner.setdefault("unknown_policy","strict")
    weight = planner.setdefault("unknown_weight",.35)
    if type(planner.setdefault("metric_target_detour", False)) is not bool:
        raise ValueError("metric_target_detour must be a boolean")
    if policy not in ("strict","soft_cost") or type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:
        raise ValueError("invalid planner unknown policy or weight")
    if type(cfg["control"].setdefault("metric_arrival_autostop", False)) is not bool:
        raise ValueError("metric_arrival_autostop must be a boolean")
    for name in ("control", "limits", "planner", "ros_clock"):
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for k,v in cfg[name].items()
                   if (name, k) not in {("planner", "unknown_policy"), ("planner", "unknown_weight"),
                                        ("control", "metric_arrival_autostop"), ("planner", "metric_target_detour")}):
            raise ValueError(f"{name}: expected positive finite parameters")
    return cfg


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def transform(self, relative):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose(self.x + c * relative.x - s * relative.y,
                    self.y + s * relative.x + c * relative.y, self.yaw + relative.yaw)


def integrate(pose, velocity, dt):
    turn = velocity.yaw_rad_s * dt
    scale = math.sin(turn / 2) / (turn / 2) if abs(turn) > 1e-8 else 1.
    angle = pose.yaw + turn / 2
    c, s = math.cos(angle), math.sin(angle)
    return Pose(pose.x + dt * scale * (c * velocity.x_m_s - s * velocity.y_m_s),
                pose.y + dt * scale * (s * velocity.x_m_s + c * velocity.y_m_s),
                pose.yaw + turn)


@dataclass(frozen=True)
class Feedback:
    stamp: float
    receipt: float
    velocity: BodyVelocity
    frame: str = "base_link"
    source_stamp_ns: int | None = None


@dataclass(frozen=True)
class Reference:
    capture: float
    received: float
    path: tuple[Pose, ...]
    frame: str = "base_link"


@dataclass(frozen=True)
class Arrival:
    stamp: float
    target_confirmed: bool
    distance_m: float | None
    source: str
    target_odom: Pose | None = None


class Odometry:
    def __init__(self, gap_s):
        self.gap_s = gap_s
        self.history = deque(maxlen=512)
        self.fault = None
        self.reanchor_stamp = None

    def push(self, sample):
        if (sample.frame != "base_link" or not all(map(math.isfinite, (sample.stamp, sample.receipt)))
                or not isinstance(sample.velocity, BodyVelocity) or sample.stamp > sample.receipt):
            self.fault = "feedback_invalid_timestamp_or_frame"
            return False
        reset = False
        pose = Pose(0., 0., 0.)
        if self.history:
            previous, pose = self.history[-1]
            dt = sample.stamp - previous.stamp
            if dt <= 0:
                self.fault = "feedback_out_of_order"
                return False
            if dt > self.gap_s:
                self.history.clear()
                reset = True
                self.reanchor_stamp = sample.stamp
            else:
                before, after = previous.velocity, sample.velocity
                average = BodyVelocity((before.x_m_s + after.x_m_s) / 2,
                                       (before.y_m_s + after.y_m_s) / 2,
                                       (before.yaw_rad_s + after.yaw_rad_s) / 2)
                pose = integrate(pose, average, dt)
        self.fault = None
        self.history.append((sample, pose))
        return reset

    def at_reported_pose(self, stamp):
        # The RGB publisher serializes Unix nanoseconds through a float in seconds.
        # Accept only its sub-microsecond roundoff at the newest measured sample.
        if self.history and 0 <= stamp - self.history[-1][0].stamp <= 5e-7:
            return self.history[-1][1]
        return self.at(stamp)

    def at(self, stamp):
        samples = list(self.history)
        index = bisect_left([v.stamp for v, _ in samples], stamp)
        if index < len(samples) and samples[index][0].stamp == stamp:
            return samples[index][1]
        if not 0 < index < len(samples):
            return None
        left, a = samples[index - 1]
        right, b = samples[index]
        f = (stamp - left.stamp) / (right.stamp - left.stamp)
        return Pose(a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.yaw + f * (b.yaw - a.yaw))


def mode_age(now, stamp, ttl):
    """Check normalized monotonic stamps. A future or nonfinite stamp is invalid."""
    return math.isfinite(stamp) and stamp <= now < stamp + ttl
