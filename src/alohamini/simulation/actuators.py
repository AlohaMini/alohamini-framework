"""Engine-independent SI actuator profiles and causal servo references."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np


def gravity_position_targets(target, gravity, kp, lower, upper, *, calibrated_gain=None):
    """Bounded gravity position feedforward in named model coordinates.

    Positive G requires positive actuator effort. For tau=kp*(goal-q), the
    corresponding offset is +G/kp. A measured inverse stiffness can replace
    1/kp. Targets and gains use radians and N·m for revolute joints.
    """
    gain = 1 / np.asarray(kp) if calibrated_gain is None else np.asarray(calibrated_gain)
    return np.clip(np.asarray(target) + np.asarray(gravity) * gain, lower, upper)


def advance_motion(position, velocity, goal, max_velocity, acceleration, timestep):
    """Integrate a minimum-time trapezoid, including switches within a step.

    A changed target inside the stopping distance may be passed while braking.
    Clamping position there would introduce an unbounded acceleration impulse.
    """
    remaining = timestep
    while remaining > 1e-12:
        error = goal - position
        if abs(error) < 1e-14 and abs(velocity) < 1e-12:
            return float(goal), 0.0
        direction = (
            math.copysign(1.0, error) if abs(error) > 1e-14 else -math.copysign(1.0, velocity)
        )
        speed = direction * velocity
        distance = abs(error)
        if speed < 0 or speed * speed / (2 * acceleration) > distance + 1e-14:
            duration = min(remaining, abs(velocity) / acceleration)
            signed_acceleration = -math.copysign(acceleration, velocity)
        else:
            peak = min(max_velocity, math.sqrt(acceleration * distance + 0.5 * speed * speed))
            acceleration_time = max(0.0, (peak - speed) / acceleration)
            cruise_distance = distance - (2 * peak * peak - speed * speed) / (2 * acceleration)
            if acceleration_time > 1e-12:
                duration = min(remaining, acceleration_time)
                signed_acceleration = direction * acceleration
            elif cruise_distance > 1e-14:
                duration = min(remaining, cruise_distance / peak)
                signed_acceleration = 0.0
            else:
                duration = min(remaining, speed / acceleration)
                signed_acceleration = -direction * acceleration
        if duration <= 1e-14:
            return float(goal), 0.0
        position += velocity * duration + 0.5 * signed_acceleration * duration * duration
        velocity += signed_acceleration * duration
        remaining -= duration
    return float(position), float(velocity)


@dataclass(frozen=True)
class JointActuator:
    kp_Nm_per_rad: float
    kd_Nm_s_per_rad: float
    friction_Nm: float
    peak_effort_Nm: float
    velocity_rad_s: float
    acceleration_rad_s2: float
    delay_s: float = 0.0
    reference_velocity_gain_Nm_s_per_rad: float = 0.0
    gravity_position_gain_rad_per_Nm: float | None = None
    ki_Nm_per_rad_s: float = 0.0
    integral_effort_limit_Nm: float = 0.0
    integral_deadband_rad: float = 0.0
    integral_period_s: float = 0.0

    def __post_init__(self):
        positive = {"kp_Nm_per_rad", "peak_effort_Nm", "velocity_rad_s", "acceleration_rad_s2"}
        for name, value in vars(self).items():
            if name == "gravity_position_gain_rad_per_Nm" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                or (name in positive and value == 0)
            ):
                raise ValueError(f"Invalid actuator parameter: {name}")
        if self.ki_Nm_per_rad_s and not 0 < self.integral_effort_limit_Nm <= self.peak_effort_Nm:
            raise ValueError("Integral action requires a bounded effort within the actuator peak")


class ServoIntegral:
    """Clamped integral effort with a sampled update and saturation recovery."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.effort = 0.0
        self.remainder_s = 0.0

    def advance(self, error, timestep, effort_without_integral, *, blocked=False):
        p = self.parameters
        self.remainder_s += timestep
        period = p.integral_period_s or timestep
        while self.remainder_s + 1e-12 >= period:
            effective = math.copysign(max(0.0, abs(error) - p.integral_deadband_rad), error)
            delta = p.ki_Nm_per_rad_s * effective * period
            requested = effort_without_integral + self.effort
            saturated_outward = abs(requested) >= p.peak_effort_Nm and requested * delta > 0
            if not saturated_outward and not (blocked and self.effort * delta >= 0):
                self.effort = float(
                    np.clip(
                        self.effort + delta, -p.integral_effort_limit_Nm, p.integral_effort_limit_Nm
                    )
                )
            self.remainder_s = max(0.0, self.remainder_s - period)
        return self.effort

    def state(self):
        return {"effort": self.effort, "remainder_s": self.remainder_s}

    def restore(self, state):
        effort, remainder = float(state["effort"]), float(state["remainder_s"])
        if (
            not math.isfinite(effort)
            or not math.isfinite(remainder)
            or abs(effort) > self.parameters.integral_effort_limit_Nm
            or remainder < 0
            or (
                self.parameters.integral_period_s and remainder >= self.parameters.integral_period_s
            )
        ):
            raise ValueError("Invalid integral controller state")
        self.effort, self.remainder_s = effort, remainder


@dataclass(frozen=True)
class ActuatorProfile:
    robot_model: str
    joints: Mapping
    provenance: Mapping

    def __post_init__(self):
        if not isinstance(self.robot_model, str) or not self.robot_model:
            raise ValueError("Actuator profile requires a robot_model")
        if not isinstance(self.joints, Mapping) or not self.joints:
            raise ValueError("Actuator profile requires named joints")
        joints = {}
        for name, values in self.joints.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Actuator joint names must be nonempty strings")
            if not isinstance(values, (JointActuator, Mapping)):
                raise ValueError(f"Actuator parameters must be a mapping: {name}")
            try:
                joints[name] = (
                    values if isinstance(values, JointActuator) else JointActuator(**values)
                )
            except TypeError as exc:
                raise ValueError(f"Invalid actuator fields for {name}: {exc}") from exc
        object.__setattr__(self, "joints", MappingProxyType(joints))
        # Keep caller mutations out of the compiled model's provenance.
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Actuator provenance must be a mapping")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(json.loads(json.dumps(dict(self.provenance), allow_nan=False))),
        )

    @classmethod
    def load(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).expanduser().read_text())
        if not isinstance(value, Mapping) or value.get("schema") != "alohamini-actuators-v1":
            raise ValueError("Expected alohamini-actuators-v1 actuator profile")
        if set(value) != {"schema", "robot_model", "joints", "provenance"}:
            raise ValueError("Actuator profile requires schema, robot_model, joints and provenance")
        return cls(value["robot_model"], value["joints"], value["provenance"])

    def to_dict(self):
        return {
            "schema": "alohamini-actuators-v1",
            "robot_model": self.robot_model,
            "joints": {name: asdict(joint) for name, joint in self.joints.items()},
            "provenance": json.loads(json.dumps(dict(self.provenance))),
        }


class ServoReference:
    """Delayed position goals followed by a trapezoidal servo reference model.

    The state is separate from MuJoCo physics and must accompany saved episodes.
    All timestamps are simulation time, including commands delayed past a reset.
    """

    def __init__(self, parameters, position, time_s):
        self.parameters = parameters
        self.position = float(position)
        self.velocity = 0.0
        self.goal = float(position)
        self.time_s = float(time_s)
        self.command = float(position)
        self.pending = deque()

    def advance(self, command, time_s, timestep):
        if not math.isclose(time_s, self.time_s, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Servo reference time differs from physics; restore controller state")
        if command != self.command:
            self.pending.append((time_s + self.parameters.delay_s, float(command)))
            self.command = float(command)
        while self.pending and self.pending[0][0] <= time_s + 1e-12:
            _, self.goal = self.pending.popleft()
        position, velocity = advance_motion(
            self.position,
            self.velocity,
            self.goal,
            self.parameters.velocity_rad_s,
            self.parameters.acceleration_rad_s2,
            timestep,
        )
        self.position, self.velocity = position, velocity
        self.time_s = float(time_s + timestep)
        return position, velocity

    def state(self):
        return {
            "position": self.position,
            "velocity": self.velocity,
            "goal": self.goal,
            "command": self.command,
            "time_s": self.time_s,
            "pending": list(self.pending),
        }

    def restore(self, state):
        numeric = [state[k] for k in ("position", "velocity", "goal", "command", "time_s")]
        pending = np.asarray(state["pending"], dtype=float).reshape(-1, 2)
        if not np.isfinite(numeric).all() or not np.isfinite(pending).all():
            raise ValueError("Servo reference state must be finite")
        if len(pending) and (np.diff(pending[:, 0]) < 0).any():
            raise ValueError("Delayed servo commands must be ordered by time")
        self.position, self.velocity, self.goal, self.command, self.time_s = map(float, numeric)
        self.pending = deque(map(tuple, pending.tolist()))
