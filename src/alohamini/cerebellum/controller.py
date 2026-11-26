"""Pure local controller. Call input methods and tick from one executor."""

from dataclasses import dataclass
import math

from alohamini.kinematics.base import OmniBaseKinematics
from alohamini.model import get_robot_model

from .obstacles import METRIC_SOURCES
from .planner import brake, components, Planner
from .state import integrate, mode_age, Odometry, Pose, ZERO


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str
    command: object
    valid_until: float
    arrival_status: str
    pose: Pose | None
    selected: object = None
    candidates: tuple = ()
    rejected: dict | None = None
    feedback_consumed_stamp_ns: int | None = None
    rejected_candidates: tuple = ()


class Controller:
    def __init__(self, configuration, model_name="alohamini2pro"):
        self.c = configuration["control"]
        self.limits = configuration["limits"]
        model = get_robot_model(model_name)
        self.planner = Planner(configuration, OmniBaseKinematics(model.wheel_radius_m, model.base_radius_m))
        self.odom = Odometry(self.c["max_wheel_gap_s"])
        self.reference = None
        self.grid = None
        self.previous = ZERO
        self.selected = None
        self.stop_requested = False
        self.stop_source = None
        self.last_follow_stop = None
        self.reobserving = False
        self.arrival = None
        self.still_since = None
        self.stop_since = None
        self.last_map_error = None
        self.last_reference_error = None

    def feedback(self, sample):
        if self.odom.push(sample):
            self.arrival = None
            self.grid = self.reference = None
            self.last_map_error = self.last_reference_error = "odometry_reanchored"
            self.planner.side = 0
            self.planner.detour = ()
            self.planner.metric_goal = None
            self.selected = None
            self.still_since = None
        if self.odom.fault:
            self.still_since = None

    def set_map(self, grid):
        if grid.frame != "odom" or grid.source not in METRIC_SOURCES:
            self.map_fault("obstacle_source_or_frame_invalid")
        elif self.odom.reanchor_stamp is not None and grid.stamp < self.odom.reanchor_stamp:
            self.map_fault("obstacle_observation_precedes_odometry_epoch")
        elif self.grid is None or grid.stamp > self.grid.stamp:
            self.grid, self.last_map_error = grid, None

    def map_fault(self, reason):
        self.grid, self.last_map_error = None, reason

    def set_reference(self, reference):
        if (reference.frame not in ("base_link", "odom") or not reference.path
                or not all(math.isfinite(v) for p in reference.path for v in (p.x, p.y, p.yaw))
                or not all(map(math.isfinite, (reference.capture, reference.received)))
                or reference.capture > reference.received):
            self.reference_fault("reference_invalid_geometry_frame_or_timestamp")
        elif self.reference is None or reference.capture > self.reference.capture:
            if self.odom.reanchor_stamp is not None and reference.capture < self.odom.reanchor_stamp:
                self.reference_fault("reference_precedes_odometry_epoch")
            else:
                self.reference, self.last_reference_error = reference, None
                return True
        return False

    def reference_fault(self, reason):
        self.reference, self.last_reference_error = None, reason

    def stop(self, source="model_stop", *, received=None):
        if self.c.get("continuous_follow") and source == "model_stop":
            self.reference = None
            self.last_follow_stop = received
        if not self.stop_requested:
            self.stop_requested = True
            self.stop_source = source
            self.stop_since = None
            self.reobserving = False
            self.arrival = None

    def set_arrival(self, observation):
        if (self.stop_requested or self.c.get("metric_arrival_autostop", False)) and (
                self.arrival is None or observation.stamp > self.arrival.stamp):
            self.arrival = observation

    def valid_arrival(self, now, *, measured_only):
        e = self.arrival
        return (e is not None and isinstance(e.source, str)
                and (e.source == "measured_metric" if measured_only else e.source in METRIC_SOURCES)
                and mode_age(now, e.stamp, self.c["arrival_age_s"]) and e.target_confirmed is True
                and type(e.distance_m) in (float, int) and math.isfinite(e.distance_m) and e.distance_m >= 0)

    def diagnostics(self, now):
        sample = self.odom.history[-1][0] if self.odom.history else None
        return {
            "stop_source": self.stop_source,
            "wheel_cache_stamp_ns": sample.source_stamp_ns if sample else None,
            "wheel_age_s": now - sample.stamp if sample else None,
            "wheel_receipt_age_s": now - sample.receipt if sample else None,
            "wheel_transport_age_s": sample.receipt - sample.stamp if sample else None,
            "reference_remaining_s": self.c["reference_ttl_s"] - (now - self.reference.capture) if self.reference else None,
            "map_age_s": now - self.grid.stamp if self.grid else None,
            "map_source": self.grid.source if self.grid else None,
        }

    def tick(self, timing):
        now = timing.now
        dt = min(timing.elapsed_s, 1 / self.c["frequency_hz"])
        deadline = now + self.c["command_ttl_s"]
        pose = None
        consumed = None

        def result(state, reason, command=ZERO, *, selected=None, candidates=(), rejected=None, expires=deadline):
            self.previous = command
            self.selected = selected
            self.reobserving = state == "REOBSERVE"
            arrival = "CONFIRMED" if state == "ARRIVED" else "UNKNOWN"
            return Decision(state, reason, command, expires, arrival, pose, selected, candidates,
                            dict(rejected) if rejected is not None else None, consumed,
                            tuple(rejected.details) if rejected is not None else ())

        if (not self.stop_requested and self.c.get("metric_arrival_autostop", False)
                and self.valid_arrival(now, measured_only=True)
                and self.arrival.distance_m <= self.c["arrival_distance_m"]):
            self.stop("metric_arrival_autostop")
        if self.stop_requested and self.stop_since is None:
            self.stop_since = now
            self.still_since = None
        if self.odom.fault or not self.odom.history:
            return result("SAFETY_HOLD", self.odom.fault or "feedback_missing", expires=now)
        sample, pose = self.odom.history[-1]
        if not (mode_age(now, sample.stamp, self.c["feedback_age_s"])
                and mode_age(now, sample.receipt, self.c["feedback_age_s"])):
            self.still_since = None
            return result("SAFETY_HOLD", "feedback_stale", expires=now)
        age = now - sample.stamp
        speed = sample.velocity
        consumed = sample.source_stamp_ns
        pose = integrate(pose, speed, age)
        speed_padding = age * (math.hypot(*components(speed)[:2]) + self.planner.radius * abs(speed.yaw_rad_s))
        deadline = min(deadline, sample.stamp + self.c["feedback_age_s"])
        moving = (math.hypot(*components(speed)[:2]) > self.c["still_linear_m_s"]
                  or abs(speed.yaw_rad_s) > self.c["still_angular_rad_s"])
        map_ok = self.grid is not None and mode_age(now, self.grid.stamp, self.c["map_ttl_s"])

        def braking(reason):
            if map_ok and moving:
                _, _, fails = self.planner.rollout(self.grid, pose, speed, horizon=0., initial_error_m=speed_padding)
                limited = brake(speed, dt, self.limits)
                if fails is None and self.planner.within_limits(limited):
                    return result("BRAKING", reason, limited,
                                  expires=min(deadline, self.grid.stamp + self.c["map_ttl_s"]))
            return result("SAFETY_HOLD", reason, expires=now)

        if self.stop_requested:
            if moving:
                self.still_since = None
                return braking(self.stop_source + "_braking")
            if sample.stamp < self.stop_since:
                return result("VERIFYING_ARRIVAL", "waiting_for_post_stop_feedback", expires=now)
            if self.still_since is None:
                self.still_since = sample.stamp
                return result("STOPPED", "first_still_feedback", expires=now)
            if sample.stamp - self.still_since < self.c["stop_confirm_s"]:
                return result("VERIFYING_ARRIVAL", "confirming_measured_stillness", expires=now)
            ref = self.reference
            follow = self.c.get("continuous_follow") and self.stop_source == "model_stop"
            if follow:
                # Only an exposure taken after confirmed standstill and the latest
                # debounced STOP can release this latch, including for odom paths.
                cutoff = max(self.stop_since, self.still_since + self.c["stop_confirm_s"])
                if self.last_follow_stop is not None:
                    cutoff = max(cutoff, self.last_follow_stop)
                waiting = "waiting_for_follow_reference"
            else:
                evidence = self.arrival
                if (not self.valid_arrival(now, measured_only=self.stop_source == "metric_arrival_autostop")
                        or evidence.stamp < self.stop_since):
                    return result("REOBSERVE", "arrival_evidence_missing_or_invalid", expires=now)
                if evidence.distance_m <= self.c["arrival_distance_m"]:
                    return result("ARRIVED", "target_confirmed_within_distance_and_still", expires=now)
                cutoff = self.stop_since
                waiting = "target_outside_arrival_distance"
            if not ((follow or self.reobserving) and map_ok and ref and ref.capture > cutoff
                    and mode_age(now, ref.capture, self.c["reference_ttl_s"]) and ref.received <= now
                    and ((not follow and ref.frame == "odom") or self.odom.at(ref.capture) is not None)):
                return result("REOBSERVE", waiting, expires=now)
            self.stop_requested = False
            self.stop_source = None
            self.stop_since = self.still_since = self.arrival = None

        if not map_ok:
            return result("SAFETY_HOLD", self.last_map_error or ("map_missing" if self.grid is None else "map_stale"), expires=now)
        ref = self.reference
        if ref is None:
            return braking(self.last_reference_error or "reference_missing")
        if not (mode_age(now, ref.capture, self.c["reference_ttl_s"]) and ref.received <= now):
            return braking("reference_expired_or_future")
        if ref.frame == "base_link":
            capture = self.odom.at(ref.capture)
            if capture is None:
                return braking("reference_capture_pose_unavailable")
            path = (capture,) + tuple(capture.transform(p) for p in ref.path)
        else:
            path = ref.path
        # A near-expired reference may be valid now but expire inside rollout.
        # Leave the publisher's independent deadline check in force.
        available=min(deadline,ref.capture+self.c["reference_ttl_s"],
                      self.grid.stamp+self.c["map_ttl_s"])-now
        if available<self.c.get("min_plan_budget_s",0.):
            return braking("insufficient_time_for_rollout")
        chosen, rejected, reason, alternatives = self.planner.choose(
            now, dt, pose, speed, self.previous, path, self.grid, start_padding_m=speed_padding,
            previous_path=self.selected.path if self.selected else None,
            target_odom=self.arrival.target_odom if self.valid_arrival(now, measured_only=True) else None)
        if chosen is None:
            stopped = braking(reason)
            return result(stopped.state, stopped.reason, stopped.command, expires=stopped.valid_until,
                          candidates=alternatives, rejected=rejected)
        return result("TRACKING", reason, chosen.command, selected=chosen, candidates=alternatives,
                      rejected=rejected, expires=min(deadline, ref.capture + self.c["reference_ttl_s"],
                                                     self.grid.stamp + self.c["map_ttl_s"]))
