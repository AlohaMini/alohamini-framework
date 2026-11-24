"""Control state with separate policy-phase, human-takeover and scoring events."""

from dataclasses import dataclass

from alohamini.schema import CommandIdentity


@dataclass
class ControlState:
    mode: str = "idle"
    critical: bool = False
    generation: int = 0
    quit: bool = False
    paused_mode: str = "vla"

    @property
    def active(self):
        return self.mode in ("vla", "human", "paused")

    def start(self):
        if self.mode != "idle":
            raise ValueError("Episode already active")
        self.critical = False
        self.mode = "vla"
        self.generation += 1

    def key(self, key):
        if key in ("q", "esc"):
            self.quit = True
        elif key == "c" and self.active:
            self.critical = not self.critical
        elif key == "h" and self.active:
            self.mode = "vla" if self.mode == "human" else "human"
        elif key == " " and self.active:
            if self.mode == "paused":
                self.mode = self.paused_mode
            else:
                self.pause()
                return
        elif key == "right" and self.active:
            self.end_episode()
            return
        else:
            return
        self.generation += 1

    def pause(self):
        if self.active and self.mode != "paused":
            self.paused_mode = self.mode
            self.mode = "paused"
            self.generation += 1

    def end_episode(self):
        if not self.active:
            raise ValueError("No active episode")
        self.mode = "awaiting_reward"
        self.generation += 1

    def submit_reward(self, reward):
        if self.mode != "awaiting_reward":
            raise ValueError("End the episode before submitting its reward")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Reward must be the integer 0 or 1")
        self.mode = "idle"
        self.generation += 1
        return reward


@dataclass
class ReturnToStart:
    origin: dict[str, float]
    target: dict[str, float]
    started_s: float
    duration_s: float
    fence: CommandIdentity | None = None
    outcome: dict | None = None
    stable_since_s: float | None = None

    def targets(self, now):
        progress = min(1.0, max(0.0, (now - self.started_s) / self.duration_s))
        fraction = progress * progress * (3 - 2 * progress)
        return {
            name: value + fraction * (self.target[name] - value)
            for name, value in self.origin.items()
        }


class RelativeLeader:
    """Latch both sides on each explicit takeover, in calibrated Host units."""

    def __init__(self, names, bounds):
        self.names, self.bounds = tuple(names), bounds
        self.origin = None

    def latch(self, leader, follower):
        self.origin = (
            {n: float(leader[n]) for n in self.names},
            {n: float(follower[n]) for n in self.names},
        )

    def targets(self, leader):
        import math

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
