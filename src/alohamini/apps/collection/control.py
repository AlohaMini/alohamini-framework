"""Control state with separate policy-phase, human-takeover and scoring events."""

from dataclasses import dataclass

from alohamini.schema import CommandIdentity


@dataclass
class ControlState:
    mode: str = "idle"
    generation: int = 0
    quit: bool = False
    paused_mode: str = "policy"

    @property
    def active(self):
        return self.mode in ("policy", "human", "paused")

    def start(self):
        if self.mode != "idle":
            raise ValueError("Episode already active")
        self.mode = "policy"
        self.generation += 1

    def key(self, key):
        if key in ("q", "esc"):
            self.quit = True
        elif key == "h" and self.active:
            self.mode = "policy" if self.mode == "human" else "human"
        elif key == " " and self.active:
            if self.mode == "paused":
                self.mode = self.paused_mode
            else:
                self.pause()
                return
        elif key in ("right", "t") and self.active:
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
