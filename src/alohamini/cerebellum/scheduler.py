"""Single-owner scheduling; callers supply a steady clock."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Tick:
    now: float
    jitter_s: float
    missed_periods: int
    elapsed_s: float


class Periodic:
    def __init__(self, hz):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("control frequency must be positive")
        self.period = 1 / hz
        self.origin = None
        self.previous = None
        self.index = 0

    def poll(self, now):
        if not math.isfinite(now) or (self.previous is not None and now < self.previous):
            raise ValueError("expected a monotonic clock")
        if self.origin is None:
            self.origin = now
        expected = self.origin + self.index * self.period
        if now < expected - 1e-9:
            return None
        lag = max(0., now - expected)
        missed = int((lag + 1e-9) / self.period)
        elapsed = self.period if self.previous is None else now - self.previous
        self.previous = now
        self.index += missed + 1
        return Tick(now, lag, missed, elapsed)


    def timer_tick(self, now, next_deadline):
        """Consume one native timer firing; the middleware owns the deadline grid."""
        due = next_deadline - self.period
        if not all(map(math.isfinite, (now, due))) or (self.previous is not None and now < self.previous):
            raise ValueError("invalid steady timer timestamp")
        if self.origin is None:
            self.origin = due
        slot = round((due - self.origin) / self.period)
        missed = max(0, slot - self.index)
        elapsed = self.period if self.previous is None else now - self.previous
        expected = self.origin + self.index * self.period
        self.previous = now
        self.index = slot + 1
        return Tick(now, now - expected, missed, elapsed)
