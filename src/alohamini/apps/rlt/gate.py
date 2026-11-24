"""Measured gripper release detector for human-in-the-loop collection."""

import math


class GripperRelease:
    """Detect a measured closed-to-open transition within one placement phase."""

    def __init__(self, closed, opened, confirmations):
        self.closed = closed
        self.opened = opened
        self.confirmations = confirmations
        self.last_sample = None
        self.last_time = None
        self.block_reentry = False
        self.new_phase()

    def new_phase(self):
        self.armed = False
        self.votes = 0

    def observe(self, opening, sample, now, max_gap, *, critical):
        if sample is None or not math.isfinite(opening) or not 0 <= opening <= 100:
            self.votes = 0
            return False
        if self.last_sample is not None and sample <= self.last_sample:
            return False
        if self.last_time is not None and now - self.last_time > max_gap:
            self.votes = 0
        self.last_sample, self.last_time = sample, now
        if opening <= self.closed:
            self.block_reentry = False
            self.armed = critical
        self.votes = self.votes + 1 if critical and self.armed and opening >= self.opened else 0
        if self.votes < self.confirmations:
            return False
        self.new_phase()
        self.block_reentry = True
        return True
