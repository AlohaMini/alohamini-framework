"""Hysteresis on fresh policy predictions. No model or device dependency."""

import math


class GateLatch:
    def __init__(self, enter_threshold, exit_threshold, confirmations):
        if not 0 <= exit_threshold < enter_threshold <= 1:
            raise ValueError("Gate thresholds must satisfy 0 <= exit < enter <= 1")
        if type(confirmations) is not int or confirmations < 1:
            raise ValueError("Gate confirmations must be a positive integer")
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.confirmations = confirmations
        self.reset()

    def reset(self):
        self.active = False
        self.votes = 0

    def update(self, probability):
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid gate probability")
        switch = (
            probability <= self.exit_threshold
            if self.active
            else probability >= self.enter_threshold
        )
        self.votes = self.votes + 1 if switch else 0
        if self.votes == self.confirmations:
            self.active = not self.active
            self.votes = 0
        return self.active


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
