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
