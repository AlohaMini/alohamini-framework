import unittest
from types import SimpleNamespace

from lightnav_clearance import ClearanceGate, Evidence, GuardedPrediction


class ClearanceTests(unittest.TestCase):
    def test_block_immediately_and_release_after_sustained_clear(self):
        gate = ClearanceGate()
        stamps_ms = [1000, 1250, 1500, 1750, 2000, 2150, 2300, 2500, 2500, 3100]
        observations = [True, True, True, False, True, True, True, True, True, True]
        actual = [gate.update(clear, stamp * 1_000_000)
                  for clear, stamp in zip(observations, stamps_ms)]
        self.assertEqual(actual, [False, False, True, False, False, False, False, True, False, False])

    def test_lost_visibility_also_suppresses_yaw(self):
        guard = GuardedPrediction.__new__(GuardedPrediction)
        guard.gate = ClearanceGate()
        for reason, yaw in [("local_obstruction", .4), ("poor_visibility", 0.)]:
            with self.subTest(reason=reason):
                cmd = SimpleNamespace(linear=SimpleNamespace(x=.2, y=.1),
                                      angular=SimpleNamespace(z=.4))
                guard.constrain(cmd, Evidence(False, reason, 0., 0, 0))
                self.assertEqual((cmd.linear.x, cmd.linear.y, cmd.angular.z), (0., 0., yaw))
