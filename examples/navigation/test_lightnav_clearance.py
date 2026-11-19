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
        for reason, yaw in [("local_obstruction", 0.), ("poor_visibility", 0.)]:
            with self.subTest(reason=reason):
                cmd = SimpleNamespace(linear=SimpleNamespace(x=.2, y=.1),
                                      angular=SimpleNamespace(z=.4))
                guard.constrain(cmd, Evidence(False, reason, 0., 0, 0))
                self.assertEqual((cmd.linear.x, cmd.linear.y, cmd.angular.z), (0., 0., yaw))

    def test_arc_persists_when_policy_reverts_to_straight(self):
        guard = GuardedPrediction.__new__(GuardedPrediction)
        guard.gate = ClearanceGate()
        guard.turn, guard.encounter_ns = 0, None
        evidence = Evidence(False, "local_obstruction", .99, 2, 67, True, True, False, 1)
        for stamp, yaw, expected in [(1000, -.4, "arc"), (1250, 0., "arc"), (2000, 0., "turn")]:
            guard.gate.update(False, stamp * 1_000_000)
            cmd = SimpleNamespace(linear=SimpleNamespace(x=.2, y=.05),
                                  angular=SimpleNamespace(z=yaw))
            self.assertEqual(guard.constrain(cmd, evidence, max_yaw=.3), expected)
            self.assertEqual(cmd.angular.z, -.3)
            self.assertEqual(cmd.linear.x, .045 if expected == "arc" else 0.)
            self.assertEqual(cmd.linear.y, 0.)

    def test_near_obstruction_only_turns_toward_clear_side(self):
        guard = GuardedPrediction.__new__(GuardedPrediction)
        guard.gate = ClearanceGate()
        guard.turn, guard.encounter_ns = 0, None
        evidence = Evidence(False, "nonfloor_or_uncertain", .8, 200, 67, True, False, True, -1)
        cmd = SimpleNamespace(linear=SimpleNamespace(x=.2, y=.05), angular=SimpleNamespace(z=-.5))
        self.assertEqual(guard.constrain(cmd, evidence), "turn")
        self.assertEqual((cmd.linear.x, cmd.linear.y, cmd.angular.z), (0., 0., .5))
