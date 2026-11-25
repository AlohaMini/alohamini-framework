from types import SimpleNamespace
import unittest

from alohamini.cerebellum.controller import Decision
from alohamini.cerebellum.ros_adapter import ClockMonitor, require_dry_run, validate_publish_deadline, yaw_from_quaternion
from alohamini.cerebellum.state import load_config, ZERO
from alohamini.schema import BodyVelocity


class AdapterTest(unittest.TestCase):
    def test_normalizes_unix_stamp(self):
        wall_ns = 1_795_000_000_000_000_000
        clock = ClockMonitor(load_config()["ros_clock"])
        self.assertAlmostEqual(clock.convert(wall_ns - 20_000_000, wall_ns, 300.), 299.98)
        with self.assertRaises(ValueError):
            clock.convert(100_000_000, wall_ns, 300.)
        with self.assertRaises(ValueError):
            clock.convert(wall_ns + 1, wall_ns, 300.)
        with self.assertRaisesRegex(ValueError, "wall_clock_step"):
            clock.convert(wall_ns, wall_ns + 100_000_000, 300.)
        with self.assertRaisesRegex(ValueError, "wall_clock_step"):
            clock.convert(wall_ns, wall_ns, 300.)

    def test_only_dry_run_mode(self):
        require_dry_run(True)
        with self.assertRaises(ValueError):
            require_dry_run(False)

    def test_bad_quaternion(self):
        self.assertEqual(yaw_from_quaternion(SimpleNamespace(x=0, y=0, z=0, w=1)), 0.)
        with self.assertRaises(ValueError):
            yaw_from_quaternion(SimpleNamespace(x=0, y=0, z=0, w=0))

    def test_refuses_already_expired_result_at_publish_time(self):
        motion = BodyVelocity(.1, 0., 0.)
        controller = SimpleNamespace(previous=motion, selected=object())
        plan = Decision("TRACKING", "best_score", motion, 1.1, "UNKNOWN", None)
        self.assertIs(validate_publish_deadline(controller, plan, 1.09), plan)
        stopped = validate_publish_deadline(controller, plan, 1.1)
        self.assertEqual(stopped.reason, "expired_during_control_tick")
        self.assertEqual(stopped.command, ZERO)
        self.assertIsNone(controller.selected)
