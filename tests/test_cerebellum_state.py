import math
import unittest

from alohamini.cerebellum.scheduler import Periodic
from alohamini.cerebellum.state import Feedback, Odometry, Pose, integrate
from alohamini.schema import BodyVelocity


class StateTest(unittest.TestCase):
    def test_periodic_never_bursts_after_delay(self):
        loop = Periodic(20)
        self.assertEqual(loop.poll(3).missed_periods, 0)
        self.assertIsNone(loop.poll(3.02))
        late = loop.poll(3.16)
        self.assertEqual(late.missed_periods, 2)
        self.assertAlmostEqual(late.jitter_s, .11)
        self.assertIsNone(loop.poll(3.16))
        self.assertIsNotNone(loop.poll(3.20))

    def test_odometry_interpolates_capture_in_same_epoch(self):
        odom = Odometry(.25)
        sideways = BodyVelocity(0, .2, 0)
        odom.push(Feedback(1, 1.02, sideways))
        odom.push(Feedback(1.2, 1.22, sideways))
        self.assertAlmostEqual(odom.at(1.1).y, .02)
        self.assertIsNone(odom.at(1.3))
        self.assertTrue(odom.push(Feedback(2., 2., sideways)))
        self.assertIsNone(odom.at(1.1))
        odom.push(Feedback(1., 2., sideways))
        self.assertEqual(odom.fault, "feedback_out_of_order")

    def test_lateral_in_body_rotates_into_odom(self):
        end = integrate(Pose(0, 0, math.pi / 2), BodyVelocity(0, .2, 0), 1)
        self.assertAlmostEqual(end.x, -.2)
        self.assertAlmostEqual(end.y, 0)
