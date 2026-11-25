import math
import unittest

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.obstacles import Grid
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Periodic, Tick
from alohamini.cerebellum.state import Arrival, Feedback, Pose, Reference, ZERO, load_config
from alohamini.schema import BodyVelocity


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.core = Controller(self.cfg)

    def test_per_tick_reference_ttl_and_command_deadline(self):
        loop = Periodic(20)
        self.core.set_map(synthetic_grid(1.))
        self.core.set_reference(Reference(1., 1., (Pose(1, 0, 0),)))
        decisions = []
        for i in range(11):
            time = 1 + i * .05
            self.core.feedback(Feedback(time, time, ZERO))
            decisions.append(self.core.tick(loop.poll(time)))
        self.assertTrue(all(d.state == "TRACKING" for d in decisions[:9]))
        self.assertTrue(all(d.valid_until <= 1.45 for d in decisions[:9]))
        self.assertTrue(all(d.command == ZERO for d in decisions[9:]))
        self.assertEqual(decisions[9].reason, "reference_expired_or_future")

    def test_narrow_slot_rejects_body_even_if_center_is_free(self):
        grid = synthetic_grid(1., rectangles=((.3, .2, 1., .6), (.3, -.6, 1., -.2)), resolution=.05)
        _, _, reason = self.core.planner.rollout(grid, Pose(0, 0, 0), BodyVelocity(.25, 0, 0))
        self.assertEqual(reason, "occupied_footprint")

    def test_rotating_corner_and_lateral_unknown(self):
        grid = synthetic_grid(1., rectangles=((.20, .20, .25, .25),), resolution=.05)
        _, _, reason = self.core.planner.rollout(grid, Pose(0, 0, 0), BodyVelocity(0, 0, .4))
        self.assertEqual(reason, "occupied_footprint")
        grid = synthetic_grid(1., unknown=((-1., .25, 2., 1.),), resolution=.05)
        _, _, reason = self.core.planner.rollout(grid, Pose(0, 0, 0), BodyVelocity(0, .15, 0))
        self.assertEqual(reason, "unknown_footprint")

    def test_omni_candidates_and_no_lateral_into_unknown(self):
        c = self.core.planner
        route = (Pose(0, 0, 0), Pose(.02, .6, 0))
        clear = synthetic_grid(0., rectangles=((.30,-.10,.45,.10),), resolution=.05)
        winner, _, _, candidates = c.choose(1., .05, Pose(0, 0, 0), ZERO, ZERO, route, clear)
        self.assertGreater(winner.command.y_m_s, 0)
        self.assertTrue(any(u.command.y_m_s < 0 for u in candidates))
        hidden = synthetic_grid(0., unknown=((-1., .23, 2., 1.),), resolution=.01)
        winner, rejected, _, options = c.choose(1.1, .05, Pose(0, 0, 0), ZERO, ZERO, route, hidden)
        self.assertIn("unknown_footprint", rejected)
        self.assertFalse(any(u.command.y_m_s > .025 for u in options))

    def test_direction_hysteresis_and_infeasible_side(self):
        c = self.core.planner
        origin = Pose(0, 0, 0)
        clear = synthetic_grid(0., resolution=.05)
        observed = []
        for i in range(6):
            route = (origin, Pose(.2, .05 * (-1)**i, 0))
            winner, _, _, _ = c.choose(1. + i*.05, .05, origin, ZERO, ZERO, route, clear)
            observed.append(winner.side)
        self.assertEqual(len(set(observed)), 1)
        c.side = 1
        blocked_left = synthetic_grid(1., rectangles=((-1., .22, 2., .6),), resolution=.01)
        winner, reasons, why, _ = c.choose(1.31, .05, origin, ZERO, ZERO, (origin, Pose(.7,-.4,0)), blocked_left)
        self.assertIn("occupied_footprint", reasons)
        self.assertTrue(winner is None or winner.side != 1, why)

    def test_stop_requires_stillness_and_independent_metric_target(self):
        c = self.core
        c.set_map(synthetic_grid(1.))
        c.stop()
        c.feedback(Feedback(1., 1., BodyVelocity(.1, 0, 0)))
        self.assertEqual(c.tick(Tick(1., 0., 0, .05)).state, "BRAKING")
        for t, expected in ((1.1, "STOPPED"), (1.2, "VERIFYING_ARRIVAL"), (1.3, "REOBSERVE")):
            c.feedback(Feedback(t, t, ZERO))
            self.assertEqual(c.tick(Tick(t, 0., 0, .05)).state, expected)
        for t, observation, expected in (
            (1.35, Arrival(1.35, True, .7, "manual_metric"), "REOBSERVE"),
            (1.4, Arrival(1.4, False, .1, "manual_metric"), "REOBSERVE"),
            (1.45, Arrival(1.45, True, .1, "manual_metric"), "ARRIVED"),
        ):
            c.set_arrival(observation)
            c.feedback(Feedback(t, t, ZERO))
            self.assertEqual(c.tick(Tick(t, 0., 0, .05)).state, expected)

    def test_invalid_frame_nan_unknown_map_and_feedback_safe_fail(self):
        c = self.core
        c.feedback(Feedback(1., 1., ZERO))
        with self.assertRaises(ValueError):
            BodyVelocity(math.nan, 0, 0)
        c.set_map(Grid(1., "uncalibrated_rgb", "odom", .1, (0, 0), {}))
        self.assertEqual(c.tick(Tick(1.,0.,0,.05)).reason, "obstacle_source_or_frame_invalid")
        c.set_map(synthetic_grid(1.))
        c.set_reference(Reference(1., 1., (Pose(math.nan, 0, 0),), "camera"))
        self.assertEqual(c.tick(Tick(1.,0.,0,.05)).command, ZERO)
        c.feedback(Feedback(1.1, 1.1, ZERO, "camera"))
        self.assertEqual(c.tick(Tick(1.1,0.,0,.05)).reason, "feedback_invalid_timestamp_or_frame")
        self.assertEqual(c.tick(Tick(1.3,0.,0,.05)).command, ZERO)
