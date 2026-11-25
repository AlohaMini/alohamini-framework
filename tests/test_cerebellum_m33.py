import json
from pathlib import Path
import tempfile
import unittest

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Tick
from alohamini.cerebellum.state import Arrival, Feedback, Pose, Reference, ZERO, load_config
from alohamini.schema import BodyVelocity


class MetricArrivalTest(unittest.TestCase):
    def core(self, enabled=True, rectangles=()):
        cfg = load_config()
        cfg["control"]["metric_arrival_autostop"] = enabled
        c = Controller(cfg)
        c.set_map(synthetic_grid(1., rectangles=rectangles, resolution=.01 if rectangles else .1))
        c.set_reference(Reference(1., 1., (Pose(0, 0, 0), Pose(1, 0, 0)), "odom"))
        return c

    def tick(self, c, t, v=0.):
        c.feedback(Feedback(t, t, BodyVelocity(v, 0., 0.)))
        return c.tick(Tick(t, 0, 0, .05))

    def test_default_and_config_types(self):
        cfg = load_config()
        self.assertIs(cfg["control"]["metric_arrival_autostop"], False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for invalid in (1, "true", None):
                cfg["control"]["metric_arrival_autostop"] = invalid
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_config(path)
        c = self.core(False)
        c.set_arrival(Arrival(1., True, .1, "measured_metric"))
        self.assertEqual(self.tick(c, 1.).state, "TRACKING")

    def test_far_stale_future_unconfirmed_wrong_source_invalid_distance(self):
        for t, confirmed, distance, source in (
            (1., True, .201, "measured_metric"), (.4, True, .1, "measured_metric"),
            (1.2, True, .1, "measured_metric"), (1., False, .1, "measured_metric"),
            (1., True, .1, "manual_metric"), (1., True, -1., "measured_metric"),
            (1., True, float("nan"), "measured_metric")):
            with self.subTest(t=t, confirmed=confirmed, distance=distance, source=source):
                c = self.core()
                c.set_arrival(Arrival(t, confirmed, distance, source))
                self.assertEqual(self.tick(c, 1.).state, "TRACKING")

    def test_near_stops_then_requires_post_stop_exposure_and_stillness(self):
        c = self.core()
        c.set_arrival(Arrival(.99, True, .20, "measured_metric"))
        d = self.tick(c, 1., .12)
        self.assertEqual((d.state, d.reason), ("BRAKING", "metric_arrival_autostop_braking"))
        self.assertGreater(d.command.x_m_s, 0.)
        self.assertLess(d.command.x_m_s, .12)
        self.assertEqual(self.tick(c, 1.1).state, "STOPPED")
        self.assertEqual(self.tick(c, 1.15).state, "VERIFYING_ARRIVAL")
        self.assertEqual(self.tick(c, 1.3).state, "REOBSERVE")
        c.set_arrival(Arrival(1.32, True, .18, "manual_metric"))
        self.assertEqual(self.tick(c, 1.32).state, "REOBSERVE")
        c.set_arrival(Arrival(1.35, True, .18, "measured_metric"))
        self.assertEqual(self.tick(c, 1.35).state, "ARRIVED")

    def test_occupied_only_in_braking_tail_forces_zero(self):
        c = self.core(rectangles=((.25, -.04, .26, .04),))
        c.set_arrival(Arrival(1., True, .18, "measured_metric"))
        d = self.tick(c, 1., .2)
        self.assertEqual((d.state, d.command), ("SAFETY_HOLD", ZERO))

