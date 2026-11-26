import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from alohamini.cerebellum import cli
from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.ros_adapter import actuator_enabled
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Tick
from alohamini.cerebellum.state import Feedback, Pose, Reference, ZERO, load_config
from alohamini.schema import BodyVelocity

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "evidence/m4-follow-shared.json"
with patch.object(sys, "path", [str(ROOT / "examples/navigation"), *sys.path]):
    from lightnav_vln import request_prediction
    from m3_trial_monitor import completion


class FollowTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROFILE)
        self.core = Controller(self.config)
        self.core.set_map(synthetic_grid(1.))
        self.reference(1.)

    def reference(self, t):
        self.core.set_reference(Reference(t, t, (Pose(1, 0, 0),)))

    def tick(self, t, speed=0.):
        self.core.feedback(Feedback(t, t, BodyVelocity(speed, 0., 0.)))
        return self.core.tick(Tick(t, 0, 0, .05))

    def standby(self):
        self.assertEqual(self.tick(1.).state, "TRACKING")
        self.core.stop(received=1.05)
        braking = self.tick(1.05, .12)
        self.assertEqual(braking.state, "BRAKING")
        self.assertLess(braking.command.x_m_s, .12)
        self.assertEqual(self.tick(1.1).state, "STOPPED")
        self.reference(1.12)
        self.assertEqual(self.tick(1.2).state, "VERIFYING_ARRIVAL")
        self.assertEqual(self.tick(1.3).state, "REOBSERVE")

    def test_stop_then_resume_only_on_new_measurable_reference(self):
        self.standby()
        self.reference(1.31)
        self.core.stop(received=1.34)
        self.reference(1.32)  # delayed delivery from before the repeated STOP
        self.assertEqual(self.tick(1.35).state, "REOBSERVE")
        self.reference(1.381)
        self.assertEqual(self.tick(1.38).state, "REOBSERVE")
        self.core.map_fault("lost")
        self.assertEqual(self.tick(1.40).state, "REOBSERVE")
        self.core.set_map(synthetic_grid(1.41))
        decision = self.tick(1.45)
        self.assertEqual(decision.state, "TRACKING")
        self.assertGreater(decision.command.x_m_s, 0.)

    def test_occupied_blocks_braking_and_resumption(self):
        # The obstacle starts outside the standing footprint+margin, inside its brake tail.
        self.core.set_map(synthetic_grid(1.01, rectangles=((.33,-.05,.35,.05),), resolution=.01))
        self.core.stop()
        decision = self.tick(1.05, .20)
        self.assertEqual((decision.state, decision.command), ("SAFETY_HOLD", ZERO))
        self.assertEqual(self.tick(1.1).state, "STOPPED")
        self.assertEqual(self.tick(1.3).state, "REOBSERVE")
        self.core.set_map(synthetic_grid(1.31, rectangles=((0,0,.1,.1),)))
        self.reference(1.32)
        decision = self.tick(1.35)
        self.assertEqual((decision.state, decision.command), ("SAFETY_HOLD", ZERO))

    def test_default_and_profile_caps(self):
        self.assertIs(load_config()["control"]["continuous_follow"], False)
        def enabled():
            return actuator_enabled(False, True, "measured_metric", PROFILE, self.config, "alohamini2pro")
        self.assertTrue(enabled())
        for key, cap in (("forward_m_s", .20), ("lateral_m_s", .14), ("linear_m_s", .20),
                         ("reverse_m_s", .06), ("yaw_rad_s", .45), ("brake_m_s2", .45)):
            with self.subTest(key=key):
                self.assertEqual(self.config["limits"][key], cap)
                with patch.dict(self.config["limits"], {key:cap+.001}), self.assertRaises(ValueError):
                    enabled()
        with patch.dict(self.config["control"], continuous_follow=False), self.assertRaises(ValueError):
            enabled()
        with self.assertRaises(ValueError):
            actuator_enabled(False, True, "synthetic", PROFILE, self.config, "alohamini2pro")

    def test_follow_does_not_complete_on_model_stop_or_arrived(self):
        for state in ("REOBSERVE", "ARRIVED"):
            decision = dict(state=state, stop_source="model_stop", reason="arrival_evidence_missing_or_invalid")
            self.assertIsNone(completion(decision, .1 if state=="ARRIVED" else None, (0,0,0),
                                         self.config["control"], False, follow=True))
        args = cli.parser().parse_args(["follow","--instruction","Follow the person in blue.","--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertFalse(hasattr(args, "color"))

    def test_visual_history_rollover_preserves_sequence_even_after_stop(self):
        class Socket:
            def __init__(self):
                self.calls = []
            def send(self, raw):
                self.calls.append(json.loads(raw))
            def recv(self):
                last = self.calls[-1]
                seq = last["data"].get("seq")
                return json.dumps(dict(action=last["action"], data=dict(rc=0, seq=seq, stop=seq==64)))
        for follow in (False, True):
            ws = Socket()
            for seq in range(1, 131):
                result = request_prediction(ws, seq, b"jpeg", "Follow", continuous_follow=follow)
                self.assertEqual((result["seq"],result["stop"]), (seq,seq==64))
            resets = [ws.calls[i+1]["data"]["seq"] for i,c in enumerate(ws.calls) if c["action"]=="reset"]
            self.assertEqual(resets, [65,129] if follow else [])
