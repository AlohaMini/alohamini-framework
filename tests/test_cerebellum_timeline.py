import os
from pathlib import Path
import tempfile
import unittest

from alohamini.cerebellum.replay import main, simulate
from alohamini.cerebellum.state import load_config
from alohamini.cerebellum.timeline import Wheel, audit, load_run


class TimelineTest(unittest.TestCase):
    def test_injected_faults_on_fixed_replay_clock(self):
        t = 1_795_000_000_000_000_000
        events = [{"seq": i, "decision_ns": t + i*100_000_000, "forward_ns": t + i*100_000_000,
                   "front_path": [[1, 0, 0]]} for i in range(11)]
        wheels = [Wheel(t + i*10_000_000, 0, 0, 0) for i in range(151)]
        for fault, expected in (
            ("wheel_stale", "feedback_stale"), ("reference_stale", "reference_expired_or_future"),
            ("unknown", "insufficient_observed_swept_space"), ("stop", "arrival_evidence_missing_or_invalid")
        ):
            with self.subTest(fault=fault):
                samples, stats = simulate(events, wheels, load_config(), scene="open", injections=[(fault, .3)])
                self.assertGreater(stats["moving_ticks"], 0)
                self.assertIn(expected, stats["reasons"])
                self.assertFalse(any(r["vx_m_s"] or r["vy_m_s"] or r["omega_rad_s"] for r in samples[18:]))
                self.assertTrue(all(r["jitter_s"] == 0 for r in samples))

    def test_nearest_logged_is_not_consumed(self):
        t = 1_795_000_000_000_000_000
        events = [
            {"seq": 0, "decision_ns": t, "forward_ns": t - 200_000_000,
             "chest_ns": t - 250_000_000, "mode": "HOLD", "reason": "wheel_feedback_stale"},
            {"event": "input_timeout", "decision_ns": t + 50_000_000},
        ]
        rows, summary = audit(events, [Wheel(t - 20_000_000, 0, 0, 0), Wheel(t + 1, 0, 0, 0)],
                              camera_ttl_s=.45, feedback_age_s=.12)
        self.assertEqual(summary["other_events"], 1)
        self.assertEqual(rows[0]["wheel_previous_age_ms"], 20.)
        self.assertEqual(rows[0]["wheel_nearest_stamp_ns"], t + 1)
        self.assertIsNone(rows[0]["wheel_consumed_stamp_ns"])
        self.assertEqual(rows[0]["wheel_log_status"], "previous_sample_within_limit")
        rows, _ = audit(events, [Wheel(100, 0, 0, 0)], camera_ttl_s=.45, feedback_age_s=.12)
        self.assertEqual(rows[0]["wheel_log_status"], "timestamps_untrusted")

    @unittest.skipUnless(os.getenv("CEREBELLUM_EVIDENCE"), "set CEREBELLUM_EVIDENCE to run real-log tests")
    def test_real_logs_and_cli(self):
        root = Path(os.environ["CEREBELLUM_EVIDENCE"])
        for run, modes, reasons in (
            ("reg6LOUn", {"HOLD": 82},
             {"side_corridor_unobserved": 44, "wheel_feedback_stale": 20, "stale_or_unpaired": 18}),
            ("At98ZJIA", {"HOLD": 47, "LEFT": 22},
             {"wheel_feedback_stale": 26, "chest_detour": 22, "no_full_width_detour": 15,
              "stale_or_unpaired": 4, "waiting_for_post_enable_pair": 1, "expired_or_pre_enable_input": 1}),
        ):
            with self.subTest(run=run):
                events, wheels = load_run(root, run)
                _, info = audit(events, wheels, camera_ttl_s=.45, feedback_age_s=.12)
                self.assertEqual(info["modes"], modes)
                self.assertEqual(info["reasons"], reasons)
                self.assertEqual(info["numbered_decisions"] + info["other_events"], len(events))
                self.assertEqual(info["stale_consumed_stamp_logged"], 0)
                if run == "At98ZJIA":
                    self.assertEqual(info["movement_expired_before_next"], 22)
                    self.assertAlmostEqual(info["movement_remaining_median_ms"], 88.77, places=2)
                    self.assertAlmostEqual(info["next_decision_after_motion_median_ms"], 276.59, places=2)
                with tempfile.TemporaryDirectory() as out:
                    main(["--evidence", str(root), "--run", run, "--out", out])
                    self.assertTrue((Path(out) / "timeline.csv").is_file())
