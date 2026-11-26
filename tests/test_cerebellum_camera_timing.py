"""Exposure-clock correction for the Pi camera publisher, independent of ROS and SSH."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

NAVIGATION = Path(__file__).resolve().parents[1] / "examples/navigation"
with patch.object(sys, "path", [str(NAVIGATION), *sys.path]):
    from m3_camera_timing import CameraClock
    from m3_clock_check import assess


def exchanges(offset, rtts):
    epoch = 1_800_000_000_000_000_000
    return [(sent, sent + rtt // 2 + offset, sent + rtt, rtt)
            for i, rtt in enumerate(rtts) for sent in [epoch + i * 30_000_000]]


class CameraClockTest(unittest.TestCase):
    def test_large_signed_offset_corrects_exposure_not_receipt(self):
        for offset in (50_000_000, 10_000_000_000, -10_000_000_000):
            with self.subTest(offset=offset):
                clock = CameraClock()
                samples = exchanges(offset, [400_000] * 16)
                clock.refresh(samples, 40.)
                capture = samples[-1][1]
                actual_pc_capture = capture - offset
                corrected = clock.corrected_stamp(capture, 41., actual_pc_capture + 8_000_000)
                self.assertEqual(corrected, actual_pc_capture - 200_000)
                self.assertFalse(assess(samples)["passed"])  # Raw host_wall is never corrected.

    def test_fastest_eight_exclude_two_inconsistent_slow_exchanges(self):
        samples = exchanges(100_000_000, [300_000, 1_000_000] * 8)
        for i in (0, 15):
            sent = samples[i][0]
            samples[i] = (sent, sent + 200_000_000, sent + 8_000_000, 8_000_000)
        result = assess(samples, bound_ns=None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["used"], 8)
        self.assertEqual(result["pi_minus_pc_ns"], [99_850_000, 100_150_000])

    def test_failed_probe_keeps_old_anchor_only_until_deadline(self):
        clock = CameraClock()
        samples = exchanges(60_000_000, [2_000_000] * 16)
        capture = samples[0][1]
        now_wall = samples[0][2]
        with self.assertRaisesRegex(ValueError, "no valid"):
            clock.corrected_stamp(capture, 99., now_wall)
        clock.refresh(samples, 100.)
        bad = exchanges(10_000_000, [8_000_000] * 16)
        for now in (103., 129.9):
            with self.assertRaisesRegex(ValueError, "uncertainty.*lo_ns=.*hi_ns=.*min_rtt_ms="):
                clock.refresh(bad, now)
        self.assertEqual(clock.corrected_stamp(capture, 130., now_wall), samples[0][0])
        with self.assertRaisesRegex(ValueError, "older than 30"):
            clock.corrected_stamp(capture, 130.000001, now_wall)
        clock.refresh(samples, 131.)
        self.assertEqual(clock.corrected_stamp(capture, 132., now_wall), samples[0][0])

    def test_width_boundary_empty_intersection_and_frame_deadlines(self):
        clock = CameraClock()
        samples = exchanges(0, [5_000_000] * 16)
        clock.refresh(samples, 1.)
        mixed = list(samples)
        t, peer, end, elapsed = mixed[0]
        mixed[0] = (t, peer + 12_000_000, end, elapsed)
        with self.assertRaisesRegex(ValueError, "do not intersect.*min_rtt_ms="):
            clock.refresh(mixed, 2.)
        for age in (-1, 250_000_000):
            with self.assertRaisesRegex(ValueError, "capture deadline"):
                clock.corrected_stamp(peer, 3., t + age)


if __name__ == "__main__":
    unittest.main()
