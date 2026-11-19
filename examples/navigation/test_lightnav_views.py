import unittest

from lightnav_views import FrameSet, combine


class ViewTests(unittest.TestCase):
    def test_pair_is_fresh_synchronized_and_consumed_once(self):
        frames = FrameSet(("forward", "chest"))
        frames.put("forward", 1000000000, b"f")
        self.assertIsNone(frames.take(1010000000))
        frames.put("chest", 1100000000, b"c")
        self.assertIsNone(frames.take(1110000000))
        frames.put("forward", 1110000000, b"f2")
        self.assertIsNone(frames.take(1100000000))
        self.assertEqual(frames.take(1120000000), (1100000000, {"forward":b"f2", "chest":b"c"}))
        frames.put("chest", 1100000000, b"repeat")
        self.assertIsNone(frames.take(1120000000))
        frames.put("forward", 1200000000, b"f3")
        frames.put("chest", 1200000000, b"c3")
        self.assertIsNone(frames.take(1700000000))

    def test_chest_checks_direction_without_steering(self):
        forward = [0.15, 0.0, 0.2]
        for chest in ([-.15, 0., .2], [0., .15, .2], [.15, 0., -.2]):
            self.assertEqual(combine({"forward":forward,"chest":chest})[0], (0., 0., 0.))
        self.assertEqual(combine({"forward":forward,"chest":[.16,.01,.1]})[0], forward)
        self.assertEqual(combine({"forward":forward})[0], forward)
        self.assertIsNone(combine({"forward":forward,"chest":None})[0])
        self.assertIsNone(combine({"forward":None,"chest":forward})[0])
