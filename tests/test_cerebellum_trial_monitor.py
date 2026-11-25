from pathlib import Path
import sys
import unittest

from alohamini.cerebellum.obstacles import Grid
from alohamini.cerebellum.rgb_navigation import PoseHistory
from alohamini.cerebellum.state import Pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/navigation"))
from m3_trial_monitor import confirmed_target_odom, occupancy_guard


class TrialMonitorTest(unittest.TestCase):
    def test_mixed_votes_revoke_ownership_until_cell_clears(self):
        from m2b_rgb_nodes import TargetVoteHistory
        history=TargetVoteHistory()
        key=(8,0)
        anchor=Pose(.42,0.,0.)
        def update(t, occupied, votes, mixed):
            history.reconcile(occupied,votes,mixed,4,anchor,t)
            return key in history.owned
        self.assertTrue(update(1,{key},{key:5},set()))
        self.assertTrue(update(2,{key},{},set()))
        self.assertFalse(update(3,{key},{key:5},{key}))
        self.assertFalse(update(4,{key},{key:5},set()))
        self.assertFalse(update(5,set(),{},set()))
        self.assertTrue(update(6,{key},{key:4},set()))
        self.assertFalse(update(500_000_006,{key},{},set()))

    def test_only_confirmed_target_body_is_exempt_on_live_map(self):
        footprint = ((-.26,-.29),(.28,-.29),(.28,.29),(-.26,.29))
        pose = Pose(0,0,0)
        # Cell starts 2 cm in front of the nominal footprint.
        body, leg = (6,0), (6,1)
        def grid(*keys):
            return Grid(1., "measured_metric", "odom", .05, (0.,0.),
                        {k: 100 for k in keys})
        def check(initial, live, target, owned):
            return occupancy_guard(initial, live, footprint, pose, .025, target, owned)
        target = Pose(.415,.025,0)
        self.assertIsNone(check(grid(), grid(body), target, {body}))
        self.assertTrue(check(grid(body), grid(body), target, {body}))
        self.assertTrue(check(grid(), grid(body,leg), target, {body}))
        self.assertTrue(check(grid(), grid(body), target, set()))
        self.assertTrue(check(grid(), grid(body), None, {body}))
        self.assertTrue(check(grid(), grid(body), Pose(.55,0.,0), {body}))

    def test_goal_uses_pose_at_measurement_and_live_epoch(self):
        poses = PoseHistory()
        poses.push(1., Pose(1.,2.,0.), 3)
        poses.push(1.2, Pose(1.2,2.,0.), 3)
        data = dict(target_id="purple_toy", target_confirmed=True, source="measured_metric",
                    stamp_ns=1_100_000_000, observed_ns=1_070_000_000, x_m=.41, y_m=.03)
        def project(obj=data, epoch=3, now=1_250_000_000):
            return confirmed_target_odom(obj, poses, epoch, now, "purple_toy")
        self.assertAlmostEqual(project().x, 1.51)
        self.assertAlmostEqual(project().y, 2.03)
        for update in ({"source":"preview"}, {"target_confirmed":False},
                       {"x_m":float("nan")}, {"target_id":"other"}, {"stamp_ns":1_240_000_000}):
            self.assertIsNone(project({**data, **update}))
        self.assertIsNone(project(epoch=4))
        self.assertIsNone(project(now=1_580_000_000))
