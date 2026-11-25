import unittest

import numpy as np

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.local_grid import CalibrationSource, PREVIEW, RollingLocalGrid, ground_ray_distance, vote_mask
from alohamini.cerebellum.obstacles import Cell
from alohamini.cerebellum.scheduler import Tick
from alohamini.cerebellum.state import Feedback, Pose, Reference, ZERO, load_config
from alohamini.schema import BodyVelocity


ORIGIN = Pose(0., 0., 0.)
K = np.array([[75.,0.,63.5], [0.,75.,47.5], [0.,0.,1.]])
T = np.array([[0.,-1.,0.,.32], [-1.,0.,0.,0.], [0.,0.,-1.,1.], [0.,0.,0.,1.]])
CALIBRATION = CalibrationSource(
    {"camera_name":"fixture", "status":"accepted_test_fixture", "image_height":96, "image_width":128,
     "image_geometry":"rectified", "rectification":{"K_new":K.tolist()},
     "distortion_coefficients":{"data":[0.]*5}, "camera_matrix":{"data":K.flatten().tolist()}},
    {"camera_name":"fixture", "status":"accepted_test_fixture", "parent_frame":"base_link",
     "T_base_from_camera_optical":T.tolist(), "T_parent_from_camera_optical":{"matrix":T.tolist()}},
)


def render(*, height=1., scale=1., brick=True, wall=False):
    v,u = np.indices((96,128))
    dx,dy = -(v-K[1,2])/K[1,1], -(u-K[0,2])/K[0,0]
    x,y = .32+dx*(height-.06), dy*(height-.06)
    block = (x>=.34) & (x<=.55) & ((abs(y)<.12) | wall) & brick
    return np.where(block,height-.06,height)*scale, ~block


def at(grid,x,y):
    i,j = np.floor((np.array([x,y])-grid.origin)/grid.resolution).astype(int)
    return grid.cells.get((i,j),Cell.UNKNOWN)


class MappingTest(unittest.TestCase):
    def test_self_points_and_optical_minimum(self):
        footprint=[(.28,.29),(-.26,.29),(-.26,-.29),(.28,-.29)]
        points=np.array([[.08,-.01,.1],[.281,0,.1],[.28,0,.1],[.4,0,.3]])
        for poly in (footprint, footprint[::-1]):
            self.assertEqual(vote_mask(points,np.array([.3,.2,.21,.19]),poly,None).tolist(),
                             [False,True,True,False])
        depth, floor=render()
        # Extend a synthetic body over the left half of the brick.
        poly=[(-.26,-.29),(.45,-.29),(.45,.29),(-.26,.29)]
        m=RollingLocalGrid(stride=1)
        self.assertTrue(m.update(depth,K,T,stamp=1.,pose_odom=ORIGIN,floor_mask=floor,self_footprint_m=poly).accepted)
        grid=m.export(1.)
        self.assertEqual(at(grid,.425,.025),Cell.UNKNOWN)
        self.assertEqual(at(grid,.525,.025),Cell.OCCUPIED)

    def test_confirmation_distinct_exposures_reset_free_expiry_and_epoch(self):
        self.mapper=RollingLocalGrid(stride=1,min_hit_frames=2)
        def hit(t):
            self.assertTrue(self.push(t).accepted)
            return at(self.mapper.export(t),.425,.025)==Cell.OCCUPIED
        self.assertFalse(hit(1.))
        self.assertFalse(hit(1.))
        self.push(1.05,brick=False)
        self.assertFalse(hit(1.10))
        self.assertFalse(hit(1.50))
        self.assertTrue(hit(1.60))
        self.mapper.reanchor(2.)
        self.assertFalse(hit(2.))
        self.assertTrue(hit(2.10))

    def test_moving_brick_confirmed_on_second_exposure(self):
        m=RollingLocalGrid(stride=1,min_hit_frames=2)
        depth,floor=render()
        for frame in range(4):
            # Shift the observed block sideways by two pixels per exposure.
            t=1.+.1*frame
            fit=m.update(np.roll(depth,2*frame,axis=1),K,T,stamp=t,pose_odom=ORIGIN,
                         floor_mask=np.roll(floor,2*frame,axis=1))
            self.assertTrue(fit.accepted)
            self.assertEqual(bool(m.export(t).occupied),frame>=1)

    def setUp(self):
        self.mapper = RollingLocalGrid(stride=1)

    def push(self, stamp, *, pose=ORIGIN, calibration=None, **scene):
        depth,floor = render(**scene)
        return self.mapper.update(depth,K,T,stamp=stamp,pose_odom=pose,calibration=calibration,
                                  floor_mask=floor,lift_height_m=scene.get("height",1.)-1.)

    def test_floor_and_six_cm_brick_with_scale_and_lift(self):
        fit = self.push(1., scale=1.3, height=1.18)
        self.assertTrue(fit.accepted, fit)
        self.assertAlmostEqual(fit.scale,1/1.3,places=8)
        grid=self.mapper.export(1.)
        self.assertEqual(at(grid,.425,.025),Cell.OCCUPIED)
        self.assertEqual(at(grid,.025,.425),Cell.FREE)
        self.assertEqual(at(grid,-1.6,-1.6),Cell.UNKNOWN)
        self.assertEqual(grid.source,PREVIEW)

    def test_range_limits_both_votes_in_base_frame_after_floor_fit(self):
        depth, floor = render(scale=1.3)
        pose = Pose(.6, .35, 0.)
        args = dict(stamp=1., pose_odom=pose, floor_mask=floor)
        full = self.mapper.update(depth, K, T, **args)
        limited = RollingLocalGrid(stride=1)
        self.assertEqual(full, limited.update(depth, K, T, max_range_m=.45, **args))
        for x, y, expected in ((.375,.025,Cell.OCCUPIED), (.525,.025,Cell.OCCUPIED),
                               (.025,.225,Cell.FREE), (.025,.625,Cell.FREE)):
            self.assertEqual(at(self.mapper.export(1.), pose.x+x, pose.y+y), expected)
            clipped = Cell.UNKNOWN if np.hypot(x,y) > .45 else expected
            self.assertEqual(at(limited.export(1.), pose.x+x, pose.y+y), clipped)
        for bad in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                limited.update(depth, K, T, max_range_m=bad, **args)

    def test_memory_blind_zone_decay_and_epoch(self):
        self.push(1.)
        self.push(2.,pose=Pose(1.4,0.,.03),brick=False)
        self.assertEqual(at(self.mapper.export(2.),.425,.025),Cell.OCCUPIED)
        self.assertEqual(at(self.mapper.export(2.76),.425,.025),Cell.UNKNOWN)
        self.mapper.reanchor(3.)
        self.assertFalse(self.mapper.export(3.).cells)
        self.assertFalse(self.push(2.9).accepted)

    def test_single_noise_point_and_configurable_support(self):
        depth, floor = render(brick=False)
        depth[40,62] -= .08
        for threshold, expected in ((1, True), (3, False)):
            mapper = RollingLocalGrid(stride=1, min_occupied_points=threshold)
            fit = mapper.update(depth, K, T, stamp=1., pose_odom=ORIGIN, floor_mask=floor)
            self.assertTrue(fit.accepted)
            self.assertEqual(bool(mapper.export(1.).occupied), expected)

    def test_departed_obstacle_clears_on_third_new_free_observation(self):
        for stamp in (1., 1.1, 1.2):
            self.push(stamp)
        for stamp, expected in ((1.3, Cell.OCCUPIED), (1.4, Cell.OCCUPIED), (1.5, Cell.FREE)):
            self.push(stamp, brick=False)
            self.assertEqual(at(self.mapper.export(stamp), .425, .025), expected)

    def test_calibrated_and_preview_floor_tilt_limits(self):
        v, _ = np.indices((96,128))
        for angle,calibration,accepted in ((10,CALIBRATION,True),(15,CALIBRATION,False),(15,None,True),(20,None,False)):
            with self.subTest(angle=angle,measured=calibration is not None):
                self.mapper.reanchor()
                depth=1./(1.+np.tan(np.deg2rad(angle))*(v-K[1,2])/K[1,1])
                fit=self.mapper.update(depth,K,T,stamp=1.,pose_odom=ORIGIN,calibration=calibration,
                                       floor_mask=np.ones_like(depth,bool))
                self.assertEqual(fit.accepted,accepted)
                if accepted:
                    self.assertAlmostEqual(np.rad2deg(np.arccos(fit.normal[2])),angle)
                else:
                    self.assertIn(fit.reason,("no_ground_plane","degenerate_or_tilted_plane"))
                self.assertEqual(bool(self.mapper.export(1.).cells),accepted)

    def test_ray_intersection_uses_lift_and_base_origin(self):
        xy, distance = ground_ray_distance(78.5, 17.5, K, T, .25)
        np.testing.assert_allclose(xy, (.82, -.25), atol=1e-12, rtol=0)
        self.assertAlmostEqual(distance, np.hypot(.82,.25), places=12)
        front = np.array([[0.,0.,1.,.1],[-1.,0.,0.,0.],[0.,-1.,0.,.6],[0.,0.,0.,1.]])
        self.assertIsNone(ground_ray_distance(63.5, 47.5, K, front))
        self.assertIsNone(ground_ray_distance(63.5, 17.5, K, front))
        self.assertIsNone(ground_ray_distance(63.5, 77.5, K, front, -.61))
        self.assertIsNone(ground_ray_distance(0, 0, np.zeros((3,3)), T))
        np.testing.assert_allclose(ground_ray_distance(63.5,77.5,K,front)[0], (1.6,0.), atol=1e-12)

    def test_mask_scale_failure_and_atomic_rejection(self):
        depth,floor=render()
        fit=self.mapper.update(depth,K,T,stamp=1.,pose_odom=ORIGIN,self_mask=~floor)
        self.assertTrue(fit.accepted)
        old=self.mapper.export(1.)
        self.assertFalse(old.occupied)
        for scale,mask in ((2.,floor),(1.,np.zeros_like(floor))):
            fit=self.mapper.update(depth*scale,K,T,stamp=1.1,pose_odom=Pose(.5,.2,1.),floor_mask=mask)
            self.assertFalse(fit.accepted)
            self.assertEqual(old,self.mapper.export(1.1))
        self.assertFalse(self.mapper.update(depth,K,np.zeros((4,4)),stamp=1.2,pose_odom=ORIGIN).accepted)

    def test_height_bands_and_capture_rotation(self):
        transform=T.copy()
        transform[2,3]=1.6
        d=np.full((96,128),1.6)
        d[10:22,20:35]=.2  # z=1.4, ignore
        d[35:50,85:99]=1.54
        pose=Pose(.5,-.2,np.pi/2)
        fit=self.mapper.update(d,K,transform,stamp=2.,pose_odom=pose,floor_mask=(d==1.6))
        self.assertTrue(fit.accepted)
        grid=self.mapper.export(2.)
        for u,v,expected in ((90,42,Cell.OCCUPIED),(25,15,Cell.UNKNOWN)):
            z=d[v,u]
            x,y=.32-(v-47.5)*z/75, -(u-63.5)*z/75
            # The ignored cell can also receive a real floor sample; it cannot be occupied.
            value=at(grid,.5-y,-.2+x)
            if expected==Cell.UNKNOWN:
                self.assertNotEqual(value,Cell.OCCUPIED)
            else:
                self.assertEqual(value,expected)

    def test_nominal_and_mismatched_calibrations_cannot_promote_map(self):
        for changed in ({"status":"tooling_available_no_real_result"}, {"camera_name":"other"},
                        {"T_parent_from_camera_optical":{"matrix":np.eye(4).tolist()}}):
            self.mapper.reanchor()
            docs = CalibrationSource(CALIBRATION.intrinsics, {**CALIBRATION.extrinsics, **changed})
            self.assertTrue(self.push(1., calibration=docs).accepted)
            self.assertEqual(self.mapper.export(1.).source, PREVIEW)

    def test_controller_refuses_preview_and_brakes_before_observed_wall(self):
        c=Controller(load_config())
        self.push(1.,wall=True)
        c.set_map(self.mapper.export(1.))
        self.assertIsNone(c.grid)
        self.mapper.reanchor(1.)
        self.push(1.,wall=True,calibration=CALIBRATION)
        grid=self.mapper.export(1.)
        self.assertEqual(grid.source,"measured_metric")
        c.set_map(grid)
        c.feedback(Feedback(1.,1.,ZERO))
        c.set_reference(Reference(1.,1.,(ORIGIN,Pose(.9,0.,0.)),"odom"))
        d=c.tick(Tick(1.,0.,0,.05))
        self.assertEqual((d.state,d.command),("SAFETY_HOLD",ZERO))
        self.assertEqual(c.planner.rollout(grid,ORIGIN,BodyVelocity(.25,0.,0.),horizon=1.)[2],
                         "occupied_footprint")
        self.push(1.1)
        self.push(1.2,calibration=CALIBRATION)
        self.assertEqual(self.mapper.export(1.2).source,PREVIEW)
