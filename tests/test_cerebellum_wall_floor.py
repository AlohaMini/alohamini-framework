"""Ray-traced wall/carpet seam and wall-only negatives at 640 x 480."""
import math
import unittest

import numpy as np

from alohamini.cerebellum.local_grid import CalibrationSource, RollingLocalGrid
from alohamini.cerebellum.state import Pose


class WallFloorTest(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[320.,0,319.5],[0,320.,239.5],[0,0,1.]])
        self.t = np.array([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,.40],[0.,0.,0.,1.]])
        self.v,self.u = np.indices((480,640))
        self.cal = CalibrationSource(
            dict(status="accepted_fixture", camera_name="test", image_height=480, image_width=640,
                 image_geometry="rectified", rectification={"K_new":self.k.tolist()},
                 camera_matrix={"data":self.k.ravel().tolist()},distortion_coefficients={"data":[0.]*5}),
            dict(status="accepted_fixture", camera_name="test",parent_frame="base_link",
                 T_base_from_camera_optical=self.t.tolist(),T_parent_from_camera_optical={"matrix":self.t.tolist()}))
        # Floor hits before the vertical wall only in the bottom 72 of 240 lower rows.
        self.wall_x = .40*320./168.

    def fit(self, depth):
        mapper = RollingLocalGrid(stride=4,min_occupied_points=4,obstacle_min_z_m=.07)
        fit = mapper.update(depth,self.k,self.t,stamp=1.,pose_odom=Pose(0,0,0),calibration=self.cal)
        return fit,mapper.export(1.)

    def test_seventy_percent_wall_keeps_obstacle_and_floor(self):
        floor = .40*320./np.maximum(self.v-239.5,.001)
        for scale in (.85,1.,1.28):
            rng=np.random.default_rng(131)
            depth=np.minimum(floor,self.wall_x)*scale + rng.normal(0,.00075,floor.shape)
            fit,grid=self.fit(depth)
            self.assertTrue(fit.accepted,fit)
            self.assertEqual(grid.source,"measured_metric")
            self.assertAlmostEqual(fit.scale,1./scale,delta=.02)
            self.assertLess(np.degrees(np.arccos(fit.normal[2])),2.)
            walls=[grid.box(*ij) for ij in grid.occupied]
            self.assertTrue(any(box[0]-.025<=self.wall_x<=box[2]+.025 and abs(box[1])<.15 for box in walls))
            self.assertFalse(any(box[0]<self.wall_x-.10 and abs(box[1])<.15 for box in walls))

    def test_only_wall_rejects_with_noise_and_oblique_yaw(self):
        for yaw in (0.,.35,-.45):
            for sigma in (0.,.001,.004):
                wall=self.wall_x/(math.cos(yaw)-math.sin(yaw)*(self.u-319.5)/320.)
                noise=np.random.default_rng(211).normal(0,sigma,wall.shape)
                fit,grid=self.fit(wall+noise)
                self.assertFalse(fit.accepted,(yaw,sigma,fit))
                self.assertFalse(grid.cells)

