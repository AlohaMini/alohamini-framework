import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from alohamini.cerebellum.rgb_navigation import Rectifier,HSVContact,LiftHistory,PoseHistory,TargetMemory,depth_canvas,mask_depth_padding
from alohamini.cerebellum.local_grid import CalibrationSource
from alohamini.cerebellum.state import Pose,load_config


class RGBNavigationTest(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        self.T=np.array([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,.205],[0.,0.,0.,1.]])
        self.cam=SimpleNamespace(K=self.K,T=self.T,arrival_excluded=np.zeros((160,140),bool),camera="chest")

    def test_depth_canvas_and_invalid_interpolated_edges(self):
        self.assertEqual(depth_canvas(518),(388,(2,2,91,91)))
        self.assertEqual(depth_canvas(384),(288,(10,10,88,88)))
        for width,cols,rows in [(384,24,12),(518,12,0)]:
            d=np.full((480,640),.021,np.float32)
            mask_depth_padding(d,width)
            good=np.zeros_like(d,bool);good[rows:480-rows,cols:640-cols]=True
            np.testing.assert_array_equal(np.isfinite(d),good)
            self.assertTrue((d[good]==np.float32(.021)).all())

    def test_rectification_provenance(self):
        i=dict(status="accepted_ground_plane_20261124",image_geometry="rectified",camera_name="chest",
               image_height=160,image_width=140,camera_matrix={"data":self.K.ravel().tolist()},
               rectification={"K_new":self.K.tolist()},distortion_coefficients={"data":[0]*5})
        e=dict(status=i["status"],camera_name="chest",parent_frame="base_link",
               T_parent_from_camera_optical={"matrix":self.T.tolist()},T_base_from_camera_optical=self.T.tolist())
        self.assertTrue(CalibrationSource(i,e).matches(self.K,self.T,(160,140)))
        for key,value in (("image_geometry","raw"),("distortion_coefficients",{"data":[.01,0,0,0,0]})):
            doc={**i,key:value}
            self.assertFalse(CalibrationSource(doc,e).matches(self.K,self.T,(160,140)))
        for M in (self.K+.001,self.K*1.000000001):
            self.assertFalse(CalibrationSource(i,e).matches(M,self.T,(160,140)))
        other=deepcopy(e);other["T_base_from_camera_optical"][0][3]+=.001
        self.assertFalse(CalibrationSource(i,other).matches(self.K,self.T,(160,140)))

    def test_static_lift_baseline_and_gap(self):
        h=LiftHistory();self.assertIsNone(h.delta(1.))
        for t in (1.,1.1,1.2,1.31):h.push(t,-.27)
        self.assertAlmostEqual(h.baseline,-.27)
        h.push(1.41,-.25)
        self.assertAlmostEqual(h.delta(1.36),.01)
        self.assertIsNone(h.delta(1.46))
        h.push(2.,-.25)
        self.assertIsNone(h.delta(1.7))
        self.assertIsNone(h.delta(2.5))

    def test_lift_at_homed_stop_one_tick(self):
        history = LiftHistory()
        for t in (1., 1.1, 1.2, 1.31):
            history.push_joint_state(t, ["vertical_move"], [-.300031982421875])
        self.assertEqual(history.baseline, -.30)
        self.assertEqual(history.delta(1.31), 0.)
        history.push(1.41, -.30 - 2 * .131 / 4096)
        self.assertIsNone(history.delta(1.41))
        history.push(1.51, -.298)
        self.assertAlmostEqual(history.delta(1.51), .002)

    def test_bumper_clearance_memory_and_blind_zone(self):
        m=TargetMemory("purple_toy",load_config()["footprint_m"]);p=Pose(0.,0.,0.)
        self.assertFalse(m.observe(1.,p,(50.,100.),self.cam,None))
        self.assertTrue(m.observe(1.,p,(50.,100.),self.cam,0.))
        msg=m.message(1.,p)
        self.assertAlmostEqual(msg["distance_m"],.20)
        self.assertAlmostEqual(msg["range_from_base_m"],.41)
        self.assertAlmostEqual(m.message(4.49,Pose(.08,0.,0.))["bumper_clearance_m"],.12)
        self.assertIsNone(m.message(4.5,p))
        self.assertFalse(m.observe(3.,p,(50.,40.),self.cam,0.))
        self.cam.arrival_excluded[100,50]=True
        self.assertFalse(m.observe(3.,p,(50.,100.),self.cam,0.))

    def test_pose_interpolation_and_epoch(self):
        h=PoseHistory();h.push(1.,Pose(0.,0.,0.),5);h.push(1.1,Pose(.02,.04,.1),5)
        self.assertAlmostEqual(h.at(1.05).y,.02)
        self.assertIsNone(h.at(1.2))
        h.push(1.2,Pose(0.,0.,0.),6);self.assertIsNone(h.at(1.05))

    def test_chest_compensation_and_unseen_observation_age(self):
        m=TargetMemory("purple_toy",load_config()["footprint_m"])
        p=Pose(0.,0.,0.)
        self.assertTrue(m.observe(10.,p,(50.,100.),self.cam,.030))
        self.assertAlmostEqual(m.message(10.,p)["range_from_base_m"],.47)
        self.assertFalse(m.observe(9.99,p,(50.,95.),self.cam,0.))
        self.assertEqual(m.message(11.4,p)["observed_ns"],10_000_000_000)
        self.assertIsNone(m.message(13.5,p))

    def test_lift_movement_cannot_acquire_baseline(self):
        h=LiftHistory()
        for i in range(8):h.push(i/10,.01*i)
        self.assertIsNone(h.delta(.7))
        for i in range(9):h.push(1.+i/10,.07)
        self.assertAlmostEqual(h.delta(1.8),0.)
        self.assertIsNone(h.delta(3.))

    def test_single_color_flash_cannot_confirm_arrival(self):
        m=TargetMemory("purple_toy",load_config()["footprint_m"],confirm_pair=True)
        p=Pose(0.,0.,0.)
        for t in (10.,10.,11.):
            self.assertFalse(m.observe(t,p,(50.,100.),self.cam,0.))
            self.assertIsNone(m.message(t,p))
        self.assertTrue(m.observe(11.12,p,(51.,100.),self.cam,0.))
        self.assertIsNotNone(m.message(11.5,p))
        m.reset()
        self.assertIsNone(m.message(11.51,p))
        self.assertFalse(m.observe(11.6,p,(51.,100.),self.cam,0.))

    def test_front_arrival_floor_pixel_and_reflection_limits(self):
        y,x=np.indices((480,640))
        depth,arrival=Rectifier.masks(x,y,"forward")
        self.assertTrue(depth[460,320]);self.assertFalse(arrival[460,320])
        self.assertTrue(arrival[469,230]);self.assertTrue(arrival[465,432])
        self.assertTrue(arrival[476,320]);self.assertTrue(arrival[430,8])
        # Horizontal optical axis: depth = focal * height / (v - cy).
        K=np.array([[200.,0.,320],[0.,200.,240],[0.,0.,1.]])
        T=self.T.copy();T[2,3]=.396
        cam=SimpleNamespace(K=K,T=T,arrival_excluded=arrival,camera="forward")
        m=TargetMemory("purple_toy",load_config()["footprint_m"])
        p=Pose(0.,0.,0.)
        self.assertTrue(m.observe(1.,p,(320.,460.),cam,0.))
        self.assertAlmostEqual(m.message(1.,p)["range_from_base_m"],.36)
        self.assertFalse(m.observe(2.,p,(320.,265.5),cam,0.))  # 3.1 m
        self.assertFalse(m.observe(2.,p,(100.,315.),cam,0.))  # side wall
        self.assertIsNotNone(m.message(4.499,p))
        self.assertIsNone(m.message(4.5,p))

    def test_color_bottom_contact(self):
        import cv2
        hsv=np.zeros((80,100,3),np.uint8);hsv[35:51,40:61]=[140,200,200]
        bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.assertEqual(HSVContact("purple")(bgr),(50.,50))
        self.assertIsNone(HSVContact("green")(bgr))
