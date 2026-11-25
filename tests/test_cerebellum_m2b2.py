import unittest

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.obstacles import Cell,Grid,inspect_pose
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Tick
from alohamini.cerebellum.state import Feedback,Pose,Reference,ZERO,load_config
from alohamini.schema import BodyVelocity

ORIGIN=Pose(0.,0.,0.)

class SoftUnknownTest(unittest.TestCase):
    def core(self):
        cfg=load_config()
        self.assertEqual(cfg["planner"]["unknown_policy"],"strict")
        cfg["planner"]["unknown_policy"]="soft_cost"
        return Controller(cfg)

    def decision(self,core,grid):
        core.feedback(Feedback(1.,1.,ZERO))
        core.set_map(grid)
        core.set_reference(Reference(1.,1.,(Pose(1.4,0.,0.),)))
        return core.tick(Tick(1.,0.,0,.05))

    def test_blind_detour_penalizes_unknown_and_keeps_margin(self):
        grid=synthetic_grid(1.,rectangles=((.53,-.09,.66,.09),),
                            unknown=((-1.4,-1.4,.45,1.4),),resolution=.05)
        core=self.core();d=self.decision(core,grid)
        self.assertEqual((d.state,d.reason),("TRACKING","detour_subgoal"))
        self.assertGreater(abs(d.command.y_m_s),.01)
        self.assertLess(d.selected.terms["unknown"],0.)
        self.assertGreater(d.selected.clearance_m,core.planner.p["margin_m"])
        for p in d.selected.path:
            self.assertIsNone(inspect_pose(grid,core.planner.poly,p,.025,.35,unknown_policy="soft_cost")[0])

    def test_choose_more_observed_side(self):
        grid=synthetic_grid(1.,rectangles=((.49,-.08,.62,.08),),
                            unknown=((-1.,.18,1.5,1.2),),resolution=.05)
        d=self.decision(self.core(),grid)
        self.assertEqual(d.reason,"detour_subgoal")
        self.assertLess(d.command.y_m_s,0.)

    def test_clear_lane_with_near_unknown_advances(self):
        grid=synthetic_grid(1.,unknown=((-1.4,-1.4,.45,1.4),),resolution=.05)
        d=self.decision(self.core(),grid)
        self.assertEqual(d.state,"TRACKING");self.assertGreater(d.command.x_m_s,0.)
        self.assertLess(d.selected.terms["unknown"],0.)

    def test_shared_budget_refuses_to_start_near_expiry(self):
        core=self.core()
        core.c["min_plan_budget_s"]=.02
        grid=synthetic_grid(1.)
        self.decision(core,grid)
        now=1.+core.c["reference_ttl_s"]-.004
        # Continuous wheel samples keep the same odometry epoch.
        t=1.05
        while t<now:
            core.feedback(Feedback(t,t,ZERO));t+=.05
        core.feedback(Feedback(now,now,ZERO))
        result=core.tick(Tick(now,0.,0.,.05))
        self.assertEqual(result.reason,"insufficient_time_for_rollout")
        self.assertEqual(result.command,ZERO)

    def test_rgb_noise_band_never_becomes_free_from_height_alone(self):
        import numpy as np
        from alohamini.cerebellum.local_grid import RollingLocalGrid
        from test_cerebellum_local_grid import K,T,ORIGIN,render
        depth,floor=render(brick=True)
        for z,expected in ((.06,False),(.10,True)):
            d=np.where(floor,depth,1.-z)
            m=RollingLocalGrid(stride=1,obstacle_min_z_m=.07,min_occupied_points=4)
            fit=m.update(d,K,T,stamp=1.,pose_odom=ORIGIN,floor_mask=floor)
            self.assertTrue(fit.accepted)
            grid=m.export(1.)
            i,j=np.floor((np.array([.425,.025])-grid.origin)/grid.resolution).astype(int)
            self.assertEqual(grid.cells.get((i,j),Cell.UNKNOWN),Cell.OCCUPIED if expected else Cell.UNKNOWN)

    def test_sealed_wall_and_braking_tail_stay_hard(self):
        wall=synthetic_grid(1.,rectangles=((.35,-1.5,.55,1.5),))
        self.assertEqual(self.decision(self.core(),wall).state,"SAFETY_HOLD")
        grid=Grid(1.,'synthetic','odom',.01,(0.,0.),{(27,0):Cell.OCCUPIED})
        core=self.core();core.set_map(grid)
        core.feedback(Feedback(1.,1.,BodyVelocity(.25,0.,0.)))
        core.stop()
        d=core.tick(Tick(1.,0.,0.,.05))
        self.assertEqual((d.state,d.command),("SAFETY_HOLD",ZERO))
