import unittest
from pathlib import Path
from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.obstacles import inspect_pose
from alohamini.cerebellum.state import Pose, ZERO, Feedback, Odometry, integrate, load_config
from alohamini.cerebellum.scenarios import synthetic_grid

class ChairDetourTest(unittest.TestCase):
    def setUp(self):
        cfg=load_config(Path(__file__).resolve().parents[1]/"evidence/m3-low-speed-shared.json")
        self.p=Controller(cfg).planner
        self.grid=synthetic_grid(0.,rectangles=((.72,-.24,1.03,.24),),resolution=.05)
        self.home=Pose(0.,0.,0.)
        self.chair=Pose(.92,0.,0.)
        self.target=Pose(1.6,-.75,0.)

    def test_separate_target_right_then_advance_no_contact(self):
        p=self.p;pose=self.home;v=ZERO
        route=(self.home,self.chair)
        for i in range(480):
            selected,_,reason,_=p.choose(i*.05,.05,pose,v,v,route,self.grid,target_odom=self.target)
            if i==0:
                self.assertEqual(reason,"detour_subgoal")
                self.assertLess(selected.command.y_m_s,0.)
            self.assertIsNotNone(selected, (i,pose,reason))
            v=selected.command;pose=integrate(pose,v,.05)
            self.assertIsNone(inspect_pose(self.grid,p.poly,pose,p.p["margin_m"],.35,unknown_policy=p.policy)[0])
            if pose.x>1.09 and abs(pose.y-self.target.y)<.18:break
        else:self.fail(f"did not clear chair: {pose}")

    def test_target_near_back_wall_checks_point_and_driven_footprint(self):
        grid = synthetic_grid(0., rectangles=((.72,-.24,1.03,.24), (.95,.15,1.84,.23),
                                              (1.80,-1.1,1.90,.24)), resolution=.05)
        pose, speed = self.home, ZERO
        route = (pose, Pose(1.01,0.,0.))
        self.p.guide(pose, route, grid, frozenset(), self.target)
        component = self.p.terminal_component[2]
        self.assertFalse(component.occupied_within_radius(self.target.x,self.target.y,.15))
        self.assertTrue(inspect_pose(grid,self.p.poly,self.target,0.,0.,unknown_policy=self.p.policy)[0])
        self.assertEqual(self.p.metric_goal,self.target)
        for i in range(550):
            candidate,_,reason,_ = self.p.choose(i*.05,.05,pose,speed,speed,route,grid,target_odom=self.target)
            self.assertIsNotNone(candidate,(pose,reason))
            if i == 0:
                self.assertEqual(reason,"detour_subgoal")
                self.assertLess(candidate.command.y_m_s,0.)
            for sample in candidate.path:
                self.assertIsNone(inspect_pose(grid,self.p.poly,sample,self.p.p["margin_m"],0.,
                                               unknown_policy=self.p.policy)[0])
            speed = candidate.command
            pose = integrate(pose,speed,.05)
            if abs(pose.x-(self.target.x-self.p.dock_offset)) < .05 and abs(pose.y-self.target.y) < .05:
                break
        else:
            self.fail(f"did not reach the standoff pose: {pose}")

    def test_target_distance_uses_closed_cell_edges(self):
        from alohamini.cerebellum.obstacles import Grid, Cell
        cell = Grid(0., "synthetic", "odom", .05, (0.,0.), {(0,0): Cell.OCCUPIED})
        for x,y in ((.2,.025),(.14,.17),(-.15,0.)):
            self.assertTrue(cell.occupied_within_radius(x,y,.15))
        self.assertFalse(cell.occupied_within_radius(.20001,.025,.15))

    def test_same_component_and_no_target_keep_terminal_clipping(self):
        for goal in (None,self.chair):
            self.p.guide(self.home,(self.home,self.chair),self.grid,frozenset(),goal)
            self.assertIsNone(self.p.metric_goal)

    def test_last_measured_pose_allows_only_unix_roundoff(self):
        odom=Odometry(.25)
        odom.push(Feedback(1.,1.,ZERO))
        self.assertEqual(odom.at_reported_pose(1.+.0000002),self.home)
        self.assertIsNone(odom.at_reported_pose(1.00001))
        self.assertIsNone(odom.at(1.+.0000002))

    def test_blocked_exit_and_loss_of_target(self):
        self.p.guide(self.home,(self.home,self.chair),self.grid,frozenset(),self.target)
        self.assertIsNotNone(self.p.metric_goal)
        blocked=synthetic_grid(0.,rectangles=((.32,-1.4,1.2,1.4),),resolution=.05)
        selected,*_=self.p.choose(0.,.05,self.home,ZERO,ZERO,(self.home,self.chair),blocked,target_odom=self.target)
        self.assertIsNone(selected)
        self.p.guide(self.home,(self.home,self.chair),self.grid,frozenset())
        self.assertIsNone(self.p.metric_goal)

