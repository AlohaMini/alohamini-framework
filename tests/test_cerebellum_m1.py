"""Independent geometry, closed-loop and STOP regressions for milestone M1."""

import math
import random
from time import perf_counter
import unittest

from alohamini.cerebellum.controller import Controller
from alohamini.cerebellum.obstacles import (
    Cell, Grid, _distance, _lower_distance, _support, initial_unknown, inspect_pose, world_polygon,
)
from alohamini.cerebellum.ros_adapter import StopDebounce
from alohamini.cerebellum.scenarios import synthetic_grid
from alohamini.cerebellum.scheduler import Periodic, Tick
from alohamini.cerebellum.state import Arrival, Feedback, Pose, Reference, ZERO, load_config
from alohamini.schema import BodyVelocity


ORIGIN = Pose(0., 0., 0.)


def brute(grid, poly, pose, margin, cap):
    """Pre-M1 search: inspect individual cells, then every occupied cell for clearance."""
    world = world_polygon(poly, pose)
    support = _support(world)
    low, high = [], []
    for axis in (0, 1):
        values = [p[axis] for p in world]
        low.append(math.floor((min(values)-margin-grid.origin[axis])/grid.resolution-1e-10))
        high.append(math.floor((max(values)+margin-grid.origin[axis])/grid.resolution+1e-10))
    unknown = False
    for i in range(low[0], high[0]+1):
        for j in range(low[1], high[1]+1):
            cell = grid.cells.get((i,j), Cell.UNKNOWN)
            if cell == Cell.FREE:
                continue
            box = grid.box(i,j)
            lower = _lower_distance(support, box)
            if lower <= margin and _distance(world, box, lower) <= margin:
                if cell == Cell.OCCUPIED:
                    return "occupied_footprint", 0.
                unknown = True
    if unknown:
        return "unknown_footprint", None
    if not grid.occupied:
        return None, None
    nearest = cap
    for key in grid.occupied:
        box = grid.box(*key)
        lower = _lower_distance(support, box)
        if lower < nearest:
            nearest = min(nearest, _distance(world, box, lower))
    return None, nearest


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.planner = Controller(load_config()).planner

    def test_local_index_agrees_with_exhaustive_search(self):
        rng = random.Random(59)
        res = .04
        layouts = [
            lambda i,j: Cell.FREE,
            lambda i,j: Cell.OCCUPIED if j in (10,19) and 0 <= i <= 7 else Cell.FREE,
            lambda i,j: Cell.OCCUPIED if i < -5 or i > 10 else Cell.FREE,
            lambda i,j: Cell.OCCUPIED if (i+3*j) % 67 == 2 else Cell.FREE,
            lambda i,j: rng.choices(list(Cell), [1,995,4])[0],
            lambda i,j: Cell.UNKNOWN if i < 0 else Cell.OCCUPIED,
        ]
        counts = set()
        for layout in layouts:
            cells = {(i,j): layout(i,j) for i in range(-24,26) for j in range(-21,22)}
            cells.pop((-3,7))
            grid = Grid(1., "synthetic", "odom", res, (.013,-.027), cells)
            for k in range(80):
                pose = Pose(rng.uniform(-1.,1.), rng.uniform(-.9,.9), rng.uniform(-math.pi,math.pi))
                margin = (0.,.025,.081)[k % 3]
                cap = (0.,.03,.35,.6)[k % 4]
                expected = brute(grid,self.planner.poly,pose,margin,cap)
                actual = inspect_pose(grid,self.planner.poly,pose,margin,cap)
                counts.add(expected[0])
                self.assertEqual(actual[0],expected[0], (k,pose))
                if expected[1] is None:
                    self.assertIsNone(actual[1])
                else:
                    self.assertAlmostEqual(actual[1],expected[1],delta=1e-12)
        self.assertEqual(counts,{None,"occupied_footprint","unknown_footprint"})

    def test_polygon_wholly_inside_solid_has_no_nearby_surface(self):
        grid = Grid(1., "synthetic", "odom", .05, (0., 0.),
                    {(i,j): Cell.OCCUPIED for i in range(-30,31) for j in range(-30,31)})
        self.assertFalse(list(grid.surface_runs((-8,8,-8,8))))
        for yaw in (0., .37, 1.5):
            pose = Pose(.1, -.1, yaw)
            expected = brute(grid, self.planner.poly, pose, .025, .35)
            self.assertEqual(expected, ("occupied_footprint", 0.))
            self.assertEqual(inspect_pose(grid, self.planner.poly, pose, .025, .35), expected)

    def test_initial_unknown_only_under_body_can_start_and_brake(self):
        for unknown in (((-.01,-.01,.01,.01),), ((-.15,-.08,.15,.08),)):
            grid = synthetic_grid(1., unknown=unknown, resolution=.025)
            core = Controller(load_config())
            self.assertTrue(initial_unknown(grid,core.planner.poly,ORIGIN))
            core.set_map(grid)
            core.feedback(Feedback(1.,1.,ZERO))
            core.set_reference(Reference(1.,1.,(Pose(.8,0.,0.),)))
            self.assertEqual(core.tick(Tick(1.,0,0,.05)).state,"TRACKING")
            core.stop()
            core.feedback(Feedback(1.05,1.05,BodyVelocity(.12,0.,0.)))
            d = core.tick(Tick(1.05,0,0,.05))
            self.assertEqual(d.state,"BRAKING")
            self.assertAlmostEqual(d.command.x_m_s,.09)

    def test_new_cells_in_lateral_turn_and_braking_tail(self):
        for region, velocity, horizon in (
            ((-.1,.245,.1,.27), BodyVelocity(0.,.15,0.), .5),
            ((.155,.22,.19,.245), BodyVelocity(0.,0.,.4), .7),
            ((.275,-.02,.29,.02), BodyVelocity(.25,0.,0.), 0.),
        ):
            with self.subTest(region=region):
                grid = synthetic_grid(1., unknown=(region,(-.04,-.04,.04,.04)), resolution=.01)
                _, _, reason = self.planner.rollout(grid,ORIGIN,velocity,horizon=horizon)
                self.assertEqual(reason,"unknown_footprint")
        for region in ((-.02,-.02,.02,.02),(.25,-.02,.27,.02)):
            grid = synthetic_grid(1.,rectangles=(region,),unknown= ((-.1,-.1,.1,.1),),resolution=.025)
            self.assertEqual(self.planner.rollout(grid,ORIGIN,BodyVelocity(.2,0.,0.))[2],"occupied_footprint")


def closed_loop(grid, path, ticks):
    cfg = load_config()
    # Stationary, immutable synthetic map in odom throughout this experiment.
    cfg["control"]["map_ttl_s"] = ticks*.05 + 1.
    core = Controller(cfg)
    core.set_map(grid)
    velocity = ZERO
    decisions = []
    for index in range(ticks):
        now = 1. + index*.05
        core.feedback(Feedback(now,now,velocity))
        core.set_reference(Reference(now,now,path,"odom"))
        decision = core.tick(Tick(now,0.,0,.05))
        decisions.append(decision)
        velocity = decision.command
    return core, decisions


class GuidanceTest(unittest.TestCase):
    def test_closed_loop_clears_front_block_and_returns_to_reference(self):
        grid = synthetic_grid(1.,rectangles=((.25,-.05,.4,.05),),resolution=.01)
        core, decisions = closed_loop(grid,(ORIGIN,Pose(1.,0.,0.)),205)
        self.assertTrue(all(d.state == "TRACKING" for d in decisions))
        self.assertAlmostEqual(decisions[0].command.x_m_s,0.)
        self.assertGreater(abs(decisions[0].command.y_m_s),0.)
        signs = []
        previous = ZERO
        for d in decisions:
            self.assertTrue(core.planner.within_limits(d.command))
            self.assertLessEqual(math.hypot(d.command.x_m_s-previous.x_m_s,d.command.y_m_s-previous.y_m_s),.04+1e-9)
            self.assertLessEqual(abs(d.command.yaw_rad_s-previous.yaw_rad_s),.075+1e-9)
            previous = d.command
            # Independent individual-cell oracle checks the executed poses.
            self.assertIsNone(brute(grid,core.planner.poly,d.pose,.025,.0)[0])
            if .25 <= d.pose.x <= .4:
                self.assertGreater(abs(d.pose.y),.05+.19+.025)
            if abs(d.command.y_m_s) > .01:
                signs.append(math.copysign(1.,d.command.y_m_s))
        self.assertEqual(sum(a != b for a,b in zip(signs,signs[1:])),1)
        self.assertLess(math.hypot(decisions[-1].pose.x-1.,decisions[-1].pose.y),.01)

    def test_sealed_sides_narrow_gap_and_unobserved_detours_hold(self):
        front = ((.25,-.05,.4,.05),)
        scenes = (
            (front+((-.8,.24,1.2,1.4),(-.8,-1.4,1.2,-.24)),()),
            (((.25,.18,.7,1.49),(.25,-1.49,.7,-.18)),()),
            (front,((-.8,.23,1.2,1.49),(-.8,-1.49,1.2,-.23))),
        )
        for rectangles,unknown in scenes:
            grid = synthetic_grid(1.,rectangles=rectangles,unknown=unknown,resolution=.025)
            _, decisions = closed_loop(grid,(ORIGIN,Pose(.9,0.,0.)),6)
            self.assertTrue(all(d.state == "SAFETY_HOLD" and d.command == ZERO for d in decisions))

    def test_left_unknown_selects_observed_right_and_respects_tight_wheel_limits(self):
        grid = synthetic_grid(1., rectangles=((.25,-.05,.4,.05),),
                              unknown=((-1.,.23,2.,1.4),), resolution=.025)
        cfg = load_config()
        cfg["limits"]["wheel_ticks_s"] = 400
        planner = Controller(cfg).planner
        command, _, _, options = planner.choose(1., .05, ORIGIN, ZERO, ZERO,
                                                 (ORIGIN, Pose(.9,0.,0.)), grid)
        self.assertIsNotNone(command)
        self.assertLess(command.command.y_m_s, 0.)
        for candidate in options:
            wheels = planner.kinematics.body_to_wheels(candidate.command)
            self.assertLessEqual(max(map(abs, wheels)), 400 * math.tau / 4096 + 1e-12)

    def assert_closed_loop_bypass(self, grid, route):
        core, steps = closed_loop(grid, route, 360)
        self.assertTrue(any(d.reason == "detour_subgoal" for d in steps))
        self.assertTrue(all(d.state == "TRACKING" for d in steps),
                        [(i, d.reason) for i, d in enumerate(steps) if d.state != "TRACKING"][:5])
        for d in steps:
            self.assertIsNone(brute(grid, core.planner.poly, d.pose, core.planner.p["margin_m"], 0.)[0])
            self.assertGreaterEqual(d.selected.clearance_m, core.planner.p["margin_m"] - 1e-10)
        final = steps[-1].pose
        self.assertLess(math.hypot(final.x-route[-1].x, final.y-route[-1].y), .005)
        return steps

    def test_rejoin_search_across_six_obstacle_distances(self):
        for x in (.35, .55, .75, .95, 1.15, 1.35):
            with self.subTest(center=x):
                obstacle = (x-.075, -.05, x+.075, .05)
                grid = synthetic_grid(1., rectangles=(obstacle,), resolution=.025)
                steps = self.assert_closed_loop_bypass(grid, (ORIGIN, Pose(2., 0., 0.)))
                crossing = [d.pose for d in steps if abs(d.pose.x-x) < .075]
                self.assertTrue(crossing)
                self.assertGreater(min(abs(p.y) for p in crossing), .05+.19+.025)

    def test_long_block_and_curved_reference(self):
        curved = (ORIGIN, Pose(.65, 0., 0.), Pose(1.25, .4, .45),
                  Pose(1.65, .5, .15), Pose(2., .5, 0.))
        for obstacle, route in (
            ((.35, -.05, 1.15, .05), (ORIGIN, Pose(2., 0., 0.))),
            ((.9, .13, 1.05, .23), curved),
        ):
            with self.subTest(obstacle=obstacle):
                grid = synthetic_grid(1., rectangles=(obstacle,), resolution=.025)
                self.assert_closed_loop_bypass(grid, route)

    def test_occupied_terminal_tapers_to_zero_at_footprint_standoff(self):
        grid = synthetic_grid(1., rectangles=((.80, -.05, .92, .05),), resolution=.025)
        end = Pose(.85, 0., 0.)
        core, steps = closed_loop(grid, (ORIGIN, end, end), 150)
        speeds = [d.command.x_m_s for d in steps]
        peak = speeds.index(max(speeds))
        self.assertGreater(speeds[peak], .20)
        self.assertTrue(all(-1e-9 <= a-b <= .04+1e-9 for a,b in zip(speeds[peak:], speeds[peak+1:])))
        for d in steps:
            self.assertIsNone(brute(grid, core.planner.poly, d.pose, .025, 0.)[0])
        self.assertTrue(all(d.reason == "terminal_standoff" and d.command == ZERO for d in steps[-20:]))
        self.assertFalse(core.stop_requested)
        gap = .80 - steps[-1].pose.x - .21
        self.assertGreaterEqual(gap, .025)
        self.assertLess(gap, .027)

    def test_free_endpoint_with_front_footprint_contact(self):
        grid = synthetic_grid(1., rectangles=((.8, -.05, .92, .05),), resolution=.025)
        core, steps = closed_loop(grid, (ORIGIN, Pose(.70, 0., 0.)), 160)
        self.assertTrue(all(d.command == ZERO and d.reason == "terminal_standoff" for d in steps[-20:]))
        self.assertGreater(steps[-1].pose.x, .56)
        self.assertLess(steps[-1].pose.x + .21 + .025, .8)
        self.assertTrue(all(brute(grid, core.planner.poly, d.pose, .025, 0.)[0] is None for d in steps))

    def test_large_terminal_wall_refresh_cost(self):
        cfg = load_config()
        core = Controller(cfg)
        wall = synthetic_grid(1., rectangles=((.80,-1.5,3.95,1.5),), resolution=.05)
        self.assertGreater(len(wall.occupied), 3000)
        elapsed = []
        previous = ZERO
        for i in range(40):
            t = 1. + i*.05
            if i % 4 == 0:
                core.set_map(Grid(t,wall.source,wall.frame,wall.resolution,wall.origin,wall.cells))
            core.feedback(Feedback(t,t,previous))
            core.set_reference(Reference(t,t,(ORIGIN,Pose(.70,0.,0.)),"odom"))
            start = perf_counter()
            decision = core.tick(Tick(t,0.,0,.05))
            elapsed.append((perf_counter()-start)*1000)
            previous = decision.command
            self.assertLess(len(core.planner.terminal_component[2].occupied),len(wall.occupied)//2)
        print("large wall 40 tick_compute_ms", [round(v,3) for v in elapsed])
        self.assertLess(max(elapsed), 50.)

    def test_terminal_behind_separate_wall_and_unknown_terminal(self):
        end = Pose(.85, 0., 0.)
        for obstacles, unknown in (
            (((.80, -.05, .92, .05), (.28, -1.45, .34, 1.45)), ()),
            ((), ((.80, -.05, .92, .05),)),
        ):
            grid = synthetic_grid(1., rectangles=obstacles, unknown=unknown, resolution=.05)
            _, steps = closed_loop(grid, (ORIGIN, end), 4)
            self.assertTrue(all(d.state == "SAFETY_HOLD" and d.command == ZERO for d in steps))

    def test_pure_rotation_and_feedback_lag_projection(self):
        planner = Controller(load_config()).planner
        turn = Pose(1., 0., .3)
        targets, _, _ = planner.velocity_targets(ORIGIN, turn, 1.)
        self.assertIn(BodyVelocity(0., 0., .3), targets)
        measured = BodyVelocity(.03, 0., .04)
        first, _, _, _ = planner.choose(1., .05, ORIGIN, measured, ZERO,
                                        (ORIGIN, turn), synthetic_grid(1.))
        self.assertIsNotNone(first)
        self.assertGreater(first.command.x_m_s, .03)
        for center in (measured, ZERO):
            self.assertLessEqual(math.hypot(first.command.x_m_s-center.x_m_s,
                                            first.command.y_m_s-center.y_m_s), .04+1e-12)
            self.assertLessEqual(abs(first.command.yaw_rad_s-center.yaw_rad_s), .075+1e-12)
        self.assertTrue(planner.within_limits(first.command))
        # A lens vertex where clipping against just the previous disk would leave
        # the measured disk. Also cover a tangent lens and disjoint disks.
        for sep in (.076, .08):
            v = planner.project_command(BodyVelocity(0., .04, .075), ZERO, BodyVelocity(sep, 0., 0.), .05)
            self.assertAlmostEqual(v.x_m_s, sep/2)
            self.assertAlmostEqual(math.hypot(v.x_m_s, v.y_m_s), .04)
        self.assertIsNone(planner.project_command(ZERO, ZERO, BodyVelocity(.09, 0., 0.), .05))
        self.assertIsNone(planner.project_command(ZERO, ZERO, BodyVelocity(0., 0., .16), .05))

    def test_speed_tapers_at_duplicate_endpoint_without_overshoot(self):
        end = Pose(.32,0.,0.)
        _, ds = closed_loop(synthetic_grid(1.),(ORIGIN,Pose(.14,0.,0.),end,end,end),100)
        speeds = [d.command.x_m_s for d in ds]
        peak = speeds.index(max(speeds))
        self.assertGreater(max(speeds),.20)
        self.assertTrue(all(-1e-9 <= a-b <= .04+1e-9 for a,b in zip(speeds[peak:],speeds[peak+1:])))
        self.assertLess(speeds[-1],.002)
        self.assertGreater(ds[-1].pose.x,.318)
        self.assertLessEqual(max(d.pose.x for d in ds),end.x)


class TimerTest(unittest.TestCase):
    def test_native_timer_phase_and_actual_missed_firings(self):
        timer = Periodic(20)
        ticks = [timer.timer_tick(now, deadline) for now, deadline in (
            (1.003, 1.05), (1.0501, 1.10), (1.102, 1.15), (1.251, 1.30))]
        self.assertEqual([t.missed_periods for t in ticks], [0, 0, 0, 2])
        self.assertAlmostEqual(ticks[1].jitter_s, .0001)
        self.assertAlmostEqual(ticks[-1].jitter_s, .101)


class StopTest(unittest.TestCase):
    def test_debounce_reset_and_compatibility(self):
        d = StopDebounce(2)
        self.assertEqual([d.observe(flag) for flag in (True,False,True,True)], [False,False,False,True])
        self.assertTrue(StopDebounce(1).observe(True))
        for value in (0,True,1.5):
            with self.assertRaises(ValueError):
                StopDebounce(value)

    def test_far_evidence_and_post_stop_valid_reference_are_both_required(self):
        for mode in ("far","unknown","stale_evidence","wrong_source","old_ref","expired_ref","at_goal"):
            with self.subTest(mode=mode):
                core = Controller(load_config())
                core.set_map(synthetic_grid(1.))
                core.stop()
                def tick(t):
                    core.feedback(Feedback(t,t,ZERO))
                    return core.tick(Tick(t,0,0,.05)).state
                for t, state in ((1.,"STOPPED"),(1.1,"VERIFYING_ARRIVAL"),(1.2,"REOBSERVE")):
                    self.assertEqual(tick(t),state)
                ref_stamp = {"old_ref":1., "expired_ref":.7}.get(mode,1.25)
                core.set_reference(Reference(ref_stamp,ref_stamp,(Pose(.8,0.,0.),),"odom"))
                if mode != "unknown":
                    core.set_arrival(Arrival(.6 if mode == "stale_evidence" else 1.25,True,
                                             .1 if mode == "at_goal" else .7,
                                             "rgb" if mode == "wrong_source" else "measured_metric"))
                expected = {"far":"TRACKING","at_goal":"ARRIVED"}.get(mode,"REOBSERVE")
                self.assertEqual(tick(1.25),expected)
                self.assertEqual(core.stop_requested,mode != "far")

