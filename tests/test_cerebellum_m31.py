"""M3.1 output interlocks and read-only two-clock preflight."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alohamini.cerebellum.controller import Decision
from alohamini.cerebellum.ros_adapter import BaseOutput, ClockMonitor, actuator_enabled
from alohamini.cerebellum.state import ZERO, load_config
from alohamini.schema import BodyVelocity

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "evidence/m3-low-speed.json"
spec = importlib.util.spec_from_file_location("m3_clock_check", ROOT/"examples/navigation/m3_clock_check.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class M3OutputTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROFILE)
        self.sent = []
        self.output = BaseOutput(self.sent.append, self.cfg["limits"], clock=lambda: 3.5)
        self.core = SimpleNamespace(previous=ZERO, selected=None)
        self.v = BodyVelocity(.08, -.02, .1)

    def plan(self, state="TRACKING", v=None, until=4.):
        return Decision(state, "test", self.v if v is None else v, until, "UNKNOWN", None)

    def test_locks_and_profile(self):
        args = ("measured_metric", PROFILE, self.cfg, "alohamini2pro")
        self.assertFalse(actuator_enabled(True, False, *args))
        self.assertTrue(actuator_enabled(False, True, *args))
        for dry, enable in ((True,True), (False,False)):
            with self.assertRaises(ValueError): actuator_enabled(dry, enable, *args)
        for source,path,cfg,model in (("synthetic",*args[1:]),(args[0],"",*args[2:]),
                (*args[:2],load_config(),args[3]),(*args[:3],"alohamini2")):
            with self.assertRaises(ValueError):
                actuator_enabled(False,True,source,path,cfg,model)
        self.cfg["limits"]["forward_m_s"] = .13
        with self.assertRaises(ValueError): actuator_enabled(False,True,*args)

    def test_output_and_stop_states(self):
        for state in ("TRACKING","BRAKING"):
            self.output.apply(self.core,self.plan(state),3.)
            self.assertEqual(self.sent[-1],self.v)
        for state in ("SAFETY_HOLD","STOPPED","REOBSERVE","ARRIVED","VERIFYING_ARRIVAL","unexpected"):
            self.output.apply(self.core,self.plan(state),3.)
            self.assertEqual(self.sent[-1],ZERO)

    def test_caps_and_deadline_at_last_write(self):
        for v in [BodyVelocity(.121,0.,0.), BodyVelocity(-.061,0.,0.), BodyVelocity(0.,.101,0.),
                  BodyVelocity(.10,.09,0.),BodyVelocity(0.,0.,.251)]:
            self.output.apply(self.core,self.plan(v=v),3.)
            self.assertEqual(self.sent[-1],ZERO)
        for t in [2.9,3.5,float("nan")]:
            result=self.output.apply(self.core,self.plan(until=t),3.)
            self.assertEqual((self.sent[-1],result.reason),(ZERO,"expired_during_control_tick"))

    def test_diagonal_full_speed_roundoff(self):
        for overshoot, expected in ((5e-10, "TRACKING"), (2e-9, "SAFETY_HOLD")):
            speed = .12 + overshoot
            command = BodyVelocity(.8 * speed, .6 * speed, 0.)
            result = self.output.apply(self.core, self.plan(v=command), 3.)
            self.assertEqual(result.state, expected)
            self.assertEqual(self.sent[-1], command if expected == "TRACKING" else ZERO)

    def test_estop_immediate_latched_and_exit_burst(self):
        self.output.estop(True)
        self.assertEqual(self.sent,[ZERO])
        self.output.estop(False)
        decision=self.output.apply(self.core,self.plan(),3.)
        self.assertEqual((decision.reason,self.sent[-1]),("estop_latched",ZERO))
        with patch("alohamini.cerebellum.ros_adapter.time.sleep"):
            self.output.close()
        self.assertEqual(self.sent,[ZERO]*5)

    def test_native_and_readonly_lift_joint_names(self):
        from alohamini.cerebellum.rgb_navigation import LiftHistory
        for name in ("vertical_move", "lift_axis"):
            lift = LiftHistory()
            for stamp in (1.,1.15,1.31):
                lift.push_joint_state(stamp, ["arm",name], [.1,-.3])
                lift.push_joint_state(stamp, ["wheel1_joint"], [.2])
            self.assertEqual(lift.baseline,-.30)
            lift.push_joint_state(1.4, ["vertical_move","lift_axis"], [-.2,-.1])
            self.assertEqual(len(lift.samples),3)
            lift.push_joint_state(1.45,[name],[-.28])
            self.assertAlmostEqual(lift.delta(1.45),.02)

    def test_dry_output_never_writes(self):
        gate=BaseOutput(None,self.cfg["limits"])
        self.assertEqual(gate.apply(self.core,self.plan(),3.).command,self.v)
        gate.estop(True)
        self.assertEqual(gate.apply(self.core,self.plan(),3.).command,ZERO)
        gate.close()

    def test_clock_probe_signed_bounds_and_future_rejection(self):
        def samples(offset, jitter=0):
            epoch=1_795_000_000_000_000_000
            return [(t,t+50_000+offset,t+100_000,100_000+jitter)
                    for t in range(epoch,epoch+600_000_000,100_000_000)]
        for skew in (-17_000_000,0,17_000_000):
            result=probe.assess(samples(skew))
            self.assertEqual(result["passed"],skew==0)
            self.assertEqual(result["pi_minus_pc_ns"],[skew-50_000,skew+50_000])
        with self.assertRaises(ValueError): probe.assess(samples(0,1_000_000))
        mixed=samples(-17_000_000)+samples(17_000_000)
        self.assertFalse(probe.assess(mixed)["passed"])
        clock=ClockMonitor(self.cfg["ros_clock"])
        wall=samples(0)[0][0]
        with self.assertRaisesRegex(ValueError,"future"):
            clock.convert(wall+17_000_000,wall,100.)
        self.assertAlmostEqual(clock.convert(wall-17_000_000,wall,100.),99.983)

