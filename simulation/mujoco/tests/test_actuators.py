import json
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.actuators import ActuatorProfile, JointActuator, ServoReference
from alohamini_mujoco.control import (
    DynamicFollower,
    FollowerSettings,
    gravity_generalized_force,
    gravity_load_report,
    gravity_position_targets,
)
from alohamini_mujoco.model import ModelProfile, build_model

from alohamini.simulation.actuators import ServoIntegral


def parameters(**changes):
    return JointActuator(
        **{
            "kp_Nm_per_rad": 20.0,
            "kd_Nm_s_per_rad": 1.0,
            "friction_Nm": 0.01,
            "peak_effort_Nm": 5.0,
            "velocity_rad_s": 0.3,
            "acceleration_rad_s2": 1.0,
            "delay_s": 0.02,
            **changes,
        }
    )


def pendulum(*, gravity=(0, 0, -9.80665), **changes):
    p = parameters(**changes)
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="0.001" integrator="implicitfast"
      gravity="{" ".join(map(str, gravity))}"/><worldbody>
      <body><joint name="arm" axis="0 1 0" range="-2 2"/>
      <inertial pos="0.2 0 0" mass="1" diaginertia="0.002 0.02 0.02"/>
      </body></worldbody><actuator><general name="arm" joint="arm"
      gainprm="{p.kp_Nm_per_rad}" biastype="affine"
      biasprm="0 {-p.kp_Nm_per_rad} {-p.kd_Nm_s_per_rad}"
      forcelimited="true" forcerange="{-p.peak_effort_Nm} {p.peak_effort_Nm}"
      ctrllimited="true" ctrlrange="-2 2"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.dof_frictionloss[:] = p.friction_Nm
    robot = SimpleNamespace(
        model=model,
        xml=xml,
        scene=None,
        manifest={
            "actuation": {
                "joints": {"arm": asdict(p)},
                "continuous_effort_limits_Nm": {"arm": 2.21},
            }
        },
        description=SimpleNamespace(joints={"arm": SimpleNamespace(kind="revolute", velocity=1.0)}),
    )
    robot.data = lambda positions=None: mujoco.MjData(model)
    return robot


@pytest.mark.parametrize(
    "field,value",
    [
        ("kp_Nm_per_rad", 0),
        ("velocity_rad_s", -1),
        ("delay_s", -0.01),
        ("friction_Nm", float("nan")),
        ("peak_effort_Nm", True),
    ],
)
def test_profile_rejects_invalid_si_parameters(field, value):
    with pytest.raises(ValueError):
        parameters(**{field: value})


def test_integral_requires_a_finite_bounded_effort():
    for limit in (0, -1, 6, float("nan")):
        with pytest.raises(ValueError):
            parameters(ki_Nm_per_rad_s=2, integral_effort_limit_Nm=limit)


def test_sampled_integral_deadband_cap_and_saturation_recovery():
    integral = ServoIntegral(
        parameters(
            ki_Nm_per_rad_s=10,
            integral_effort_limit_Nm=0.1,
            integral_deadband_rad=0.01,
            integral_period_s=0.02,
        )
    )
    integral.advance(0.1, 0.01, 0)
    assert integral.effort == 0
    integral.advance(0.1, 0.01, 0)
    assert integral.effort == pytest.approx(0.018)
    integral.advance(0.005, 0.02, 0)
    assert integral.effort == pytest.approx(0.018)
    integral.advance(1, 0.02, 0)
    assert integral.effort == 0.1
    integral.advance(-1, 0.02, -6)  # Saturation freezes outward integration.
    assert integral.effort == 0.1
    integral.advance(-0.1, 0.02, 6)  # Reversed error unwinds even when saturated.
    assert integral.effort < 0.1
    previous = integral.effort
    integral.advance(0.1, 0.02, 0, blocked=True)
    assert integral.effort == previous
    clone = ServoIntegral(integral.parameters)
    clone.restore(integral.state())
    assert clone.advance(-0.1, 0.01, 0) == integral.advance(-0.1, 0.01, 0)
    for state in ({"effort": 0.2, "remainder_s": 0}, {"effort": 0, "remainder_s": 0.02}):
        with pytest.raises(ValueError):
            clone.restore(state)


def test_integral_removes_static_error_and_snapshot_retains_accumulation():
    from alohamini_mujoco.collection import physics_state, restore_state

    robot = pendulum(
        delay_s=0,
        friction_Nm=0,
        ki_Nm_per_rad_s=8,
        integral_effort_limit_Nm=3,
        integral_period_s=0.02,
    )
    follower = DynamicFollower(robot)
    for _ in range(500):
        follower.step()
    controller = follower.controller_state()
    assert abs(controller["integrals"]["arm"]["effort"]) > 0.1
    clone = DynamicFollower(robot)
    restore_state(robot.model, clone.data, physics_state(robot.model, follower.data))
    clone.restore_controller_state(controller)
    for _ in range(15000):
        follower.step()
        clone.step()
    np.testing.assert_array_equal(follower.data.qpos, clone.data.qpos)
    assert follower.controller_state() == clone.controller_state()
    assert abs(follower.data.qpos[0]) < 0.001
    assert abs(follower.data.actuator_force[0]) <= 5
    assert follower.data.qfrc_applied[0] == 0


def test_pd_snapshot_accepts_legacy_state_without_integrals():
    follower = DynamicFollower(pendulum())
    state = follower.controller_state()
    del state["integrals"]
    follower.restore_controller_state(state)


def test_reference_delays_commands_and_limits_speed_acceleration():
    p = parameters(delay_s=0.025)
    ref = ServoReference(p, 0, 0)
    values = np.array([ref.advance(0.1, i * 0.001, 0.001) for i in range(2000)])
    np.testing.assert_array_equal(values[:25], 0)
    assert values[25, 0] > 0
    assert np.max(np.abs(values[:, 1])) <= p.velocity_rad_s
    assert np.max(np.abs(np.diff(values[:, 1]))) <= p.acceleration_rad_s2 * 0.001 + 1e-12
    assert values[-1, 0] == 0.1
    assert values[-1, 1] == 0


def test_reference_preserves_short_commands_in_delayed_queue():
    ref = ServoReference(parameters(delay_s=0.01), 0, 0)
    for i in range(5):
        ref.advance(0.1, i * 0.001, 0.001)
    for i in range(5, 10):
        ref.advance(-0.1, i * 0.001, 0.001)
    clone = ServoReference(ref.parameters, 0, 0)
    clone.restore(ref.state())
    for i in range(10, 25):
        assert ref.advance(-0.1, i * 0.001, 0.001) == clone.advance(-0.1, i * 0.001, 0.001)
        assert ref.goal == (0.1 if i < 15 else -0.1)
    with pytest.raises(ValueError, match="restore controller state"):
        ref.advance(0, 5, 0.001)


def test_motion_reversal_brakes_without_an_acceleration_impulse():
    ref = ServoReference(parameters(delay_s=0), 0, 0)
    values = []
    for i in range(2000):
        values.append(ref.advance(0.1 if i < 200 else -0.05, i * 0.001, 0.001))
    values = np.asarray(values)
    assert np.max(np.abs(values[:, 1])) <= 0.3 + 1e-12
    assert np.max(np.abs(np.diff(values[:, 1]))) <= 0.001 + 1e-12
    assert values[-1, 0] == pytest.approx(-0.05, abs=1e-12)
    assert values[-1, 1] == 0


def test_gravity_is_independent_of_velocity_and_uses_world_frame(robot):
    data = robot.data({"right_shoulder_lift": -1.2, "right_elbow_flex": 1.0})
    expected = data.qfrc_bias.copy()
    data.qvel[:] = np.linspace(0.2, 2.0, robot.model.nv)
    mujoco.mj_forward(robot.model, data)
    np.testing.assert_allclose(gravity_generalized_force(robot.model, data), expected, atol=1e-12)
    assert np.max(np.abs(data.qfrc_bias - expected)) > 0.01
    rotated = pendulum(gravity=(2.0, -3.0, -8.0))
    d = rotated.data()
    d.qpos[0] = 0.4
    mujoco.mj_forward(rotated.model, d)
    np.testing.assert_allclose(gravity_generalized_force(rotated.model, d), d.qfrc_bias, atol=1e-12)


def test_position_feedforward_sign_and_joint_boundaries():
    np.testing.assert_allclose(
        gravity_position_targets([0.2, 0.2, 0.2], [2.0, -2.0, 100.0], 20.0, -0.3, 0.3),
        [0.3, 0.1, 0.3],
    )
    np.testing.assert_allclose(
        gravity_position_targets([0.2, 0.2], [2.0, -2.0], 20.0, -0.3, 0.3, calibrated_gain=0.02),
        [0.24, 0.16],
    )


def test_follower_uses_measured_gravity_position_gain():
    robot = pendulum(gravity_position_gain_rad_per_Nm=0.02)
    follower = DynamicFollower(robot, settings=FollowerSettings(feedforward="gravity"))
    follower.step()
    assert follower.position_feedforward[0] == pytest.approx(-1.96133 * 0.02)


def test_measured_mode_defaults_to_no_external_gravity_and_obeys_peak_force():
    robot = pendulum()
    follower = DynamicFollower(robot)
    follower.step()
    assert follower.data.ctrl[0] == 0
    assert follower.data.actuator_force[0] == 0
    assert follower.data.qvel[0] > 0  # Physical gravity remains active.
    # Native position control still clamps torque even for a large disturbance.
    follower.data.qpos[0] = -1.9
    follower.step()
    assert abs(follower.data.actuator_force[0]) <= 5.0
    report = gravity_load_report(robot)
    assert report["effort_limits"] == [2.21]
    assert follower.effort_limits[0] == 5.0  # Separate continuous and peak values.


def test_gravity_position_compensation_improves_static_hold_without_bypassing_drive():
    robot = pendulum(delay_s=0.0, friction_Nm=0.0)
    baseline = DynamicFollower(robot)
    compensated = DynamicFollower(robot, settings=FollowerSettings(feedforward="gravity"))
    for _ in range(5000):
        baseline.step()
        compensated.step()
    assert abs(baseline.data.qpos[0]) > 0.08
    assert abs(compensated.data.qpos[0]) < 1e-4
    assert compensated.data.actuator_force[0] == pytest.approx(-1.96133, abs=0.001)
    assert compensated.data.qfrc_applied[0] == 0
    assert abs(compensated.position_feedforward[0]) == pytest.approx(1.96133 / 20, abs=1e-4)


def test_snapshot_restores_pending_profile_commands_and_physics():
    from alohamini_mujoco.collection import physics_state, restore_state

    robot = pendulum()
    follower = DynamicFollower(robot)
    follower.target[:] = 0.05
    for _ in range(10):
        follower.step()
    state = physics_state(robot.model, follower.data)
    controller = follower.controller_state()
    assert controller["references"]["arm"]["pending"]
    clone = DynamicFollower(robot)
    restore_state(robot.model, clone.data, state)
    clone.restore_controller_state(controller)
    for _ in range(200):
        follower.step()
        clone.step()
    np.testing.assert_array_equal(
        physics_state(robot.model, follower.data), physics_state(robot.model, clone.data)
    )


def test_viewer_reset_restores_delayed_controller_history():
    from alohamini_mujoco.viewer import DynamicPlayback

    robot = pendulum()
    playback = DynamicPlayback(robot)
    initial_controller = playback.follower.controller_state()
    playback.manual_target[:] = 0.1
    for _ in range(13):
        playback.step()
    assert playback.follower.references[0].pending
    playback.reset()
    assert playback.follower.controller_state() == initial_controller
    assert playback.data.time == 0


def test_compiled_profile_keeps_geometry_inertia_and_continuous_effort(robot, tmp_path):
    profile = ActuatorProfile(
        "alohamini2pro", {"right_elbow_flex": parameters()}, {"status": "test"}
    )
    path = tmp_path / "actuators.json"
    path.write_text(json.dumps(profile.to_dict()))
    measured = build_model(
        profile=ModelProfile(initial_positions={"vertical_move": 0.2}, actuator_profile=path)
    )
    np.testing.assert_array_equal(measured.model.body_mass, robot.model.body_mass)
    np.testing.assert_array_equal(measured.model.body_ipos, robot.model.body_ipos)
    np.testing.assert_array_equal(measured.model.body_inertia, robot.model.body_inertia)
    np.testing.assert_array_equal(measured.model.geom_size, robot.model.geom_size)
    np.testing.assert_array_equal(measured.model.geom_contype, robot.model.geom_contype)
    actuator = measured.model.actuator("right_elbow_flex").id
    assert measured.model.actuator_forcerange[actuator, 1] == 5.0
    assert measured.description.joints["right_elbow_flex"].effort == 2.21
    assert measured.manifest["actuation"]["joints"]["right_elbow_flex"] == asdict(parameters())
    with pytest.raises(ValueError, match="identified gains"):
        DynamicFollower(
            measured,
            settings=FollowerSettings(joint_gains={"right_elbow_flex": {"kp": 2, "kd": 1}}),
        )


@pytest.mark.parametrize("integral", [False, True])
def test_identified_episode_replay_restores_nontrivial_controller_state(tmp_path, integral):
    h5py = pytest.importorskip("h5py")
    from alohamini_mujoco.collection import EpisodeReplay, physics_state
    from alohamini_mujoco.libero import sha256

    options = (
        {"ki_Nm_per_rad_s": 8, "integral_effort_limit_Nm": 3, "integral_period_s": 0.02}
        if integral
        else {}
    )
    robot = pendulum(**options)
    # The persisted XML must include the same passive friction as the model.
    robot.xml = robot.xml.replace('axis="0 1 0"', 'axis="0 1 0" frictionloss="0.01"')
    follower = DynamicFollower(robot)
    follower.target[:] = 0.1
    for _ in range(13):
        follower.step()
    initial_controller = follower.controller_state()
    states = [physics_state(robot.model, follower.data)]
    actions = [[0.1], [-0.05], [0.0]]
    for action in actions:
        follower.target[:] = action
        for _ in range(50):
            follower.step()
        states.append(physics_state(robot.model, follower.data))
    (tmp_path / "model").mkdir()
    model_path = tmp_path / "model/model.xml"
    model_path.write_text(robot.xml)
    with h5py.File(tmp_path / "episode.hdf5", "w") as file:
        file.attrs["env_args"] = json.dumps(robot.manifest)
        group = file.create_group("data/demo_0")
        group.attrs["joint_names"] = json.dumps(["arm"])
        group.attrs["initial_controller_state"] = json.dumps(initial_controller)
        group.attrs["controller"] = json.dumps(
            {
                "kp": follower.kp.tolist(),
                "kd": follower.kd.tolist(),
                "kinds": follower.kinds,
                "settings": follower.settings.to_dict(),
            }
        )
        group["states"] = states
        group["actions"] = actions
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "mujoco_version": mujoco.__version__,
                "environment": robot.manifest,
                "model_files": {"model/model.xml": sha256(model_path)},
                "episode": "episode.hdf5",
                "physics_steps_per_command": 50,
                "collection_settings": {"replay_tolerance": 1e-12},
            }
        )
    )
    replay = EpisodeReplay(tmp_path)
    while replay.step(single=True):
        pass
    assert replay.report()["passed"], replay.report()
    assert replay.maximum_error == 0.0
    replay.reset()
    assert replay.follower.controller_state() == initial_controller


def test_cli_loads_actuator_profile_and_selects_gravity_without_opening_window(
    robot, tmp_path, monkeypatch
):
    from alohamini_mujoco import cli, viewer

    path = tmp_path / "actuators.json"
    path.write_text(
        json.dumps(
            ActuatorProfile(
                "alohamini2pro", {"right_elbow_flex": parameters()}, {"status": "test"}
            ).to_dict()
        )
    )
    loaded, displayed = [], []
    monkeypatch.setattr(cli, "_route_viewer_files", lambda arguments: None)

    def build(*, profile, scene):
        loaded.append(profile)
        return robot

    monkeypatch.setattr(cli, "build_model", build)
    monkeypatch.setattr(viewer, "run_viewer", lambda *args, **kwargs: displayed.append(kwargs))
    assert (
        cli.main(["view", "--actuator-profile", str(path), "--feedforward", "gravity", "--paused"])
        == 0
    )
    assert loaded[0].actuator_profile.joints["right_elbow_flex"] == parameters()
    assert displayed[0]["follower_settings"].feedforward == "gravity"
    assert displayed[0]["paused"]
