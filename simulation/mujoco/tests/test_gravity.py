import numpy as np
import pytest
from alohamini_mujoco.model import ModelProfile, build_model

from alohamini.simulation.frames import transform
from alohamini.simulation.gravity import GravityModel


def configuration(description):
    positions = {n: 0.0 for n, j in description.joints.items() if j.kind != "fixed"}
    positions.update(description.home)
    positions.update(
        vertical_move=0.2,
        right_shoulder_lift=-1.2,
        right_elbow_flex=1.0,
        left_shoulder_lift=-1.4,
        left_elbow_flex=0.7,
    )
    return positions


@pytest.mark.parametrize("rotated", [False, True])
def test_engine_independent_gravity_matches_mujoco_in_world_coordinates(robot, rotated):
    if rotated:
        robot = build_model(
            profile=ModelProfile(
                T_world_root=transform((0.3, -0.2, 0.1), (0.2, -0.1, 0.5)),
                gravity=(1.2, -0.7, -9.0),
            )
        )
    positions = configuration(robot.description)
    data = robot.data(positions)
    names = [robot.model.actuator(i).name for i in range(robot.model.nu)]
    provider = GravityModel(robot.description, joint_names=names, gravity=robot.model.opt.gravity)
    result = provider.loads(positions, T_world_root=robot.profile.T_world_root)
    dofs = [robot.model.jnt_dofadr[robot.model.joint(n).id] for n in names]
    np.testing.assert_allclose([result[n] for n in names], data.qfrc_bias[dofs], atol=1e-12)
    assert result["vertical_move"] > 0


def test_gravity_is_the_gradient_of_potential_and_root_translation_cancels(robot):
    description = robot.description
    positions = configuration(description)
    provider = GravityModel(description)
    expected = provider.loads(positions)
    shifted = provider.loads(positions, T_world_root=transform((2, -3, 5)))
    np.testing.assert_allclose(list(expected.values()), list(shifted.values()), atol=1e-12)

    def potential(q):
        poses = description.forward(q)
        return -sum(
            m * provider.gravity @ (poses[n][:3, :3] @ inertial[:3, 3] + poses[n][:3, 3])
            for n, (m, inertial, _) in description.inertials.items()
        )

    step = 1e-6
    for name in ("right_shoulder_lift", "right_elbow_flex", "left_wrist_flex", "vertical_move"):
        derivative = (
            potential({**positions, name: positions[name] + step})
            - potential({**positions, name: positions[name] - step})
        ) / (2 * step)
        assert expected[name] == pytest.approx(derivative, abs=1e-6)


def test_gravity_requires_explicit_measured_or_fixed_coordinates(robot):
    provider = GravityModel(robot.description, joint_names=["right_shoulder_lift"])
    positions = configuration(robot.description)
    positions.pop("right_elbow_flex")
    with pytest.raises(ValueError, match="right_elbow_flex"):
        provider.loads(positions)


def test_single_arm_gravity_does_not_require_unrelated_arm_coordinates(robot):
    positions = configuration(robot.description)
    provider = GravityModel(robot.description, joint_names=["right_shoulder_lift"])
    partial = {name: positions[name] for name in provider.required_positions}
    assert not any(n.startswith("left_") for n in partial)
    assert provider.loads(partial) == provider.loads(positions)
