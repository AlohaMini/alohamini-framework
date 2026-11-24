import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.runtime.model import build_model
from alohamini_mujoco.scenes.editor import SceneEditor
from alohamini_mujoco.scenes.research import ResearchScene


def test_editor_saves_world_pose_and_camera_and_keeps_last_valid_scene(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples/simulation/pick_place"
    source = example / "tabletop.json"
    source_text = source.read_text()
    output = tmp_path / "scene.json"
    editor = SceneEditor(source, output, task_config=example / "task.json")
    selected = editor.model.body("scene__cube")
    mocap = selected.mocapid[0]
    initial = editor.positions[mocap].copy()
    np.testing.assert_allclose(editor.data.xipos[selected.id], initial)
    perturb = mujoco.MjvPerturb()
    perturb.select = selected.id
    perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    perturb.refpos[:] = initial + [0.005, 0.01, 0.03]
    editor.cycle_axis()
    editor.update_drag(perturb)
    perturb.active = 0
    editor.update_drag(perturb)
    editor.publish()
    expected = initial + [0.005, 0, 0]
    np.testing.assert_allclose(editor.data.body("scene__cube").xpos, expected)

    perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE
    yaw = np.deg2rad(8)
    perturb.refquat[:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    editor.update_drag(perturb)
    perturb.active = 0
    editor.update_drag(perturb)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.32, -0.21, 0.75]
    camera.distance, camera.azimuth, camera.elevation = 0.8, 35, -65
    editor.capture_camera(camera)
    editor.save()

    scene = ResearchScene(output)
    robot = build_model(profile=editor.profile, scene=scene)
    data = robot.data()
    np.testing.assert_allclose(data.body("scene__cube").xpos, expected)
    np.testing.assert_allclose(data.body("scene__cube").xquat, perturb.refquat)
    assert robot.model.body("scene__cube").mass[0] == 0.05
    assert robot.model.body("scene__cube").mocapid[0] == -1
    pose = np.asarray(scene.configuration["cameras"][0]["T_body_camera"])
    np.testing.assert_allclose(data.camera("workspace").xpos, pose[:3, 3])
    np.testing.assert_allclose(data.camera("workspace").xmat.reshape(3, 3), pose[:3, :3])
    assert scene.configuration["cameras"][1] == json.loads(source_text)["cameras"][1]

    saved = output.read_bytes()
    editor.positions[mocap, 2] = 0.72
    with pytest.raises(ValueError, match="overlap"):
        editor.save()
    assert output.read_bytes() == saved

    editor.positions[mocap] = editor.initial_mocap[0][mocap]
    editor.quaternions[mocap] = editor.initial_mocap[1][mocap]
    editor.save()
    objects = json.loads(output.read_text())["objects"]
    assert objects == json.loads(source_text)["objects"]
    assert source.read_text() == source_text
