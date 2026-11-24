import json
from pathlib import Path

import numpy as np
import pytest
from alohamini_mujoco.cli import main
from PIL import Image

from alohamini.model.joint_spec import ModelJointSpecification


@pytest.mark.parametrize("use_catalog", [False, True], ids=["primitive", "catalog"])
def test_tabletop_collect_and_replay(tmp_path, capsys, use_catalog):
    h5py = pytest.importorskip("h5py")
    example = Path(__file__).resolve().parents[3] / "examples/simulation/pick_place"
    output = tmp_path / "episode"
    scene = [
        "--scene-config",
        str(example / ("tabletop_asset.json" if use_catalog else "tabletop.json")),
        "--task-config",
        str(example / "task.json"),
    ]
    if use_catalog:
        scene.extend(["--asset-catalog", str(example / "assets/catalog.json")])
        preview = tmp_path / "preview.png"
        assert (
            main(
                [
                    "inspect-scene",
                    *scene,
                    "--preview",
                    str(preview),
                    "--camera",
                    "workspace",
                    "--image-size",
                    "64",
                    "64",
                ]
            )
            == 0
        )
        inspection = json.loads(capsys.readouterr().out)
        assert inspection["task"]["reset"]["passed"]
        assert inspection["object_assets"]["cube"]["id"] == "blue_cube"
    assert (
        main(
            [
                "collect",
                str(example / "motion.json"),
                *scene,
                "--output",
                str(output),
                "--image-size",
                "64",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert all(result["quality"]["checks"].values())
    replay = result["quality"]["replay"]
    assert replay["completed"] and replay["maximum_state_error"] == 0
    assert result["quality"]["workspace_camera_framing"]["performed"]

    manifest = json.loads((output / "dataset.json").read_text())
    assert manifest["split"] == "train"
    assert manifest["environment"]["scene"]["benchmark"] == "alohamini-pick-place-v1"
    spec = ModelJointSpecification.from_model()
    trajectory = json.loads((example / "motion.json").read_text())
    with h5py.File(output / manifest["episode"], "r") as file:
        group = file["data/demo_0"]
        if use_catalog:
            snapshot = manifest["environment"]["scene"]["object_assets"]
            assert snapshot == inspection["object_assets"]
            np.testing.assert_array_equal(
                inspection["preview"]["T_world_camera"],
                group["camera/workspace/T_world_camera"][0],
            )
            with Image.open(preview) as image:
                np.testing.assert_array_equal(image, group["obs/workspace_rgb"][0])
        for camera in ("workspace", "right_wrist"):
            assert group[f"obs/{camera}_rgb"].shape == (replay["frames"], 64, 64, 3)
        actions = group["actions"][:]
        assert group["obs/joint_positions"].shape == actions.shape
        np.testing.assert_allclose(np.diff(group["simulation_time_s"]), 0.05, atol=1e-12)
        names = json.loads(group.attrs["joint_names"])
        for name in trajectory["joint_names"]:
            lower, upper = spec.joints[name].command_limits
            targets = actions[:, names.index(name)]
            assert targets.min() >= lower and targets.max() <= upper
