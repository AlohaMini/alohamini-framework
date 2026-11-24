import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.cli import main
from alohamini_mujoco.runtime.model import ModelProfile, build_model
from alohamini_mujoco.scenes.research import ResearchScene
from PIL import Image

from alohamini.simulation.assets.catalog import AssetCatalog
from alohamini.simulation.frames import transform

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/simulation/pick_place"


def test_scene_inspection_rejects_overlap_without_writing_preview(tmp_path, capsys):
    scene = json.loads((EXAMPLE / "tabletop.json").read_text())
    cube = next(obj for obj in scene["objects"] if obj["name"] == "cube")
    cube["T_world_object"][2][3] = 0.725
    layout = tmp_path / "overlap.json"
    layout.write_text(json.dumps(scene))
    image = tmp_path / "preview.png"

    assert main(["inspect-scene", "--scene-config", str(layout), "--preview", str(image)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not image.exists()
    assert any(
        set(hit["geoms"]) == {"scene__cube_collision", "scene__table_collision"}
        for hit in report["placement"]["overlaps"]
    )


def test_catalog_compiles_textured_asset_and_distinct_collision_shapes(tmp_path):
    definition = json.loads((EXAMPLE / "assets/catalog.json").read_text())
    asset = definition["assets"][0]
    obj = tmp_path / "cube.obj"
    obj.write_bytes((EXAMPLE / "assets/cube.obj").read_bytes())
    texture = tmp_path / "paint.png"
    Image.new("RGB", (8, 8), (120, 180, 210)).save(texture)
    asset["visual"]["texture"] = {
        "file": texture.name,
        "sha256": hashlib.sha256(texture.read_bytes()).hexdigest(),
    }
    asset["collision"].append(
        {
            "type": "mesh",
            "mesh": {"file": obj.name, "sha256": asset["visual"]["sha256"], "scale": 0.5},
            "T_asset_geometry": transform([0.06, 0, 0]).tolist(),
        }
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(definition))
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "name": "prop",
                        "asset": {"id": "blue_cube", "version": 1},
                        "T_world_object": transform([0.7, 0.25, 0.8]).tolist(),
                    }
                ]
            }
        )
    )
    scene = ResearchScene(layout, asset_catalog=catalog)
    robot = build_model(profile=ModelProfile(), scene=scene)
    model, data = robot.model, robot.data()
    box, convex = map(model.geom, scene.collision_geometries("prop"))
    visual = model.geom("scene__prop_visual")
    assert box.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert convex.type[0] == visual.type[0] == mujoco.mjtGeom.mjGEOM_MESH
    np.testing.assert_allclose(box.size, [0.02, 0.02, 0.02])
    np.testing.assert_allclose(data.geom(convex.id).xpos, [0.76, 0.25, 0.8], atol=1e-8)
    assert visual.matid[0] >= 0 and model.texture("scene__prop_texture").width[0] == 8
    assert (visual.contype[0], box.contype[0], convex.contype[0]) == (0, 1, 1)

    with pytest.raises(ValueError, match="definition checksum mismatch"):
        AssetCatalog(catalog).resolve({"id": "blue_cube", "version": 1, "sha256": "0" * 64})
    obj.write_text("# tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        AssetCatalog(catalog).resolve({"id": "blue_cube", "version": 1})
