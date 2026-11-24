"""Coordinate-independent pick and place goals for ordinary research scenes."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .checks import fixed_scene_intersections, settle_scene
from .research import ResearchScene, primitive_dimensions


class PickPlaceScene(ResearchScene):
    def __init__(self, path, task_path, *, asset_catalog=None):
        super().__init__(path, asset_catalog=asset_catalog)
        task = json.loads(Path(task_path).read_text())
        if (
            not isinstance(task, dict)
            or not {"schema_version", "kind", "source", "target", "language"} <= task.keys()
            or task.keys() - {"schema_version", "kind", "source", "target", "language", "split"}
            or task["schema_version"] != 1
            or task["kind"] != "pick_place"
        ):
            raise ValueError("Expected a schema_version 1 pick_place task")
        if self.configuration["base_mode"] != "fixed":
            raise ValueError("Pick-place collection requires a fixed base")
        if self.configuration["control_frequency_hz"] != 20:
            raise ValueError("Pick-place collection requires 20 Hz scene control")
        self.source, self.target = task["source"], task["target"]
        if (
            not all(isinstance(n, str) and n in self.objects for n in (self.source, self.target))
            or self.source == self.target
            or self.mass(self.source) <= 0
        ):
            raise ValueError("Pick-place requires a named dynamic source and distinct target")
        target = self.objects[self.target]
        pose = np.asarray(target.get("T_world_object", np.eye(4)))
        if (
            target.get("shape") != "box"
            or target.get("mass_kg", 0) != 0
            or not np.allclose(pose[:3, 2], [0, 0, 1], rtol=0, atol=1e-10)
        ):
            raise ValueError("Pick-place target must be a fixed, upright box support")
        language = task["language"]
        split = task.get("split", "train")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Provide a nonempty task language instruction")
        if split not in ("train", "evaluation"):
            raise ValueError("Task split must be train or evaluation")
        self.task_objects = {
            name: {"root_body": f"scene__{name}", "contact_geoms": self.collision_geometries(name)}
            for name in (self.source, self.target)
        }
        self.manifest.update(
            {
                "benchmark": "alohamini-pick-place-v1",
                "language": language,
                "goals": [["On", self.source, self.target]],
            }
        )
        identity = {key: self.manifest[key] for key in ("configuration", "goals")}
        if self.assets:
            identity["object_assets"] = self.manifest["object_assets"]
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.dataset_partition = {"split": split, "split_group": f"pick-place:{digest}"}

    def settle(self, follower):
        settle_scene(follower)

    def goal_status(self, robot, data):
        source = self.task_objects[self.source]
        target = self.task_objects[self.target]
        cx, cy, cz = robot.model.geom(target["contact_geoms"][0]).size
        body = data.body(target["root_body"])
        offset = body.xmat.reshape(3, 3).T @ (data.body(source["root_body"]).xpos - body.xpos)
        source_ids = {robot.model.geom(name).id for name in source["contact_geoms"]}
        target_id = robot.model.geom(target["contact_geoms"][0]).id
        top = body.xpos[2] + cz
        contact = any(
            (
                (c.geom1 == target_id and c.geom2 in source_ids)
                or (c.geom2 == target_id and c.geom1 in source_ids)
            )
            and c.dist <= 0
            and abs(c.frame[2]) >= 0.9
            and abs(c.pos[2] - top) <= 0.001
            for c in data.contact
        )
        status = {
            "source": self.source,
            "target": self.target,
            "top_contact": contact,
            "center_in_support_frame_m": offset.tolist(),
            "passed": bool(
                contact and abs(offset[0]) <= cx and abs(offset[1]) <= cy and offset[2] > cz
            ),
        }
        return {"passed": status["passed"], "predicates": [status]}

    def validate(self, robot):
        model, data = robot.model, robot.data()
        errors, scene_geoms = [], []
        for name, obj in self.objects.items():
            body = model.body(f"scene__{name}")
            pose = np.asarray(obj.get("T_world_object", np.eye(4)))
            if not np.allclose(
                data.xpos[body.id], pose[:3, 3], rtol=0, atol=1e-10
            ) or not np.allclose(
                data.xmat[body.id].reshape(3, 3), pose[:3, :3], rtol=0, atol=1e-10
            ):
                errors.append(f"{name}: object pose differs from the scene configuration")
            mass = self.mass(name)
            if not np.isclose(body.mass[0], mass, rtol=0, atol=1e-12):
                errors.append(f"{name}: object mass differs from the scene configuration")
            visual = model.geom(f"scene__{name}_visual")
            colliders = self.collision_geometries(name)
            scene_geoms.extend(colliders)
            if name in self.assets:
                if (
                    visual.bodyid[0] != body.id
                    or visual.type[0] != mujoco.mjtGeom.mjGEOM_MESH
                    or visual.group[0] != 2
                    or visual.contype[0]
                    or visual.conaffinity[0]
                    or any(
                        geom.bodyid[0] != body.id
                        or geom.group[0] != 3
                        or not (geom.contype[0] and geom.conaffinity[0])
                        for geom in map(model.geom, colliders)
                    )
                ):
                    errors.append(
                        f"{name}: compiled catalog asset lost geometry or collision roles"
                    )
                continue
            collision = model.geom(colliders[0])
            size = primitive_dimensions(obj["shape"], obj["size"])
            kind = getattr(mujoco.mjtGeom, "mjGEOM_" + obj["shape"].upper())
            if (
                visual.type[0] != kind
                or collision.type[0] != kind
                or not np.allclose(collision.size[: len(size)], size, rtol=0, atol=1e-10)
                or not np.array_equal(visual.size, collision.size)
                or not np.array_equal(visual.pos, collision.pos)
                or not np.array_equal(visual.quat, collision.quat)
                or not (collision.contype[0] and collision.conaffinity[0])
                or (visual.contype[0] or visual.conaffinity[0])
            ):
                errors.append(f"{name}: visual mesh or collision geometry differs from the scene")
        return {
            "passed": not errors,
            "errors": errors,
            "static_robot_scene": fixed_scene_intersections(robot, scene_geoms),
        }
