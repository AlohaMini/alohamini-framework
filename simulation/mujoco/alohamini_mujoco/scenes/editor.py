"""Edit scene placements in the native viewer and save ordinary research JSON."""

from __future__ import annotations

import json
import queue
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from ..runtime.control import DynamicFollower
from ..runtime.model import ModelProfile, build_model
from ..runtime.viewer import check_display, native_window, set_camera, set_geometry
from .checks import layout_intersections
from .pick_place import PickPlaceScene
from .research import ResearchScene


class SceneEditor:
    def __init__(
        self,
        source,
        output,
        *,
        task_config=None,
        camera_name="workspace",
        collision_bundle=None,
        inertial_bundle=None,
        asset_catalog=None,
    ):
        source = Path(source).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        if self.output == source or self.output.exists():
            raise FileExistsError(f"Choose a new scene JSON file: {self.output}")
        if not camera_name.isidentifier():
            raise ValueError("Camera name must be an identifier")

        scene = ResearchScene(source, placement_preview=True, asset_catalog=asset_catalog)
        self.asset_catalog = asset_catalog
        self.configuration = scene.configuration
        self.profile = ModelProfile(
            T_world_root=self.configuration["T_world_base"],
            initial_positions=self.configuration["initial_positions"],
            timestep=self.configuration["timestep_s"],
            locked_positions={} if self.configuration["base_mode"] == "kinematic" else None,
            collision_bundle=collision_bundle,
            inertial_bundle=inertial_bundle,
        )
        self.robot = build_model(profile=self.profile, scene=scene)
        self.model, self.data = self.robot.model, self.robot.data()
        self.home_qpos = self.data.qpos.copy()
        self.positions = self.data.mocap_pos.copy()
        self.quaternions = self.data.mocap_quat.copy()
        self.initial_mocap = self.positions.copy(), self.quaternions.copy()
        self.published_mocap = self.initial_mocap
        self.objects = {}
        for obj in self.configuration["objects"]:
            body = self.model.body("scene__" + obj["name"])
            self.objects[body.id] = (obj, int(body.mocapid[0]))

        self.task_config, self.camera_name = task_config, camera_name
        self.camera_scene = mujoco.MjvScene(self.model, maxgeom=1)
        self.selected = None
        self.axis = None
        self.drag = None
        self.saved = False

    def cycle_axis(self):
        if self.drag is None:
            self.axis = {None: 0, 0: 1, 1: 2, 2: None}[self.axis]

    def update_drag(self, perturbation):
        self.selected = self.objects.get(perturbation.select)
        if self.selected is None:
            self.drag = None
            return
        _, mocap = self.selected
        active = perturbation.active
        if active:
            if self.drag is None or self.drag[:2] != (mocap, active):
                self.drag = (mocap, active, self.positions[mocap].copy())
        elif self.drag is None or self.drag[0] != mocap:
            self.drag = None
            return
        _, mode, origin = self.drag
        # The preview mocap bodies have identity body-to-inertia frames.
        # Keep the last reference on release, including mouse motion between syncs.
        if mode & mujoco.mjtPertBit.mjPERT_TRANSLATE:
            position = perturbation.refpos.copy()
            if self.axis is not None:
                origin = origin.copy()
                origin[self.axis] = position[self.axis]
                position = origin
            self.positions[mocap] = position
        if mode & mujoco.mjtPertBit.mjPERT_ROTATE:
            self.quaternions[mocap] = perturbation.refquat
        if not active:
            self.drag = None

    def publish(self):
        if (
            np.array_equal(self.data.qpos, self.home_qpos)
            and np.array_equal(self.data.mocap_pos, self.positions)
            and np.array_equal(self.data.mocap_quat, self.quaternions)
            and np.array_equal(self.published_mocap[0], self.positions)
            and np.array_equal(self.published_mocap[1], self.quaternions)
        ):
            return
        self.data.qpos[:] = self.home_qpos
        self.data.mocap_pos[:] = self.positions
        self.data.mocap_quat[:] = self.quaternions
        mujoco.mj_forward(self.model, self.data)
        self.published_mocap = self.positions.copy(), self.quaternions.copy()

    def capture_camera(self, camera):
        if camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            raise ValueError("Choose Rendering > Camera > Free before capturing a view")
        cameras = self.configuration["cameras"]
        existing = next((c for c in cameras if c["name"] == self.camera_name), None)
        if existing is not None and (existing["body"] != "world" or "intrinsics" in existing):
            raise ValueError("Use a new --camera-name to preserve the mounted or calibrated camera")
        mujoco.mjv_updateCamera(self.model, self.data, camera, self.camera_scene)
        if self.camera_scene.camera[0].orthographic:
            raise ValueError("Capture a perspective view for collection")
        position, forward, up = np.empty(3), np.empty(3), np.empty(3)
        mujoco.mjv_cameraInModel(position, forward, up, self.camera_scene)
        z = -forward / np.linalg.norm(forward)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        pose[:3, 3] = position
        definition = {
            "name": self.camera_name,
            "body": "world",
            "role": (existing or {}).get("role", "workspace"),
            "fovy_deg": self.model.vis.global_.fovy,
            "T_body_camera": pose.tolist(),
        }
        if existing is None:
            cameras.append(definition)
        else:
            existing.update(definition)

    def save(self):
        configuration = deepcopy(self.configuration)
        objects = {obj["name"]: obj for obj in configuration["objects"]}
        original_pos, original_quat = self.initial_mocap
        moved = set()
        for obj, mocap in self.objects.values():
            if np.array_equal(self.positions[mocap], original_pos[mocap]) and np.array_equal(
                self.quaternions[mocap], original_quat[mocap]
            ):
                continue
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, self.quaternions[mocap])
            pose = np.eye(4)
            pose[:3, :3] = matrix.reshape(3, 3)
            pose[:3, 3] = self.positions[mocap]
            objects[obj["name"]]["T_world_object"] = pose.tolist()
            moved.add(obj["name"])

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=self.output.parent,
            delete=False,
        ) as stream:
            candidate = Path(stream.name)
            try:
                json.dump(configuration, stream, indent=2, allow_nan=False, ensure_ascii=False)
                stream.write("\n")
                stream.close()
                self._validate(candidate, moved)
                if self.saved:
                    candidate.replace(self.output)
                else:
                    self.output.hardlink_to(candidate)
                self.saved = True
            finally:
                candidate.unlink(missing_ok=True)

    def _validate(self, path, moved):
        scene = (
            ResearchScene(path, asset_catalog=self.asset_catalog)
            if self.task_config is None
            else PickPlaceScene(path, self.task_config, asset_catalog=self.asset_catalog)
        )
        if scene.manifest.get("object_assets") != self.robot.scene.manifest.get("object_assets"):
            raise ValueError("Asset catalog changed while editing; reopen the scene")
        robot = build_model(profile=self.profile, scene=scene)
        if moved:
            placement = layout_intersections(robot, names=moved)
            if placement["invalid_distance_queries"]:
                raise ValueError(placement["invalid_distance_queries"][0]["error"])
            if placement["overlaps"]:
                hit = placement["overlaps"][0]
                names = " / ".join(hit["geoms"])
                raise ValueError(f"{names} overlap by {-hit['distance_m'] * 1000:.2f} mm")
        if self.task_config is not None:
            report = scene.validate(robot)
            if not report["passed"]:
                raise ValueError(f"Invalid task scene: {report['errors']}")
            if not report["static_robot_scene"]["passed"]:
                raise ValueError(f"Fixed robot intersects scene: {report['static_robot_scene']}")
            follower = DynamicFollower(robot)
            scene.settle(follower)
            if scene.goal_status(robot, follower.data)["passed"]:
                raise ValueError("The task goal is already satisfied at reset")


def run_editor(editor):
    import glfw

    check_display()
    events = queue.SimpleQueue()
    collisions, status = False, "Double-click an object to select it"
    with native_window(
        editor.model,
        editor.data,
        key_callback=events.put,
        show_right_ui=False,
    ) as window:
        with window.lock():
            set_geometry(window.opt, collisions=collisions)
            set_camera(editor.robot, window.cam)
        while window.is_running():
            start = time.monotonic()
            with window.lock():
                editor.update_drag(window.perturb)
                editor.publish()
            while not events.empty():
                key = events.get()
                try:
                    if key == glfw.KEY_F8:
                        collisions = not collisions
                        with window.lock():
                            set_geometry(window.opt, collisions=collisions)
                    elif key == glfw.KEY_F10:
                        editor.save()
                        status = f"Saved {editor.output.name}"
                        print(f"Saved scene: {editor.output}", flush=True)
                    elif key == glfw.KEY_F11:
                        with window.lock():
                            editor.capture_camera(window.cam)
                        status = f"Captured {editor.camera_name}. F10 to save"
                    elif key == glfw.KEY_F12:
                        editor.cycle_axis()
                except (ValueError, OSError, RuntimeError, mujoco.FatalError) as exc:
                    status = str(exc)
                    print(f"Scene editor: {status}", flush=True)
            name = "None"
            if editor.selected:
                obj, index = editor.selected
                x, y, z = editor.positions[index]
                name = f"{obj['name']}  ({x:.3f}, {y:.3f}, {z:.3f}) m"
            axis = "Free" if editor.axis is None else "XYZ"[editor.axis]
            window.set_texts(
                (
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "Scene editor\nSelected\nMove axis\nDrag\nView\nSave",
                    f"{status[:140]}\n{name}\n{axis} (F12)\n"
                    "Ctrl+right: move | Ctrl+left: rotate | Shift: horizontal\n"
                    "F8: collision | F11: capture camera\nF10: save scene",
                )
            )
            window.sync(state_only=True)
            time.sleep(max(0, start + 1 / 30 - time.monotonic()))
