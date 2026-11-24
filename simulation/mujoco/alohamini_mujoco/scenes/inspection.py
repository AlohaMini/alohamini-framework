"""Headless verification and image preview using the runtime's compiled scene."""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ..runtime.control import DynamicFollower
from ..runtime.viewer import set_camera, set_geometry
from .cameras import CameraCapture
from .checks import SceneResetError, layout_intersections


def inspect_scene(robot, *, preview=None, camera=None, image_size=(480, 640), collisions=False):
    scene = robot.scene
    model, data = robot.model, robot.data()
    placement = layout_intersections(robot)
    report = {
        "schema_version": 1,
        "passed": placement["passed"],
        "robot_model": robot.description.robot_model.model_id,
        "placement": placement,
        "objects": list(scene.objects),
        "object_assets": scene.manifest.get("object_assets", {}),
        "cameras": [model.camera(index).name for index in range(model.ncam)],
        "preview": None,
    }
    if hasattr(scene, "task_objects"):
        equivalent = scene.validate(robot)
        reset = {"passed": False}
        if (
            placement["passed"]
            and equivalent["passed"]
            and equivalent["static_robot_scene"]["passed"]
        ):
            try:
                follower = DynamicFollower(robot)
                scene.settle(follower)
                if scene.goal_status(robot, follower.data)["passed"]:
                    raise SceneResetError("Initial task goal is already satisfied")
                data = follower.data
                reset["passed"] = True
            except SceneResetError as exc:
                reset["error"] = str(exc)
        report["task"] = {"scene": equivalent, "reset": reset}
        report["passed"] = report["passed"] and reset["passed"]
    if preview is not None and report["passed"]:
        path = Path(preview).expanduser()
        if path.suffix.lower() != ".png":
            raise ValueError("Use a .png preview path")
        height, width = image_size
        if any(type(value) is not int or value < 32 for value in (height, width)):
            raise ValueError("Preview image dimensions must be integers of at least 32 pixels")
        if camera is not None and camera not in report["cameras"]:
            raise ValueError(f"Unknown scene camera: {camera}")
        view = camera
        if view is None:
            view = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(view)
            set_camera(robot, view)
        options = mujoco.MjvOption()
        set_geometry(options, collisions=collisions)
        options.sitegroup[:] = 0
        with CameraCapture(
            robot,
            [camera] if camera is not None else [],
            width=width,
            height=height,
        ) as capture:
            image = Image.fromarray(capture.image(data, view, options))
            metadata = capture.metadata(camera) if camera is not None else None
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".png", dir=path.parent, delete=False) as stream:
            staged = Path(stream.name)
            try:
                image.save(stream, format="PNG")
                stream.close()
                staged.replace(path)
            finally:
                staged.unlink(missing_ok=True)
        report["preview"] = {
            "path": str(path.resolve()),
            "camera": camera,
            "resolution": [width, height],
            "geometry": "collision" if collisions else "visual",
        }
        if camera is not None:
            frame = data.camera(camera)
            pose = np.eye(4)
            pose[:3, :3] = frame.xmat.reshape(3, 3)
            pose[:3, 3] = frame.xpos
            report["preview"].update(T_world_camera=pose.tolist(), camera_model=metadata)
    return report
