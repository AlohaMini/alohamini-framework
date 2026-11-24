"""Reset and collision checks shared by task adapters."""

import mujoco
import numpy as np

from ..runtime.native_distance import InvalidNativeDistance, checked_geom_distance

RESET_SETTLE_S = 0.5
PLACEMENT_TOLERANCE_M = 0.0005


class SceneResetError(ValueError):
    """A task initial state violates the declared reset conditions."""


def layout_intersections(robot, *, names=None, tolerance_m=PLACEMENT_TOLERANCE_M):
    """Check scene placement using the compiled collision geoms, including welded bodies."""
    model, data = robot.model, robot.data()
    scene = robot.scene
    pieces = {
        name: {model.geom(geom).id for geom in scene.collision_geometries(name)}
        for name in scene.objects
    }
    robot_geoms = {
        model.geom(name).id for name, geom in robot.geometry.items() if geom["kind"] == "collision"
    }
    all_geoms = robot_geoms.union(*pieces.values())
    checked, overlaps, invalid = set(), [], []
    for name in sorted(pieces if names is None else names):
        for first in sorted(pieces[name]):
            for second in sorted(all_geoms - pieces[name]):
                pair = tuple(sorted((first, second)))
                if pair in checked:
                    continue
                checked.add(pair)
                geoms = [model.geom(index).name for index in pair]
                try:
                    distance, _ = checked_geom_distance(model, data, *pair, tolerance_m)
                except InvalidNativeDistance as exc:
                    invalid.append({"geoms": geoms, "error": str(exc)})
                    continue
                if distance < -tolerance_m:
                    overlaps.append({"geoms": geoms, "distance_m": distance})
    return {
        "passed": not overlaps and not invalid,
        "tolerance_m": tolerance_m,
        "overlaps": overlaps,
        "invalid_distance_queries": invalid,
    }


def settle_scene(follower, duration_s=RESET_SETTLE_S):
    """Settle a task reset without allowing objects to hit the waiting robot."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Scene settling duration must be finite and positive")
    robot, model, data = follower.robot, follower.model, follower.data
    robot_geoms = {model.geom(name).id for name in robot.geometry}

    def check_contacts():
        for contact in data.contact:
            first = contact.geom1 in robot_geoms
            second = contact.geom2 in robot_geoms
            if (first != second and contact.dist <= 0) or (
                (first or second) and contact.dist < -follower.settings.penetration_tolerance_m
            ):
                raise SceneResetError(
                    f"Scene reset collision at {data.time:.3f} s: "
                    f"{model.geom(contact.geom1).name} / {model.geom(contact.geom2).name}, "
                    f"distance {contact.dist:.6f} m; "
                    "use an initial robot pose clear of the scene and falling objects"
                )

    check_contacts()
    for _ in range(int(np.ceil(duration_s / model.opt.timestep))):
        follower.step()
        check_contacts()
    mujoco.mj_forward(model, data)
    check_contacts()


def fixed_scene_intersections(
    robot, scene_geom_names, *, tolerance_m=0.0005, distance_query=checked_geom_distance
):
    """Check real shapes for contacts filtered by MuJoCo's world-weld rule."""
    model, data = robot.model, robot.data()

    def world_welded(geom):
        return model.body_weldid[geom.bodyid[0]] == 0

    robot_geoms = [model.geom(n) for n, g in robot.geometry.items() if g["kind"] == "collision"]
    scene_geoms = [model.geom(name) for name in scene_geom_names]
    intersections, invalid_queries = [], []
    for a in filter(world_welded, robot_geoms):
        for b in scene_geoms:
            if not world_welded(b) or not (b.contype[0] or b.conaffinity[0]):
                continue
            pair = {"robot_geom": a.name, "scene_geom": b.name}
            try:
                distance, _ = distance_query(model, data, a.id, b.id, tolerance_m)
            except InvalidNativeDistance as exc:
                invalid_queries.append({**pair, "error": str(exc)})
                continue
            if distance < -tolerance_m:
                intersections.append({**pair, "distance_m": float(distance)})
    return {
        "passed": not intersections and not invalid_queries,
        "tolerance_m": tolerance_m,
        "intersections": intersections,
        "invalid_distance_queries": invalid_queries,
    }
