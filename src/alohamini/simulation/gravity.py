"""Static gravity loads from the canonical SI robot tree, without a physics engine."""

from __future__ import annotations

import numpy as np

from .frames import rigid_transform, vector


class GravityModel:
    """Compute G(q) in named URDF coordinates using link masses and COMs.

    ``positions`` contains each queried joint and its moving ancestors and
    descendants. A fixture supplies its locked coordinates explicitly.
    ``T_world_root`` maps
    root coordinates into the world, where ``gravity`` is expressed in m/s².
    External contact and grasped-object loads require separate load information.
    """

    def __init__(self, description, *, joint_names=None, gravity=(0, 0, -9.80665)):
        self.description = description
        self.gravity = vector(gravity, 3, "World gravity").copy()
        self.gravity.flags.writeable = False
        moving = {name for name, joint in description.joints.items() if joint.kind != "fixed"}
        self.names = (
            tuple(description.joints[n].name for n in description.joints if n in moving)
            if joint_names is None
            else tuple(joint_names)
        )
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or not set(self.names) <= moving
        ):
            raise ValueError("Gravity output requires distinct moving URDF joints")
        self.links = tuple(description.inertials)
        self.masses = np.array([description.inertials[n][0] for n in self.links])
        self.coms = np.array([description.inertials[n][1][:3, 3] for n in self.links])
        indices = {name: i for i, name in enumerate(self.links)}
        self.subtrees = {}
        parents = {joint.child: joint for joint in description.joints.values()}
        required = set(self.names)
        for name in self.names:
            parent = description.joints[name].parent
            while parent in parents:
                ancestor = parents[parent]
                if ancestor.kind != "fixed":
                    required.add(ancestor.name)
                parent = ancestor.parent
            pending = [description.joints[name].child]
            subtree = []
            while pending:
                link = pending.pop()
                if link in indices:
                    subtree.append(indices[link])
                for child_joint in description.by_parent.get(link, ()):
                    if child_joint.kind != "fixed":
                        required.add(child_joint.name)
                    pending.append(child_joint.child)
            if not subtree:
                raise ValueError(f"Gravity output has no declared subtree mass: {name}")
            self.subtrees[name] = np.asarray(subtree, dtype=int)
        self.required_positions = frozenset(required)

    def loads(self, positions, *, T_world_root=None):
        missing = self.required_positions - positions.keys()
        if missing:
            raise ValueError(
                f"Gravity requires measured or explicitly fixed positions: {sorted(missing)}"
            )
        root_pose = np.eye(4) if T_world_root is None else rigid_transform(T_world_root)
        poses = self.description.forward(positions, root_pose)
        centers = np.array(
            [
                poses[name][:3, :3] @ self.coms[i] + poses[name][:3, 3]
                for i, name in enumerate(self.links)
            ]
        )
        forces = self.masses[:, None] * self.gravity
        result = {}
        for name in self.names:
            joint = self.description.joints[name]
            T_world_joint = poses[joint.parent] @ joint.T_parent_joint
            axis = T_world_joint[:3, :3] @ joint.axis
            indices = self.subtrees[name]
            if joint.kind == "prismatic":
                load = -axis @ forces[indices].sum(axis=0)
            else:
                arms = centers[indices] - T_world_joint[:3, 3]
                load = -axis @ np.cross(arms, forces[indices]).sum(axis=0)
            result[name] = float(load)
        return result
