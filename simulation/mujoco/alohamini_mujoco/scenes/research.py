"""Research scenes compiled from shared layouts and object assets."""

import xml.etree.ElementTree as ET

import numpy as np

from alohamini.simulation.assets.catalog import AssetCatalog, MeshGeometry
from alohamini.simulation.assets.inertia import primitive_inertia
from alohamini.simulation.scene import load_scene

from ..assets.inertia import mjcf_attributes
from ..runtime.model import pose_attributes, text
from .cameras import add_cameras


def primitive_dimensions(shape, size):
    if shape == "box":
        return np.asarray(size) / 2
    if shape == "cylinder":
        return [size[0], size[1] / 2]
    return size


class ResearchScene:
    def __init__(self, path, *, placement_preview=False, asset_catalog=None):
        self.configuration = load_scene(path)
        self.placement_preview = placement_preview
        self.manifest = {"benchmark": "alohamini-research-v1", "configuration": self.configuration}
        self.objects = {obj["name"]: obj for obj in self.configuration["objects"]}
        self.assets = {}
        catalog = AssetCatalog(asset_catalog) if asset_catalog is not None else None
        for name, obj in self.objects.items():
            if "asset" in obj:
                if catalog is None:
                    raise ValueError(f"{name}: provide --asset-catalog to resolve object assets")
                self.assets[name] = catalog.resolve(obj["asset"])
        if self.assets:
            self.manifest["object_assets"] = {
                name: asset.manifest for name, asset in self.assets.items()
            }

    def mass(self, name):
        asset = self.assets.get(name)
        if asset is None:
            return self.objects[name].get("mass_kg", 0)
        return asset.properties.mass_kg if asset.properties is not None else 0

    def collision_geometries(self, name):
        prefix = f"scene__{name}_collision"
        if name not in self.assets:
            return [prefix]
        return [f"{prefix}_{index}" for index in range(len(self.assets[name].colliders))]

    def augment(self, root, assets):
        world = root.find("worldbody")
        definitions = root.find("asset")
        if self.configuration["base_mode"] == "kinematic":
            world.find("body[@name='base_link']").set("mocap", "true")
        for obj in self.configuration["objects"]:
            name = "scene__" + obj["name"]
            body = ET.SubElement(
                world, "body", name=name, **pose_attributes(obj.get("T_world_object", np.eye(4)))
            )
            mass = self.mass(obj["name"])
            if self.placement_preview:
                body.set("mocap", "true")
                # The native drag reference is the inertial frame. Match it
                # to the editable body frame even when inertiafromgeom is off.
                ET.SubElement(body, "inertial", pos="0 0 0", mass="0", diaginertia="0 0 0")
            elif mass:
                ET.SubElement(body, "freejoint", name=name + "_joint")
                asset = self.assets.get(obj["name"])
                inertial = (
                    mjcf_attributes(asset.properties)
                    if asset is not None
                    else {
                        "mass": text([mass]),
                        "pos": "0 0 0",
                        "diaginertia": text(primitive_inertia(obj["shape"], obj["size"], mass)),
                    }
                )
                ET.SubElement(body, "inertial", **inertial)
            if obj["name"] in self.assets:
                self._add_asset(body, definitions, assets, obj, name)
                continue
            shape = obj["shape"]
            dimensions = primitive_dimensions(shape, obj["size"])
            attributes = {"type": shape, "size": text(dimensions), "mass": "0"}
            collision = ET.SubElement(
                body,
                "geom",
                name=name + "_collision",
                **attributes,
                group="3",
                friction=text([obj.get("friction", 1), 0.005, 0.0001]),
                rgba=".1 .8 .3 .3",
            )
            _contact_override(collision, obj)
            ET.SubElement(
                body,
                "geom",
                name=name + "_visual",
                **attributes,
                group="2",
                contype="0",
                conaffinity="0",
                rgba=text(obj.get("color", [0.7, 0.7, 0.7, 1])),
            )
        add_cameras(root, self.configuration["cameras"])

    def _add_asset(self, body, definitions, files, obj, name):
        asset = self.assets[obj["name"]]

        def add_mesh(mesh, mesh_name):
            files[mesh.file.filename] = mesh.file.data
            ET.SubElement(
                definitions,
                "mesh",
                name=mesh_name,
                file=mesh.file.filename,
                scale=text([mesh.scale] * 3),
            )
            return mesh_name

        visual = {"mesh": add_mesh(asset.visual, f"{name}_visual_mesh")}
        if asset.texture is not None:
            texture_name, material_name = f"{name}_texture", f"{name}_material"
            files[asset.texture.filename] = asset.texture.data
            ET.SubElement(
                definitions,
                "texture",
                type="2d",
                name=texture_name,
                file=asset.texture.filename,
            )
            ET.SubElement(
                definitions,
                "material",
                name=material_name,
                texture=texture_name,
                rgba=text(obj.get("color", asset.rgba)),
            )
            visual["material"] = material_name
        else:
            visual["rgba"] = text(obj.get("color", asset.rgba))
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="mesh",
            group="2",
            mass="0",
            contype="0",
            conaffinity="0",
            **visual,
            **pose_attributes(asset.visual.pose),
        )
        for geom, shape in zip(
            self.collision_geometries(obj["name"]), asset.colliders, strict=True
        ):
            if isinstance(shape, MeshGeometry):
                attributes = {"type": "mesh", "mesh": add_mesh(shape, f"{geom}_mesh")}
                pose = shape.pose
            else:
                size = shape.parameters.get("size", [])
                if shape.kind == "sphere":
                    size = [shape.parameters["radius"]]
                elif shape.kind == "cylinder":
                    size = [shape.parameters["radius"], shape.parameters["length"]]
                attributes = {
                    "type": shape.kind,
                    "size": text(primitive_dimensions(shape.kind, size)),
                }
                pose = shape.T_link_geometry
            collision = ET.SubElement(
                body,
                "geom",
                name=geom,
                mass="0",
                group="3",
                rgba=".1 .8 .3 .3",
                friction=text([obj.get("friction", 1), 0.005, 0.0001]),
                **attributes,
                **pose_attributes(pose),
            )
            _contact_override(collision, obj)

    def initialize(self, robot, data):
        pass

    def settle(self, follower):
        # Research scenes start at their declared state. Tasks can provide
        # explicit settling and acceptance checks in their own adapter.
        return


def _contact_override(collision, obj):
    if "mujoco_contact_time_constant_s" in obj:
        collision.set("solref", text([obj["mujoco_contact_time_constant_s"], 1]))
        collision.set("priority", "1")
