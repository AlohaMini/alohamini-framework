"""Portable object assets with separately declared rendering and collision shapes."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..frames import rigid_transform
from .inertia import MassProperties
from .meshes import Mesh, read_mesh
from .shapes import Primitive


def _fields(value, required, optional=()):
    return (
        isinstance(value, dict)
        and set(required) <= value.keys()
        and value.keys() <= set(required) | set(optional)
    )


def asset_key(reference):
    if (
        not _fields(reference, ("id", "version"), ("sha256",))
        or not isinstance(reference["id"], str)
        or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", reference["id"]) is None
        or type(reference["version"]) is not int
        or reference["version"] < 1
    ):
        raise ValueError("Asset references require an id and a positive integer version")
    if "sha256" in reference and not (
        isinstance(reference["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is not None
    ):
        raise ValueError("Asset reference sha256 must be a lowercase SHA-256")
    return reference["id"], reference["version"]


@dataclass(frozen=True)
class AssetFile:
    sha256: str
    suffix: str
    data: bytes

    @property
    def filename(self):
        return f"catalog_{self.sha256}{self.suffix}"


@dataclass(frozen=True)
class MeshGeometry:
    file: AssetFile
    pose: np.ndarray
    scale: float


@dataclass(frozen=True)
class ObjectAsset:
    visual: MeshGeometry
    colliders: tuple
    properties: MassProperties | None
    rgba: tuple
    texture: AssetFile | None
    manifest: dict


class AssetCatalog:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        self.directory = path.parent
        data = json.loads(path.read_text())
        if (
            not _fields(data, ("schema_version", "assets"))
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or not isinstance(data["assets"], list)
        ):
            raise ValueError("Expected a schema_version 1 asset catalog")
        self.entries, self.resolved, self.files = {}, {}, {}
        for entry in data["assets"]:
            if not _fields(
                entry,
                ("id", "version", "visual", "collision"),
                ("mass_properties", "color"),
            ):
                raise ValueError("Invalid asset catalog entry")
            key = asset_key({"id": entry["id"], "version": entry["version"]})
            if key in self.entries:
                raise ValueError(f"Duplicate asset {key[0]}@{key[1]}")
            self.entries[key] = entry

    def _file(self, record, suffixes):
        name, digest = record["file"], record["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Asset files require a relative path and a SHA-256")
        path = (self.directory / name).resolve()
        suffix = path.suffix.lower()
        if not path.is_relative_to(self.directory) or suffix not in suffixes:
            raise ValueError(f"Invalid asset file location or type: {name}")
        key = path, digest
        if key not in self.files:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Asset file checksum mismatch: {name}")
            self.files[key] = AssetFile(digest, suffix, data)
        return self.files[key]

    def _mesh(self, row, pose, *, convex):
        if not _fields(row, ("file", "sha256"), ("scale",)):
            raise ValueError("Meshes require file, sha256 and optional positive scale")
        scale = row.get("scale", 1.0)
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Mesh scale must be a positive conversion to metres")
        file = self._file(row, {".obj"})
        if any(line.split()[:1] == [b"mtllib"] for line in file.data.splitlines()):
            raise ValueError("Declare OBJ appearance in the catalog instead of an MTL reference")
        mesh = read_mesh(file.data, file.filename)
        if convex:
            from .collision_geometry import signed_volume, validate_convex

            validate_convex(Mesh(mesh.vertices * scale, mesh.faces))
            if signed_volume(mesh) <= 0:
                raise ValueError("Asset collision mesh needs outward face winding")
        return MeshGeometry(file, pose, float(scale))

    def resolve(self, reference):
        key = asset_key(reference)
        if key not in self.entries:
            raise ValueError(f"Unknown asset {key[0]}@{key[1]}")
        if key not in self.resolved:
            self.resolved[key] = self._load(self.entries[key])
        asset = self.resolved[key]
        if "sha256" in reference and reference["sha256"] != asset.manifest["sha256"]:
            raise ValueError(f"Asset definition checksum mismatch: {key[0]}@{key[1]}")
        return asset

    def _load(self, entry):
        visual = entry["visual"]
        if not _fields(visual, ("file", "sha256"), ("scale", "T_asset_visual", "texture")):
            raise ValueError("Invalid asset visual mesh or texture definition")
        pose = rigid_transform(visual.get("T_asset_visual", np.eye(4)))
        mesh_fields = {k: v for k, v in visual.items() if k in ("file", "sha256", "scale")}
        visual_mesh = self._mesh(mesh_fields, pose, convex=False)
        texture = None
        if "texture" in visual:
            if not _fields(visual["texture"], ("file", "sha256")):
                raise ValueError("A texture requires a file and SHA-256")
            faces = [line.split(b"#", 1)[0].split() for line in visual_mesh.file.data.splitlines()]
            if not any(row[:1] == [b"vt"] for row in faces) or any(
                not all(len(ref.split(b"/")) > 1 and ref.split(b"/")[1] for ref in row[1:])
                for row in faces
                if row[:1] == [b"f"]
            ):
                raise ValueError("Textured OBJ faces require explicit UV coordinates")
            texture = self._file(visual["texture"], {".png"})
        if not isinstance(entry["collision"], list) or not entry["collision"]:
            raise ValueError("An asset needs explicit collision geometry")
        colliders = []
        for row in entry["collision"]:
            if not _fields(row, ("type",), ("parameters", "mesh", "T_asset_geometry")):
                raise ValueError("Invalid asset collision geometry")
            pose = rigid_transform(row.get("T_asset_geometry", np.eye(4)))
            if row["type"] == "mesh" and "mesh" in row and "parameters" not in row:
                colliders.append(self._mesh(row["mesh"], pose, convex=True))
            elif "mesh" not in row and "parameters" in row:
                colliders.append(Primitive(row["type"], row["parameters"], pose))
            else:
                raise ValueError("Use primitive parameters or an explicit convex mesh collider")
        properties = None
        if "mass_properties" in entry:
            data = entry["mass_properties"]
            if not _fields(data, ("mass_kg", "com_m", "central_inertia_kg_m2")) or type(
                data["mass_kg"]
            ) not in (int, float):
                raise ValueError("Dynamic assets need mass, centre of mass and central inertia")
            properties = MassProperties("asset", **data)
        rgba = np.asarray(entry.get("color", [1, 1, 1, 1]), dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError("Asset color requires four RGBA values in [0,1]")
        fingerprint = hashlib.sha256(
            json.dumps(entry, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return ObjectAsset(
            visual_mesh,
            tuple(colliders),
            properties,
            tuple(rgba),
            texture,
            {**entry, "sha256": fingerprint},
        )
