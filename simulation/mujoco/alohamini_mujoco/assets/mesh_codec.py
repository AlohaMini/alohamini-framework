"""MuJoCo binary mesh codec; ordinary STL and OBJ use the shared reader."""

import struct
from pathlib import Path

import numpy as np

from alohamini.simulation.assets.meshes import Mesh
from alohamini.simulation.assets.meshes import read_mesh as read_common_mesh


def msh_bytes(mesh):
    # MSH has no STL decoder's 200,000-face restriction. Indexing exact
    # repeated float32 coordinates changes neither triangles nor winding.
    header = struct.pack("<4i", len(mesh.vertices), 0, 0, len(mesh.faces))
    return header + mesh.vertices.astype("<f4").tobytes() + mesh.faces.astype("<i4").tobytes()


def binary_msh(data):
    if len(data) < 16:
        raise ValueError("Truncated MSH")
    vertices, normals, textures, faces = struct.unpack_from("<4i", data)
    if (
        vertices < 4
        or faces < 4
        or normals not in (0, vertices)
        or textures not in (0, vertices)
        or len(data) != 16 + 12 * (vertices + normals + faces) + 8 * textures
    ):
        raise ValueError("Invalid MSH sizes")
    points = np.frombuffer(data, dtype="<f4", offset=16, count=3 * vertices).reshape(-1, 3)
    offset = 16 + 12 * (vertices + normals) + 8 * textures
    indices = np.frombuffer(data, dtype="<i4", offset=offset, count=3 * faces).reshape(-1, 3)
    if not np.isfinite(points).all() or indices.min() < 0 or indices.max() >= vertices:
        raise ValueError("Invalid MSH coordinates or indices")
    return Mesh(points.copy(), indices.copy())


def read_mesh(data, path):
    if Path(path).suffix.lower() == ".msh":
        return binary_msh(data)
    return read_common_mesh(data, path)
