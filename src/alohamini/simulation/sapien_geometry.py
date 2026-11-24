"""PhysX-specific preparation of candidate shared convex meshes."""

import tempfile
from pathlib import Path

import numpy as np

from .assets.collision_geometry import quantized_convex
from .assets.meshes import Mesh, obj_bytes, surface_distances
from .assets.solid_geometry import clip_convex_halfspace, convex_boundary


def prepare_shared_convex(piece, *, tolerance_m=None):
    """Bake PhysX preparation into the shared source and check its fixed point.

    Cooking may merge near-coplanar points even below the vertex limit. The
    resulting source must be certified against CAD before either backend loads
    it. Millimetre transport matches the generated OBJ files and URDF scales.
    """
    import sapien

    original = piece
    for _ in range(8):
        # Use the actual OBJ import path: the file loader can reorder vertices,
        # which changes near-coplanar QuickHull choices compared with array input.
        with tempfile.TemporaryDirectory(prefix="alohamini-convex-") as temporary:
            path = Path(temporary) / "piece.obj"
            path.write_bytes(obj_bytes(Mesh(piece.vertices.astype(float) * 1000, piece.faces)))
            shape = sapien.physx.PhysxCollisionShapeConvexMesh(
                str(path), [0.001] * 3, sapien.physx.get_default_material()
            )
        actual = quantized_convex(
            convex_boundary(np.asarray(shape.get_vertices()).astype(float) / 1000)
        )[0]
        error = max(
            float(surface_distances(first.vertices, second.triangles).max())
            for first, second in ((piece, actual), (actual, piece))
        )
        if error < 0.000001:
            break
        piece = actual
    else:
        raise ValueError("PhysX convex preparation did not reach a stable shared mesh")
    if (
        tolerance_m is None
        or max(
            float(surface_distances(first.vertices, second.triangles).max())
            for first, second in ((original, piece), (piece, original))
        )
        < tolerance_m
    ):
        return [piece]
    # A large flat contact pad can lose curved boundary detail during cooking.
    # Split its footprint along its longest extent until each cooked patch
    # meets the declared contact tolerance; retain the original plane.
    axis = np.eye(3)[np.argmax(np.ptp(original.vertices, axis=0))]
    offset = float((original.vertices @ axis).min() + (original.vertices @ axis).max()) / 2
    halves = [clip_convex_halfspace(original, sign * axis, sign * offset) for sign in (1, -1)]
    return [
        p
        for half in halves
        for p in prepare_shared_convex(quantized_convex(half)[0], tolerance_m=tolerance_m)
    ]
