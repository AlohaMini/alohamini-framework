"""Mesh topology, convexity and geometry in named URDF frames."""

import hashlib
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from ..description import numbers, origin
from .meshes import Mesh


def topology(mesh):
    faces = mesh.faces
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, direction_counts = np.unique(directed, axis=0, return_counts=True)
    triangles = mesh.triangles.astype(float)
    area_twice = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    return {
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "repeated_directed_edges": int((direction_counts > 1).sum()),
        "zero_area_faces": int((area_twice == 0).sum()),
    }


def components(mesh):
    faces = mesh.faces
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    )
    count, labels = connected_components(graph, directed=False)
    result = []
    for index in range(count):
        selected = faces[labels[faces[:, 0]] == index]
        indices, inverse = np.unique(selected, return_inverse=True)
        result.append(Mesh(mesh.vertices[indices].copy(), inverse.reshape(-1, 3)))
    return result


def manifold_parts(mesh):
    """Partition triangle surfaces across edges with exactly two incident faces.

    Vertex contact and nonmanifold edges do not establish surface adjacency.
    This partitions indices only: it changes no triangle or vertex coordinate.
    """
    faces = mesh.faces
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, counts.cumsum()[:-1]][counts == 2]
    a, b = order[starts] % len(faces), order[starts + 1] % len(faces)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(faces), len(faces)))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.bincount(labels).cumsum()[:-1])


def cad_components(mesh):
    """Separate closed solids that share vertices/edges without remeshing them.

    Only complete groups of outward closed surfaces without potential nesting
    can split. Ambiguous, inward, open and degenerate patches retain their
    original grouping: independent solidification could fill a real cavity.
    """
    result, records = [], []
    for parent, part in enumerate(components(mesh)):
        surfaces, clean = [], []
        for indices in manifold_parts(part):
            vertices, inverse = np.unique(part.faces[indices], return_inverse=True)
            surface = Mesh(part.vertices[vertices], inverse.reshape(-1, 3))
            valid = (
                len(surface.vertices) >= 4
                and not any(topology(surface).values())
                and np.linalg.matrix_rank(surface.vertices - surface.vertices[0]) == 3
            )
            valid = valid and signed_volume(surface) > 0
            surfaces.append(surface)
            clean.append(valid)
        split = all(clean)
        if split and len(surfaces) > 1:
            lower = np.array([s.vertices.min(0) for s in surfaces])
            upper = np.array([s.vertices.max(0) for s in surfaces])
            for i in range(len(surfaces)):
                contains = np.all(lower[i] <= lower, axis=1) & np.all(upper[i] >= upper, axis=1)
                contains[i] = False
                if contains.any():
                    split = False
                    break
        for surface in surfaces if split else [part]:
            result.append(surface)
            records.append(
                {"vertex_component": parent, "faces": len(surface.faces), "closed_solid": split}
            )
    if sum(len(p.faces) for p in result) != len(mesh.faces):
        raise RuntimeError("CAD partition did not preserve all source triangles")
    return result, {
        "algorithm": "closed-manifold-surface-partition-v1",
        "source_triangles": len(mesh.faces),
        "parts": records,
    }


def signed_volume(mesh):
    triangles = mesh.triangles.astype(float) - mesh.vertices.mean(axis=0)
    return float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    )


def weld_precision_seams(mesh, tolerance_m):
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Weld tolerance must be finite and positive")
    points = mesh.vertices.astype(float)
    parent = np.arange(len(points))
    members, centers = {}, {}
    skipped = 0

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in sorted(cKDTree(points).query_pairs(tolerance_m)):
        a, b = find(a), find(b)
        if a == b:
            continue
        merged = members.get(a, [a]) + members.get(b, [b])
        center = points[merged].mean(axis=0)
        # Nearby edges do not bound the diameter of their transitive closure.
        # Admit each merge only if every original vertex stays within budget.
        if np.linalg.norm(points[merged] - center, axis=1).max() > tolerance_m:
            skipped += 1
            continue
        parent[a] = b
        members[b], centers[b] = merged, center
        members.pop(a, None)
        centers.pop(a, None)
    roots, inverse = np.unique([find(i) for i in range(len(points))], return_inverse=True)
    vertices = np.array([centers.get(i, points[i]) for i in roots])
    displacement = float(np.linalg.norm(points - vertices[inverse], axis=1).max())
    if displacement > tolerance_m:
        raise ValueError("Weld cluster exceeds the declared metric displacement bound")
    faces = inverse[mesh.faces]
    valid = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    )
    if not valid.any():
        raise ValueError("Welding left no nondegenerate faces")
    used, indexed = np.unique(faces[valid], return_inverse=True)
    return Mesh(vertices[used], indexed.reshape(-1, 3)), {
        "algorithm": "bounded-centroid-seam-weld-v1",
        "maximum_vertex_displacement_m": displacement,
        "merged_vertices": len(points) - len(vertices),
        "collapsed_faces": int((~valid).sum()),
        "skipped_merges_over_displacement_bound": skipped,
    }


def visual_sources(description, link):
    if link not in description.links:
        raise ValueError(f"Unknown link: {link}")
    sources = []
    for node in description.links[link].findall("visual"):
        shape = node.find("geometry/mesh")
        if shape is None:
            raise ValueError("CAD decomposition requires explicit visual meshes")
        path = description.mesh_path(shape)
        scale = numbers(shape.get("scale", "1 1 1"), 3, "visual mesh scale")
        if (scale <= 0).any():
            raise ValueError("Visual mesh scales must be positive")
        data = path.read_bytes()
        sources.append(
            {
                "path": path.relative_to(description.robot_model.directory).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "T_link_visual": origin(node).tolist(),
                "scale": scale.tolist(),
            }
        )
    if not sources:
        raise ValueError(f"Link has no CAD visual geometry: {link}")
    return sources


def collision_sources(description, link):
    """Bind collision provenance to the source shapes as well as visual CAD."""
    if link not in description.links:
        raise ValueError(f"Unknown link: {link}")
    result = []
    for index, node in enumerate(description.links[link].findall("collision")):
        geometry = node.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError(f"Expected one source collision shape: {link}")
        shape = geometry[0]
        record = {
            "index": index,
            "name": node.get("name"),
            "T_link_geometry": origin(node).tolist(),
            "type": shape.tag,
            "parameters": dict(shape.attrib),
        }
        if shape.tag == "mesh":
            path = description.mesh_path(shape)
            record["mesh"] = {
                "path": path.relative_to(description.robot_model.directory).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scale": numbers(shape.get("scale", "1 1 1"), 3, "collision scale").tolist(),
            }
        result.append(record)
    return result


def ordered_convex_parts(mesh):
    """Recover consecutive closed convex shells from an assembled collision STL.

    Shared vertices/edges cannot join neighboring pieces when each shell's
    original face range remains intact. Every triangle is assigned once; no
    weld, cap, normal reversal, simplification or decomposition is performed.
    Unordered, concave, inward or incomplete input is explicitly rejected.
    """
    start, unpaired, counts, parts, ranges = 0, 0, {}, [], []
    for index, face in enumerate(mesh.faces):
        for a, b in zip(face, np.roll(face, -1), strict=True):
            key = tuple(sorted((int(a), int(b))))
            count = counts.get(key, 0) + 1
            counts[key] = count
            if count > 2:
                raise ValueError("Source is not a sequence of closed convex shells")
            unpaired += 1 if count == 1 else -1
        if unpaired == 0:
            selected = mesh.faces[start : index + 1]
            used, faces = np.unique(selected, return_inverse=True)
            part = Mesh(mesh.vertices[used].copy(), faces.reshape(-1, 3))
            validate_convex(part)
            if signed_volume(part) <= 0:
                raise ValueError("Inward source shell cannot be interpreted as a convex solid")
            parts.append(part)
            ranges.append([start, index + 1])
            start, counts = index + 1, {}
    if counts or not parts:
        raise ValueError("Incomplete source convex shell; no missing faces were added")
    return parts, {
        "algorithm": "ordered-closed-convex-shells-v1",
        "source_triangles": len(mesh.faces),
        "face_ranges": ranges,
        "geometry_changed": False,
    }


def validate_convex(mesh):
    if (
        not np.isfinite(mesh.vertices).all()
        or len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or mesh.faces.min() < 0
        or mesh.faces.max() >= len(mesh.vertices)
    ):
        raise ValueError("Invalid convex mesh coordinates or indices")
    if any(topology(mesh).values()):
        raise ValueError("Convex collision pieces must be closed oriented triangle manifolds")
    if len(np.unique(mesh.faces)) != len(mesh.vertices) or len(components(mesh)) != 1:
        raise ValueError(
            "Convex collision pieces must have one connected surface and no unused vertices"
        )
    hull = ConvexHull(mesh.vertices)
    reference = mesh.vertices.mean(axis=0)
    triangles = mesh.triangles.astype(float) - reference
    volume = abs(
        float(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
            / 6
        )
    )
    if hull.volume <= 0 or not np.isclose(volume, hull.volume, rtol=2e-5, atol=1e-18):
        raise ValueError("Collision piece is concave or has inconsistent surface winding")


def quantized_convex(mesh):
    """Triangulate the convex boundary of the transported float32 vertices.

    Merely casting vertices can make a previously planar triangle fan concave.
    Reconstructing its hull retains the quantized convex envelope without a
    tolerance relaxation, vertex decimation, hull merging or feature removal.
    """
    validate_convex(mesh)
    vertices = mesh.vertices.astype(np.float32)
    error = float(np.linalg.norm(mesh.vertices - vertices.astype(float), axis=1).max())
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces].astype(float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    used, indexed = np.unique(faces, return_inverse=True)
    result = Mesh(vertices[used], indexed.reshape(-1, 3))
    validate_convex(result)
    return result, {
        "maximum_vertex_quantization_m": error,
        "native_vertices": len(mesh.vertices),
        "boundary_vertices": len(used),
        "native_triangles": len(mesh.faces),
    }
