"""Extract an exact open rectangular prism from its original CAD solid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..shapes import Primitive
from .step import file_sha256


def hollow_prism(shape, axis):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.gp import gp_Pnt
    from OCP.GProp import GProp_GProps
    from OCP.Precision import Precision
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    if type(axis) is not int or axis not in (0, 1, 2) or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("Select an explicit part-frame axis and a valid CAD solid")
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    if not solids.More():
        raise ValueError("A hollow prism needs exactly one original solid")
    solids.Next()
    if solids.More():
        raise ValueError("A hollow prism needs exactly one original solid")
    offsets = [[], [], []]
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        surface = BRepAdaptor_Surface(TopoDS.Face_s(faces.Current()))
        if surface.GetType() != GeomAbs_Plane:
            raise ValueError("Hollow-prism extraction accepts planar faces only")
        plane = surface.Plane()
        normal = np.array(plane.Axis().Direction().Coord())
        coordinate = int(np.abs(normal).argmax())
        expected = np.eye(3)[coordinate] * np.sign(normal[coordinate])
        if np.linalg.norm(normal - expected) > 256 * np.finfo(float).eps:
            raise ValueError("CAD faces are not aligned to the declared part frame")
        offsets[coordinate].append(plane.Location().Coord()[coordinate] * 0.001)
        faces.Next()
    planes = []
    for values in offsets:
        values = sorted(values)
        if not values:
            raise ValueError("Missing prism boundary planes")
        guard = 128 * np.finfo(float).eps * max(1.0, np.abs(values).max())
        unique = [values[0]]
        for value in values[1:]:
            if value - unique[-1] > guard:
                unique.append(value)
        planes.append(unique)
    transverse = [i for i in range(3) if i != axis]
    if len(planes[axis]) != 2 or any(len(planes[i]) != 4 for i in transverse):
        raise ValueError("CAD plane arrangement is not a hollow rectangular prism")
    u, v = transverse
    outer_low = np.array([p[0] for p in planes])
    outer_high = np.array([p[-1] for p in planes])
    intervals = [
        (u, planes[u][0], planes[u][1], v, planes[v][0], planes[v][-1]),
        (u, planes[u][2], planes[u][3], v, planes[v][0], planes[v][-1]),
        (v, planes[v][0], planes[v][1], u, planes[u][1], planes[u][2]),
        (v, planes[v][2], planes[v][3], u, planes[u][1], planes[u][2]),
    ]
    primitives, native = [], []
    for first, lo, hi, second, other_lo, other_hi in intervals:
        low, high = outer_low.copy(), outer_high.copy()
        low[first], high[first], low[second], high[second] = lo, hi, other_lo, other_hi
        size, center = high - low, low + (high - low) / 2
        pose = np.eye(4)
        pose[:3, 3] = center
        primitives.append(Primitive("box", {"size": size.tolist()}, pose))
        native.append(BRepPrimAPI_MakeBox(gp_Pnt(*(low * 1000)), *(size * 1000)).Shape())
    union = native[0]
    for wall in native[1:]:
        operation = BRepAlgoAPI_Fuse(union, wall)
        if not operation.IsDone():
            raise ValueError("Native hollow-prism union failed")
        union = operation.Shape()
    if not BRepCheck_Analyzer(union).IsValid():
        raise ValueError("Native hollow-prism union is invalid")
    for first, second in ((shape, union), (union, shape)):
        operation = BRepAlgoAPI_Cut(first, second)
        if not operation.IsDone():
            raise ValueError("Native hollow-prism equivalence check failed")
        residual = operation.Shape()
        if residual.IsNull() or not BRepCheck_Analyzer(residual).IsValid():
            raise ValueError("Native hollow-prism difference is invalid")
        if TopExp_Explorer(residual, TopAbs_FACE).More():
            raise ValueError("Native hollow-prism difference is nonempty; no feature was discarded")
    tolerances = []
    for kind in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            convert = {
                TopAbs_VERTEX: TopoDS.Vertex_s,
                TopAbs_EDGE: TopoDS.Edge_s,
                TopAbs_FACE: TopoDS.Face_s,
            }[kind]
            tolerances.append(BRep_Tool.Tolerance_s(convert(explorer.Current())) * 0.001)
            explorer.Next()
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, True, False, False)
    if not np.isfinite(properties.Mass()) or properties.Mass() <= 0:
        raise ValueError("CAD prism must have a finite positive oriented volume")
    return primitives, {
        "method": "axis-aligned CAD planes and bidirectional native solid difference",
        "axis": "xyz"[axis],
        "plane_offsets_m": planes,
        "source_minus_boxes_empty": True,
        "boxes_minus_source_empty": True,
        "source_brep_volume_m3": properties.Mass() * 1e-9,
        "box_volume_sum_m3": sum(np.prod(p.parameters["size"]) for p in primitives),
        "maximum_source_tolerance_m": max(tolerances),
        "occt_confusion_m": Precision.Confusion_s() * 0.001,
        "scope": (
            "native OCCT solid equivalence at kernel/source tolerances; not physical calibration"
        ),
    }


def read_hollow_prism(directory, definition, *, axis):
    """Read a frozen STEP definition and return the part-frame boxes and proof."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    directory = Path(directory).expanduser().resolve()
    fingerprint = file_sha256(directory / "cad.json")
    cad = json.loads((directory / "cad.json").read_text())
    record = cad["definitions"][definition]
    path = (directory / record["file"]).resolve()
    if (
        cad.get("schema_version") != 1
        or record.get("brep_units") != "mm"
        or not record.get("valid_brep")
        or record.get("solids") != 1
        or not path.is_relative_to(directory)
        or file_sha256(path) != cad["files"][record["file"]]
        or file_sha256(directory / "source.step") != cad["source_sha256"]
    ):
        raise ValueError("CAD primitive source contract or checksum mismatch")
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise ValueError("Cannot read original CAD prism")
    primitives, proof = hollow_prism(shape, axis)
    metadata = {
        "schema_version": 1,
        "units": "m",
        "frame": "CAD part definition frame",
        "status": "verified CAD component; robot rigid-body assignment required",
        "source_cad_sha256": cad["source_sha256"],
        "cad_manifest_sha256": fingerprint,
        "brep_sha256": cad["files"][record["file"]],
        "definition": definition,
        "name": record["name"],
        "occurrences": [o for o in cad["occurrences"] if o["definition"] == definition],
        "primitives": [
            {
                "type": p.kind,
                "parameters": p.parameters,
                "T_part_geometry": p.T_link_geometry.tolist(),
            }
            for p in primitives
        ],
        "proof": proof,
    }
    if (
        file_sha256(directory / "cad.json") != fingerprint
        or file_sha256(path) != metadata["brep_sha256"]
        or file_sha256(directory / "source.step") != cad["source_sha256"]
    ):
        raise ValueError("CAD source changed during primitive extraction")
    return primitives, metadata
