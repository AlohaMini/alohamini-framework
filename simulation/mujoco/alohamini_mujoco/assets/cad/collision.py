"""MuJoCo mesh transport for qualified CAD convex collision candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from alohamini.simulation.assets.cad.collision import (
    _decomposition,
    _file,
    _mesh_identity,
    _parameters,
    _registration,
    _source,
    _target_mesh,
    qualify_collision_geometry,
    validate_cad_qualification,
)
from alohamini.simulation.assets.collision_geometry import (
    collision_sources,
    validate_convex,
    visual_sources,
)
from alohamini.simulation.assets.provenance import file_sha256

from ..mesh_codec import binary_msh, msh_bytes


def build_cad_collision(
    description,
    mesh_bundle,
    registrations,
    output,
    *,
    tolerance_m,
    concavity_m=0.0005,
    seed=0,
    reuse_decomposition=None,
    refine=False,
    merge_verified=False,
    component_only=False,
):
    """Export either complete-link replacements or isolated component assets.

    Component assets never publish collision.json and cannot replace a link.
    """
    directory, output = (
        Path(mesh_bundle).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    if not registrations or not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Supply complete-link registrations and a positive metric error budget")
    source, metadata, fingerprint = _source(directory)
    bindings = [
        _registration(
            description, path, fingerprint, metadata, source, component_only=component_only
        )
        for path in registrations
    ]
    if len({b[0] for b in bindings}) != len(bindings):
        raise ValueError("CAD collision registrations must select distinct links")
    parameters = _parameters(concavity_m, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        (stage / "geometry").mkdir()
        raw_pieces, decomposition_hash = _decomposition(
            source, metadata, fingerprint, stage / "decomposition", parameters, reuse_decomposition
        )
        records, reports, cached = {}, {}, {}
        for link, pose, registration, registration_hash in bindings:
            key = pose.tobytes()
            if key not in cached:
                cached[key] = qualify_collision_geometry(
                    source, raw_pieces, pose, tolerance_m=tolerance_m,
                    refine=refine, merge_verified=merge_verified,
                )
            parts, boundary, certificate, refinement, simplification = cached[key]
            pieces = []
            for part in parts:
                data = msh_bytes(part)
                digest = hashlib.sha256(data).hexdigest()
                relative = f"hulls/{digest}.msh"
                (stage / relative).write_bytes(data)
                pieces.append(
                    {
                        "type": "mesh",
                        "path": relative,
                        "sha256": digest,
                        "triangles": len(part.faces),
                    }
                )
            qualification = {
                "passed": certificate["passed"],
                "coverage": "visual-component" if component_only else "complete-link",
                "target_geometry_sha256": _mesh_identity(
                    _target_mesh(description, registration, component_only=component_only)
                ),
                "source_mesh_manifest_sha256": fingerprint,
                "registration_sha256": registration_hash,
                "registration": registration,
                "certificate": certificate,
                "refinement": refinement,
                "simplification": simplification,
                "collision_piece_sha256": [p["sha256"] for p in pieces],
            }
            reports[link] = qualification
            records[link] = {
                "frame": "URDF link-local",
                "source_visuals": visual_sources(description, link),
                "source_collisions": collision_sources(description, link),
                "pieces": pieces,
                "cad_geometry": qualification,
            }
            if certificate["passed"]:
                validate_cad_qualification(records[link], link, component_only=component_only)
            np.savez_compressed(
                stage / "geometry" / f"{link}.npz", vertices=boundary.vertices, faces=boundary.faces
            )
        summary = {"passed": all(r["passed"] for r in reports.values()), "links": reports}
        if (
            file_sha256(directory / "mesh.json") != fingerprint
            or any(
                _registration(
                    description, path, fingerprint, metadata, source, component_only=component_only
                )[3]
                != binding[3]
                for path, binding in zip(registrations, bindings, strict=True)
            )
            or any(path.read_bytes() != value for path, value in description.source_bytes.items())
        ):
            raise ValueError("CAD or robot registration changed during collision generation")
        for record in metadata["solids"]:
            _file(directory, record["file"], record["sha256"])
        (stage / "qualification.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if summary["passed"]:
            manifest = {
                "schema_version": 2,
                "units": "m",
                "robot_model": description.robot_model.model_id,
                "algorithm": "registered closed reference convex collision",
                "status": (
                    "reference material-distance verified candidate; "
                    "contact and dynamics qualification required"
                ),
                "source_mesh_manifest_sha256": fingerprint,
                "source_mesh_manifest": metadata,
                "decomposition_sha256": decomposition_hash,
                "parameters": parameters,
                "geometry_options": {"refine": refine, "merge_verified": merge_verified},
                "coverage": "visual-component" if component_only else "complete-link",
                "links": records,
            }
            (stage / ("component.json" if component_only else "collision.json")).write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return summary


def load_cad_components(description, directory):
    """Load isolated components without making a full-link coverage claim."""
    directory = Path(directory).expanduser().resolve()
    fingerprint = file_sha256(directory / "component.json")
    metadata = json.loads((directory / "component.json").read_text())
    if (
        metadata.get("schema_version") != 2
        or metadata.get("units") != "m"
        or metadata.get("robot_model") != description.robot_model.model_id
        or metadata.get("coverage") != "visual-component"
        or not metadata.get("links")
    ):
        raise ValueError("Use an explicitly scoped registered CAD component bundle")
    meshes = {}
    for link, record in metadata["links"].items():
        if (
            record.get("frame") != "URDF link-local"
            or record.get("source_visuals") != visual_sources(description, link)
            or record.get("source_collisions") != collision_sources(description, link)
        ):
            raise ValueError("CAD component does not match current robot sources")
        validate_cad_qualification(record, link, component_only=True)
        selected = _target_mesh(
            description, record["cad_geometry"]["registration"], component_only=True
        )
        if _mesh_identity(selected) != record["cad_geometry"].get("target_geometry_sha256"):
            raise ValueError("CAD component target selection changed since qualification")
        parts = []
        for piece in record["pieces"]:
            if (
                piece.get("type") != "mesh"
                or {"parameters", "scale", "T_link_geometry"} & piece.keys()
            ):
                raise ValueError("CAD component meshes must already use the link frame")
            path = _file(directory, piece["path"], piece["sha256"])
            mesh = binary_msh(path.read_bytes())
            validate_convex(mesh)
            if len(mesh.faces) != piece.get("triangles"):
                raise ValueError("CAD component triangle count mismatch")
            parts.append(mesh)
        if not parts:
            raise ValueError("CAD component needs at least one qualified convex piece")
        meshes[link] = parts
    if file_sha256(directory / "component.json") != fingerprint:
        raise ValueError("CAD component changed during loading")
    return meshes, {"manifest_sha256": fingerprint, "definition": metadata}
