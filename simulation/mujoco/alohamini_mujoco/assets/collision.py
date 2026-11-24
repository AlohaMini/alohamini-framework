"""Reproducible CAD-derived convex candidates in named URDF link frames.

No task dimensions, jaw angle correction, hull-count cap, vertex decimation,
or extrusion enters this generator. Concavity is specified in metres.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from alohamini.simulation.assets.collision_geometry import (
    cad_components,
    collision_sources,
    quantized_convex,
    signed_volume,
    topology,
    validate_convex,
    visual_sources,
    weld_precision_seams,
)
from alohamini.simulation.assets.meshes import Mesh
from alohamini.simulation.assets.shapes import Primitive
from alohamini.simulation.frames import points_in_frame

from .mesh_codec import binary_msh, msh_bytes, read_mesh


def generate_bundle(
    description, links, output, *, tolerance_m=0.0005, seed=0, nonmanifold="reject"
):
    try:
        import coacd
    except ImportError as exc:
        raise ValueError(
            "Install the simulation geometry extra to generate collision candidates"
        ) from exc

    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Concavity tolerance must be a positive distance in metres")
    if type(seed) is not int or seed < 0 or not links or len(set(links)) != len(links):
        raise ValueError("Use unique links and a nonnegative integer seed")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if nonmanifold not in ("reject", "remesh"):
        raise ValueError("Nonmanifold handling must be reject or remesh")
    # Preflight every source before spending time in the native decomposition.
    prepared = {}
    for link in links:
        sources = visual_sources(description, link)
        prepared[link] = []
        for source in sources:
            data = (description.robot_model.directory / source["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"CAD source changed during preparation: {source['path']}")
            raw = read_mesh(data, source["path"])
            vertices = points_in_frame(
                source["T_link_visual"], raw.vertices * np.asarray(source["scale"])
            )
            # Weld before splitting: float32 seams can disconnect a solid.
            welded, weld = weld_precision_seams(Mesh(vertices, raw.faces), 5e-8)
            parts, partition = cad_components(welded)
            for index, part in enumerate(parts):
                shape_topology = topology(part)
                if not any(shape_topology.values()) and (
                    np.linalg.matrix_rank(part.vertices - part.vertices[0]) < 3
                    or signed_volume(part) <= 0
                ):
                    raise ValueError(
                        f"{link} CAD component {index} is a degenerate or inward shell; "
                        "solid/cavity attribution requires the original CAD. "
                        "No collision thickness or filled cavity was fabricated."
                    )
                if any(shape_topology.values()) and nonmanifold == "reject":
                    raise ValueError(
                        f"{link} CAD component {index} is not a closed oriented "
                        f"manifold: {shape_topology}. No geometry was generated; "
                        "repair the source or explicitly select remesh candidates."
                    )
            prepared[link].append((source, parts, weld, partition))
    output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(
        threshold=tolerance_m,
        real_metric=True,
        max_convex_hull=-1,
        decimate=False,
        extrude=False,
        seed=seed,
        resolution=2000,
        mcts_iterations=150,
        mcts_nodes=20,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        apx_mode="ch",
        preprocess_resolution=100,
    )
    manifest = {
        "schema_version": 1,
        "status": "candidate; requires geometry and contact validation",
        "robot_model": description.robot_model.model_id,
        "algorithm": "CoACD",
        "algorithm_version": version("coacd"),
        "parameters": options,
        "nonmanifold_handling": nonmanifold,
        "threshold_interpretation": (
            "CoACD collision-aware concavity target; not a certified surface error bound"
        ),
        "weld_displacement_bound_m": 5e-8,
        "transport": "float32 MSH; convex boundary retriangulated after quantization",
        "runtime": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "units": "m",
        "links": {},
    }
    cache = {}
    coacd.set_log_level("error")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        for link in links:
            sources = [source for source, _, _, _ in prepared[link]]
            # File paths are not part of geometric identity: mirrored entries
            # may share work only when their bytes, scale and local pose match.
            geometry_key = hashlib.sha256(
                json.dumps(
                    [{k: v for k, v in source.items() if k != "path"} for source in sources],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if geometry_key in cache:
                record = json.loads(json.dumps(cache[geometry_key]))
                paths = {
                    old["path"]: new["path"]
                    for old, new in zip(record["source_visuals"], sources, strict=True)
                }
                for component in record["components"]:
                    component["source_path"] = paths[component["source_path"]]
                for piece in record["pieces"]:
                    piece["source_path"] = paths[piece["source_path"]]
                record["source_visuals"] = sources
                manifest["links"][link] = record
                continue
            record = {
                "frame": "URDF link-local",
                "source_visuals": sources,
                "components": [],
                "pieces": [],
            }
            for source, parts, weld, partition in prepared[link]:
                for component_index, component in enumerate(parts):
                    shape_topology = topology(component)
                    # Native remeshing is opt-in and recorded. An edge check is
                    # not a proof against self intersections; candidates still
                    # require geometric and physical contact validation.
                    mode = "on" if any(shape_topology.values()) else "off"
                    print(
                        f"Decomposing {link}: component {component_index + 1}/{len(parts)}, "
                        f"{len(component.faces)} triangles, remesh={mode}, "
                        f"concavity target {tolerance_m:g} m",
                        file=sys.stderr,
                        flush=True,
                    )
                    pieces = coacd.run_coacd(
                        coacd.Mesh(component.vertices, component.faces),
                        preprocess_mode=mode,
                        **options,
                    )
                    if not pieces:
                        raise RuntimeError(f"CoACD returned no collision pieces for {link}")
                    print(f"Generated {len(pieces)} convex pieces", file=sys.stderr, flush=True)
                    record["components"].append(
                        {
                            "source_component": component_index,
                            "source_faces": len(component.faces),
                            "source_path": source["path"],
                            "whole_source_weld": weld,
                            "source_partition": {
                                "algorithm": partition["algorithm"],
                                **partition["parts"][component_index],
                            },
                            "topology_after_weld": shape_topology,
                            "preprocess_mode": mode,
                            "generated_pieces": len(pieces),
                        }
                    )
                    for piece_index, (vertices, faces) in enumerate(pieces):
                        try:
                            hull, transport = quantized_convex(
                                Mesh(
                                    np.asarray(vertices, dtype=np.float64),
                                    np.asarray(faces, dtype=np.int32),
                                )
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"{link} CAD component {component_index} hull {piece_index}: {exc}"
                            ) from exc
                        data = msh_bytes(hull)
                        digest = hashlib.sha256(data).hexdigest()
                        name = f"hulls/{digest}.msh"
                        (stage / name).write_bytes(data)
                        record["pieces"].append(
                            {
                                "path": name,
                                "sha256": digest,
                                "triangles": len(hull.faces),
                                "source_path": source["path"],
                                "source_component": component_index,
                                "native_piece": piece_index,
                                "transport": transport,
                            }
                        )
            record["geometry_identity"] = geometry_key
            cache[geometry_key] = record
            manifest["links"][link] = record
        for link, record in manifest["links"].items():
            if record["source_visuals"] != visual_sources(description, link):
                raise ValueError(f"CAD source changed during decomposition: {link}")
        (stage / "collision.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during generation: {output}")
        os.rename(stage, output)
    return output


def load_bundle(description, directory):
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "collision.json"
    manifest_bytes = manifest_path.read_bytes()
    metadata = json.loads(manifest_bytes)
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") not in (1, 2)
        or metadata.get("units") != "m"
        or metadata.get("robot_model") != description.robot_model.model_id
        or not isinstance(metadata.get("links"), dict)
        or (not metadata["links"] and "assembly_inventory" not in metadata)
        or metadata.get("coverage", "complete-link") != "complete-link"
    ):
        raise ValueError("Collision bundle has an incompatible coordinate contract")

    def validate_generation(definition, represented_links):
        if "assembly_inventory" in definition:
            from .organization import validate_inventory

            inventory = definition["assembly_inventory"]
            if not isinstance(inventory, dict):
                raise ValueError("Invalid whole-robot collision inventory")
            validate_inventory(description, inventory)
            if set(inventory.get("replaced_links", [])) != represented_links:
                raise ValueError("Whole-robot collision replacement coverage changed")
            for link in represented_links:
                if len(metadata["links"][link].get("pieces", [])) != inventory["links"][link].get(
                    "organized_shape_count", inventory["links"][link].get("recovered_convex_count")
                ):
                    raise ValueError(f"Whole-robot collision piece coverage changed: {link}")
        for parent in definition.get("source_manifests", []):
            if (
                not isinstance(parent, dict)
                or not isinstance(parent.get("generator"), dict)
                or not isinstance(parent.get("links"), list)
                or not set(parent["links"]) <= represented_links
            ):
                raise ValueError("Invalid collision composition provenance")
            validate_generation(parent["generator"], set(parent["links"]))

    validate_generation(metadata, set(metadata["links"]))
    result = {}
    for link, record in metadata["links"].items():
        if (
            not isinstance(record, dict)
            or record.get("frame") != "URDF link-local"
            or record.get("source_visuals") != visual_sources(description, link)
        ):
            raise ValueError(f"Collision candidate does not match current CAD frames: {link}")
        if "source_collisions" in record and record["source_collisions"] != collision_sources(
            description, link
        ):
            raise ValueError(
                f"Collision candidate does not match current collision sources: {link}"
            )
        if not isinstance(record.get("pieces"), list):
            raise ValueError(f"Invalid collision pieces: {link}")
        if "cad_geometry" in record:
            from alohamini.simulation.assets.cad.collision import validate_cad_qualification

            validate_cad_qualification(record, link)
        pieces = []
        for piece in record.get("pieces", []):
            if (
                isinstance(piece, dict)
                and metadata["schema_version"] == 1
                and piece.get("type", "mesh") != "mesh"
            ):
                raise ValueError("Native collision shapes require schema 2")
            if isinstance(piece, dict) and metadata["schema_version"] == 2:
                if piece.get("type") in ("box", "sphere", "cylinder"):
                    pieces.append(Primitive.from_record(piece))
                    continue
                if piece.get("type") != "mesh":
                    raise ValueError(f"Unknown collision shape type: {link}")
                if {"parameters", "scale", "T_link_geometry"} & piece.keys():
                    raise ValueError("Convex mesh vertices must already be in the link frame")
            if (
                not isinstance(piece, dict)
                or not isinstance(piece.get("path"), str)
                or not piece["path"]
                or not isinstance(piece.get("sha256"), str)
                or len(piece["sha256"]) != 64
            ):
                raise ValueError(f"Invalid collision piece definition: {link}")
            path = (directory / piece["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Collision mesh path escapes its bundle")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != piece["sha256"]:
                raise ValueError("Collision mesh checksum mismatch")
            mesh = binary_msh(data)
            validate_convex(mesh)
            if piece.get("triangles") != len(mesh.faces):
                raise ValueError("Collision mesh triangle count mismatch")
            pieces.append(mesh)
        if not pieces:
            raise ValueError(f"Empty collision replacement: {link}")
        result[link] = pieces
    return result, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "definition": metadata,
    }


def merge_bundles(description, directories, output):
    """Combine validated, disjoint link replacements without regenerating any hull."""
    if len(directories) < 2:
        raise ValueError("Use at least two collision bundles")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    sources, links = [], {}
    for directory in directories:
        directory = Path(directory).expanduser().resolve()
        _, provenance = load_bundle(description, directory)
        definition = provenance["definition"]
        if links.keys() & definition["links"].keys():
            raise ValueError("Collision bundles replace overlapping links")
        links.update(json.loads(json.dumps(definition["links"])))
        sources.append((directory, provenance))
    manifest = {
        "schema_version": max(p["definition"]["schema_version"] for _, p in sources),
        "units": "m",
        "robot_model": description.robot_model.model_id,
        "status": "composed CAD candidate; requires geometry and contact validation",
        "algorithm": "composition of independently validated bundles",
        "source_manifests": [
            {
                "sha256": p["manifest_sha256"],
                "generator": {k: v for k, v in p["definition"].items() if k != "links"},
                "links": list(p["definition"]["links"]),
            }
            for _, p in sources
        ],
        "links": links,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        for directory, provenance in sources:
            for link, record in provenance["definition"]["links"].items():
                for index, piece in enumerate(record["pieces"]):
                    if "path" not in piece:
                        continue
                    data = (directory / piece["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != piece["sha256"]:
                        raise ValueError("Collision hull changed during composition")
                    relative = f"hulls/{piece['sha256']}.msh"
                    (stage / relative).write_bytes(data)
                    manifest["links"][link]["pieces"][index]["path"] = relative
                    if manifest["schema_version"] == 2:
                        manifest["links"][link]["pieces"][index]["type"] = "mesh"
            if (
                hashlib.sha256((directory / "collision.json").read_bytes()).hexdigest()
                != provenance["manifest_sha256"]
            ):
                raise ValueError("Collision definition changed during composition")
        (stage / "collision.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return output
