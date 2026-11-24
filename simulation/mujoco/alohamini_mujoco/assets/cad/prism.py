"""Write and ray-check the MuJoCo model of a shared hollow prism."""

from __future__ import annotations

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from alohamini.simulation.assets.cad.prism import read_hollow_prism
from alohamini.simulation.assets.provenance import file_sha256

from ...runtime.compilation import compile_xml


def export_hollow_prism(directory, definition, output, *, axis):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    primitives, metadata = read_hollow_prism(directory, definition, axis=axis)
    proof = metadata["proof"]
    model = ET.Element("mujoco", model="CAD hollow prism")
    ET.SubElement(model, "option", gravity="0 0 0")
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.65 0.65 0.65")
    asset = ET.SubElement(model, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.94 0.96 0.98",
        rgb2="0.88 0.91 0.94",
        width="128",
        height="128",
    )
    bounds = np.array([[p[0], p[-1]] for p in proof["plane_offsets_m"]])

    def fmt(values):
        return " ".join(format(float(x), ".17g") for x in values)

    ET.SubElement(
        model,
        "statistic",
        center=fmt(bounds.mean(1)),
        extent=format(float(np.ptp(bounds, axis=1).max()) * 1.5, ".17g"),
    )
    world = ET.SubElement(model, "worldbody")
    extent = float(np.ptp(bounds, axis=1).max())
    ET.SubElement(
        world,
        "light",
        pos=fmt(bounds.mean(1) + np.array([1, -1, 1]) * extent),
        dir="-1 1 -1",
        diffuse="0.7 0.7 0.7",
    )
    for i, primitive in enumerate(primitives):
        ET.SubElement(
            world,
            "geom",
            name=f"wall_{i}",
            type="box",
            mass="0",
            size=fmt(np.array(primitive.parameters["size"]) / 2),
            pos=fmt(primitive.T_link_geometry[:3, 3]),
            rgba=f"{0.2 + i * 0.12} 0.65 0.75 1",
        )
    xml = ET.tostring(model, encoding="unicode")
    native_model = compile_xml(xml)
    data = mujoco.MjData(native_model)
    mujoco.mj_forward(native_model, data)
    ray = np.eye(3)[axis]
    origin = bounds.mean(1)
    origin[axis] = bounds[axis, 0] - max(np.ptp(bounds, axis=1))
    geom_id = np.array([-1], dtype=np.int32)
    if mujoco.mj_ray(native_model, data, origin, ray, None, 1, -1, geom_id) != -1:
        raise ValueError("Native collision compilation filled the prism through-hole")
    for primitive in primitives:
        probe = primitive.T_link_geometry[:3, 3].copy()
        probe[axis] = origin[axis]
        if mujoco.mj_ray(native_model, data, probe, ray, None, 1, -1, geom_id) < 0:
            raise ValueError("Native collision compilation omitted a prism wall")
    proof["mujoco_through_hole_and_four_walls"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "component"
        stage.mkdir()
        (stage / "component.xml").write_text(xml + "\n")
        metadata["files"] = {"component.xml": file_sha256(stage / "component.xml")}
        (stage / "component.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return metadata
