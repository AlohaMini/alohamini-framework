"""MuJoCo representation of shared link-local mass properties."""

import mujoco
import numpy as np

from alohamini.simulation.assets.inertia import principal_inertia


def mjcf_attributes(properties):
    """Explicit inertial attributes for a body whose local frame is properties.frame."""
    moments, axes = principal_inertia(properties.central_inertia_kg_m2)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(axes).ravel())
    return {
        key: " ".join(format(float(v), ".17g") for v in values)
        for key, values in {
            "mass": [properties.mass_kg],
            "pos": properties.com_m,
            "quat": quaternion,
            "diaginertia": moments,
        }.items()
    }
