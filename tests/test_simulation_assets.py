import subprocess
import sys

import pytest


def test_shared_asset_modules_import_without_simulator_packages():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import pkgutil
import sys

for name in ("mujoco", "alohamini_mujoco", "sapien"):
    sys.modules[name] = None

from alohamini.simulation import assets

for module in pkgutil.walk_packages(assets.__path__, assets.__name__ + "."):
    importlib.import_module(module.name)
""",
        ],
        check=True,
    )
