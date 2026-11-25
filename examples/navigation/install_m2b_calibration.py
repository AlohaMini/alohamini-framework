"""Install the reviewed 2026-11-24 documents for the stationary-pose M2b dry run."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import zipfile

import json
import yaml

STATUS = "accepted_ground_plane_20261124"
REVIEW_SHA256 = "f5c088a1457020e253e81aa7eb5c50e29981ccc508d56366f206c86818b23046"


def install(archive, root):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != REVIEW_SHA256:
        raise ValueError("review archive digest mismatch")
    documents = {}
    with zipfile.ZipFile(archive) as z:
        for cam in ("forward", "chest"):
            raw = yaml.safe_load(z.read(f"intrinsic_review/historical_candidates/{cam}.yaml"))
            rect = json.loads(z.read(f"floor_pair/review_config/{cam}.json"))
            ext = yaml.safe_load(z.read(f"floor_pair/{cam}-ground-candidate.yaml"))
            raw.update(status=STATUS, image_geometry="raw", review_sha256=REVIEW_SHA256)
            rectified = deepcopy(raw)
            matrix = rect["K_new"]
            rectified.update(image_geometry="rectified", rectification=rect)
            rectified["camera_matrix"]["data"] = sum(matrix, [])
            rectified["distortion_coefficients"]["data"] = [0.] * 5
            rectified["projection_matrix"]["data"] = sum((row + [0.] for row in matrix), [])
            ext.update(status=STATUS, review_sha256=REVIEW_SHA256, deployment_scope="m2b_dry_run")
            documents[f"intrinsics/{cam}.yaml"] = raw
            documents[f"intrinsics/{cam}_rectified.yaml"] = rectified
            documents[f"extrinsics/{cam}.yaml"] = ext
    files = {root / k: yaml.safe_dump(v, sort_keys=False).encode() for k, v in documents.items()}
    for f, data in files.items():
        if f.exists() and f.read_bytes() != data:
            raise FileExistsError(f"existing different calibration: {f}")
    for f, data in files.items():
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=f.parent, delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(f)
    return list(files)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--root", type=Path, default=Path.home()/"Alohamini_workspace/calibration/cameras")
    a = p.parse_args()
    print(*install(a.archive, a.root), sep="\n")
