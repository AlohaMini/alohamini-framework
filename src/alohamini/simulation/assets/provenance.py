"""File checksums and robot description provenance."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def description_hashes(description):
    return {
        path.relative_to(description.robot_model.directory).as_posix(): hashlib.sha256(
            data
        ).hexdigest()
        for path, data in description.source_bytes.items()
    }


def matches_description_hashes(description, expected):
    """Keep prepared bundles usable when identical source files are renamed.

    Both URDF and SRDF payloads must match, including multiplicity. Path changes
    never permit different joint frames, geometry references or inertial values.
    """
    actual = description_hashes(description)
    return (isinstance(expected, dict) and len(expected) == len(actual)
            and all(isinstance(v, str) for v in expected.values())
            and sorted(expected.values()) == sorted(actual.values()))
