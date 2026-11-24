"""Registered v3 extensions that require metadata relocation or target validity rules."""

import json
from pathlib import Path

from alohamini.datasets.experience.merge import ScoredEpisodeMerge as ExperienceMerge
from alohamini.rl.rlt.merge import ScoredEpisodeMerge

EXTENSIONS = (ExperienceMerge, ScoredEpisodeMerge)


def merge_handlers(sources):
    supported = {handler.key for handler in EXTENSIONS}
    for source in sources:
        unknown = {k for k in source.info if k.startswith("alohamini_")} - supported
        if unknown:
            raise ValueError(f"Unsupported v3 extension for merge: {sorted(unknown)}")
    return [
        handler(sources) for handler in EXTENSIONS if any(handler.key in s.info for s in sources)
    ]


def action_validity_fields(info):
    declared = info.get("action_validity_fields")
    expected = [
        field for ext in EXTENSIONS if ext.key in info for field in ext.action_validity_fields
    ]
    if declared is None:
        return expected
    if not isinstance(declared, list) or not all(isinstance(key, str) for key in declared):
        raise ValueError("action_validity_fields must be a list of field names")
    if not set(expected) <= set(declared):
        raise ValueError("Dataset omits validity fields required by its registered extension")
    return declared


def collection_layout(root):
    """Select the raw scored-run schema from its durable collection contract."""
    from alohamini.datasets.experience.v3 import ExperienceFormat
    from alohamini.rl.rlt.v3 import RLTFormat

    contract = json.loads((Path(root) / "run-contract.json").read_bytes())
    for layout in (ExperienceFormat, RLTFormat):
        if {k: v for k, v in contract.items() if k != "base"} == layout.run_metadata:
            return layout
    raise ValueError(f"Unknown scored collection format at {root}")
