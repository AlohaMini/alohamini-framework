"""RLT replay eligibility and metadata for raw or published scored episodes."""

import json
from pathlib import Path

from alohamini.datasets.experience import storage

RUN_METADATA = {"reference_contract": "vla-start-chunk-v1"}


def replay_step(summary, row):
    whole = (
        summary.get("phase_labels") == "optional-operator-v2"
        and summary.get("critical_entries") == 0
    )
    return (whole or row.get("critical", False)) and row["source"] in ("VLA", "RLT", "HUMAN")


def validate_training_episode(directory, summary):
    if summary.get("collection_format", "rlt") != "rlt":
        raise ValueError("RLT training requires an RLT collection")
    if (
        summary.get("training_valid", True)
        and not summary.get("requires_acknowledgements")
        and "started_at" not in summary
    ):
        return  # Legacy replay fixtures and recordings predate command receipts.
    storage.validate_training_episode(
        directory, summary, include_step=replay_step, receipt_field_key="learned_names"
    )


def commit_episode(directory, reward):
    return storage.commit_episode(directory, reward, validate=validate_training_episode)


class EpisodeRecorder(storage.EpisodeRecorder):
    def commit(self, reward):
        return commit_episode(self.directory, reward)


def scored_episodes(root):
    if (root / "episodes").is_dir():
        candidates = [(p.name, p) for p in sorted((root / "episodes").iterdir()) if p.is_dir()]
    else:
        if (root / "lerobot_v3/meta/info.json").is_file():
            root = root / "lerobot_v3"
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise ValueError(f"No RLT episodes or published LeRobot v3 metadata in {root}")
        info = json.loads(info_path.read_bytes())
        ids = info.get("alohamini_rlt", {}).get("episode_ids", [])
        if len(ids) != info["total_episodes"] or len(set(ids)) != len(ids):
            raise ValueError("Invalid RLT episode index in v3 metadata")
        candidates = [(eid, root / "meta/rlt" / f"episode_{i:06d}") for i, eid in enumerate(ids)]
    result = []
    for eid, path in candidates:
        summary = json.loads((path / "episode.json").read_bytes())
        if summary.get("collection_format", "rlt") != "rlt":
            raise ValueError("RLT training requires an RLT collection")
        if summary["episode_id"] != eid:
            raise ValueError(f"Episode ID differs from its dataset index in {path}")
        if (
            summary.get("status") != "committed"
            or not summary.get("ended")
            or not summary.get("training_valid", True)
            or summary.get("evaluation")
            or summary.get("writer_error")
        ):
            continue
        reward = summary.get("reward")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError(f"Invalid operator reward in {path}")
        result.append((eid, reward, path))
    if not result:
        raise ValueError("No completed, scored, training-valid RLT episodes")
    return result


def recorded_action_names(root):
    """Read the coordinate selection when token training starts from a scored run."""
    root = Path(root).expanduser()
    for contract in (root / "run-contract.json", root / "meta/rlt/run-contract.json"):
        if contract.is_file():
            return json.loads(contract.read_bytes())["base"]["learned_names"]
    if (root / "meta/rlt").is_dir():
        # Merged scored datasets retain the original contract per episode.
        selection = None
        for _, _, directory in scored_episodes(root):
            names = json.loads((directory / "episode.json").read_bytes())["base_contract"][
                "learned_names"
            ]
            if selection is not None and names != selection:
                raise ValueError(
                    "Different scored coordinate selections: specify --learned-action-names"
                )
            selection = names
        return selection
    return None


def training_ranges(summary, step_count):
    """Return (first action, exclusive stop / next observation, local reward)."""
    annotation = summary.get("training_segments")
    if annotation is None:
        if summary.get("phase_labels") in (
            "video-reviewed-intervals-v1",
            "video-reviewed-placements-v1",
        ):
            raise ValueError("Reviewed episode is missing its training segments")
        return [(0, step_count, summary["reward"])] if step_count else []
    if not isinstance(annotation, dict) or annotation.get("version") != 1:
        raise ValueError("Unsupported training segment format")
    segments = annotation.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Training segments must be a list")
    result = []
    previous_stop = 0
    for segment in segments:
        if (
            not isinstance(segment, dict)
            or not isinstance(segment.get("name"), str)
            or not segment["name"].strip()
        ):
            raise ValueError("Training segment is missing a name")
        start, stop, reward = (
            segment.get(k) for k in ("start_frame", "end_frame_exclusive", "reward")
        )
        if (
            type(start) is not int
            or type(stop) is not int
            or not previous_stop <= start < stop <= step_count
        ):
            raise ValueError(f"Invalid or overlapping training interval [{start}, {stop})")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Training segment needs a binary outcome")
        result.append((start, stop, reward))
        previous_stop = stop
    return result
