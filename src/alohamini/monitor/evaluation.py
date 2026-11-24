"""Open local evaluation recordings by ID and project numeric Parquet columns only."""

from __future__ import annotations

import re
from array import array
from pathlib import Path

from .files import directories, document, identifier, stamp
from .history import Series


def fields(info, column):
    feature = info.get("features", {}).get(column, {})
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    shape = feature.get("shape", [])
    if len(shape) != 1 or type(shape[0]) is not int or not 1 <= shape[0] <= 4096:
        return []
    if (
        isinstance(names, list)
        and len(names) == shape[0]
        and all(isinstance(n, str) for n in names)
        and len(set(names)) == len(names)
    ):
        return names
    return [f"{column}[{i}]" for i in range(shape[0])]


def field_unit(info, column, name, index):
    units = info["features"][column].get("units")
    if isinstance(units, dict):
        return units.get(name)
    if isinstance(units, list) and index < len(units):
        return units[index]
    return None


class EvaluationLibrary:
    def __init__(self, roots):
        self.roots = [path.expanduser().resolve() for path in roots]

    def paths(self):
        paths = {}
        for base in self.roots:
            candidates = [base] if (base / "meta/info.json").is_file() else directories(base)
            for root in candidates:
                info = root / "meta/info.json"
                if info.resolve().is_relative_to(root) and stamp(info):
                    paths[identifier(root)] = root
        return paths

    def listing(self):
        return [
            {"id": key, "name": path.name}
            for key, path in sorted(self.paths().items(), key=lambda pair: pair[1].name)
        ]

    def root(self, key):
        try:
            return self.paths()[key]
        except KeyError:
            raise KeyError("Unknown evaluation dataset") from None

    def details(self, key, run=None):
        root = self.root(key)
        info = document(root / "meta/info.json")
        if info.get("format") == "alohamini-episodes":
            fmt = "AlohaMini"
            episodes = []
            lengths = []
            for path in sorted(directories(root / "episodes")):
                if re.fullmatch(r"episode_\d{6}", path.name) and stamp(path / "frames.parquet"):
                    episodes.append(int(path.name[8:]))
                    lengths.append(document(path / "episode.json").get("length"))
            frames = sum(lengths) if all(type(v) is int for v in lengths) else None
        elif str(info.get("codebase_version", "")).startswith("v3"):
            fmt = "LeRobot v3"
            count = info.get("total_episodes")
            if type(count) is not int or not 0 <= count <= 1_000_000:
                raise ValueError("Invalid total_episodes in meta/info.json")
            episodes = None
            frames = info.get("total_frames")
        else:
            raise ValueError("Expected an AlohaMini or LeRobot v3 dataset")
        metadata = root / "meta/evaluation.json"
        provenance = document(metadata) if metadata.resolve().is_relative_to(root) else {}
        checkpoint = provenance.get("checkpoint") or {}
        recorded = checkpoint.get("path")
        match = "unverified"
        # Compare only a local manifest at the explicitly recorded numbered checkpoint.
        # Saved paths may refer to a different machine.
        checkpoint_dir = Path(recorded).parent.name if isinstance(recorded, str) else ""
        if run and re.fullmatch(r"\d+", checkpoint_dir) and checkpoint.get("manifest_sha256"):
            from alohamini.analysis.provenance import digest

            manifest = run.root / "checkpoints" / checkpoint_dir / "pretrained_model/policy.json"
            if manifest.resolve().is_relative_to(run.root) and stamp(manifest):
                match = (
                    "match" if digest(manifest) == checkpoint["manifest_sha256"] else "different"
                )
        return {
            "id": key,
            "name": root.name,
            "format": fmt,
            "frames": frames,
            "episode_count": count if episodes is None else len(episodes),
            "episode_ids": episodes,
            "fps": info.get("fps"),
            "task": provenance.get("task") or info.get("task"),
            "fields": fields(info, "action"),
            "checkpoint": recorded,
            "manifest_match": match,
            "policy": checkpoint.get("kind"),
        }

    def trajectory(self, key, episode, field, budget):
        try:
            import pyarrow.compute as pc
            import pyarrow.dataset as ds
        except ImportError as exc:
            raise ValueError("Evaluation trajectories require pyarrow") from exc

        root = self.root(key)
        info = document(root / "meta/info.json")
        action_names = fields(info, "action")
        if field not in action_names:
            raise ValueError("Unknown action field")
        columns = {"action": action_names.index(field)}
        units = {k: field_unit(info, k, field, i) for k, i in columns.items()}
        states = fields(info, "observation.state")
        if field in states:
            i = states.index(field)
            state_unit = field_unit(info, "observation.state", field, i)
            if units["action"] == state_unit:
                columns["observation.state"] = i
                units["observation.state"] = state_unit

        if info.get("format") == "alohamini-episodes":
            paths = [root / f"episodes/episode_{episode:06d}/frames.parquet"]
        elif str(info.get("codebase_version", "")).startswith("v3"):
            indexes = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
            if indexes:
                if any(not p.resolve().is_relative_to(root) for p in indexes):
                    raise ValueError("Invalid episode index path")
                index = ds.dataset([str(p) for p in indexes], format="parquet")
                rows = index.to_table(
                    columns=["data/chunk_index", "data/file_index"],
                    filter=ds.field("episode_index") == episode,
                ).to_pylist()
                if len(rows) != 1:
                    raise ValueError("Missing or duplicate episode index")
                relative = info["data_path"].format(
                    chunk_index=rows[0]["data/chunk_index"], file_index=rows[0]["data/file_index"]
                )
                paths = [root / relative]
            else:
                # Numeric-only archives may retain the data shards without the episode index.
                paths = sorted((root / "data").glob("chunk-*/*.parquet"))
        else:
            raise ValueError("Unsupported dataset format")
        if not paths or any(not p.resolve().is_relative_to(root) or not p.is_file() for p in paths):
            raise ValueError("Missing episode Parquet data")
        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        projected = {
            "frame_index": ds.field("frame_index"),
            **{k: pc.list_element(ds.field(k), i) for k, i in columns.items()},
        }
        frames = array("q")
        series = {k: Series() for k in columns}
        for batch in dataset.to_batches(
            columns=projected, filter=ds.field("episode_index") == episode, batch_size=8192
        ):
            first = len(frames)
            for frame in batch.column("frame_index").to_pylist():
                if type(frame) is not int or (frames and frame <= frames[-1]):
                    raise ValueError("Invalid or unordered episode frame_index")
                frames.append(frame)
            for k, values in series.items():
                for row, v in enumerate(batch.column(k).to_pylist(), first):
                    values.append(row, v)
        if not frames:
            raise ValueError("Episode has no frames")
        return {
            "field": field,
            "unit": units["action"],
            "frames": len(frames),
            "axis": "frame",
            "series": {k: s.plot(frames, 0, len(frames), budget) for k, s in series.items()},
        }
