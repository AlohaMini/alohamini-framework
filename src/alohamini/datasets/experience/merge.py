"""Relocate scored episode journals and their references to merged v3 video files."""

import json
import shutil

from alohamini.datasets.lerobotv3 import VIDEO_PATH
from alohamini.datasets.record import _write_json


class ScoredEpisodeMerge:
    directory = "meta/experience"
    key = "alohamini_experience"
    action_validity_fields = ("experience.action_valid", "experience.applied_mask")

    def __init__(self, sources):
        self.ids = []
        self.seen = set()
        self.header = None
        for source in sources:
            metadata = source.info.get(self.key, {})
            ids = metadata.get("episode_ids", [])
            if len(ids) != len(source.episode_rows) or len(set(ids)) != len(ids):
                raise ValueError("Invalid scored episode index")
            header = {k: v for k, v in metadata.items() if k != "episode_ids"}
            if header.get("version") != 1 or (self.header and self.header != header):
                raise ValueError("Incompatible scored episode metadata")
            self.header = header

    def copy_episode(self, source, old, row, stage):
        eid = source.info[self.key]["episode_ids"][old]
        if eid in self.seen:
            raise ValueError(f"Duplicate scored episode ID: {eid}")
        source_dir = source.root / self.directory / f"episode_{old:06d}"
        new = row["episode_index"]
        dest = stage / self.directory / f"episode_{new:06d}"
        if source_dir.is_symlink() or any(p.is_symlink() for p in source_dir.rglob("*")):
            raise ValueError("Scored episode sidecars must be regular local files")
        shutil.copytree(source_dir, dest)
        for name in ("run-config.json", "run-contract.json"):
            origin = source_dir.parent / name
            if origin.is_symlink():
                raise ValueError("Scored run metadata must be local files")
            if origin.is_file() and not (dest / name).exists():
                shutil.copyfile(origin, dest / name)
        if json.loads((dest / "episode.json").read_bytes())["episode_id"] != eid:
            raise ValueError("Scored episode ID differs from its journal")
        index_path = dest / "v3.json"
        index = json.loads(index_path.read_bytes())
        index["episode_index"] = new
        for camera, spec in index["cameras"].items():
            key = "observation.images." + camera
            spec["path"] = VIDEO_PATH.format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
        _write_json(index_path, index)
        self.seen.add(eid)
        self.ids.append(eid)

    def metadata(self):
        return {self.key: {**self.header, "episode_ids": self.ids}}
