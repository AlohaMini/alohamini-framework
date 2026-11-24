"""Read scored observations from a LeRobot v3 episode and its standard MP4s."""

import json
from bisect import bisect_right
from pathlib import Path

import pyarrow.parquet as pq

from alohamini.datasets.video_reader import VideoFrameCache
from alohamini.protocol import HostSnapshot


class V3Observations:
    def __init__(self, directory, cameras=None):
        self.directory = Path(directory)
        root = self.directory.parents[2].resolve()
        manifest = json.loads((self.directory / "v3.json").read_bytes())
        self.paths = {}
        for camera in cameras if cameras is not None else manifest["cameras"]:
            path = (root / manifest["cameras"][camera]["path"]).resolve()
            if not path.is_relative_to(root / "videos"):
                raise ValueError(f"Invalid episode camera path in {self.directory}")
            self.paths[camera] = path
        self.videos = VideoFrameCache(max_size=len(self.paths))
        self.snapshots = pq.ParquetFile(self.directory / "observations.parquet")
        self.ends = []
        self.group = None
        self.records = None
        rows = 0
        for i in range(self.snapshots.num_row_groups):
            rows += self.snapshots.metadata.row_group(i).num_rows
            self.ends.append(rows)
        if rows != manifest["frames"]:
            self.close()
            raise ValueError("Snapshot count differs from episode video length")

    def read(self, index):
        if not 0 <= index < self.ends[-1]:
            raise IndexError(f"Frame {index} is outside this episode")
        group = bisect_right(self.ends, index)
        if group != self.group:
            self.records = self.snapshots.read_row_group(group, columns=["snapshot"])["snapshot"]
            self.group = group
        local = index - (self.ends[group - 1] if group else 0)
        snapshot = HostSnapshot(**json.loads(self.records[local].as_py()), images={})
        images = {camera: self.videos.read(path, index) for camera, path in self.paths.items()}
        return snapshot, images

    def close(self):
        self.videos.close()
        self.snapshots.close()
