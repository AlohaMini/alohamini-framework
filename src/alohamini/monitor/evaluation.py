"""Open local evaluation recordings by ID and project numeric Parquet columns only."""

from __future__ import annotations

import math
import re
from array import array
from pathlib import Path

from .files import directories, document, identifier, stamp
from .history import Series, number, quantile

TRAJECTORY_MODES = ("position", "residual", "velocity", "acceleration", "jerk")


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


def _finite_difference(frames, values, dt, order):
    """Order-k finite difference respecting non-contiguous frame indices."""
    cur_f = list(frames)
    cur_v = list(values)
    for _ in range(order):
        next_f, next_v = [], []
        for i in range(1, len(cur_f)):
            df = cur_f[i] - cur_f[i - 1]
            v0, v1 = cur_v[i - 1], cur_v[i]
            if df > 0 and math.isfinite(v0) and math.isfinite(v1):
                next_f.append(cur_f[i])
                next_v.append((v1 - v0) / (df * dt))
        cur_f, cur_v = next_f, next_v
    return cur_f, cur_v


def _residual_stats(action_vals, state_vals):
    diffs = [
        a - s
        for a, s in zip(action_vals, state_vals, strict=True)
        if math.isfinite(a) and math.isfinite(s)
    ]
    if not diffs:
        return None
    abs_diffs = sorted(abs(v) for v in diffs)
    n = len(diffs)
    return {
        "n": n,
        "bias": math.fsum(diffs) / n,
        "mae": math.fsum(abs_diffs) / n,
        "rmse": math.sqrt(math.fsum(v * v for v in diffs) / n),
        "p50": quantile(abs_diffs, 0.5),
        "p90": quantile(abs_diffs, 0.9),
        "p95": quantile(abs_diffs, 0.95),
        "max_abs": abs_diffs[-1],
    }


def _kinematic_stats(values):
    finite = sorted(abs(v) for v in values if math.isfinite(v))
    if not finite:
        return None
    n = len(finite)
    return {
        "n": n,
        "rms": math.sqrt(math.fsum(v * v for v in finite) / n),
        "p95_abs": quantile(finite, 0.95),
        "max_abs": finite[-1],
    }


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

    def trajectory(self, key, episode, field, budget, mode="position"):
        if mode not in TRAJECTORY_MODES:
            raise ValueError(f"Unsupported trajectory mode: {mode}")
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

        fps = info.get("fps")
        has_fps = number(fps) and fps > 0
        dt = 1.0 / float(fps) if has_fps else 1.0
        base_unit = units["action"]
        summary = {}
        if "observation.state" in series:
            summary["residual"] = _residual_stats(
                series["action"].values, series["observation.state"].values
            )

        if mode == "position":
            plotted = {k: s.plot(frames, 0, len(frames), budget) for k, s in series.items()}
            out_unit = base_unit
        elif mode == "residual":
            if "observation.state" not in series:
                raise ValueError("Residual requires matching observation.state with the same unit")
            res = Series()
            for row, (a, s) in enumerate(
                zip(series["action"].values, series["observation.state"].values, strict=True)
            ):
                res.append(row, a - s if math.isfinite(a) and math.isfinite(s) else None)
            plotted = {"residual": res.plot(frames, 0, len(frames), budget)}
            out_unit = base_unit
        else:
            order = {"velocity": 1, "acceleration": 2, "jerk": 3}[mode]
            plotted = {}
            for k, s in series.items():
                df_frames, df_vals = _finite_difference(frames, s.values, dt, order)
                if k == "action":
                    summary["kinematics"] = _kinematic_stats(df_vals)
                if not df_frames:
                    plotted[k] = {"count": 0, "points": []}
                    continue
                d_axis = array("q", df_frames)
                d_series = Series()
                for row, v in enumerate(df_vals):
                    d_series.append(row, v)
                plotted[k] = d_series.plot(d_axis, 0, len(d_axis), budget)
            time_unit = "s" if has_fps else "frame"
            suffix = {1: f"/{time_unit}", 2: f"/{time_unit}²", 3: f"/{time_unit}³"}[order]
            out_unit = (base_unit + suffix) if base_unit else ("unit" + suffix)

        return {
            "field": field,
            "unit": out_unit,
            "frames": len(frames),
            "fps": fps if has_fps else None,
            "mode": mode,
            "axis": "frame",
            "summary": summary,
            "series": plotted,
        }
