"""Publish scored RLT episodes as portable LeRobot v3, with RGB video and command provenance."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alohamini.datasets.image_statistics import ImageStatistics
from alohamini.datasets.lerobot_tools import aggregate_stats
from alohamini.datasets.lerobotv3 import (
    DATA_PATH,
    DEFAULT_FEATURES,
    EPISODE_PATH,
    VIDEO_PATH,
    _arrow_schema,
)
from alohamini.rlt.segments import training_ranges
from alohamini.rlt.storage import RunLock, atomic_json, json_lines, validate_training_episode
from alohamini.rlt.v3_video import prepare_videos, snapshot_directories
from alohamini.schema import CommandIdentity

SOURCES = {"HOLD": 0, "VLA": 1, "RLT": 2, "HUMAN": 3, "TERMINAL": 4}
DISPOSITIONS = {"none": 0, "applied": 1, "superseded": 2, "rejected": 3}
VERSION = 1


def _json(path):
    return json.loads(path.read_bytes())


def _features(names, videos):
    vector = {"dtype": "float32", "shape": [len(names)], "names": names}
    features = {
        **DEFAULT_FEATURES,
        "observation.state": vector,
        "action": vector,
        "rlt.requested_action": vector,
        "rlt.step_reference": vector,
        "rlt.applied_mask": dict(vector, dtype="bool"),
    }
    for key in ("rlt.source", "rlt.disposition"):
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    for key in (
        "rlt.is_terminal_observation",
        "rlt.action_valid",
        "rlt.has_reference",
        "rlt.critical",
        "next.done",
    ):
        features[key] = {"dtype": "bool", "shape": [1], "names": None}
    features["next.reward"] = {"dtype": "float32", "shape": [1], "names": None}
    features["rlt.control_time"] = {"dtype": "float64", "shape": [1], "names": None}
    for cam, video in videos.items():
        info = video["info"]
        features["observation.images." + cam] = {
            "dtype": "video",
            "shape": [info["video.height"], info["video.width"], 3],
            "names": ["height", "width", "channels"],
            "info": info,
        }
    return features


def _table_stats(table):
    stats = {}
    for key in table.column_names:
        values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(len(table), -1)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {key}")
        stats[key] = {
            name: getattr(values, name)(axis=0).tolist() for name in ("min", "max", "mean", "std")
        }
        stats[key]["count"] = [len(table)]
    return stats


def _lists(stats):
    return {k: {s: np.asarray(v).tolist() for s, v in group.items()} for k, group in stats.items()}


def _mkdir(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _link_file(source, target):
    _mkdir(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _write_safety_frame(stream, snapshot, episode_index, frame_index):
    payload = snapshot["payload"]
    # RLT records receipt and command provenance, but no interval for sampling a
    # teleoperator action. The native safety schema supports a single client clock.
    record = {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "client_monotonic_s": snapshot["received_s"],
        "host_timing": payload.get("_host_timing", {}),
        "safety": payload.get("_safety", {}),
    }
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _close_safety(stream, episode_index, expected, written):
    event = {
        "type": "recorder_closed",
        "frame_count": written,
        "dropped_records": expected - written,
        "source": "committed_rlt_observations",
    }
    stream.write(
        json.dumps(
            {"episode_index": episode_index, "frame_index": None, "event": event},
            separators=(",", ":"),
        )
        + "\n"
    )


def _frame_table(episode, summary, side, safety_path, features, episode_index, offset, task_index):
    steps = json_lines(episode / "steps.jsonl")
    outcomes = {CommandIdentity(**r["command"]): r for r in json_lines(episode / "acks.jsonl")}
    dirs = snapshot_directories(episode)
    names = summary["base_contract"]["names"]
    learned = set(summary["base_contract"]["learned_names"])
    whole = (
        summary.get("phase_labels") == "optional-operator-v2"
        and summary.get("critical_entries") == 0
    )
    trainable = summary.get("training_valid", True) and not summary.get("evaluation", False)
    has_terminal = bool(dirs and dirs[-1].name == "terminal")
    if len(dirs) != len(steps) + int(has_terminal):
        raise ValueError(f"Step and observation counts disagree in {episode}")
    if trainable:
        validate_training_episode(episode, summary)
    rows, snapshot_batch = [], []
    schema = pa.schema([("snapshot", pa.binary())])
    with (
        pq.ParquetWriter(side / "observations.parquet", schema, compression="zstd") as snapshots,
        _mkdir(safety_path).open("w", encoding="utf-8") as safety,
    ):
        for i, directory in enumerate(dirs):
            raw = (directory / "state.json").read_bytes()
            snapshot = json.loads(raw)
            payload = snapshot["payload"]
            snapshot_batch.append(raw)
            if len(snapshot_batch) == 128 or i == len(dirs) - 1:
                snapshots.write_table(pa.table({"snapshot": snapshot_batch}, schema=schema))
                snapshot_batch.clear()
            _write_safety_frame(safety, snapshot, episode_index, i)
            terminal = i == len(steps)
            step = {} if terminal else steps[i]
            if not terminal and step["index"] != i:
                raise ValueError(f"Out-of-order step {i} in {episode}")
            outcome = {} if terminal else outcomes.get(CommandIdentity(**step["command"]), {})
            status = outcome.get("status", "applied" if outcome.get("targets") else "none")
            targets = (outcome.get("targets") or {}) if status == "applied" else {}
            state = [payload[n] for n in names]
            requested = step.get("requested", state)
            source = step.get("source", "TERMINAL")
            critical = bool(step.get("critical", False))
            valid = bool(
                trainable
                and (whole or critical)
                and source in ("HUMAN", "RLT", "VLA")
                and learned <= targets.keys()
                and status == "applied"
            )
            ref = step.get("reference")
            rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": i,
                    "index": offset + i,
                    "task_index": task_index,
                    "timestamp": i / summary["fps"],
                    "observation.state": state,
                    "action": [targets.get(n, requested[j]) for j, n in enumerate(names)],
                    "rlt.applied_mask": [n in targets for n in names],
                    "rlt.requested_action": requested,
                    "rlt.step_reference": ref if ref is not None else [0.0] * len(names),
                    "rlt.has_reference": ref is not None,
                    "rlt.source": SOURCES[source],
                    "rlt.disposition": DISPOSITIONS[status],
                    "rlt.action_valid": valid,
                    "rlt.is_terminal_observation": terminal,
                    "rlt.critical": critical,
                    "rlt.control_time": step.get("time", snapshot["received_s"]),
                    "next.done": False,
                    "next.reward": 0.0,
                }
            )
        _close_safety(safety, episode_index, len(dirs), len(rows))
    if trainable:
        for start, stop, reward in training_ranges(summary, len(steps)):
            for i in range(stop - 1, start - 1, -1):
                if rows[i]["rlt.action_valid"]:
                    rows[i].update({"next.reward": float(reward), "next.done": True})
                    break
    return pa.Table.from_pylist(rows, schema=_arrow_schema(features))


def _write_episode(root, stage, output, episode, summary, info):
    index = info["total_episodes"] if info else 0
    offset = info["total_frames"] if info else 0
    chunk, file_index = divmod(index, 1000)
    parts = {"chunk_index": chunk, "file_index": file_index}
    transaction = stage / "publish"
    # A transaction with a ready marker is recovered before entering this function.
    if transaction.exists():
        shutil.rmtree(transaction)
    transaction.mkdir(parents=True)
    side = transaction / "meta/rlt" / f"episode_{index:06d}"
    side.mkdir(parents=True)
    videos_dir = stage / "video" / episode.name
    videos = prepare_videos(episode, videos_dir, summary["fps"])
    features = _features(summary["base_contract"]["names"], videos)
    if info and (features != info["features"] or info["fps"] != summary["fps"]):
        raise ValueError(f"LeRobot v3 camera, action schema or FPS changed in {episode}")
    tasks = list(pd.read_parquet(output / "meta/tasks.parquet").index) if info else []
    if summary["task"] not in tasks:
        tasks.append(summary["task"])
    task_index = tasks.index(summary["task"])
    table = _frame_table(
        episode,
        summary,
        side,
        transaction / "meta/safety" / f"episode_{index:06d}.jsonl",
        features,
        index,
        offset,
        task_index,
    )
    length = len(table)
    pq.write_table(table, _mkdir(transaction / DATA_PATH.format(**parts)), compression="zstd")
    stats = _table_stats(table)
    epmeta = {
        "episode_index": index,
        "length": length,
        "tasks": [summary["task"]],
        "data/chunk_index": parts["chunk_index"],
        "data/file_index": parts["file_index"],
        "meta/episodes/chunk_index": parts["chunk_index"],
        "meta/episodes/file_index": parts["file_index"],
        "dataset_from_index": offset,
        "dataset_to_index": offset + length,
    }
    image_stats = ImageStatistics()
    camera_manifest = {}
    for camera, video in videos.items():
        if video["frames"] != length:
            raise ValueError(f"{episode}: {camera} does not align with data rows")
        key = "observation.images." + camera
        path = VIDEO_PATH.format(video_key=key, **parts)
        _link_file(videos_dir / f"{camera}.mp4", transaction / path)
        image_stats.merge(ImageStatistics(video["histogram"]))
        for field in ("chunk_index", "file_index"):
            epmeta[f"videos/{key}/{field}"] = parts[field]
        epmeta[f"videos/{key}/from_timestamp"] = 0.0
        epmeta[f"videos/{key}/to_timestamp"] = length / summary["fps"]
        camera_manifest[camera] = {
            "path": path,
            "frames": length,
            "sha256": video["sha256"],
            "color_check": video["color_check"],
            "encoding": video["encoding"],
        }
    stats.update(_lists(image_stats.result()))
    for key, group in stats.items():
        for stat, value in group.items():
            epmeta[f"stats/{key}/{stat}"] = value
    pq.write_table(
        pa.Table.from_pylist([epmeta]),
        _mkdir(transaction / EPISODE_PATH.format(**parts)),
        compression="zstd",
    )
    for filename in ("episode.json", "steps.jsonl", "acks.jsonl"):
        if (episode / filename).is_file():
            shutil.copyfile(episode / filename, side / filename)
    terminal = episode / "terminal/state.json"
    if terminal.is_file():
        shutil.copyfile(terminal, _mkdir(side / "terminal/state.json"))
    atomic_json(
        side / "v3.json",
        dict(version=VERSION, cameras=camera_manifest, frames=length, episode_index=index),
    )
    # The official global moment statistics are mergeable. Per-episode ranges
    # and the signed frame data remain available independently of the run journal.
    stats_path = output / "meta/stats.json"
    combined = aggregate_stats([_json(stats_path), stats]) if info else aggregate_stats([stats])
    atomic_json(transaction / "meta/stats.json", _lists(combined))
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks).to_parquet(
        transaction / "meta/tasks.parquet"
    )
    ids = [*info["alohamini_rlt"]["episode_ids"], episode.name] if info else [episode.name]
    if not info:
        first = snapshot_directories(episode)[0]
        robot = _json(first / "state.json")["payload"]["_robot_metadata"]
        source = dict(
            features=features,
            action_validity_field="rlt.action_valid",
            cameras=list(videos),
            robot_metadata=robot,
            fps=summary["fps"],
            task=summary["task"],
            fixed_dimensions=summary["base_contract"].get("fixed_dimensions"),
        )
        atomic_json(transaction / "meta/alohamini.json", {"version": 1, "source_info": source})
        for file in ("run-contract.json", "run-config.json"):
            if (root / file).is_file():
                shutil.copyfile(root / file, transaction / "meta/rlt" / file)
        info = dict(
            codebase_version="v3.0",
            robot_type=robot["robot_model"],
            fps=summary["fps"],
            chunks_size=1000,
            data_files_size_in_mb=100,
            video_files_size_in_mb=200,
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
            features=features,
            alohamini_rlt=dict(
                version=VERSION,
                sources=SOURCES,
                dispositions=DISPOSITIONS,
                color="Host decoded once into RGB; H.264 yuv420p using the recording encoder",
                terminal="observation-only final row, action_valid=false",
            ),
        )
    info["alohamini_rlt"]["episode_ids"] = ids
    info.update(
        total_episodes=index + 1,
        total_frames=offset + length,
        total_tasks=len(tasks),
        splits={"train": f"0:{index + 1}"},
    )
    atomic_json(transaction / "meta/info.json", info)
    files = sorted(
        p.relative_to(transaction).as_posix() for p in transaction.rglob("*") if p.is_file()
    )
    files.remove("meta/info.json")
    # Generic readers discover episode metadata by glob. Publish that row only
    # after every frame, MP4 and RLT sidecar it refers to is readable.
    files.sort(key=lambda p: p.startswith("meta/episodes/"))
    atomic_json(
        transaction / "ready.json", {"episode": episode.name, "files": [*files, "meta/info.json"]}
    )


def _finish(stage, output):
    transaction = stage / "publish"
    marker = transaction / "ready.json"
    if not marker.is_file():
        return
    commit = _json(marker)
    if (
        not isinstance(commit["episode"], str)
        or len(commit["episode"]) != 32
        or any(c not in "0123456789abcdef" for c in commit["episode"])
    ):
        raise ValueError(f"Invalid episode ID in {marker}")
    for name in commit["files"]:
        source, target = transaction / name, output / name
        if not source.resolve().is_relative_to(
            transaction.resolve()
        ) or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError(f"Invalid file path in {marker}")
        if source.is_file():
            os.replace(source, _mkdir(target))
        elif not target.is_file():
            raise RuntimeError(f"Missing file in v3 publication: {name}")
    marker.unlink()
    shutil.rmtree(transaction)
    cache = stage / "video" / commit["episode"]
    if cache.is_dir():
        shutil.rmtree(cache)


def export_v3(root):
    """Publish all committed episodes. The same command resumes a partial export."""
    root = Path(root).expanduser().absolute()
    if not (root / "episodes").is_dir() and not (root / "run-contract.json").is_file():
        raise ValueError(f"No RLT recording at {root}")
    stage, output = root / ".rlt-v3-stage", root / "lerobot_v3"
    with RunLock(stage / "publish.lock"), RunLock(output / "recording.lock"):
        info_path = output / "meta/info.json"
        if (
            info_path.exists()
            and _json(info_path).get("alohamini_rlt", {}).get("version") != VERSION
        ):
            raise ValueError(f"Destination contains another dataset: {output}")
        if (
            output.exists()
            and any(path.name != "recording.lock" for path in output.iterdir())
            and not info_path.exists()
            and not (stage / "publish/ready.json").exists()
        ):
            raise ValueError(f"Destination is not an empty v3 directory: {output}")
        _finish(stage, output)
        info = _json(info_path) if info_path.is_file() else None
        known = set(info["alohamini_rlt"]["episode_ids"]) if info else set()
        entries = []
        for episode in (root / "episodes").glob("*"):
            if episode.is_dir() and episode.name not in known:
                summary = _json(episode / "episode.json")
                if (
                    summary.get("status") != "committed"
                    or not summary.get("ended")
                    or summary.get("episode_id") != episode.name
                    or type(summary.get("reward")) is not int
                    or summary["reward"] not in (0, 1)
                ):
                    raise ValueError(f"Invalid committed episode: {episode}")
                entries.append((episode, summary))
        failed = []
        for episode, summary in sorted(
            entries, key=lambda item: (item[1]["started_at"], item[0].name)
        ):
            try:
                _write_episode(root, stage, output, episode, summary, info)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                # This episode still has its source journal. Later valid episodes
                # can be published while its missing data is being investigated.
                failed.append(f"{episode.name}: {exc}")
                info = _json(info_path) if info_path.is_file() else None
                continue
            _finish(stage, output)
            info = _json(info_path)
            print(
                f"LeRobot v3 已保存第 {info['total_episodes']} 回合 {episode.name}，"
                "视频 RGB 校验通过。",
                flush=True,
            )
        if failed:
            raise ValueError("未导出以下回合，原始文件已保留：" + " | ".join(failed))
        return info
