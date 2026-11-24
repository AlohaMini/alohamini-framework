import io
import json
import os
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from alohamini.apps.collection.v3_worker import collection_publisher
from alohamini.datasets.experience import v3
from alohamini.datasets.experience.reader import V3Observations
from alohamini.datasets.experience.storage import save_snapshot
from alohamini.datasets.images import decode_host_image
from alohamini.protocol import HostSnapshot
from alohamini.rl.rlt.replay import materialize_episode


def _episode(root, sequence, *, scored=True):
    directory = root / ("episodes" if scored else "pending") / f"{sequence:032x}"
    directory.mkdir(parents=True)
    names = ["left", "right"]
    # The Host uses the channels OpenCV expects for JPEG compression. A standard
    # player must see orange, green and purple in this order.
    rgb = np.zeros((32, 96, 3), dtype=np.uint8)
    for j, color in enumerate(((240, 100, 12), (25, 193, 30), (132, 22, 188))):
        rgb[:, j * 32 : (j + 1) * 32] = color
    encoded = io.BytesIO()
    Image.fromarray(rgb[:, :, ::-1]).save(encoded, format="JPEG", quality=95)
    wire = encoded.getvalue()

    def write_observation(path, i):
        payload = {
            "left": float(i),
            "right": 3.0,
            "_robot_metadata": {"robot_model": "test"},
            "_safety": {"feedback_valid": True, "host_session_id": "host", "control_epoch": 1},
        }
        save_snapshot(path, HostSnapshot(payload, {"chest": wire}, 10.0 + i / 30, 10.0 + i / 30))

    commands, rows, outcomes = [], [], []
    for i, (source, status) in enumerate(
        zip(
            ("VLA", "HUMAN", "VLA", "HOLD"),
            ("applied", "applied", "superseded", "applied"),
            strict=True,
        )
    ):
        ident = {
            "client_id": "pc",
            "host_session_id": "host",
            "control_epoch": 1,
            "sequence": i + 1,
        }
        commands.append(ident)
        rows.append(
            dict(
                index=i,
                command=ident,
                time=i / 30.0,
                critical=False,
                source=source,
                requested=[50.0, 3.0],
                reference=None if source == "HOLD" else [20.0, 3.0],
            )
        )
        outcomes.append(
            dict(
                command=ident,
                observed_s=i / 30.0,
                status=status,
                targets={"left": i + 1.0, "right": 3.0} if status == "applied" else {},
            )
        )
        write_observation(directory / "observations" / str(i), i)
    write_observation(directory / "terminal", len(rows))
    for filename, values in (("steps.jsonl", rows), ("acks.jsonl", outcomes)):
        (directory / filename).write_text("\n".join(map(json.dumps, values)), encoding="utf-8")
    contract = {"names": names, "learned_names": ["left"]}
    (root / "run-contract.json").write_text(
        json.dumps({"reference_contract": "vla-start-chunk-v1", "base": contract})
    )
    summary = dict(
        episode_id=directory.name,
        status="committed" if scored else "pending",
        base_contract=contract,
        started_at=sequence,
        task="place blocks",
        fps=30,
        training_valid=True,
        ended=True,
        reward=1,
        critical_entries=0,
        phase_labels="optional-operator-v2",
        requires_command_dispositions=True,
        requires_acknowledgements=True,
        end_fence=commands[-1],
    )
    (directory / "episode.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory, contract, wire


def test_scored_collection_publishes_compact_rgb_and_portable_replay(tmp_path):
    with collection_publisher(tmp_path):
        raw, contract, wire = _episode(tmp_path, 1)
        _episode(tmp_path, 2, scored=False)
    dataset = tmp_path / "lerobot_v3"
    info = json.loads((dataset / "meta/info.json").read_bytes())
    assert (info["total_episodes"], info["total_frames"], info["codebase_version"]) == (
        1,
        5,
        "v3.0",
    )
    feature = info["features"]["observation.images.chest"]["info"]
    assert (feature["video.codec"], feature["video.pix_fmt"]) == ("h264", "yuv420p")
    rows = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pylist()
    # Published v1 shards must remain directly mergeable by Arrow schema, including column order.
    assert [k for k in rows[0] if k.startswith(("rlt.", "next."))] == [
        "rlt.requested_action",
        "rlt.step_reference",
        "rlt.applied_mask",
        "rlt.source",
        "rlt.disposition",
        "rlt.is_terminal_observation",
        "rlt.action_valid",
        "rlt.has_reference",
        "rlt.critical",
        "next.done",
        "next.reward",
        "rlt.control_time",
    ]
    assert [r["rlt.action_valid"] for r in rows] == [True, True, False, False, False]
    assert [r["next.reward"] for r in rows] == [0, 1, 0, 0, 0]
    assert [r["next.done"] for r in rows] == [False, True, False, False, False]
    assert rows[-1]["rlt.is_terminal_observation"] and not any(rows[-1]["rlt.applied_mask"])
    side = dataset / "meta/rlt/episode_000000"
    reader = V3Observations(side)
    try:
        snap, images = reader.read(4)
        assert snap.payload["left"] == 4.0
        target = decode_host_image(wire).astype(np.int16)
        restored = images["chest"].astype(np.int16)
        assert np.abs(restored - target).mean() < 4
        assert np.abs(restored - target[:, :, ::-1]).mean() > 40
    finally:
        reader.close()
    # Demonstrate that neither images nor snapshots are read from the raw copy.
    raw.rename(tmp_path / "moved-original")

    def ref(snapshot, index, *, rgb_images):
        assert "chest" in rgb_images
        return SimpleNamespace(
            token=torch.ones(1, 2),
            proprio=torch.ones(1, 1),
            context=torch.tensor([0.0]),
            reference=torch.full((1, 2, 2), 20.0),
        )

    native = SimpleNamespace(
        names=contract["names"],
        manifest={"cameras": ["chest"]},
        fps=30,
        config=SimpleNamespace(),
        processor=SimpleNamespace(action_prefix=lambda a, context: a),
    )
    policy = SimpleNamespace(
        native=native,
        contract=contract,
        indices=[0],
        reference=ref,
        config=SimpleNamespace(chunk_size=2, replay_stride=2),
    )
    (window,) = materialize_episode(policy, side)
    np.testing.assert_array_equal(window["ref"][:, 0], [20.0, 20.0])
    np.testing.assert_array_equal(window["bc_target"][:, 0], [20.0, 2.0])
    np.testing.assert_array_equal(window["action"][:, 0], [1.0, 2.0])
    assert (
        window["valid"].sum() == 2
        and window["done"].item() == 1
        and window["reward_delay"].item() == 2
    )


def test_v3_append_recovers_partial_publication_and_is_idempotent(tmp_path, monkeypatch):
    _episode(tmp_path, 1)
    assert v3.export_v3(tmp_path)["total_episodes"] == 1
    _episode(tmp_path, 2)
    info_path = tmp_path / "lerobot_v3/meta/info.json"
    replace = os.replace

    def interrupted(source, target):
        if Path(target) == info_path:
            raise OSError("simulated interruption")
        return replace(source, target)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", interrupted)
        with pytest.raises(OSError, match="interruption"):
            v3.export_v3(tmp_path)
    assert json.loads(info_path.read_bytes())["total_episodes"] == 1
    info = v3.export_v3(tmp_path)
    assert info["total_frames"] == 10 and info["total_episodes"] == 2
    assert v3.export_v3(tmp_path) == info
    second = pq.read_table(tmp_path / "lerobot_v3/data/chunk-000/file-001.parquet")
    assert second["index"].to_pylist() == list(range(5, 10))


def test_platform_merge_and_action_training_use_scored_data(tmp_path):
    from alohamini.datasets.merge_v3 import merge_v3
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.rl.rlt.episodes import recorded_action_names, scored_episodes

    sources = []
    for number in (1, 2):
        root = tmp_path / str(number)
        _episode(root, number)
        v3.export_v3(root)
        sources.append(root / "lerobot_v3")
    merged = tmp_path / "merged"
    merge_v3(sources, merged)
    assert (merged / "recording.lock").is_file()
    assert recorded_action_names(merged) == ["left"]
    for eid, reward, directory in scored_episodes(merged):
        assert json.loads((directory / "episode.json").read_bytes())["episode_id"] == eid
        assert reward == 1
        with closing(V3Observations(directory)) as observations:
            snapshot, images = observations.read(4)
            assert snapshot.payload["left"] == 4.0 and "chest" in images
    samples = AlohaMiniDataset(
        merged, cameras=[], state="none", delta_indices={"action": [0]}, episodes=[0, 1]
    )
    assert [s["action"][0, 0].item() for s in samples] == [1, 2, 1, 2]
    assert samples.statistics(keys=["action"])["action"]["mean"][0] == 1.5


def test_token_reads_recorded_task_and_observations_at_checkpoint_fps(tmp_path):
    from alohamini.rl.rlt.token_training import TokenData

    for i in (1, 2):
        directory, _, _ = _episode(tmp_path, i)
        summary = directory / "episode.json"
        summary.write_text(json.dumps(dict(json.loads(summary.read_bytes()), task=f"task {i}")))
    v3.export_v3(tmp_path)
    root = tmp_path / "lerobot_v3"
    native = SimpleNamespace(
        manifest=dict(
            training={},
            state="none",
            cameras=["chest"],
            image_size=[32, 32],
            temporal_sampling=dict(fps=15, downsample_rate=2),
        ),
        processor=lambda batch: batch,
        fixed_input=lambda sample: sample,
    )
    config = SimpleNamespace(dataset=str(root), task="fallback", validation_fraction=0.5, seed=1)
    data = TokenData(config, SimpleNamespace(native=native))
    # Source frames 0 and 4: retain the terminal observation without supervising its action.
    observations = sorted(
        (data.dataset.locations[row], data.batch(i)["task"][0])
        for i, row in enumerate(data.dataset.sample_indices)
    )
    assert observations == [
        ((0, 0), "task 1"),
        ((0, 4), "task 1"),
        ((1, 0), "task 2"),
        ((1, 4), "task 2"),
    ]
    assert all("action" not in data.batch(i) for i in range(4))


def test_score_pending_and_export_portable_dataset(tmp_path):
    from alohamini.cli import _main

    root, output = tmp_path / "collection", tmp_path / "portable"
    pending, _, _ = _episode(root, 1, scored=False)
    score = [
        "collect",
        "score",
        "--output-dir",
        str(root),
        "--episode",
        pending.name,
        "--reward",
        "0",
    ]
    assert _main(score) == 0
    assert _main(score) == 0
    assert (
        _main(["dataset", "export", str(root), "--format", "lerobot-v3", "--output", str(output)])
        == 0
    )
    meta = json.loads((output / "meta/info.json").read_bytes())
    assert meta["alohamini_rlt"]["episode_ids"] == [pending.name]
    assert json.loads((output / "meta/rlt/episode_000000/episode.json").read_bytes())["reward"] == 0
