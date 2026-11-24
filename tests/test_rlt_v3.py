import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from alohamini.datasets.images import decode_host_image
from alohamini.protocol import HostSnapshot
from alohamini.rlt import v3
from alohamini.rlt.replay import materialize_episode
from alohamini.rlt.storage import save_snapshot
from alohamini.rlt.v3_reader import V3Observations
from alohamini.rlt.v3_worker import collection_publisher


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
    replace = v3.os.replace

    def interrupted(source, target):
        if Path(target) == info_path:
            raise OSError("simulated interruption")
        return replace(source, target)

    with monkeypatch.context() as m:
        m.setattr(v3.os, "replace", interrupted)
        with pytest.raises(OSError, match="interruption"):
            v3.export_v3(tmp_path)
    assert json.loads(info_path.read_bytes())["total_episodes"] == 1
    info = v3.export_v3(tmp_path)
    assert info["total_frames"] == 10 and info["total_episodes"] == 2
    assert v3.export_v3(tmp_path) == info
    second = pq.read_table(tmp_path / "lerobot_v3/data/chunk-000/file-001.parquet")
    assert second["index"].to_pylist() == list(range(5, 10))
