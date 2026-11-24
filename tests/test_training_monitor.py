import json
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from alohamini.monitor.history import History
from alohamini.monitor.server import MonitorServer
from alohamini.monitor.workspace import Monitor, console_tail


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_history_cursors_partial_duplicate_resume_and_rotation(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    first = {"step": 9, "loss": 1, "samples": 50, "elapsed_s": 2.25, "metric_totals": {"a": 4}}
    path.write_bytes((json.dumps(first) + '\n{"step":9').encode())
    h = History(path)
    h.refresh()
    assert h.summary()["axes"] == ["step", "samples", "elapsed_s"]
    assert h.records(None, 10, False)["records"] == [first]
    cursor = h.cursor
    ingested = []
    append = h._append

    def count(row):
        ingested.append(row)
        append(row)

    monkeypatch.setattr(h, "_append", count)
    h.refresh()
    assert not ingested
    with path.open("a") as f:
        f.write(',"loss":0.5,"samples":50,"elapsed_s":2.5}\ninvalid\n')
        f.write('{"step":10,"loss":null,"samples":56,"elapsed_s":3,"metrics":{"a":3}}\n')
    h.refresh()
    assert len(ingested) == 2
    assert h.summary()["invalid_lines"] == 1 and h.summary()["nonfinite"] == {"loss": 1}
    delta = h.chart(["loss", "a"], "elapsed_s", None, None, 64, cursor)
    assert delta["mode"] == "append"
    assert delta["series"]["loss"]["points"] == [[2.5, 0.5], [3, None]]
    assert delta["series"]["a"]["points"] == [[3, 3]]
    assert [r["step"] for r in h.records(cursor, 1, False)["records"]] == [9]
    assert h.records(cursor, 1, False)["more"]
    assert h.records(h.cursor, 100, False)["records"] == []
    # Rotation to a larger file and in-place truncation invalidate cursor generations.
    path.rename(tmp_path / "archive.jsonl")
    path.write_text('{"step":0,"loss":2}\n' * 24)
    h.refresh()
    assert h.chart(["loss"], "step", None, None, 64, cursor)["mode"] == "replace"
    assert h.records(cursor, 2, True)["reset"]
    assert h.summary()["axes"] == ["step"]
    cursor = h.cursor
    path.write_text('{"step":1,"loss":3}\n')
    h.refresh()
    assert h.records(cursor, 3, True)["records"] == [{"step": 1, "loss": 3}]
    path.unlink()
    h.refresh()
    assert not h.summary()["records"]


def test_envelope_per_metric_zoom_and_gaps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with path.open("w") as f:
        for step in range(10000):
            value = {11: 999, 9942: -183, 505: math.nan}.get(step, step % 3)
            components = {} if step % 10 else {"rare": -333 if step == 870 else 1}
            f.write(json.dumps({"step": step, "loss": value, "metrics": components}) + "\n")
    h = History(path)
    h.refresh()
    data = h.chart(["loss", "rare"], "step", None, None, 32, None)["series"]
    points = data["loss"]["points"]
    assert points[0][0] == 0 and points[-1][0] == 9999
    assert len(points) <= 32 and data["loss"]["count"] == 10000
    assert [11, 999] in points and [9942, -183] in points and [505, None] in points
    assert [870, -333] in data["rare"]["points"] and data["rare"]["count"] == 1000
    zoom = h.chart(["loss"], "step", 9941, 9943, 32, None)
    assert zoom["series"]["loss"]["points"] == [[9941, 2], [9942, -183], [9943, 1]]
    assert h.records(None, 1, False)["records"][0]["metrics"] == {"rare": 1}
    assert h.chart(["loss"], "samples", None, None, 32, None)["series"] == {}
    assert h.summary()["nonfinite"] == {"loss": 1}


def test_catalog_separates_attempts_and_verified_launch_logs(tmp_path):
    runs = tmp_path / "runs"
    root, logs = runs / "exp", tmp_path / "logs/training"
    write_json(logs / "exp.json", {"output_dir": str(root), "policy": "act", "steps": 3})
    (logs / "exp.log").write_bytes(b"start\n")
    write_json(logs / "unrelated.json", {"output_dir": str(tmp_path / "other/exp")})
    (logs / "unrelated.log").write_text("other")
    m = Monitor(runs, logs, [])
    (pending,) = m.catalog.discover()
    assert pending["policy"] == "act"
    r = m.catalog.get(pending["id"])
    assert list(r.logs) == ["exp.log"] and m.summary(r)["phases"]["train"]["records"] == 0
    tail = console_tail(r, "exp.log", None)
    assert tail["text"] == "start\n" and tail["reset"]
    with (logs / "exp.log").open("ab") as f:
        f.write("评估\n".encode())
    assert console_tail(r, "exp.log", tail["cursor"])["text"] == "评估\n"
    root.mkdir(parents=True)
    for suffix in ("", "-from-1-110", "-from-1-220"):
        (root / f"metrics{suffix}.jsonl").write_text('{"step":2,"loss":1}\n')
        (root / f"validation-metrics{suffix}.jsonl").write_text('{"step":2,"loss":3}\n')
        write_json(
            root / (f"resume-{suffix[6:]}.json" if suffix else "train.json"),
            {"policy": "act", "seed": suffix},
        )
    m = Monitor(root, logs, [])
    entries = m.catalog.discover()
    assert {a["attempt"] for a in entries} == {None, "1-110", "1-220"}
    for entry in entries:
        r = m.catalog.get(entry["id"])
        summary = m.summary(r)
        assert summary["phases"]["validation"]["latest"]["loss"] == 3
        assert summary["config"]["seed"] == ("-from-" + r.suffix if r.suffix else "")
        assert "config" not in m.summary(r, brief=True)
        with pytest.raises(KeyError):
            console_tail(r, "unrelated.log", None)


def test_http_incremental_contract_and_shared_index(tmp_path):
    root = tmp_path / "runs/r"
    write_json(root / "train.json", {"policy": "act"})
    path = root / "metrics.jsonl"
    path.write_text('{"step":1,"loss":3}\n')
    m = Monitor(root.parent, tmp_path / "logs", [])
    origin = "http://localhost:8123"
    with MonitorServer(0, m, [origin]) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def get(route):
            with urlopen(Request(base + route, headers={"Origin": origin}), timeout=5) as res:
                assert res.headers["Access-Control-Allow-Origin"] == origin
                return json.load(res)

        try:
            with urlopen(base, timeout=5) as res:
                assert b"Training Monitor" in res.read()
            key = get("/api/v1/runs")["runs"][0]["id"]
            api = f"/api/v1/runs/{key}"
            cursor = get(api + "/series")["train"]["cursor"]
            with path.open("a") as f:
                f.write('{"step":1,"loss":2}\n')
            with ThreadPoolExecutor(max_workers=3) as pool:
                replies = list(pool.map(get, [api + "/series?after_train=" + cursor] * 3))
            assert all(
                r["train"]["mode"] == "append"
                and r["train"]["series"]["loss"]["points"] == [[1, 2]]
                for r in replies
            )
            assert get(api + "/records?limit=1&tail=1")["records"][0]["loss"] == 2
            with pytest.raises(HTTPError) as error:
                get(api + "/series?points=0")
            assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join(timeout=2)


@pytest.mark.parametrize("layout", ["native", "v3", "v3-indexed"])
def test_evaluation_projection_and_provenance(tmp_path, layout):
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = tmp_path / "datasets/pick"
    run = tmp_path / "runs/policy"
    write_json(run / "train.json", {"policy": "act"})
    manifest = run / "checkpoints/000009/pretrained_model/policy.json"
    write_json(manifest, {"kind": "act"})
    info = {
        "features": {
            "action": {"names": ["a", "b"], "shape": [2], "units": ["mm", "deg"]},
            "observation.state": {"names": ["b", "a"], "shape": [2], "units": ["deg", "mm"]},
        },
        "fps": 30,
    }
    native = layout == "native"
    if native:
        info["format"] = "alohamini-episodes"
        path = ds / "episodes/episode_000002/frames.parquet"
        write_json(path.parent / "episode.json", {"length": 2})
    else:
        info.update(codebase_version="v3.0", total_frames=3, total_episodes=3)
        if layout == "v3-indexed":
            info["data_path"] = "data/part-{chunk_index}/records-{file_index}.parquet"
            path = ds / "data/part-0/records-7.parquet"
            index = ds / "meta/episodes/chunk-000/file-000.parquet"
            index.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({"episode_index": [2], "data/chunk_index": [0], "data/file_index": [7]}),
                index,
            )
        else:
            path = ds / "data/chunk-000/file-000.parquet"
    write_json(ds / "meta/info.json", info)
    write_json(
        ds / "meta/evaluation.json",
        {
            "checkpoint": {
                "path": "/old/host/checkpoints/000009/pretrained_model",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        },
    )
    path.parent.mkdir(exist_ok=True, parents=True)
    table = pa.table(
        {
            "episode_index": [0, 2, 2],
            "frame_index": [0, 0, 1],
            "action": [[99, 99], [5, 6], [7, 8]],
            "observation.state": [[0, 0], [4, 3], [6, 5]],
        }
    )
    pq.write_table(table.slice(1) if native else table, path)
    m = Monitor(run, tmp_path / "logs", [ds.parent])
    (entry,) = m.catalog.discover()
    (dataset,) = m.datasets.listing()
    key = dataset["id"]
    d = m.datasets.details(key, m.catalog.get(entry["id"]))
    assert d["manifest_match"] == "match"
    assert d["episode_ids"] == ([2] if native else None)
    trace = m.datasets.trajectory(key, 2, "a", 32)
    assert trace["unit"] == "mm" and trace["frames"] == 2
    assert trace["series"]["action"]["points"] == [[0, 5], [1, 7]]
    assert trace["series"]["observation.state"]["points"] == [[0, 3], [1, 5]]
    write_json(manifest, {"kind": "other"})
    assert m.datasets.details(key, m.catalog.get(entry["id"]))["manifest_match"] == "different"
    with pytest.raises(ValueError):
        m.datasets.trajectory(key, 2, "unknown", 32)
