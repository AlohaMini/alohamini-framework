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


def test_raw_trends_and_diagnostics_across_appends(tmp_path):
    run = tmp_path / "runs/test"
    write_json(run / "train.json", {"optimizer": {"grad_clip_norm": 4}})
    metrics = run / "metrics.jsonl"
    records = [
        {
            "step": i,
            "loss": i,
            "grad_norm": i,
            "samples": i * 4,
            "elapsed_s": i * 2,
            "lr": 1 / i,
            "metrics": {"loss_action": i / 2, "flow_loss": i / 4},
        }
        for i in range(1, 102)
    ]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in records[:99]))
    (run / "validation-metrics.jsonl").write_text(
        '{"step":50,"loss":3,"samples":200,"elapsed_s":100}\n'
        '{"step":90,"loss":2,"samples":360,"elapsed_s":180,"train_eval_loss":1.5,"generalization_gap":0.5}\n'
    )
    monitor = Monitor(run.parent, tmp_path / "logs", [])
    attempt = monitor.catalog.get(monitor.catalog.discover()[0]["id"])
    h = monitor.index(attempt).phases["train"]
    h.refresh()
    initial = h.chart(["loss", "lr", "flow_loss"], "step", None, None, 16, None, 20)
    assert len(initial["series"]["loss"]["points"]) <= 16
    assert initial["means"]["loss"]["points"] == [[20, 10.5], [40, 30.5], [60, 50.5], [80, 70.5]]
    assert "lr" not in initial["means"]
    assert initial["means"]["flow_loss"]["points"][0] == [20, 10.5 / 4]
    with metrics.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in records[99:])
    h.refresh()
    inc = h.chart(["loss"], "step", None, None, 16, initial["cursor"], 20)
    assert inc["mode"] == "append" and inc["means"]["loss"]["points"] == [[100, 90.5]]
    chart = h.chart(["loss"], "step", 96, 101, 16, None, 20)
    assert chart["means"]["loss"]["points"] == [[100, 90.5]]
    d = monitor.diagnostics(attempt, 20, "loss", 3.1)
    assert (d["from_step"], d["to_step"]) == (82, 101)
    assert d["distribution"]["mean"] == 91.5
    assert d["distribution"]["slope_per_step"] == 1 and sum(d["distribution"]["bins"]) == 20
    assert d["components"]["loss_action"]["previous_mean"] == 71.5 / 2
    assert d["clipping"]["fraction"] == 1 and d["clipping"]["source"] == "from_norm"
    assert d["validation"]["first_hit"] == {"step": 50, "samples": 200, "elapsed_s": 100, "loss": 3}
    assert d["validation"]["best"]["step"] == 90 and d["samples_per_s"] == 2
    assert d["paired"]["generalization_gap"] == 0.5

    metrics.write_text('{"step":1,"loss":1}\n{"step":2,"loss":null}\n')
    h.refresh()
    assert h.chart(["loss"], "step", None, None, 16, inc["cursor"], 2)["means"]["loss"][
        "points"
    ] == [[2, None]]


def test_archived_and_native_preflight_are_attempt_specific(tmp_path):
    run, logs = tmp_path / "runs/archived", tmp_path / "logs"
    write_json(run / "train.json", {"policy": "act", "dataset": "/data/test"})
    write_json(run / "resume-1-10.json", {"policy": "act", "dataset": "/data/test"})
    write_json(run / "resources.json", {"policy": "act", "ok": True})
    write_json(run / "metrics-from-1-10-resources.json", {"policy": "act", "ok": False})
    write_json(
        logs / "archived.preflight.json", {"passed": True, "dataset": "/data/test", "frames": 48}
    )
    (logs / "archived.log").write_text("INFO num_learnable_params=10\nINFO num_total_params=20\n")
    monitor = Monitor(run.parent, tmp_path / "other", [])
    for item in monitor.catalog.discover():
        attempt = monitor.catalog.get(item["id"])
        detail = monitor.summary(attempt)
        if item["attempt"]:
            assert detail["preflight"]["ok"] is False and detail["launch_preflight"] is None
            assert not detail["logs"] and detail["console_parameters"] is None
        else:
            assert detail["preflight"]["ok"] and detail["launch_preflight"]["frames"] == 48
            assert detail["console_parameters"] == {
                "num_learnable_params": 10,
                "num_total_params": 20,
            }
            assert "archived.log" in detail["logs"]
        assert "preflight" not in monitor.summary(attempt, brief=True)


def test_legacy_policy_outputs_and_sparse_window_observations(tmp_path):
    root = tmp_path / "runs" / "legacy"
    write_json(root / "train.json", {"policy": "smolvla"})
    records = [
        {
            "step": i,
            "metrics": {"loss": 9, "grad_norm": 2},
            "policy_metrics": {"loss": 3, **({"losses_after_forward": i} if i in (1, 4) else {})},
        }
        for i in range(1, 5)
    ]
    (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "validation-metrics.jsonl").write_text(
        '{"step":1,"loss":3}\n{"step":2,"loss":null}\n{"step":4,"loss":2}\n'
    )
    monitor = Monitor(root.parent, tmp_path / "logs", [])
    run = monitor.catalog.get(monitor.catalog.discover()[0]["id"])
    phase = monitor.summary(run)["phases"]["train"]
    assert phase["policy_metrics"] == ["policy/loss", "losses_after_forward"]
    assert phase["latest"]["loss"] == 9 and phase["latest"]["grad_norm"] == 2
    keys = ["loss", "policy/loss", "losses_after_forward"]
    chart = monitor.chart(run, keys, "step", None, None, 16, dict(train=None, validation=None))
    assert [chart["train"]["series"][k]["points"][-1] for k in keys] == [[4, 9], [4, 3], [4, 4]]
    assert monitor.records(run, "train", None, 8, False)["records"] == records
    d = monitor.diagnostics(run, 2, "losses_after_forward", 1)
    assert d["previous_window"] == {"records": 2, "from_step": 1, "to_step": 2, "n": 1, "mean": 1}
    assert d["distribution"]["n"] == d["components"]["losses_after_forward"]["previous_n"] == 1
    assert d["validation"]["latest"]["step"] == 4 and d["validation"]["previous"]["step"] == 1
    assert d["validation"]["count"] == 2 and d["validation"]["first_hit"] is None
    d = monitor.diagnostics(run, 8, "unobserved", None)
    assert d["previous_window"] is None and d["distribution"] is None


def test_legacy_asset_locators_belong_to_initial_policy_and_refresh(tmp_path):
    root = tmp_path / "runs" / "legacy"
    for name in ("train.json", "resume-1-12.json"):
        write_json(root / name, {"policy": "openwam"})
    artifact = root / "statistics.json"
    write_json(
        artifact,
        {
            "contract": {
                "policy": "openwam",
                "model": {"assets_path": "old/tokenizer", "width": 4096},
            }
        },
    )
    monitor = Monitor(root.parent, tmp_path / "logs", [])
    runs = {x["attempt"]: monitor.catalog.get(x["id"]) for x in monitor.catalog.discover()}
    summary = monitor.summary(runs[None])
    assert summary["recorded_asset_paths"] == {"assets_path": "old/tokenizer"}
    assert monitor.summary(runs["1-12"])["recorded_asset_paths"] == {}
    write_json(
        artifact, {"contract": {"policy": "other", "model": {"assets_path": "another/tokenizer"}}}
    )
    changed = monitor.summary(runs[None])
    assert (
        changed["recorded_asset_paths"] == {}
        and changed["metadata_version"] != summary["metadata_version"]
    )
