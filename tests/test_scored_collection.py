"""Native policy execution, intervention and scored v3 data without robot hardware."""

import json
import time
from contextlib import closing, contextmanager

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_dataset import jpeg
from test_native_learning import model_options, samples
from test_native_learning import recording as recording
from test_rlt_collection import session as session
from test_rlt_collection import snapshot, step, wait_writer

from alohamini.apps.collection.native import CollectionConfig, worker
from alohamini.apps.collection.scored_session import ControlSession
from alohamini.cli import _main
from alohamini.datasets.experience.reader import V3Observations
from alohamini.datasets.experience.storage import json_lines
from alohamini.datasets.merge_v3 import merge_v3
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import make_policy, save_checkpoint
from alohamini.model import get_robot_model


def test_spawned_native_worker_loads_act_and_returns_named_host_action(recording, tmp_path):
    data = samples(recording, episodes=[0])
    motors = data.info["robot_metadata"]["motors"]
    for actuator in get_robot_model("alohamini2pro").actuators:
        motors.setdefault(actuator.name, {}).update(
            id=actuator.motor_id,
            model=actuator.motor_model,
            homing_offset=0,
            range_min=0,
            range_max=4095,
            drive_mode=0,
        )
        motors[actuator.name].setdefault("normalization", "range_m100_100")
    stats = data.statistics()
    checkpoint = tmp_path / "act"
    save_checkpoint(
        checkpoint, make_policy("act", model_options(state=False)), stats, data, training={}
    )
    observation = snapshot(time.monotonic())
    observation.payload["_robot_metadata"] = data.info["robot_metadata"]
    observation.images["forward"] = jpeg()
    config = CollectionConfig(
        str(checkpoint), "test", str(tmp_path / "run"), device="cpu", intervention="right"
    )

    def receive(service):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = service.poll()
            if result is not None:
                return result
            time.sleep(0.01)
        pytest.fail("Collection worker timed out")

    with closing(worker(config, observation)) as service:
        ready = receive(service)
        assert ready["kind"] == "ready" and ready["base_contract"]["policy"] == "act"
        assert ready["handover_fields"] and all(
            name.startswith("arm_right_") for name in ready["handover_fields"]
        )
        assert set(ready["handover_fields"]) < set(ready["base_contract"]["receipt_fields"])
        assert service.submit(dict(snapshot=observation, id=1, generation=1))
        result = receive(service)
        assert result["single_action"] and result["generation"] == 1
        assert len(result["actions"][0]) == len(ready["names"])
        assert np.isfinite(result["actions"]).all()
        assert "reference" not in result


def test_scored_native_episodes_merge_with_receipts_and_distinct_boundaries(session, tmp_path):
    template, clock = session
    sources = []
    for intervention in (False, True):
        root = tmp_path / str(intervention)
        config = CollectionConfig(
            "act", "move", str(root), intervention="both" if intervention else None
        )
        names = template.names
        contract = dict(names=names, receipt_fields=list(template.relative.names))
        ready = {**template.ready, "base_contract": contract}
        leader = template.leader if intervention else None
        control = ControlSession(
            template.client, template.inference, leader, config, ready, snapshot(clock.now)
        )
        with _close_recorder(control):
            step(control, clock)
            time_sampled = clock.now
            request = [5.0] * len(names)
            # A one-action policy can take longer than one period to select its action.
            control.inference.result = dict(
                kind="prediction",
                generation=control.state.generation,
                id=1,
                sampled_s=time_sampled - 0.06,
                actions=[request],
                inference_s=0.06,
                single_action=True,
            )
            step(control, clock)
            # The Host limited one joint. Only its receipt identifies the applied value.
            template.client.sent[-1][0][names[0]] = 2.0
            step(control, clock, [("key", "h")])
            assert control.state.mode == ("human" if intervention else "policy")
            if intervention:
                step(control, clock)
            step(
                control,
                clock,
                [("key", "t" if intervention else "right"), ("reward", int(intervention))],
            )
            step(control, clock)
            wait_writer(control.record)
            journal = control.record.committed
            assert json.loads((journal / "episode.json").read_bytes())["truncated"] == intervention
            assert all("critical" not in row for row in json_lines(journal / "steps.jsonl"))

        (root / "run-contract.json").write_text(json.dumps({"collection": "experience"}))
        dataset = root / "lerobot_v3"
        assert (
            _main(
                ["dataset", "export", str(root), "--format", "lerobot-v3", "--output", str(dataset)]
            )
            == 0
        )
        sources.append(dataset)
        info = json.loads((dataset / "meta/info.json").read_bytes())
        labels = info["alohamini_experience"]["sources"]
        rows = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pylist()
        (policy,) = [row for row in rows if row["experience.source"] == labels["POLICY"]]
        assert policy["experience.action_valid"] and policy["experience.applied_mask"][0]
        assert policy["experience.requested_action"][0] == 5 and policy["action"][0] == 2
        assert any(row["experience.source"] == labels["HUMAN"] for row in rows) == intervention
        assert rows[-2]["next.done"] and rows[-2]["next.truncated"] == intervention
        assert rows[-2]["next.terminated"] != intervention and rows[-2]["next.reward"] == int(
            intervention
        )
        assert (
            sum(row["next.done"] for row in rows) == 1
            and rows[-1]["experience.is_terminal_observation"]
        )
        assert not any(rows[-1]["experience.applied_mask"])
        assert not any(key.startswith("rlt.") for key in info["features"])

    merged = tmp_path / "merged"
    merge_v3(sources, merged)
    for i in range(2):
        with closing(V3Observations(merged / f"meta/experience/episode_{i:06d}")) as obs:
            assert obs.read(0)[0].robot_model == "alohamini2pro"
    data = AlohaMiniDataset(
        merged, episodes=[0, 1], cameras=[], state="none", delta_indices={"action": [0]}
    )
    assert data[0]["action"][0, 0].item() == 2


@contextmanager
def _close_recorder(control):
    try:
        yield
    finally:
        if control.record:
            control.record.close()
