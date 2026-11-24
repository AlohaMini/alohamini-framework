import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alohamini.datasets.rlt.storage import save_snapshot
from alohamini.policies.rlt.replay import materialize_episode
from alohamini.protocol import HostSnapshot


def test_optional_phase_uses_applied_actions_and_separate_reference(tmp_path):
    contract = {"learned_names": ["left"]}
    summary = dict(
        base_contract=contract,
        status="committed",
        ended=True,
        evaluation=False,
        reward=1,
        phase_labels="optional-operator-v2",
        critical_entries=0,
    )
    meta = tmp_path / "episode.json"
    meta.write_text(json.dumps(summary))
    rows, acks = [], []
    for i, source in enumerate(("VLA", "HUMAN", "VLA", "HOLD")):
        command = dict(client_id="pc", host_session_id="pi", control_epoch=1, sequence=i + 1)
        rows.append(
            dict(
                index=i,
                time=i / 30,
                source=source,
                critical=False,
                command=command,
                requested=[1000.0],
            )
        )
        acks.append(dict(command=command, status="applied", targets={"left": float(i + 1)}))
        save_snapshot(tmp_path / "observations" / str(i), HostSnapshot({}, {}, i / 30, i / 30))
    (tmp_path / "steps.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "acks.jsonl").write_text("\n".join(map(json.dumps, acks)))
    save_snapshot(tmp_path / "terminal", HostSnapshot({}, {}, 1, 1))

    def reference(snapshot, index):
        return SimpleNamespace(
            token=torch.ones(1, 2),
            proprio=torch.ones(1, 1),
            context=torch.tensor([float(index)]),
            reference=torch.full((1, 2, 1), 20.0 + index),
        )

    def encode(actions, *, context):
        return actions - context

    policy = SimpleNamespace(
        native=SimpleNamespace(
            names=["left"],
            fps=30,
            config=SimpleNamespace(),
            processor=SimpleNamespace(action_prefix=encode),
        ),
        config=SimpleNamespace(chunk_size=2, replay_stride=2),
        contract=contract,
        indices=[0],
        reference=reference,
    )
    first, last = materialize_episode(policy, tmp_path)
    np.testing.assert_array_equal(first["ref"][:, 0], [20, 20])
    np.testing.assert_array_equal(first["bc_target"][:, 0], [20, 2])
    np.testing.assert_array_equal(first["action"][:, 0], [1, 2])
    np.testing.assert_array_equal(last["action"][:, 0], [1, 0])
    np.testing.assert_array_equal(last["valid"], [1, 0])
    np.testing.assert_array_equal(last["reward"], [1, 0])
    assert last["reward_delay"].item() == 1
    assert last["elapsed_steps"].item() == 2
    meta.write_text(json.dumps(dict(summary, critical_entries=1)))
    assert materialize_episode(policy, tmp_path) == []
    meta.write_text(json.dumps(dict(summary, evaluation=True)))
    with pytest.raises(ValueError, match="Replay only accepts"):
        materialize_episode(policy, tmp_path)
