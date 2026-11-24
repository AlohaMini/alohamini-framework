import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_rlt_collection import session as session
from test_rlt_collection import snapshot, tick
from test_rlt_v3 import _episode

from alohamini.apps.collection import rlt
from alohamini.apps.collection.workers import serve
from alohamini.cli import _main as main
from alohamini.datasets.experience.storage import json_lines
from alohamini.datasets.experience.v3 import export_v3
from alohamini.policies.rlt.networks import AC_FORMAT
from alohamini.policies.rlt.policy import RLTPolicy
from alohamini.rl.rlt import ac_training
from alohamini.rl.rlt.actor_critic import ActorCritic
from alohamini.rl.rlt.config import RLTConfig
from alohamini.rl.rlt.replay import materialize_episode

CONTRACT = dict(
    names=["left", "right"], learned_names=["left"], token_spec=dict(dim=2), chunk_size=2
)


def _config(path, **kwargs):
    return RLTConfig(
        checkpoint="base", task="place", device="cpu", output_dir=str(path), chunk_size=2, **kwargs
    )


def test_discounted_binary_reward_and_bootstrap(tmp_path):
    learner = ActorCritic(_config(tmp_path, ac_gamma=0.5), CONTRACT)
    with torch.no_grad():
        for p in learner.target_critic.parameters():
            p.zero_()
        learner.target_critic.q1[-1].bias.fill_(8)
        learner.target_critic.q2[-1].bias.fill_(4)
    batch = dict(
        valid=torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
        reward=torch.tensor([[0.0, 1.0], [0.0, 100.0], [0.0, 100.0]]),
        reward_delay=torch.tensor([[2], [0], [0]]),
        elapsed_steps=torch.tensor([[4], [1], [3]]),
        done=torch.tensor([[1.0], [1.0], [0.0]]),
        next_z=torch.ones(3, 2),
        next_state=torch.ones(3, 1),
        next_ref=torch.zeros(3, 2, 1),
        next_valid=torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
    )
    torch.testing.assert_close(learner.target(batch), torch.tensor([[0.125], [0.0], [0.5]]))
    a = torch.randn(3, 2, 1)
    a2 = a.clone()
    a2[batch["valid"] == 0] = 12345.0
    for before, after in zip(
        learner.critic(batch["next_z"], batch["next_state"], a, batch["valid"]),
        learner.critic(batch["next_z"], batch["next_state"], a2, batch["valid"]),
        strict=True,
    ):
        torch.testing.assert_close(before, after)


def test_flipping_reward_changes_critic_and_policy_with_bc_target_fixed(tmp_path):
    cfg = _config(tmp_path, lr=0.001)

    def fit(reward):
        torch.manual_seed(8)
        learner = ActorCritic(cfg, CONTRACT)
        z, state, ref = torch.ones(4, 2), torch.zeros(4, 1), torch.zeros(4, 2, 1)
        assert torch.equal(learner.actor(z, state, ref), ref)
        r = torch.zeros(4, 2)
        r[:, -1] = reward
        valid = torch.ones(4, 2)
        b = dict(
            z=z,
            state=state,
            ref=ref,
            action=ref,
            bc_target=ref,
            valid=valid,
            reward=r,
            next_z=z,
            next_state=state,
            next_ref=ref,
            next_valid=valid * 0,
            done=torch.ones(4, 1),
            elapsed_steps=torch.full((4, 1), 2),
            reward_delay=torch.zeros(4, 1, dtype=torch.int64),
        )
        for _ in range(9):
            learner.update(b)
        assert learner.actor_updates == 5
        actor = learner.actor(z, state, ref).detach()
        assert 0 < (actor - ref).abs().max() <= cfg.ac_residual_limit
        return actor, learner.critic(z, state, ref, valid)[0].detach()

    fail, q0 = fit(0)
    success, q1 = fit(1)
    assert (q1 > q0).all() and not torch.allclose(success, fail)


def _policy(config):
    native = SimpleNamespace(
        names=CONTRACT["names"],
        manifest={"cameras": ["chest"]},
        fps=30,
        reset=lambda: None,
        config=SimpleNamespace(),
        processor=SimpleNamespace(action_prefix=lambda a, context: a),
        execution_action=lambda a, context: a,
    )
    policy = RLTPolicy.__new__(RLTPolicy)
    policy.native, policy.config, policy.contract, policy.indices = native, config, CONTRACT, [0]
    policy.actor = policy.actor_identity = None
    policy.encoder, policy.token_sha256 = object(), "token-A"

    def reference(snap, i, *, rgb_images=None, encode_token=True):
        assert "chest" in (snap.images if rgb_images is None else rgb_images)
        raw = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])
        return SimpleNamespace(
            token=torch.ones(1, 2) if encode_token else None,
            proprio=torch.tensor([[snap.payload["left"]]]),
            reference=raw,
            context=torch.zeros(1),
            physical=raw,
        )

    policy.reference = reference
    return policy


@pytest.mark.parametrize("v3", [False, True], ids=["raw", "portable-v3"])
def test_scored_episode_cli_trains_and_deploys(tmp_path, monkeypatch, v3):
    root, out = tmp_path / "data", tmp_path / "model"
    for i in range(6):
        path, _, _ = _episode(root, i)
        summary_path = path / "episode.json"
        summary = json.loads(summary_path.read_bytes())
        summary.update(
            base_contract=CONTRACT, reward=i % 2, evaluation=i == 4, training_valid=i != 5
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _episode(root, 8, scored=False)
    if v3:
        # Exercise a stand-alone published copy, with no raw image fallback.
        export_v3(root)
        (root / "episodes").rename(tmp_path / "originals")
        root = root / "lerobot_v3"
    monkeypatch.setattr(ac_training, "RLTPolicy", _policy)
    cfg = _config(out)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    args = [
        "train",
        "rlt",
        "ac",
        "--config",
        str(config_file),
        "--token-checkpoint",
        "token-A",
        "--dataset",
        str(root),
        "--output-dir",
        str(out),
        "--ac-steps",
        "5",
        "--ac-batch-size",
        "4",
    ]
    assert main(args) == 0
    from alohamini.learning.stages import read_stage_weights

    saved, _ = read_stage_weights(out / "actor")
    split = saved["split"]
    assert set(split["episodes"]) == {f"{i:032x}" for i in range(4)}
    assert not set(split["train"]) & set(split["validation"])
    assert len(split["train"]) == len(split["validation"]) == 2
    assert {split["episodes"][eid]["reward"] for eid in split["train"]} == {0, 1}
    assert (saved["format"], saved["critic_updates"], saved["actor_updates"]) == (AC_FORMAT, 5, 3)
    assert np.isfinite(saved["metrics"]["validation_critic_loss"])
    assert len(json_lines(out / "ac-metrics.jsonl")) == 5
    with pytest.raises(FileExistsError):
        main(args)
    if not v3:
        resumed = tmp_path / "resumed"
        short = args.copy()
        short[short.index(str(out))] = str(resumed)
        short[short.index("--ac-steps") + 1] = "3"
        assert main(short) == 0
        short[short.index("--ac-steps") + 1] = "5"
        assert main([*short, "--resume"]) == 0
        restored, _ = read_stage_weights(resumed / "actor")
        for component in ("actor", "critic"):
            for key, param in saved[component].items():
                torch.testing.assert_close(param, restored[component][key], rtol=0, atol=0)

        def state(directory):
            return torch.load(
                directory / "checkpoints/last/training_state/state.pt", weights_only=True
            )

        for key in (
            "sampler_rng",
            "target_actor",
            "target_critic",
            "actor_optimizer",
            "critic_optimizer",
        ):
            torch.testing.assert_close(state(out)[key], state(resumed)[key], rtol=0, atol=0)
    policy = _policy(cfg)
    snap = SimpleNamespace(payload={"left": 1.0}, images={"chest": None})
    original = policy.predict(snap)
    assert original["actor_version"] == -1
    np.testing.assert_array_equal(original["actions"], original["reference"])
    policy.load_actor(out / "actor")
    outside = policy.predict(snap)
    assert outside["actor_version"] == -1
    np.testing.assert_array_equal(outside["actions"], original["reference"])
    prediction = policy.predict(snap, apply_actor=True)
    assert prediction["actor_version"] == 3
    diff = prediction["actions"] - prediction["reference"]
    assert np.any(diff[:, 0]) and np.max(np.abs(diff[:, 0])) <= cfg.ac_residual_limit
    np.testing.assert_array_equal(diff[:, 1], 0)
    policy.token_sha256 = "mismatch"
    with pytest.raises(ValueError, match="exact token"):
        policy.load_actor(out / "actor")
    policy.contract = dict(CONTRACT, chunk_size=3)
    with pytest.raises(ValueError, match="contract"):
        policy.load_actor(out / "actor")


def test_gapped_replay_reuses_the_same_stochastic_boundary_and_requires_ack(tmp_path):
    path, _, _ = _episode(tmp_path, 3)
    summary = json.loads((path / "episode.json").read_bytes())
    summary["base_contract"] = CONTRACT
    (path / "episode.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = json_lines(path / "steps.jsonl")
    rows[1]["source"] = "HOLD"
    rows[2]["time"], rows[3]["time"] = 0.50, 0.54
    (path / "steps.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    acknowledgements = json_lines(path / "acks.jsonl")
    acknowledgements[2].update(status="applied", targets={"left": 1.0, "right": 3.0})
    acks = path / "acks.jsonl"
    acks.write_text("\n".join(map(json.dumps, acknowledgements)), encoding="utf-8")
    policy = _policy(_config(tmp_path))
    reference = policy.reference
    visits = []

    def sample(snapshot, index):
        visits.append(index)
        ref = reference(snapshot, index)
        ref.reference = ref.reference + torch.rand_like(ref.reference)
        return ref

    policy.reference = sample
    first, last = materialize_episode(policy, path)
    assert visits == [0, 2, 4]
    assert first["done"].item() == 0 and first["elapsed_steps"].item() == 15
    np.testing.assert_array_equal(first["next_ref"], last["ref"])
    assert last["done"].item() == 1 and last["reward"].sum() == 1
    acks.write_text(
        "\n".join(map(json.dumps, acknowledgements[:2] + acknowledgements[3:])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Host acceptance"):
        materialize_episode(policy, path)


def test_inference_worker_passes_the_requested_stage(monkeypatch):
    policy = Mock()
    policy.native.manifest = {"cameras": []}
    policy.predict.return_value = {}
    monkeypatch.setattr("alohamini.policies.rlt.policy.RLTPolicy", lambda _: policy)
    snap = SimpleNamespace(request_started_s=0.0)
    jobs, results = Mock(), Mock()
    jobs.get.side_effect = [
        dict(id=1, generation=1, snapshot=snap),
        dict(id=2, generation=1, critical=False, snapshot=snap),
        dict(id=3, generation=2, critical=True, snapshot=snap),
        dict(id=4, generation=3, critical=False, snapshot=snap),
        InterruptedError(),
    ]
    serve(None, snap, jobs, results, SimpleNamespace(is_set=lambda: False), load=rlt.load)
    assert [c.kwargs["apply_actor"] for c in policy.predict.call_args_list] == [
        False,
        False,
        True,
        False,
    ]
    assert policy.native.reset.call_count == 3


def test_actor_command_provenance_and_late_takeover(session):
    control, clock = session
    control.ready["actor_identity"] = {"sha256": "actor", "token_sha256": "token", "version": 3}
    tick(control, clock)
    pose = snapshot(clock.now).payload
    ref = np.array([[pose[n] for n in control.names]] * 8)
    actions = ref.copy()
    axis = control.names.index(control.relative.names[0])
    actions[:, axis] += 1

    def deliver(generation, plan_id):
        control.inference.result = dict(
            kind="prediction",
            generation=generation,
            id=plan_id,
            sampled_s=clock.now,
            inference_s=0.05,
            actions=actions,
            reference=ref,
            actor_version=3,
        )

        tick(control, clock)
        return control.client.sent[-1][0][control.names[axis]]

    def toggle(key):
        tick(control, clock, [("key", key)])
        assert not control.actions
        return control.state.generation

    outside = control.state.generation
    assert control.inference.jobs[-1]["critical"] is False
    # Even a worker that erroneously returns corrected actions cannot move this axis off-reference.
    assert deliver(outside, 10) == ref[0, axis]
    inside = toggle("c")
    assert control.inference.jobs[-1]["critical"] is True
    deliver(outside, 11)
    assert not control.actions
    assert deliver(inside, 12) == actions[0, axis]
    outside = toggle("c")
    assert control.inference.jobs[-1]["critical"] is False
    deliver(inside, 13)
    assert not control.actions
    assert deliver(outside, 14) == ref[0, axis]

    inside = toggle("c")
    assert deliver(inside, 15) == actions[0, axis]
    toggle("h")
    deliver(inside, 16)
    assert not control.actions
    resumed = toggle("h")
    assert control.inference.jobs[-1]["critical"] is True
    assert deliver(resumed, 17) == actions[0, axis]

    control.record.close()
    rows = json_lines(control.record.directory / "steps.jsonl")
    actor_rows = [row for row in rows if row["source"] == "RLT"]
    assert [row["plan_id"] for row in actor_rows] == [12, 15, 17]
    assert all(r["critical"] and r["actor_version"] == 3 for r in actor_rows)
    assert all(r["reference"] != r["requested"] for r in actor_rows)
    vla_rows = [row for row in rows if row["source"] == "VLA"]
    assert [row["plan_id"] for row in vla_rows] == [10, 14]
    assert all(r["reference"] == r["requested"] and r["actor_version"] == -1 for r in vla_rows)
    human_rows = [row for row in rows if row["source"] == "HUMAN"]
    assert human_rows and all(r["plan_id"] is None and r["actor_version"] == -1 for r in human_rows)
    assert control.record.summary["actor"]["version"] == 3
