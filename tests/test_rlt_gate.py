import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_rlt_collection import session as session
from test_rlt_collection import snapshot, tick

from alohamini.rlt import gate_training as gt
from alohamini.rlt.cli import main
from alohamini.rlt.config import RLTConfig
from alohamini.rlt.gate import GateLatch, GripperRelease
from alohamini.rlt.gate_classifier import GATE_FORMAT, StageClassifier
from alohamini.rlt.policy import RLTPolicy
from alohamini.rlt.runtime import ControlSession
from alohamini.rlt.storage import json_lines
from alohamini.learning.stages import read_stage_weights as load_hashed_weights

CONTRACT = {"learned_names": ["left", "grip"], "token_spec": {"dim": 4}}


def _cfg(tmp_path):
    return RLTConfig(
        checkpoint="unused",
        output_dir=str(tmp_path / "output"),
        task="place",
        device="cpu",
        dataset=str(tmp_path / "data"),
        token_checkpoint="token.pt",
        gate_steps=3,
        gate_sample_stride=1,
        gate_batch_size=4,
        gate_confirmations=2,
    )


def _prediction(control, sampled, *, plan_id, generation, length=3, probability=None, observation=None):
    ref = np.tile([snapshot(sampled).payload[name] for name in control.names], (length, 1))
    axis = control.names.index(control.relative.names[0])
    ref[:, axis] = np.arange(1, length + 1)
    actions = ref.copy()
    actions[:, axis] += 20
    return dict(
        kind="prediction",
        generation=generation,
        id=plan_id,
        sampled_s=sampled,
        inference_s=0.25,
        actions=actions,
        reference=ref,
        actor_version=5,
        gate_probability=probability,
        observation_sequence=observation,
    )


def _recorded(control):
    control.record.close()
    return json_lines(control.record.directory / "steps.jsonl")


def test_hysteresis_requires_consecutive_votes_and_supports_reentry():
    latch = GateLatch(0.8, 0.2, 2)
    scores = [0.9, 0.6, 0.9, 0.9, 0.3, 0.1, 0.1, 0.9, 0.9]
    assert [latch.update(p) for p in scores] == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    latch.reset()
    assert not latch.update(0.9)
    for score in [float("nan"), 1.1]:
        with pytest.raises(ValueError):
            latch.update(score)


def test_release_requires_closed_then_distinct_fresh_open_feedback():
    rule = GripperRelease(18, 24, 3)

    def observe(opening, seq, time):
        return rule.observe(opening, seq, time, 0.09, critical=True)

    assert not any(observe(40, i, i / 30) for i in range(1, 5))
    assert not observe(18, 5, 5 / 30)
    assert not observe(24, 6, 6 / 30)
    assert not observe(40, 6, 7 / 30)
    assert not observe(40, 5, 8 / 30)
    assert not observe(23, 7, 9 / 30)
    assert not observe(24, 8, 10 / 30)
    assert not observe(24, 9, 1.0)  # A feedback gap breaks the confirmation streak.
    assert not observe(float("nan"), 10, 1.03)
    assert not observe(24, 11, 1.06)
    assert not observe(24, 12, 1.09)
    assert observe(24, 13, 1.12)
    assert rule.block_reentry


def _segment(name, a, b):
    return dict(name=name, start_frame=a, end_frame_exclusive=b, reward=1)


def _corpus(tmp_path):
    ds = tmp_path / "data"
    manifest = dict(version=1, fps=30, episodes=[])
    for i in range(5):
        eid = f"{i:032x}"
        path = ds / "episodes" / eid
        path.mkdir(parents=True)
        seg = [_segment("orange_placement", 2, 5), _segment("purple_placement", 6, 9)]
        summary = dict(
            episode_id=eid, reward=i % 2, status="committed", ended=True, base_contract=CONTRACT
        )
        if i < 4:
            manifest["episodes"].append(dict(episode_id=eid, segments=seg))
            summary["training_segments"] = dict(version=1, segments=seg)
        (path / "episode.json").write_text(json.dumps(summary))
        (path / "steps.jsonl").write_text("{}\n" * 12)
    (ds / "rlt-placement-intervals.json").write_text(json.dumps(manifest))
    return ds, manifest


def test_labels_use_only_reviewed_episodes_and_never_treat_excluded_as_negative(tmp_path):
    ds, labels = _corpus(tmp_path)
    eid = labels["episodes"][0]["episode_id"]
    drop = dict(labels["episodes"][0]["segments"][0], quality="exclude")
    corpus = gt.reviewed_episodes(ds, labels, {eid: [drop]})
    assert len(corpus) == 4
    assert gt._sample_frames(corpus[eid], 1) == (
        [0, 1, 5, 6, 7, 8, 9, 10, 11],
        [
            None,
            None,
            None,
            "purple_placement",
            "purple_placement",
            "purple_placement",
            None,
            None,
            None,
        ],
    )
    labels["episodes"][0]["segments"][0]["end_frame_exclusive"] -= 1
    with pytest.raises(ValueError, match="disagrees"):
        gt.reviewed_episodes(ds, labels, {})


def test_train_load_and_predict_auto_gate_without_second_policy_forward(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _corpus(tmp_path)

    class Frozen:
        def __init__(self, config):
            self.token_sha256 = "frozen-token"
            self.contract = CONTRACT
            self.native = SimpleNamespace(fps=30, reset=lambda: None)

        def gate_features(self, snapshot, *, rgb_images=None):
            return torch.tensor([[snapshot, 1.0, -1.0, 0.0]]), torch.tensor([[snapshot, 0.0]])

    monkeypatch.setattr(gt, "RLTPolicy", Frozen)
    monkeypatch.setattr(gt, "load_snapshot", lambda path: float(path.name))
    config_path = tmp_path / "gate-config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    assert (
        main(
            [
                "train-gate",
                "--config",
                str(config_path),
                "--intervals",
                str(Path(cfg.dataset) / "rlt-placement-intervals.json"),
                "--output-dir",
                cfg.output_dir,
            ]
        )
        == 0
    )
    weights = Path(cfg.output_dir) / "gate"
    saved, _ = load_hashed_weights(weights)
    assert saved["format"] == GATE_FORMAT
    report = json.loads((weights.parent / "gate-validation.json").read_text())
    assert set(report["train_episodes"]).isdisjoint(report["validation_episodes"])
    selection = set(report["selection_episodes"])
    assert selection and selection.isdisjoint(report["train_episodes"] + report["validation_episodes"])
    assert all(
        h["best_bce"] == min(s["bce"] for s in h["scores"])
        for h in report["selection_history"].values()
    )
    assert set(report["probes"]) == {"token_state", "token_only", "state_only"}
    assert set(report["probes"]["token_state"]["by_phase"]) == {
        "orange_placement",
        "purple_placement",
    }

    policy = RLTPolicy.__new__(RLTPolicy)
    policy.contract = CONTRACT
    policy.config = replace(cfg, gate_checkpoint=str(weights))
    policy.encoder, policy.token_sha256 = object(), "wrong"
    with pytest.raises(ValueError, match="exact token"):
        policy.load_gate(weights)
    policy.token_sha256 = "frozen-token"
    policy.load_gate(weights)
    assert isinstance(policy.gate, StageClassifier)
    visits = []
    base = torch.zeros(1, 2, 2)

    def reference(snapshot, observation_id, *, encode_token):
        visits.append((observation_id, encode_token))
        return SimpleNamespace(
            token=torch.ones(1, 4),
            proprio=torch.ones(1, 2),
            reference=base,
            physical=base,
            context=None,
        )

    policy.reference = reference
    policy.native = SimpleNamespace(execution_action=lambda action, context: action)
    policy.actor = lambda z, state, ref: ref + 0.01
    policy.indices = [0, 1]
    policy.actor_identity = {"version": 2}
    result = policy.predict(None, observation_id=7, apply_actor=False)
    assert visits == [(7, True)] and 0 <= result["gate_probability"] <= 1
    assert result["actor_version"] == 2 and (result["actions"] != result["reference"]).all()
    policy.actor = None
    result = policy.predict(None, observation_id=8)
    assert visits[-1] == (8, True) and result["actor_version"] == -1
    assert (result["actions"] == result["reference"]).all()


def test_validation_counts_false_entries_missing_and_timing(tmp_path):
    cfg = replace(_cfg(tmp_path), gate_confirmations=1)
    names = [None, None, "orange", "orange", None, "purple", "purple", None]
    ep = dict(frames=list(range(8)), names=names, spans=[("orange", 2, 4, 1), ("purple", 5, 7, 0)])
    report = gt._assessment({"id": ep}, {"id": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}, cfg, 30)
    assert report["false_entries_before_first_target"] == 1
    assert report["false_entries_outside_intervals"] == 1
    assert report["by_phase"]["purple"]["missed"] == 1
    assert report["by_phase"]["orange"]["entry_offsets_s"] == [1 / 30]
    assert report["by_phase"]["orange"]["exit_offsets_s"] == [0.0]

    stuck = gt._assessment({"id": ep}, {"id": [1.0] * len(names)}, cfg, 30)
    assert stuck["by_phase"]["purple"]["carried_from_previous_phase"] == 1
    assert stuck["by_phase"]["orange"]["active_at_end_after_phase"] == 1
    assert stuck["by_phase"]["purple"]["active_at_end_after_phase"] == 1


def test_excluded_first_phase_and_vote_cadence(tmp_path):
    cfg = replace(_cfg(tmp_path), gate_confirmations=2)
    ep = dict(
        frames=[0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
        first_target_start=2,
        names=[None] * 4 + ["purple"] * 6,
        spans=[("purple", 6, 12, 1)],
        blocked=[("orange", 2, 4, 0)],
    )
    scores = {"x": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]}
    dense = gt._assessment({"x": ep}, scores, cfg, 30, 1)
    sparse = gt._assessment({"x": ep}, scores, cfg, 30, 2)
    assert dense["false_entries_outside_intervals"] == 1
    assert dense["false_entries_before_first_target"] == 0
    assert sparse["false_entries_outside_intervals"] == 0
    assert sparse["by_phase"]["purple"]["entry_offsets_s"] == [4 / 30]


def test_release_exits_manual_actor_and_ignores_pre_release_plan(session):
    c, clock = session
    ready = dict(c.ready, actor_identity={"sha256": "actor", "version": 5})
    c = ControlSession(
        c.client,
        c.inference,
        c.leader,
        c.config,
        ready,
        snapshot(clock.now),
    )

    def frame(opening, *events):
        snap = snapshot(clock.now)
        snap.payload["arm_left_gripper.pos"] = opening
        if c.client.sent:
            targets, cmd = c.client.sent[-1]
            snap.payload["_safety"].update(command=asdict(cmd), accepted_targets=targets)
        c.tick(snap, [("key", k) for k in events], clock.now)
        clock.now += 1 / 30

    try:
        frame(40)
        frame(40, "c")
        for _ in range(4):
            frame(40)
        assert c.state.critical
        frame(13)
        job = c.inference.jobs[-1]
        c.inference.result = _prediction(
            c, clock.now, plan_id=job["id"], generation=job["generation"], length=12
        )
        frame(23)
        frame(24)
        frame(25)
        old_generation = c.state.generation
        c.inference.result = _prediction(c, clock.now, plan_id=99, generation=old_generation)
        frame(25)
        assert not c.state.critical and c.state.generation > old_generation
        assert not c.actions and c.phase_source == "gripper_release"
        assert not c.inference.jobs[-1]["critical"]
        frame(13, "c")
        frame(13, "h")
        for _ in range(3):
            frame(30)
        assert c.state.mode == "human" and not c.state.critical
        frame(30, "h")
        assert c.state.mode == "vla" and not c.state.critical
        rows = _recorded(c)
        assert len(c.gripper_exits) == 2
        assert any(row["source"] == "RLT" for row in rows)
        for event in c.gripper_exits:
            row = rows[event["next_step_index"]]
            assert row["source"] != "RLT" and row["critical_source"] == "gripper_release"
            assert row["critical_label"] is None and not row["critical"]
    finally:
        c.record.close()


def test_classifier_cannot_reenter_until_gripper_closes(session):
    base, clock = session
    cfg = replace(
        base.config, gate_checkpoint="gate.pt", token_checkpoint="token.pt", gate_confirmations=1
    )
    c = ControlSession(
        base.client, base.inference, base.leader, cfg, base.ready, snapshot(clock.now)
    )
    try:
        tick(c, clock)
        c.state.critical = c.gate.update(1)
        for i, opening in enumerate((13.0, 25.0, 25.0, 25.0, 50.0, 13.0)):
            snap = snapshot(clock.now)
            snap.payload["arm_left_gripper.pos"] = opening
            c._observe_gripper(snap, clock.now)
            c._accept_plan(
                _prediction(
                    c,
                    clock.now,
                    plan_id=i + 10,
                    generation=c.state.generation,
                    probability=0.99,
                    observation=i + 200,
                ),
                True,
                clock.now,
            )
            assert c.state.critical == (i not in (3, 4))
            clock.now += 1 / 30
    finally:
        c.record.close()


def test_gripper_rule_configuration_and_scoring(session):
    base, clock = session
    initial = snapshot(clock.now)
    assert base.release is None  # Plain VLA demonstrations keep manual labels by default.
    cfg = replace(base.config, gripper_exit=True, gripper_exit_open_pct=30)
    c = ControlSession(base.client, base.inference, base.leader, cfg, base.ready, initial)
    assert c.release.opened == 30
    with pytest.raises(ValueError):
        replace(cfg, gripper_exit_open_pct=cfg.gripper_exit_closed_pct)
    with pytest.raises(ValueError):
        replace(cfg, gripper_exit_confirmations=0)
    with pytest.raises(ValueError):
        replace(cfg, gripper_exit_closed_pct=float("nan"))
    ready = dict(base.ready, actor_identity={"version": 2})
    disabled = ControlSession(
        base.client, base.inference, base.leader, replace(cfg, gripper_exit=False), ready, initial
    )
    assert disabled.release is None
    try:
        tick(c, clock)
        tick(c, clock, [("key", "c")])
        tick(c, clock, [("key", "right")])
        for _ in range(4):
            snap = snapshot(clock.now)
            snap.payload["arm_left_gripper.pos"] = 45
            c._observe_gripper(snap, clock.now)
            clock.now += 1 / 30
        assert c.state.mode == "awaiting_reward" and not c.gripper_exits
    finally:
        c.record.close()


def test_auto_gate_multiple_entries_fresh_votes_only_and_manual_priority(session):
    c, clock = session
    c.config = replace(c.config, gate_checkpoint="gate.pt", token_checkpoint="token.pt")
    c.gate = GateLatch(0.75, 0.35, 2)
    c.ready["actor_identity"] = dict(sha256="actor", version=5)
    tick(c, clock)

    def frame(i, probability, sequence, events=(), generation=None):
        c.inference.result = _prediction(
            c,
            clock.now,
            plan_id=i,
            length=1,
            generation=c.state.generation if generation is None else generation,
            probability=probability,
            observation=sequence,
        )
        tick(c, clock, events)
        return c.state.critical

    # c cannot force an automatic phase. A second reply about the same Host state is one vote.
    assert not frame(10, 0.97, 1, [("key", "c")])
    assert not frame(11, 0.97, 1)
    assert frame(12, 0.97, 2)
    assert frame(13, 0.15, 3)
    assert not frame(14, 0.12, 4)
    assert not frame(15, 0.97, 5)
    assert frame(16, 0.97, 6)
    old = c.state.generation
    tick(c, clock, [("key", "h")])
    frame(17, 0.99, 7, generation=old)
    assert c.state.mode == "human"
    tick(c, clock, [("key", "h")])
    assert c.state.mode == "vla" and not c.state.critical
    frame(18, 0.99, 8, generation=old)
    assert not frame(19, 0.99, 9)
    rows = _recorded(c)
    assert all(r["critical_source"] == "classifier" and r["critical_label"] is None for r in rows)
    assert [r["plan_id"] for r in rows if r["source"] == "RLT"] == [12, 13, 16]
    assert all(r["requested"] == r["reference"] for r in rows if r["source"] == "VLA")


@pytest.mark.parametrize("bad", [float("nan"), None])
def test_auto_gate_invalid_probability_fails_closed(session, bad):
    c, clock = session
    c.gate = GateLatch(0.75, 0.35, 1)
    c.ready["actor_identity"] = dict(sha256="actor", version=5)
    tick(c, clock)
    c.inference.result = _prediction(
        c, clock.now, plan_id=7, generation=c.state.generation, probability=0.99, observation=1
    )
    tick(c, clock)
    assert c.state.critical and c.actions
    c.inference.result = _prediction(
        c,
        clock.now,
        plan_id=8,
        generation=c.state.generation - 1,
        probability=bad,
        observation=2,
    )
    tick(c, clock)
    assert c.state.critical and c.actions and c.state.mode == "vla"
    c.inference.result = _prediction(
        c,
        clock.now,
        plan_id=9,
        generation=c.state.generation,
        probability=bad,
        observation=3,
    )
    tick(c, clock)
    assert c.state.mode == "paused" and not c.state.critical and not c.actions
    rows = _recorded(c)
    assert rows[-2]["source"] == "RLT"
    assert rows[-1]["source"] == "HOLD" and rows[-1]["hold_reason"] == "invalid_gate"


def test_shadow_gate_without_actor_always_sends_reference(session):
    c, clock = session
    c.gate = GateLatch(0.75, 0.35, 1)
    tick(c, clock)
    c.inference.result = _prediction(
        c, clock.now, plan_id=10, generation=c.state.generation, probability=0.99, observation=1
    )
    tick(c, clock)
    row = _recorded(c)[-1]
    assert row["critical"] and row["source"] == "VLA" and row["requested"] == row["reference"]
