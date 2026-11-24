"""Offline checks for continuous, passive-leader demonstration collection."""

import json
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from alohamini.datasets.record import state_names
from alohamini.model import get_robot_model
from alohamini.protocol import HostSnapshot
from alohamini.rlt.config import RLTConfig
from alohamini.rlt.console import ConsoleInput
from alohamini.rlt.control import RelativeLeader, feedback_after_command
from alohamini.rlt.runtime import ControlSession
from alohamini.rlt.storage import json_lines, load_snapshot, validate_training_episode
from alohamini.schema import CommandIdentity


def config(tmp_path, **changes):
    return RLTConfig(
        checkpoint=str(tmp_path / "base"),
        output_dir=str(tmp_path),
        task="place",
        device="cpu",
        **changes,
    )


def snapshot(now=0):
    model = get_robot_model("alohamini2pro")
    motors = {}
    payload = {name: 0.0 for name in state_names("alohamini2pro")}
    payload["lift_axis.height_mm"] = 100.0
    payload["lift_axis.reference_sequence"] = 1
    for motor in model.actuators:
        if motor.name.startswith("arm_"):
            motors[motor.name] = dict(
                normalization="range_0_100" if motor.name.endswith("gripper") else "range_m100_100",
                range_min=1000,
                range_max=3000,
                drive_mode=0,
            )
    payload["_robot_metadata"] = dict(
        schema_version=1,
        robot_model="alohamini2pro",
        motors=motors,
        cameras=[],
        lift_axis=dict(soft_min_mm=0.0, soft_max_mm=600.0),
    )
    payload["_safety"] = dict(
        version=1,
        feedback_valid=True,
        lift_reference_valid=True,
        phase="ready",
        fault=None,
        host_session_id="host",
        control_epoch=0,
        control_owner=None,
        command={},
        command_watchdog_timeout_s=1.0,
        accepted_targets={},
    )
    payload["_host_timing"] = dict(
        state_sequence=round(now * 30),
        state_sample_started_monotonic_s=now,
        state_sample_finished_monotonic_s=now,
    )
    return HostSnapshot(payload, {}, now, now)


def wait_writer(recorder):
    assert recorder._done.wait(5)
    recorder.check()


@pytest.fixture
def session(tmp_path, monkeypatch):

    clock = SimpleNamespace(now=1.0)
    # Keep multiprocessing.Queue's nonblocking poll on the real monotonic clock.
    monkeypatch.setattr("alohamini.client.time", SimpleNamespace(monotonic=lambda: clock.now))
    native_names = state_names("alohamini2pro")
    learned = [n for n in native_names if n.startswith("arm_left_")]

    class Client:
        client_id = "client"
        sent = []

        def send_command(self, action, *, based_on):
            identity = CommandIdentity("client", len(self.sent) + 1, "host", 0)
            self.sent.append((dict(action), identity))
            return identity

    class Inference:
        result = None
        jobs = []

        def poll(self):
            result, self.result = self.result, None
            return result

        def submit(self, job):
            self.jobs.append(job)
            return True

    leader = SimpleNamespace(read=lambda _age: (clock.now, {n: -30 for n in learned}))
    initial = snapshot(clock.now)
    ready = dict(
        names=native_names,
        fps=30,
        fixed=None,
        metadata=initial.payload["_robot_metadata"],
        base_contract=dict(learned_names=learned),
    )
    session = ControlSession(
        Client(), Inference(), leader, config(tmp_path, return_to_start=False), ready, initial
    )
    yield session, clock
    if session.record is not None:
        session.record.close()


def tick(session, clock, events=()):
    snap = snapshot(clock.now)
    if session.client.sent:
        action, identity = session.client.sent[-1]
        snap.payload["_safety"].update(
            command=asdict(identity),
            accepted_targets=action,
            command_receipts=[
                dict(
                    command=asdict(identity),
                    targets=action,
                    host_cycle=round((clock.now - 1 / 30) * 30),
                    completed_monotonic_s=clock.now - 1 / 30,
                )
            ],
        )
    session.tick(snap, events, clock.now)
    clock.now += 1 / 30


def test_relative_leader_does_not_jump_and_relatches():
    leader = RelativeLeader(["joint", "gripper"], dict(joint=(-100, 100), gripper=(0, 100)))
    leader.latch(dict(joint=-60, gripper=90), dict(joint=20, gripper=30))
    assert leader.targets(dict(joint=-60, gripper=90)) == dict(joint=20, gripper=30)
    assert leader.targets(dict(joint=-50, gripper=0)) == dict(joint=30, gripper=0)
    leader.latch(dict(joint=80, gripper=0), dict(joint=-10, gripper=40))
    assert leader.targets(dict(joint=85, gripper=10)) == dict(joint=-5, gripper=50)


def test_human_takeover_ignores_late_gpu_result_and_waits_for_manual_reward(session):
    session, clock = session
    tick(session, clock)
    old_generation = session.state.generation
    tick(session, clock, [("key", "h")])
    assert session.state.mode == "human"
    assert session.takeovers == 1
    session.inference.result = dict(
        kind="prediction",
        generation=old_generation,
        id=1,
        sampled_s=clock.now,
        inference_s=0.1,
        actions=np.full((10, 18), 99),
        reference=np.full((10, 18), 99),
        actor_version=1000,
    )
    tick(session, clock)
    assert session.state.mode == "human" and not session.actions
    assert session.client.sent[-1][0][session.relative.names[0]] == 0
    tick(session, clock, [("key", "right")])
    for _ in range(30):
        tick(session, clock)
    assert session.state.mode == "awaiting_reward"
    assert session.record.committed is None
    assert not session.finished
    tick(session, clock, [("reward", 1)])
    wait_writer(session.record)
    assert session.finished[0]["takeovers"] == 1
    assert session.record.committed is not None


def test_zero_takeovers_and_multiple_takeovers_are_both_valid(session):
    session, clock = session
    tick(session, clock)
    tick(session, clock, [("key", "c")])
    tick(session, clock, [("key", "right")])
    tick(session, clock, [("reward", 0)])
    wait_writer(session.record)
    assert session.finished[-1]["takeovers"] == 0
    tick(session, clock)
    for _ in range(3):
        tick(session, clock, [("key", "h")])
        tick(session, clock, [("key", "h")])
    tick(session, clock, [("key", "right")])
    tick(session, clock, [("reward", 1)])
    wait_writer(session.record)
    assert session.finished[-1]["takeovers"] == 3


def pose(session, joint=5, gripper=40):
    return {
        name: float(gripper if name.endswith("_gripper.pos") else joint)
        for name in session.relative.names
    }


def step(session, clock, events=(), *, positions=None, accept=True, post_application=True, age=0):
    reply = snapshot(clock.now - age)
    if session.client.sent and accept:
        action, identity = session.client.sent[-1]
        reply.payload.update(action)
        completed = clock.now - 1 / session.fps
        reply.payload["_safety"].update(
            command=asdict(identity),
            accepted_targets=action,
            command_receipts=[
                dict(
                    command=asdict(identity),
                    targets=action,
                    host_cycle=round(completed * session.fps),
                    completed_monotonic_s=completed,
                )
            ],
        )
        if not post_application:
            reply.payload["_host_timing"]["state_sample_started_monotonic_s"] = completed - 0.001
    if positions is not None:
        reply.payload.update(positions)
    session.tick(reply, events, clock.now)
    clock.now += 1 / session.fps
    return reply


def end_episode(session, clock, *, reward=None):
    step(session, clock, positions=pose(session))
    step(session, clock, [("key", "c"), ("key", "h")], positions=pose(session, 15, 10))
    events = ConsoleInput().feed("\x1b[C")
    if reward is not None:
        events.append(("reward", reward))
    step(session, clock, events)
    step(session, clock)
    assert session.record.terminal_saved.wait(5)
    if reward is not None:
        wait_writer(session.record)
    step(session, clock)
    assert session.return_motion is not None


def finish_return(session, clock):
    for _ in range(240):
        if not session.return_pending:
            break
        step(session, clock)
    assert not session.return_pending


@pytest.fixture
def returning(session):
    session, clock = session
    session.config = replace(session.config, return_to_start=True, return_duration_s=0.2)
    return session, clock


def test_collect_starts_on_valid_feedback_without_n_and_h_toggles(session):
    session, clock = session
    leader = session.leader
    session.leader = SimpleNamespace(read=lambda _age: None, error=None)
    step(session, clock)
    assert session.record is None and not session.inference.jobs
    session.leader = leader
    step(session, clock, age=1)
    assert session.record is None
    step(session, clock, positions=pose(session))
    first = session.record
    assert session.state.mode == "vla" and session.start_pose == pose(session)
    step(session, clock, [("key", "h")])
    assert session.state.mode == "human"
    step(session, clock, [("key", "h")])
    assert session.state.mode == "vla" and session.record is first and session.takeovers == 1


def test_collect_score_after_reset_waits_for_disk_and_new_observation(returning, monkeypatch):
    from alohamini.rlt import storage

    session, clock = returning
    end_episode(session, clock)
    finish_return(session, clock)
    record = session.record
    for _ in range(3):
        step(session, clock)
    assert session.record is record and session.state.mode == "awaiting_reward"
    entered, release = threading.Event(), threading.Event()
    commit = storage.commit_episode

    def delayed_commit(*args):
        entered.set()
        assert release.wait(5)
        return commit(*args)

    monkeypatch.setattr(storage, "commit_episode", delayed_commit)
    key_time = clock.now
    try:
        session.event("reward", 1, snapshot(key_time), key_time, defer_feedback=True)
        assert entered.wait(5)
        step(session, clock)
        assert session.record is record
    finally:
        release.set()
    wait_writer(record)
    step(session, clock, age=clock.now - key_time + 0.001)
    assert session.record is record
    step(session, clock)
    assert session.state.mode == "vla" and session.record.id != record.id
    assert len(session.finished) == 1 and session.finished[0]["reward"] == 1


def test_early_score_stays_stopped_until_a_separate_enter_after_return(returning):
    session, clock = returning
    end_episode(session, clock, reward=0)
    record = session.record
    step(session, clock, [("key", "enter")])
    finish_return(session, clock)
    for _ in range(3):
        step(session, clock)
    assert session.record is record and session.awaiting_reset_confirmation
    console = ConsoleInput()
    assert console.feed("0\r", awaiting_reward=True) == [("reward", 0)]
    assert console.feed("\n", confirm_reset=True) == []  # LF from the previous CRLF.
    step(session, clock, console.feed("\r", confirm_reset=True))
    assert session.record is not record and session.state.mode == "vla"
    assert len(session.finished) == 1 and session.finished[0]["reward"] == 0


def test_collect_quit_or_failed_leader_at_start_creates_no_episode(session):
    session, clock = session
    session.leader = SimpleNamespace(read=lambda _age: None, error=ConnectionError("0x01"))
    with pytest.raises(RuntimeError, match="0x01"):
        step(session, clock)
    step(session, clock, [("key", "q")])
    assert session.state.quit and session.record is None and not session.inference.jobs


def test_return_waits_until_terminal_is_durably_saved(returning, monkeypatch):
    from alohamini.rlt import storage

    session, clock = returning
    entered, release = threading.Event(), threading.Event()
    save = storage.save_snapshot

    def delayed_save(path, value):
        if path.name == "terminal":
            entered.set()
            assert release.wait(5)
        save(path, value)

    monkeypatch.setattr(storage, "save_snapshot", delayed_save)
    try:
        step(session, clock, positions=pose(session))
        step(session, clock, positions=pose(session, 15, 10))
        step(session, clock, [("key", "right")])
        step(session, clock)
        assert entered.wait(5)
        for _ in range(3):
            step(session, clock)
            assert session.return_motion is None
            assert session.client.sent[-1][0][session.relative.names[0]] == 15
        release.set()
        assert session.record.terminal_saved.wait(5)
        step(session, clock)
        assert session.return_motion is not None
    finally:
        release.set()


def test_feedback_gap_pauses_return_until_operator_resumes(returning):
    session, clock = returning
    end_episode(session, clock)
    clock.now += 0.2
    step(session, clock)
    assert session.return_paused and session.return_motion is None
    for _ in range(3):
        session.pause(snapshot(clock.now), clock.now, feedback_lost=True)
        step(session, clock)
        assert session.return_paused and session.return_motion is None
    step(session, clock, [("key", " ")])
    assert not session.return_paused and session.return_motion is not None


def test_return_goal_waits_for_its_own_receipt_and_later_feedback(returning):
    session, clock = returning
    end_episode(session, clock, reward=0)
    for _ in range(150):
        step(session, clock)
        if session.return_motion.fence is not None:
            break
    motion = session.return_motion
    assert motion.fence is not None and motion.outcome is None
    count = len(session.client.sent)
    for _ in range(2):
        step(session, clock, accept=False, positions=session.start_pose)
        assert len(session.client.sent) == count and session.return_pending
    step(session, clock, post_application=False)
    assert motion.outcome["status"] == "applied" and motion.stable_since_s is None
    finish_return(session, clock)
    assert session.finished[0]["reward"] == 0


@pytest.mark.parametrize("status", ["applied", "superseded", "rejected"])
def test_return_consumes_new_host_dispositions_after_episode_commit(returning, status):
    session, clock = returning
    end_episode(session, clock, reward=1)
    for _ in range(150):
        step(session, clock)
        if session.return_motion.fence is not None:
            break
    motion = session.return_motion
    action, identity = session.client.sent[-1]
    before = json_lines(session.record.committed / "acks.jsonl")
    reply = snapshot(clock.now)
    reply.payload.update(action)
    completed = clock.now - 1 / session.fps
    reply.payload["_safety"].update(
        command_result_schema=1,
        command_results=[
            dict(
                command=asdict(identity),
                status=status,
                result_id=1,
                targets=action if status == "applied" else {},
                host_cycle=round(completed * session.fps),
                completed_monotonic_s=completed,
                last_sequence=identity.sequence,
            )
        ],
    )
    session.tick(reply, (), clock.now)
    clock.now += 1 / session.fps
    assert motion.outcome["status"] == status
    assert session.control_suspended is (status != "applied")
    assert json_lines(session.record.committed / "acks.jsonl") == before
    if status == "applied":
        finish_return(session, clock)


def outcome(identity, status, result_id, targets=None, *, cycle=40, completed=1.2, last=None):
    return dict(
        command=asdict(identity),
        status=status,
        result_id=result_id,
        targets=targets or {},
        last_sequence=identity.sequence if last is None else last,
        host_cycle=cycle,
        completed_monotonic_s=completed,
    )


def deliver(session, clock, results, position, *, cycle, sampled, request=None):
    reply = snapshot(clock.now if request is None else request)
    reply.payload.update({name: float(position) for name in session.relative.names})
    reply.payload["_host_timing"].update(
        state_sequence=cycle,
        state_sample_started_monotonic_s=sampled,
        state_sample_finished_monotonic_s=sampled,
    )
    reply.payload["_safety"].update(command_result_schema=1, command_results=results)
    applied = [row for row in results if row["status"] == "applied"]
    if applied:
        reply.payload["_safety"].update(
            command=applied[-1]["command"], accepted_targets=applied[-1]["targets"]
        )
    session.tick(reply, [], clock.now)
    clock.now += 1 / 30
    return reply


@pytest.mark.parametrize("key", [" ", "right"])
@pytest.mark.parametrize("backpressure", [False, True])
def test_pre_key_reply_only_supplies_temporary_hold_and_never_consumes_anchor(
    session, key, backpressure, monkeypatch
):
    session, clock = session
    tick(session, clock)
    pressed = clock.now
    session.event("key", key, snapshot(pressed - 0.02), pressed, defer_feedback=True)
    send = session.client.send_command
    if backpressure:
        monkeypatch.setattr(session.client, "send_command", lambda *_args, **_kwargs: None)
    deliver(session, clock, [], 10, cycle=40, sampled=1.0, request=pressed - 0.01)
    assert session.needs_hold_latch
    if not backpressure:
        assert session.client.sent[-1][0][session.relative.names[0]] == 10
    monkeypatch.setattr(session.client, "send_command", send)
    assert session.hold[session.relative.names[0]] != 10
    deliver(session, clock, [], 20, cycle=41, sampled=1.1)
    assert not session.needs_hold_latch and session.hold[session.relative.names[0]] == 20
    deliver(session, clock, [], 30, cycle=42, sampled=1.2)
    assert session.hold[session.relative.names[0]] == 20
    assert session.client.sent[-1][0][session.relative.names[0]] == 20


def test_terminal_waits_for_late_application_then_a_later_feedback_cycle(session):
    session, clock = session
    tick(session, clock)
    tick(session, clock, [("key", "c"), ("key", "h")])
    assert session.state.mode == "human"
    targets, command = session.client.sent[-1]
    session.event("key", "right", snapshot(clock.now), clock.now, defer_feedback=True)
    session.event("reward", 1, snapshot(clock.now), clock.now, defer_feedback=True)
    deliver(session, clock, [], 10, cycle=40, sampled=1.1)
    assert session.terminal_pending and not (session.record.directory / "terminal").exists()
    fence = session.end_fence
    results = [outcome(command, "applied", 1, targets, cycle=41, completed=1.2)]
    deliver(session, clock, results, 20, cycle=41, sampled=1.19)
    assert session.terminal_pending
    results.append(
        outcome(fence, "applied", 2, session.client.sent[-1][0], cycle=42, completed=1.3)
    )
    deliver(session, clock, results, 30, cycle=42, sampled=1.29)
    assert session.terminal_pending and not session.finished
    terminal = deliver(session, clock, results, 40, cycle=43, sampled=1.4)
    wait_writer(session.record)
    saved = load_snapshot(session.record.committed / "terminal")
    assert saved.payload[session.relative.names[0]] == 40
    assert saved.payload["_host_timing"] == terminal.payload["_host_timing"]
    assert saved.payload["_host_timing"]["state_sample_started_monotonic_s"] > 1.3
    state_path = session.record.committed / "terminal/state.json"
    altered = json.loads(state_path.read_text())
    altered["payload"]["_host_timing"].update(
        state_sequence=42,
        state_sample_started_monotonic_s=1.29,
    )
    state_path.write_text(json.dumps(altered))
    summary = json.loads((session.record.committed / "episode.json").read_text())
    with pytest.raises(ValueError, match="predates a relevant command application"):
        validate_training_episode(session.record.committed, summary)


def test_unproven_history_after_applied_fence_is_scored_but_excluded_from_training(session):
    session, clock = session
    tick(session, clock)
    tick(session, clock, [("key", "c"), ("key", "h")])
    assert session.state.mode == "human"
    missing = session.client.sent[-1][1]
    session.event("key", "right", snapshot(clock.now), clock.now, defer_feedback=True)
    session.event("reward", 0, snapshot(clock.now), clock.now, defer_feedback=True)
    deliver(session, clock, [], 10, cycle=40, sampled=1.1)
    fence = session.end_fence
    results = [outcome(fence, "applied", 1, session.client.sent[-1][0])]
    deliver(session, clock, results, 10, cycle=41, sampled=1.3)
    wait_writer(session.record)
    assert session.command_outcomes[missing]["status"] == "unknown"
    assert not session.finished[0]["training_valid"] and not session.control_suspended
    summary = json.loads((session.record.committed / "episode.json").read_text())
    with pytest.raises(ValueError, match="excluded from training"):
        validate_training_episode(session.record.committed, summary)
    tick(session, clock)
    assert session.state.active


def test_settle_timeout_keeps_reward_without_fabricating_a_terminal(session):
    session, clock = session
    tick(session, clock)
    tick(session, clock, [("key", "c"), ("key", "h")])
    assert session.state.mode == "human"
    session.event("key", "right", snapshot(clock.now), clock.now, defer_feedback=True)
    session.event("reward", 1, snapshot(clock.now), clock.now, defer_feedback=True)
    clock.now += session.config.command_settle_timeout_s + 0.01
    session.check_end_timeout(clock.now)
    wait_writer(session.record)
    saved = json.loads((session.record.committed / "episode.json").read_text())
    assert saved["reward"] == 1 and not saved["training_valid"] and not saved["terminal_valid"]
    assert not (session.record.committed / "terminal").exists()
    assert session.state.mode == "idle" and not session.pending_commands


def test_rejected_end_fence_saves_invalid_score_and_does_not_wait_forever(session):
    session, clock = session
    tick(session, clock)
    session.event("key", "right", snapshot(clock.now), clock.now, defer_feedback=True)
    session.event("reward", 0, snapshot(clock.now), clock.now, defer_feedback=True)
    deliver(session, clock, [], 10, cycle=40, sampled=1.1)
    fence = session.end_fence
    deliver(session, clock, [outcome(fence, "rejected", 1)], 10, cycle=41, sampled=1.3)
    wait_writer(session.record)
    saved = json.loads((session.record.committed / "episode.json").read_text())
    assert saved["reward"] == 0 and not saved["training_valid"]
    assert not saved["terminal_valid"] and session.control_suspended
    assert session.state.mode == "idle" and not session.pending_commands


@pytest.mark.parametrize(
    "field,value", [("host_session_id", "different-host"), ("control_epoch", 1)]
)
def test_terminal_ordering_proof_is_bound_to_the_command_lease(field, value):
    frame = snapshot(2.0)
    frame.payload["_safety"][field] = value
    result = outcome(CommandIdentity("client", 1, "host", 0), "applied", 1, cycle=1, completed=1.0)
    assert not feedback_after_command(frame, result)


def test_stock_host_allows_only_one_normal_command_in_flight(session):
    session, clock = session
    first = snapshot(clock.now)
    session.tick(first, (), clock.now)
    _, sent = session.client.sent[-1]
    for _ in range(3):
        clock.now += 1 / 60
        session.tick(snapshot(clock.now), (), clock.now)
    assert len(session.client.sent) == 1
    assert sent in session.pending_commands
    clock.now += 1 / 60
    applied = snapshot(clock.now)
    applied.payload["_safety"].update(
        command=asdict(sent), accepted_targets=session.client.sent[0][0]
    )
    session.tick(applied, (), clock.now)
    assert len(session.client.sent) == 2
    assert sent in session.acknowledged_commands


def test_no_manual_start_hotkey_and_passive_only_config(tmp_path):
    assert ConsoleInput().feed("n") == []
    cfg = config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(cfg.to_dict(), leader_output_limits={"shoulder_lift": 0.5})))
    with pytest.raises(ValueError, match="passive"):
        RLTConfig.read(path)


@pytest.mark.parametrize("use_json", [False, True])
def test_collect_cli_selects_task_checkpoint_hardware_and_fixed_axes(tmp_path, monkeypatch, use_json):
    from alohamini.rlt import runtime
    from alohamini.rlt.cli import main

    monkeypatch.chdir(tmp_path)
    saved = {}

    def capture(config, *, resume):
        saved.update(config=config, resume=resume)

    monkeypatch.setattr(runtime, "run", capture)
    args = ["collect"]
    folder = tmp_path / "settings"
    if use_json:
        folder.mkdir()
        path = folder / "defaults.json"
        path.write_text(json.dumps(dict(
            host="test-host", calibration_dir="calibration", task="old task",
            checkpoint="old-model", fixed_dimensions=["lift_axis"], fps=15,
        )))
        args += ["--config", str(path)]
    else:
        args += ["--host", "test-host", "--calibration-dir", "calibration"]
    args += [
        "--policy.path", "model", "--task", "place the orange module", "--output-dir", "episodes",
        "--robot_model", "alohamini2pro", "--device", "cuda", "--fps", "30",
        "--chunk-size", "15", "--policy.n_action_steps", "8", "--fixed-dimensions", "arm_right", "lift_axis",
        "--left-port", "/dev/test-left", "--leader-id", "leader", "--resume",
    ]
    assert main(args) == 0
    cfg = saved["config"]
    assert saved["resume"] is True
    assert (cfg.task, cfg.checkpoint) == ("place the orange module", str(tmp_path / "model"))
    assert cfg.output_dir == str(tmp_path / "episodes")
    assert cfg.fixed_dimensions == ["arm_right", "lift_axis"]
    assert (cfg.fps, cfg.chunk_size, cfg.execution_steps, cfg.device) == (30, 15, 8, "cuda")
    assert (cfg.host, cfg.robot_model, cfg.left_port, cfg.leader_id) == (
        "test-host", "alohamini2pro", "/dev/test-left", "leader",
    )
    assert cfg.calibration_dir == str((folder if use_json else tmp_path) / "calibration")


def test_collect_command_runs_episode_scoring_and_cleanup_through_cli(tmp_path, monkeypatch):
    from alohamini.rlt import runtime
    from alohamini.rlt.cli import main

    names = state_names("alohamini2pro")
    learned = [n for n in names if n.startswith("arm_left_")]
    sent = []
    command_times = {}
    command_cycles = {}

    class Client:
        client_id = "client"

        def __init__(self, *_args, **_kwargs):
            self.host_cycle = 0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def connect_control(self):
            return snapshot(time.monotonic())

        def read(self, **_kwargs):
            result = snapshot(time.monotonic())
            self.host_cycle += 2
            result.payload["_host_timing"]["state_sequence"] = self.host_cycle
            if sent:
                result.payload["_safety"].update(
                    command=asdict(sent[-1][1]),
                    accepted_targets=sent[-1][0],
                    command_receipts=[
                        dict(
                            command=asdict(sent[-1][1]),
                            targets=sent[-1][0],
                            host_cycle=command_cycles[sent[-1][1]],
                            completed_monotonic_s=command_times[sent[-1][1]],
                        )
                    ],
                )
            return result

        def send_command(self, action, **_kwargs):
            identity = CommandIdentity("client", len(sent) + 1, "host", 0)
            sent.append((dict(action), identity))
            command_times[identity] = time.monotonic()
            command_cycles[identity] = self.host_cycle + 1
            return identity

    class Console:
        def __init__(self):
            self.events = iter(
                [
                    [],
                    [],  # The first ready collection tick starts the episode.
                    [("key", "c")],
                    [("key", "h")],
                    [],
                    [],
                    [],
                    [("key", "h")],
                    [("key", "right")],
                    [("reward", 1)],
                    [("key", "q")],
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def poll(self, **_kwargs):
            return next(self.events)

    class Inference:
        def __init__(self, _config, initial, **_kwargs):
            self.ready = dict(
                kind="ready",
                names=names,
                fps=30,
                fixed=None,
                metadata=initial.payload["_robot_metadata"],
                base_contract=dict(learned_names=learned),
                cameras=[],
            )

        def poll(self):
            result, self.ready = self.ready, None
            return result

        def submit(self, _job):
            return True

        def close(self):
            pass

    class Leader:
        def __init__(self, *_args):
            pass

        def read(self, _age):
            return time.monotonic(), {name: 0 for name in learned}

        def close(self):
            pass

    stopped = []
    monkeypatch.setattr(runtime, "HostClient", Client)
    monkeypatch.setattr(runtime, "ConsoleInput", Console)
    monkeypatch.setattr(runtime, "InferenceWorker", Inference)
    monkeypatch.setattr(runtime, "LeaderReader", Leader)
    monkeypatch.setattr(runtime, "stop_owned_robot", lambda *_args: stopped.append(True))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(tmp_path).to_dict()))
    assert main(["collect", "--config", str(path)]) == 0
    summaries = list((tmp_path / "episodes").glob("*/episode.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary["reward"] == 1 and summary["takeovers"] == 1
    assert stopped == [True]
    assert summary["evaluation"] is False and summary["training_valid"]


def test_stock_host_manual_episode_finalizes_using_only_its_last_application(session):
    session, clock = session

    def legacy_tick(events=()):
        clock.now += 1 / session.fps
        frame = snapshot(clock.now)
        if session.client.sent:
            targets, identity = session.client.sent[-1]
            frame.payload["_safety"].update(command=asdict(identity), accepted_targets=targets)
            frame.payload.update(targets)
        session.tick(frame, events, clock.now)

    legacy_tick()
    legacy_tick([("key", "h")])
    legacy_tick([("key", "h")])
    legacy_tick([("key", "right")])
    legacy_tick()
    assert session.terminal_pending
    legacy_tick([("reward", 1)])
    wait_writer(session.record)
    summary = json.loads((session.record.committed / "episode.json").read_text())
    assert summary["reward"] == 1 and summary["takeovers"] == 1
    assert summary["training_valid"] and summary["critical_entries"] == 0
    assert {row["source"] for row in json_lines(session.record.committed / "steps.jsonl")} >= {
        "HUMAN"
    }
    validate_training_episode(session.record.committed, summary)
