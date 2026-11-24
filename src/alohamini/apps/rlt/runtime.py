"""Responsive robot control with user-latched interventions and manual scoring."""

import json
import math
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from alohamini.apps.replay import TargetRanges
from alohamini.apps.rlt.console import ConsoleInput
from alohamini.apps.rlt.control import (
    ControlState,
    RelativeLeader,
    ReturnToStart,
)
from alohamini.apps.rlt.gate import GripperRelease
from alohamini.apps.rlt.v3_worker import collection_publisher
from alohamini.apps.rlt.workers import InferenceWorker, LeaderReader
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.client import HostClient, control_feedback_valid
from alohamini.datasets.rlt.storage import EpisodeRecorder, RunLock, atomic_json, recording_history
from alohamini.errors import ResponseTimeoutError
from alohamini.fixed import FixedGuard, limits, restore
from alohamini.policies.rlt.gate import GateLatch
from alohamini.remote.protocol import context_for
from alohamini.runtime.command_feedback import feedback_after_command
from alohamini.schema import CommandIdentity

HELP = "h 人工接管/交还 | 空格 暂停/继续 | 右箭头 结束并回位 | 0/1+回车 评分 | q 退出"


class ControlSession:
    def __init__(self, client, inference, leader, config, ready, initial):
        self.client, self.inference, self.leader = client, inference, leader
        self.config, self.ready = config, ready
        self.state = ControlState()
        # A queued start consumes only feedback requested after the operator's confirmation.
        self.collect_start_after = -math.inf
        self.names, self.fps = ready["names"], ready["fps"]
        self.ranges = TargetRanges(self.names, initial)
        bounds = {f"{n}.pos": (low, high) for n, _, low, high, _ in self.ranges.joints}
        self.relative = RelativeLeader(ready["base_contract"]["learned_names"], bounds)
        self.context = context_for(initial)
        self.metadata = deepcopy(initial.payload["_robot_metadata"])
        self.fixed = (ready["fixed"] or {}).get("targets", {})
        self.fixed_guard = FixedGuard(ready["fixed"], ready["metadata"])
        self.actions = deque()
        self.gate = (
            GateLatch(
                config.gate_enter_threshold, config.gate_exit_threshold, config.gate_confirmations
            )
            if config.gate_checkpoint
            else None
        )
        self.gate_probability = None
        enable_release = config.gripper_exit
        if enable_release is None:
            enable_release = bool(ready.get("actor_identity") or self.gate)
        self.release = (
            GripperRelease(
                config.gripper_exit_closed_pct,
                config.gripper_exit_open_pct,
                config.gripper_exit_confirmations,
            )
            if enable_release
            else None
        )
        self.phase_source = "classifier" if self.gate else "manual"
        self.release_gate_after_s = -math.inf
        self.gripper_exits = []
        self.gate_sampled_s = -math.inf
        self.gate_sequence = None
        self.record = None
        self.identity = None
        self.step_index = self.request_id = 0
        self.hold = self._hold(initial)
        self.inference_s = 0.1
        self.last_ack = None
        self.last_ack_time = None
        self.legacy_inflight = None
        self.urgent_hold = False
        self.human_since = None
        self.critical_since = None
        self.critical_seconds = 0.0
        self.critical_entries = 0
        self.takeovers = 0
        self.human_seconds = 0.0
        self.finished = []
        self.started = None
        self.ended_summary = None
        self.plan_id = None
        self.plan_actor_version = -1
        self.needs_hold_latch = False
        self.needs_human_latch = False
        self.terminal_pending = False
        self.pending_reward = None
        self.reward_submitted = False
        self.episode_commands = set()
        self.pending_commands = set()
        self.acknowledged_commands = set()
        self.last_control_s = None
        self.feedback_after_s = -math.inf
        self.command_outcomes = {}
        self.last_result_id = 0
        self.end_fence = None
        self.end_started_s = None
        self.control_suspended = False
        self.start_pose = None
        self.return_pending = False
        self.return_paused = False
        self.return_motion = None

    def _hold(self, snapshot):
        targets = {n: float(snapshot.payload[n]) for n in self.names}
        targets.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        targets.update(self.fixed if hasattr(self, "fixed") else {})
        return targets

    def _start_episode(self, now):
        root = Path(self.config.output_dir)
        self.record = EpisodeRecorder(
            root,
            dict(
                base_contract=self.ready["base_contract"],
                actor=self.ready.get("actor_identity"),
                gate=self.ready.get("gate_identity"),
                gripper_exit_rule=(
                    dict(
                        closed_pct=self.release.closed,
                        open_pct=self.release.opened,
                        confirmations=self.release.confirmations,
                    )
                    if self.release
                    else None
                ),
                evaluation=False,
                started_at=time.time(),
                fps=self.fps,
                task=self.config.task,
                phase_labels="classifier-output-v1" if self.gate else "optional-operator-v2",
                requires_acknowledgements=True,
                requires_command_dispositions=True,
                return_to_start=self.config.return_to_start,
            ),
        )
        self.step_index = self.takeovers = 0
        self.plan_id = None
        self.gate_probability = self.gate_sequence = None
        self.gate_sampled_s = -math.inf
        if self.gate:
            self.gate.reset()
        if self.release:
            self.release.new_phase()
            self.release.block_reentry = False
        self.gripper_exits = []
        self.release_gate_after_s = -math.inf
        self.phase_source = "classifier" if self.gate else "manual"
        self.human_seconds = 0.0
        self.human_since = self.critical_since = None
        self.critical_seconds = 0.0
        self.critical_entries = 0
        self.pending_reward = None
        self.reward_submitted = False
        self.terminal_pending = False
        self.episode_commands.clear()
        self.pending_commands.clear()
        self.acknowledged_commands.clear()
        self.command_outcomes.clear()
        self.end_fence = None
        self.end_started_s = None
        self.start_pose = None
        self.return_pending = False
        self.return_paused = False
        self.return_motion = None
        self.started = now
        self.collect_start_after = None
        print(f"回合 {self.record.id} 已开始。", flush=True)

    @property
    def awaiting_reset_confirmation(self):
        return (
            self.reward_submitted
            and not self.return_pending
            and not self.control_suspended
            and self.collect_start_after is None
        )

    def _start_collection_if_ready(self, snapshot, now):
        if (
            self.collect_start_after is None
            or snapshot.request_started_s < self.collect_start_after
            or self.state.mode != "idle"
            or self.control_suspended
            or self.return_pending
            or (self.record is not None and self.record.committed is None)
        ):
            return
        if self.leader.read(self.config.leader_max_age_s) is None:
            error = getattr(self.leader, "error", None)
            if error is not None:
                raise RuntimeError(f"无法开始采集，左主臂读取失败。{error}") from error
            return
        self._start_episode(now)
        self.state.start()
        self.hold = self._hold(snapshot)

    def _leave_human(self, now):
        self.needs_human_latch = False
        if self.human_since is not None:
            self.human_seconds += now - self.human_since
            self.human_since = None

    def pause(self, snapshot, now, *, feedback_lost=False):
        if self.release:
            self.release.votes = 0
        if self.state.active and self.state.mode != "paused":
            self.urgent_hold = True
        self.state.pause()
        self._reset_gate(now)
        if self.return_pending:
            if not self.return_paused:
                print("回位已暂停，反馈恢复后按空格继续。", flush=True)
            self.return_paused = True
            self.return_motion = None
        self.actions.clear()
        self._leave_human(now)
        if feedback_lost:
            self.feedback_after_s = max(self.feedback_after_s, now)
        if feedback_lost or snapshot.request_started_s < self.feedback_after_s:
            self.needs_hold_latch = True
        else:
            self.hold = self._hold(snapshot)
            self.needs_hold_latch = False

    def _phase_changed(self, now, source):
        self.actions.clear()
        self.phase_source = source
        if self.release:
            self.release.new_phase()
        if self.state.critical:
            self.critical_since = now
            self.critical_entries += 1
        elif self.critical_since is not None:
            self.critical_seconds += now - self.critical_since
            self.critical_since = None
        phase = "开启" if self.state.critical else "关闭"
        policy = "RLT" if self.state.critical and self.ready.get("actor_identity") else "π0.5"
        print(f"对槽阶段已{phase}，自动控制时使用 {policy}。", flush=True)

    def _reset_gate(self, now):
        if self.gate is None:
            return
        self.gate.reset()
        self.gate_probability = None
        if self.state.critical and self.state.active:
            self.state.critical = False
            self._phase_changed(now, "classifier")

    def _observe_gripper(self, snapshot, now):
        if self.release is None or self.state.mode not in ("vla", "human"):
            return
        opening = snapshot.payload.get("arm_left_gripper.pos")
        sequence = snapshot.payload.get("_host_timing", {}).get("state_sequence")
        if type(sequence) is not int or type(opening) not in (float, int):
            self.release.votes = 0
            return
        blocked = self.release.block_reentry
        released = self.release.observe(
            opening, sequence, now, 2.5 / self.fps, critical=self.state.critical
        )
        if blocked and not self.release.block_reentry:
            self.release_gate_after_s = snapshot.request_started_s
        if not released:
            return
        self.state.critical = False
        self.state.generation += 1
        self._phase_changed(now, "gripper_release")
        if self.gate:
            self.gate.reset()
        self.gripper_exits.append(
            dict(
                next_step_index=self.step_index,
                time=now,
                state_sequence=sequence,
                opening_pct=opening,
                generation=self.state.generation,
            )
        )
        self.record.put("summary", dict(gripper_exits=list(self.gripper_exits)))
        print(f"左夹爪实测开度 {opening:.1f}%，已退出 RLT。", flush=True)

    def _finish_reward(self):
        if (
            self.pending_reward is None
            or self.terminal_pending
            or self.pending_commands
            or self.reward_submitted
        ):
            return
        value = self.pending_reward
        self.record.submit_reward(value)
        self.reward_submitted = True
        self.state.submit_reward(value)
        result = dict(self.ended_summary, reward=value, episode_id=self.record.id)
        self.finished.append(result)
        atomic_json(Path(self.config.output_dir) / "session-metrics.json", self.statistics())
        if self.control_suspended:
            print("奖励已保存，停止尚未确认。请确认机器人停止并重连控制后再复位。", flush=True)
        elif self.return_pending:
            print("已收到评分。回位结束、物体摆好后，按回车开始下一回合。", flush=True)
        else:
            print("已收到评分，正在保存回合。", flush=True)

    def _resolve_command(self, identity, result, now):
        status = result["status"]
        targets = result.get("targets", {})
        if status == "applied" and not set(self.relative.names) <= targets.keys():
            return
        outcome = dict(result, command=asdict(identity), observed_s=now, time=now)
        if self.return_motion is not None and identity == self.return_motion.fence:
            if self.return_motion.outcome is None:
                self.return_motion.outcome = outcome
            return
        if (
            self.record is None
            or self.reward_submitted
            or identity not in self.episode_commands
            or identity in self.command_outcomes
        ):
            return
        self.record.acknowledgement(outcome)
        self.command_outcomes[identity] = outcome
        self.pending_commands.discard(identity)
        if status == "applied":
            self.acknowledged_commands.add(identity)

    def _resolve_unknown(self, reason, now):
        for identity in tuple(self.pending_commands):
            self._resolve_command(identity, dict(status="unknown", reason=reason), now)
        self.ended_summary.update(training_valid=False, invalid_reason=reason)

    def _invalidate_end(self, reason, now):
        self._resolve_unknown(reason, now)
        self.control_suspended = True
        self.ended_summary["terminal_valid"] = False
        self.record.end_episode(None, self.ended_summary)
        self.terminal_pending = False
        self.return_pending = False
        self.return_motion = None
        print(f"回合记录不可用于训练，人工评分仍会保存。{reason}", flush=True)
        self._finish_reward()

    def check_end_timeout(self, now):
        if (
            self.terminal_pending
            and now - self.end_started_s >= self.config.command_settle_timeout_s
        ):
            self._invalidate_end("Stop command or post-application feedback was not confirmed", now)

    def _seal_terminal(self, snapshot, now, observation_ready):
        if not self.terminal_pending or self.end_fence is None:
            return
        fence = self.command_outcomes.get(self.end_fence)
        if fence is None:
            return
        if fence["status"] != "applied":
            self._invalidate_end("Final HOLD command was not applied", now)
            return
        applications = [row for row in self.command_outcomes.values() if row["status"] == "applied"]
        if not observation_ready or not all(
            feedback_after_command(snapshot, row) for row in applications
        ):
            return
        if self.pending_commands:
            self._resolve_unknown(
                "Command history does not prove earlier command dispositions", now
            )
        if any(row["status"] == "unknown" for row in self.command_outcomes.values()):
            self.ended_summary["training_valid"] = False
            self.ended_summary.setdefault("invalid_reason", "Unknown command application")
        self.ended_summary.update(terminal_valid=True, end_fence=asdict(self.end_fence))
        self.record.end_episode(snapshot, self.ended_summary)
        self.terminal_pending = False

    def _latch_human(self, snapshot, now, sample=None):
        if sample is None:
            sample = self.leader.read(self.config.leader_max_age_s)
        self.needs_human_latch = False
        if sample is None:
            self.state.pause()
            print("主臂数据不可用，保持当前位置。主臂恢复后按 h。", flush=True)
            return
        self.relative.latch(sample[1], snapshot.payload)
        self.takeovers += 1
        self.human_since = now

    def feedback_fresh(self, snapshot, now):
        return control_feedback_valid(snapshot)

    def _return_targets(self, snapshot, now, observation_ready):
        if self.return_paused or not observation_ready or not self.record.terminal_saved.is_set():
            return self._hold(snapshot) if self.needs_hold_latch else dict(self.hold)
        if self.start_pose is None:
            self.return_pending = False
            print("本回合起点尚未采集，保持当前位置。", flush=True)
            return dict(self.hold)
        if self.return_motion is None:
            origin = {name: float(snapshot.payload[name]) for name in self.start_pose}
            duration = max(
                self.config.return_duration_s,
                max(
                    1.5 * abs(self.start_pose[name] - value) / limits(name, self.metadata)[0]
                    for name, value in origin.items()
                ),
            )
            self.return_motion = ReturnToStart(origin, dict(self.start_pose), now, duration)
            self.hold = self._hold(snapshot)
            print(f"正在回到本回合左臂和夹爪起点，预计 {duration:.1f} 秒。", flush=True)
        motion = self.return_motion
        failure = None
        if now > motion.started_s + motion.duration_s + max(
            5.0, self.config.command_settle_timeout_s
        ):
            failure = "回位未在预期时间内到达起点"
        elif motion.outcome is not None and motion.outcome["status"] != "applied":
            failure = "回位终点命令未被 Host 应用"
        if failure is not None:
            self.control_suspended = True
            self.return_pending = False
            self.return_motion = None
            print(f"{failure}。已停止回位，请确认机器人状态并重连控制。", flush=True)
            return None
        if motion.fence is not None:
            if motion.outcome is None:
                return None
            at_target = all(
                abs(snapshot.payload[name] - value) <= limits(name, self.metadata)[1]
                for name, value in motion.target.items()
            )
            if feedback_after_command(snapshot, motion.outcome) and at_target:
                if motion.stable_since_s is None:
                    motion.stable_since_s = now
                if now - motion.stable_since_s >= 0.5:
                    self.hold.update(motion.target)
                    self.return_pending = False
                    self.return_motion = None
                    prompt = (
                        "摆好物体后按回车开始下一回合。"
                        if self.reward_submitted
                        else "摆好物体后输入 0 或 1，按回车保存评分并开始下一回合。"
                    )
                    print(f"左臂和夹爪已回到起点。{prompt}", flush=True)
                    return dict(self.hold)
            else:
                motion.stable_since_s = None
        return {**self.hold, **motion.targets(now)}

    def event(self, kind, value, snapshot, now, *, defer_feedback=False):
        if kind == "key" and value == "enter":
            if self.awaiting_reset_confirmation:
                self.collect_start_after = now if defer_feedback else snapshot.request_started_s
            return
        if kind == "reward":
            if self.state.mode != "awaiting_reward":
                return
            if type(value) is not int or value not in (0, 1):
                raise ValueError("Reward must be the integer 0 or 1")
            if self.pending_reward is None:
                self.pending_reward = value
                self.record.confirm_reward(value)
                if not self.return_pending and not self.control_suspended:
                    self.collect_start_after = now if defer_feedback else snapshot.request_started_s
            self._finish_reward()
            if not self.reward_submitted:
                print("奖励已确认，等待终点反馈和未决 Host 回执，记录保留在 pending。", flush=True)
            return
        if value == " " and self.return_pending:
            self.return_paused = not self.return_paused
            self.return_motion = None
            self.actions.clear()
            self.state.generation += 1
            self.urgent_hold = True
            if not defer_feedback:
                self.hold = self._hold(snapshot)
            self.needs_hold_latch = defer_feedback
            self.feedback_after_s = now if defer_feedback else snapshot.request_started_s
            print("回位已暂停。" if self.return_paused else "继续回位。", flush=True)
            return
        before = self.state.mode
        if value == "c" and self.gate is not None:
            return
        previous_phase = self.state.critical
        generation = self.state.generation
        self.state.key(value)
        if self.release and (self.state.mode == "paused" or before == "paused"):
            self.release.votes = 0
        if self.state.critical != previous_phase:
            self._phase_changed(now, "manual")
        if self.state.generation == generation:
            return
        if self.state.mode == "vla" or self.state.mode == "paused":
            self._reset_gate(now)
        self.actions.clear()
        self.urgent_hold = value in (" ", "right", "h")
        if not defer_feedback:
            self.hold = self._hold(snapshot)
        self.needs_hold_latch = defer_feedback
        self.feedback_after_s = now if defer_feedback else snapshot.request_started_s
        if before == "human" and (self.state.mode != "human" or self.state.quit):
            self._leave_human(now)
        if self.state.mode == "human" and before != "human":
            self.needs_human_latch = True
            if not defer_feedback:
                self._latch_human(snapshot, now)
        if self.state.mode == "awaiting_reward" and before != "awaiting_reward":
            self.ended_summary = dict(
                takeovers=self.takeovers,
                human_seconds=self.human_seconds,
                duration_s=now - self.started,
                critical_seconds=self.critical_seconds
                + (0.0 if self.critical_since is None else now - self.critical_since),
                critical_entries=self.critical_entries,
                training_valid=True,
            )
            self.terminal_pending = True
            self.end_started_s = now
            self.end_fence = None
            self.return_pending = self.config.return_to_start
            self.return_paused = False
            prompt = "请在物体复位后输入 0 或 1，按回车保存评分并开始下一回合。"
            if self.return_pending:
                prompt = "请等左臂回位并摆好物体，再输入 0 或 1，按回车保存评分并开始下一回合。"
            print(f"\n回合已结束，正在封存终点。{prompt}", flush=True)

    def _acknowledge(self, snapshot, now):
        safety = snapshot.payload["_safety"]
        command = safety.get("command", {})
        if command.get("client_id") == self.client.client_id:
            identity = CommandIdentity(**command)
            if identity != self.last_ack:
                self.last_ack = identity
                self.last_ack_time = now
            if self.legacy_inflight == identity:
                self.legacy_inflight = None
        if safety.get("command_result_schema") == 1:
            for result in safety.get("command_results", []):
                if result["result_id"] <= self.last_result_id:
                    continue
                self.last_result_id = result["result_id"]
                first = CommandIdentity(**result["command"])
                if first.client_id != self.client.client_id:
                    continue
                if result["status"] not in ("applied", "superseded", "rejected", "unknown"):
                    raise ValueError("Invalid Host command disposition")
                if result["status"] == "superseded":
                    pending = set(self.pending_commands)
                    if self.return_motion is not None and self.return_motion.fence is not None:
                        pending.add(self.return_motion.fence)
                    matches = [
                        identity
                        for identity in pending
                        if identity.host_session_id == first.host_session_id
                        and identity.control_epoch == first.control_epoch
                        and first.sequence <= identity.sequence <= result["last_sequence"]
                    ]
                    for identity in matches:
                        self._resolve_command(identity, result, now)
                else:
                    self._resolve_command(first, result, now)
            return
        receipts = list(safety.get("command_receipts", []))
        if command.get("client_id") == self.client.client_id:
            receipts.append(dict(command=command, targets=safety.get("accepted_targets", {})))
        for receipt in receipts:
            identity = CommandIdentity(**receipt["command"])
            if identity.client_id != self.client.client_id:
                continue
            self._resolve_command(identity, dict(receipt, status="applied"), now)

    def _accept_plan(self, result, observation_ready, now):
        if not result or result["kind"] != "prediction":
            return None
        if (
            not observation_ready
            or result["generation"] != self.state.generation
            or self.state.mode != "vla"
            or (self.plan_id is not None and result["id"] <= self.plan_id)
        ):
            return "cancelled_plan"
        sampled = result["sampled_s"]
        age = now - sampled
        if not 0 <= age < self.config.prediction_max_age_s or sampled < self.feedback_after_s:
            return "expired_plan"
        first = math.floor(age * self.fps)
        actions = result["actions"]
        if first >= len(actions):
            return "expired_plan"
        if self.gate is not None:
            sequence = result.get("observation_sequence")
            if sampled <= self.gate_sampled_s or (
                sequence is not None
                and self.gate_sequence is not None
                and sequence <= self.gate_sequence
            ):
                return "duplicate_observation"
            probability = result.get("gate_probability")
            if type(probability) not in (float, int) or not 0 <= probability <= 1:
                return "invalid_gate"
            self.gate_sampled_s, self.gate_sequence = sampled, sequence
            self.gate_probability = float(probability)
            active = (
                False
                if self.release
                and (self.release.block_reentry or sampled < self.release_gate_after_s)
                else self.gate.update(self.gate_probability)
            )
            if active != self.state.critical:
                self.state.critical = active
                self.state.generation += 1
                self._phase_changed(now, "classifier")
        self.inference_s = result["inference_s"]
        self.actions.clear()
        self.actions.extend(
            (sampled + index / self.fps, action, reference)
            for index, (action, reference) in enumerate(
                zip(actions, result["reference"], strict=True)
            )
            if index >= first
        )
        self.plan_id = result["id"]
        self.plan_actor_version = result.get("actor_version", -1)
        return None

    def tick(self, snapshot, events, now):
        if self.record is not None:
            self.record.check()
        self.check_end_timeout(now)
        if (
            context_for(snapshot) != self.context
            or snapshot.payload["_robot_metadata"] != self.metadata
        ):
            if self.terminal_pending:
                self._invalidate_end(
                    "Host control context changed before terminal confirmation", now
                )
            if self.control_suspended:
                return
            raise RuntimeError("Host control session, calibration or lift reference changed")
        feedback_ready = ready_units(
            snapshot, self.config.robot_model, self.client.client_id
        ) is not None and self.feedback_fresh(snapshot, now)
        self._acknowledge(snapshot, now)
        for kind, value in events:
            self.event(kind, value, snapshot, now, defer_feedback=not feedback_ready)
        if self.state.quit:
            return
        if not feedback_ready:
            self.pause(snapshot, now, feedback_lost=True)
            return
        self.fixed_guard.check(snapshot)
        observation_ready = snapshot.request_started_s >= self.feedback_after_s
        if observation_ready:
            self._start_collection_if_ready(snapshot, now)
        if observation_ready and self.needs_hold_latch:
            self.hold = self._hold(snapshot)
            self.needs_hold_latch = False
        if observation_ready and self.needs_human_latch and self.state.mode == "human":
            self._latch_human(snapshot, now)
        if observation_ready and self.state.active and self.start_pose is None:
            self.start_pose = {name: float(snapshot.payload[name]) for name in self.relative.names}
            for name, value in self.start_pose.items():
                lower, upper = self.relative.bounds[name]
                if not math.isfinite(value) or not lower <= value <= upper:
                    raise ValueError(f"Episode start pose exceeds calibration range: {name}")
            self.record.put("summary", dict(start_pose=self.start_pose))
        self._seal_terminal(snapshot, now, observation_ready)
        self._finish_reward()
        if self.control_suspended or (self.terminal_pending and self.end_fence is not None):
            return
        if observation_ready:
            self._observe_gripper(snapshot, now)
        if (
            (
                self.state.mode in ("vla", "human")
                or (self.return_pending and not self.return_paused)
            )
            and self.last_ack_time is not None
            and now - self.last_ack_time > self.config.ack_timeout_s
        ):
            raise RuntimeError("Host stopped acknowledging commands")
        # Stock Host publishes only its latest application. Allow one outstanding
        # command so that an ordinary next target cannot erase the only receipt.
        # Operator cancellation can still send a priority HOLD; any overtaken
        # command without its own receipt will invalidate this episode.
        stock_host = not (
            snapshot.payload["_safety"].get("command_result_schema") == 1
            or "command_receipts" in snapshot.payload["_safety"]
        )
        if stock_host and self.legacy_inflight is not None:
            if not (self.urgent_hold and observation_ready):
                return
        priority_hold = stock_host and self.legacy_inflight is not None and self.urgent_hold
        if (
            self.state.mode == "vla"
            and self.last_control_s is not None
            and now - self.last_control_s > 2.5 / self.fps
        ):
            self.actions.clear()
            self.state.generation += 1
            self._reset_gate(now)
        if (
            self.return_motion is not None
            and self.last_control_s is not None
            and now - self.last_control_s > 2.5 / self.fps
            and self.return_motion.fence is None
        ):
            self.pause(snapshot, now)
        plan_issue = self._accept_plan(self.inference.poll(), observation_ready, now)
        if plan_issue == "invalid_gate":
            self.pause(snapshot, now)
            print("阶段分类器缺少有效概率，已暂停。", flush=True)
        if observation_ready and self.state.mode == "vla":
            if len(self.actions) <= math.ceil(self.inference_s * self.fps) + 1:
                self.request_id += 1
                self.inference.submit(
                    dict(
                        id=self.request_id,
                        generation=self.state.generation,
                        critical=self.state.critical,
                        snapshot=snapshot,
                    )
                )
        source, reference = "HOLD", None
        targets = self._hold(snapshot) if self.needs_hold_latch else dict(self.hold)
        while self.actions and self.actions[0][0] < now - 1 / self.fps:
            self.actions.popleft()
        if self.state.mode == "vla" and not self.actions and not self.needs_hold_latch:
            self.hold = self._hold(snapshot)
            targets = dict(self.hold)
        if self.return_pending and not self.terminal_pending:
            targets = self._return_targets(snapshot, now, observation_ready)
            if targets is None:
                return
        elif priority_hold:
            targets = dict(self.hold)
        elif self.state.mode == "vla" and self.actions:
            _, action, ref = self.actions.popleft()
            # Enforce the operator's stage selection at dispatch as well as in the GPU worker.
            use_actor = (
                self.state.critical
                and self.ready.get("actor_identity") is not None
                and self.plan_actor_version >= 0
            )
            selected = action if use_actor else ref
            targets = dict(zip(self.names, map(float, selected), strict=True))
            reference = list(map(float, ref))
            source = "RLT" if use_actor else "VLA"
        elif self.state.mode == "human" and not self.needs_human_latch:
            sample = self.leader.read(self.config.leader_max_age_s)
            if sample is None:
                self.pause(snapshot, now)
                targets = dict(self.hold)
                print("主臂数据已过期，保持当前位置，按 h 重新接管。", flush=True)
            else:
                targets = self._hold(snapshot)
                targets.update(self.relative.targets(sample[1]))
                source = "HUMAN"
        targets.update(self.fixed)
        if any(not math.isfinite(value) for value in targets.values()):
            raise ValueError("Non-finite control target")
        self.ranges.check_policy_action(targets)
        identity = self.client.send_command(targets, based_on=snapshot)
        if identity is None:
            self.pause(snapshot, now)
            return
        self.identity = identity
        if stock_host:
            self.legacy_inflight = identity
        if observation_ready:
            self.urgent_hold = False
        self.last_control_s = now
        if self.last_ack_time is None:
            self.last_ack_time = now
        final_hold = self.terminal_pending and not self.needs_hold_latch
        if self.state.active or final_hold:
            self.episode_commands.add(identity)
            self.pending_commands.add(identity)
        if final_hold:
            self.end_fence = identity
        motion = self.return_motion
        if (
            motion is not None
            and motion.fence is None
            and now >= motion.started_s + motion.duration_s
        ):
            motion.fence = identity
        if self.state.active:
            self.record.step(
                self.step_index,
                snapshot,
                dict(
                    time=now,
                    command=asdict(identity),
                    command_status="queued",
                    source=source,
                    critical=self.state.critical,
                    critical_label=self.state.critical if self.phase_source == "manual" else None,
                    critical_source=self.phase_source,
                    gate_probability=self.gate_probability,
                    requested=[targets[n] for n in self.names],
                    reference=reference,
                    actor_version=self.plan_actor_version if source == "RLT" else -1,
                    plan_id=self.plan_id if source in ("RLT", "VLA") else None,
                    hold_reason="invalid_gate"
                    if source == "HOLD" and plan_issue == "invalid_gate"
                    else None,
                    generation=self.state.generation,
                ),
            )
            self.step_index += 1
        if source != "HOLD" and not self.needs_hold_latch:
            # A gap after any moving plan holds measured feedback at the next tick.
            self.hold = self._hold(snapshot)

    def statistics(self):
        autonomous = [r for r in self.finished if r["takeovers"] == 0]
        assisted = [r for r in self.finished if r["takeovers"] > 0]

        def rate(rows):
            return sum(r["reward"] for r in rows) / len(rows) if rows else None

        return dict(
            episodes=self.finished,
            success_rate=rate(self.finished),
            autonomous_success_rate=rate(autonomous),
            assisted_success_rate=rate(assisted),
        )


def run(config, *, resume=False):
    if config.learned_action_names is not None:
        from alohamini.fixed import dimensions

        if not set(config.learned_action_names) <= set(
            dimensions(["arm_left"], config.robot_model)
        ):
            raise ValueError("This passive-leader collector controls left-arm coordinates only")
    root = Path(config.output_dir)
    with ExitStack() as stack:
        stack.enter_context(RunLock(root / "control.lock"))
        history = recording_history(root, resume=resume)
        stack.enter_context(collection_publisher(root))
        pending = sum(path.is_dir() for path in (root / "pending").glob("*"))
        print(
            f"记录目录 {root.resolve()}，已评分回合 {len(history)}，待处理回合 {pending}。",
            flush=True,
        )
        console = stack.enter_context(ConsoleInput())
        client = stack.enter_context(
            HostClient(
                config.host,
                expected_model=config.robot_model,
                timeout_s=0.3,
                request_window=3,
                prefetch_before_decode=True,
            )
        )
        initial = client.connect_control()
        units = ready_units(initial, config.robot_model, client.client_id)
        if units is None:
            raise RuntimeError("Host is not ready for control")
        inference = InferenceWorker(config, initial)
        stack.callback(inference.close)
        print("正在加载冻结的 π0.5，按 q 可退出。", flush=True)
        ready = None
        while ready is None:
            if any(value == "q" for _, value in console.poll()):
                return
            ready = inference.poll()
            time.sleep(0.02)
        if ready["kind"] != "ready":
            raise RuntimeError("Inference worker did not initialize")
        if ready.get("actor_identity"):
            identity = ready["actor_identity"]
            print(
                f"已加载 RLT Actor，更新次数 {identity['version']}，SHA256 {identity['sha256']}。",
                flush=True,
            )
        from alohamini.policies.rlt.policy import require_contract

        for summary in history:
            try:
                saved_contract = summary.get("base_contract")
                if not isinstance(saved_contract, dict):
                    raise ValueError("Missing episode base contract")
                require_contract(saved_contract, ready["base_contract"])
            except ValueError as exc:
                raise ValueError(
                    "Saved episode belongs to another base model or action contract"
                ) from exc
        contract_path = root / "run-contract.json"
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            canonical_base = json.loads(json.dumps(ready["base_contract"]))
            if existing["base"] != canonical_base:
                raise ValueError(
                    "output_dir belongs to another base model, task or action contract"
                )
        else:
            atomic_json(
                contract_path,
                {"base": ready["base_contract"], "reference_contract": "vla-start-chunk-v1"},
            )
        atomic_json(root / "run-config.json", config.to_dict())
        leader = LeaderReader(config, units)
        stack.callback(leader.close)
        while leader.read(config.leader_max_age_s) is None:
            if leader.error is not None:
                raise RuntimeError(f"左主臂读取失败。{leader.error}") from leader.error
            if any(value == "q" for _, value in console.poll()):
                return
            time.sleep(0.02)
        prepared_identity = None
        session = None

        def stop_session():
            identity = (
                session.identity if session is not None and session.identity else prepared_identity
            )
            if identity is not None:
                stop_owned_robot(client, config.robot_model, identity)

        stack.callback(stop_session)
        if ready["fixed"]:
            try:
                prepared_identity = restore(
                    client,
                    config.robot_model,
                    ready["fixed"],
                    cancelled=lambda: any(value == "q" for _, value in console.poll()),
                    expected_metadata=initial.payload["_robot_metadata"],
                    expected_snapshot=initial,
                )
            except InterruptedError:
                return
        initial_context = context_for(initial)
        initial_metadata = initial.payload["_robot_metadata"]
        while True:
            try:
                initial = client.read(include_images=True)
                if (
                    context_for(initial) != initial_context
                    or initial.payload["_robot_metadata"] != initial_metadata
                ):
                    raise RuntimeError("Host calibration or control context changed during setup")
                missing = set(ready["cameras"]) - initial.images.keys()
                if missing:
                    raise ValueError(f"Host lacks required policy cameras: {sorted(missing)}")
                break
            except ResponseTimeoutError:
                if any(value == "q" for _, value in console.poll()):
                    return
                continue
        session = ControlSession(client, inference, leader, config, ready, initial)
        summary_fields = (
            "episode_id",
            "reward",
            "takeovers",
            "human_seconds",
            "duration_s",
            "critical_seconds",
            "critical_entries",
            "training_valid",
        )
        session.finished = [
            {key: row[key] for key in summary_fields if key in row}
            for row in history
            if not row.get("evaluation")
        ]
        phase_prompt = (
            "分类器自动进入/退出对槽阶段" if config.gate_checkpoint else "c 对槽阶段开/关"
        )
        if session.release:
            phase_prompt += f" | 夹持后实测张开至 {session.release.opened:g}% 自动退出"
        print(f"就绪后自动开始首回合 | {phase_prompt} | {HELP}", flush=True)
        last_snapshot = initial
        try:
            while not session.state.quit:
                started = time.monotonic()
                events = console.poll(
                    awaiting_reward=session.state.mode == "awaiting_reward",
                    returning=session.return_pending,
                    confirm_reset=session.awaiting_reset_confirmation,
                )
                for kind, value in events:
                    session.event(kind, value, last_snapshot, started, defer_feedback=True)
                session.check_end_timeout(started)
                if session.state.quit:
                    break
                try:
                    snapshot = client.read(include_images=True, timeout_s=0.02)
                except ResponseTimeoutError:
                    # Operator cancellation already ran before this bounded receive.
                    if not session.feedback_fresh(last_snapshot, time.monotonic()):
                        session.pause(last_snapshot, started, feedback_lost=True)
                    time.sleep(max(0, 1 / session.fps - (time.monotonic() - started)))
                    continue
                last_snapshot = snapshot
                session.tick(snapshot, (), time.monotonic())
                time.sleep(max(0, 1 / session.fps - (time.monotonic() - started)))
        finally:
            # Stop motion before waiting for the disk writer. Also covers faults
            # during setup through the stack callback if no session was built.
            try:
                stop_session()
            finally:
                session.identity = prepared_identity = None
                if session.record is not None:
                    session.record.close()
