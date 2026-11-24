"""Responsive robot control with user-latched interventions and manual scoring."""

import json
import math
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from alohamini.apps.collection.console import ConsoleInput
from alohamini.apps.collection.control import (
    ControlState,
    ReturnToStart,
)
from alohamini.apps.collection.v3_worker import collection_publisher
from alohamini.apps.collection.workers import LeaderReader
from alohamini.apps.replay import TargetRanges
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.client import HostClient, control_feedback_valid
from alohamini.datasets.experience.storage import (
    RUN_METADATA,
    EpisodeRecorder,
    RunLock,
    atomic_json,
    recording_history,
)
from alohamini.errors import ResponseTimeoutError
from alohamini.fixed import FixedGuard, limits, restore
from alohamini.hardware.leader import leader_sides
from alohamini.remote.protocol import context_for
from alohamini.runtime.command_feedback import feedback_after_command
from alohamini.runtime.handover import RelativeLeader, measured_hold_targets
from alohamini.schema import CommandIdentity

HANDOVER_HELP = "h 人工接管/交还 | "
CONTROL_HELP = "空格 暂停/继续 | 右箭头 任务结束"
SCORE_HELP = "0/1+回车 评分 | q 退出"


class ControlSession:
    receipt_field_key = "receipt_fields"
    run_metadata = RUN_METADATA

    @staticmethod
    def check_setup(config, ready, initial):
        pass

    @staticmethod
    def resume_config(config, saved):
        return config

    hotkeys = ("h", " ", "t")
    state_type = ControlState
    recorder_type = EpisodeRecorder
    summary_fields = (
        "episode_id",
        "reward",
        "takeovers",
        "human_seconds",
        "duration_s",
        "training_valid",
    )

    @property
    def help(self):
        leader = HANDOVER_HELP if self.leader is not None else ""
        return f"{leader}{CONTROL_HELP} | t 截断回合 | {SCORE_HELP}"

    def __init__(self, client, inference, leader, config, ready, initial):
        self.client, self.inference, self.leader = client, inference, leader
        self.config, self.ready = config, ready
        self.state = self.state_type()
        # A queued start consumes only feedback requested after the operator's confirmation.
        self.collect_start_after = -math.inf
        self.names, self.fps = ready["names"], ready["fps"]
        self.ranges = TargetRanges(self.names, initial)
        self.receipt_fields = set(ready["base_contract"][self.receipt_field_key])
        self.relative = RelativeLeader(
            self.handover_fields(ready),
            self.ranges.arm_bounds,
            max_joint_velocity_deg_s=config.max_joint_velocity_deg_s,
        )
        self.context = context_for(initial)
        self.metadata = deepcopy(initial.payload["_robot_metadata"])
        self.fixed = (ready["fixed"] or {}).get("targets", {})
        self.fixed_guard = FixedGuard(ready["fixed"], ready["metadata"])
        self.actions = deque()
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
        self.takeovers = 0
        self.human_seconds = 0.0
        self.finished = []
        self.started = None
        self.ended_summary = None
        self.plan_id = None
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

    @classmethod
    def handover_fields(cls, ready):
        return ready.get("handover_fields", ready["base_contract"][cls.receipt_field_key])

    def _hold(self, snapshot):
        targets = measured_hold_targets(snapshot, self.names)
        targets.update(self.fixed)
        return targets

    def _start_episode(self, now):
        root = Path(self.config.output_dir)
        self.record = self.recorder_type(
            root,
            dict(
                base_contract=self.ready["base_contract"],
                **self.episode_metadata(),
                intervention_fields=list(self.relative.names) if self.leader is not None else [],
                evaluation=False,
                started_at=time.time(),
                fps=self.fps,
                task=self.config.task,
                requires_acknowledgements=True,
                requires_command_dispositions=True,
                return_to_start=self.config.return_to_start,
            ),
        )
        self.step_index = self.takeovers = 0
        self.plan_id = None
        self.human_seconds = 0.0
        self.human_since = None
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
        if self.leader is not None and self.leader.read(self.config.leader_max_age_s) is None:
            error = getattr(self.leader, "error", None)
            if error is not None:
                raise RuntimeError(f"无法开始采集，主臂读取失败。{error}") from error
            return
        self._start_episode(now)
        self.state.start()
        self.hold = self._hold(snapshot)

    def _leave_human(self, now):
        self.relative.release()
        self.needs_human_latch = False
        if self.human_since is not None:
            self.human_seconds += now - self.human_since
            self.human_since = None

    def pause(self, snapshot, now, *, feedback_lost=False):
        if self.state.active and self.state.mode != "paused":
            self.urgent_hold = True
        self.state.pause()
        self.invalidate_plan(now)
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

    def episode_metadata(self):
        return {"collection_format": "experience"}

    def episode_metrics(self, now):
        return {}

    def invalidate_plan(self, now):
        self.actions.clear()

    def observe(self, snapshot, now):
        pass

    def inference_options(self):
        return {}

    def plan_rows(self, result, sampled, now):
        return ((action, {}) for action in result["actions"])

    def handle_plan_issue(self, issue, snapshot, now):
        pass

    def policy_command(self, action, details):
        return action, "POLICY", {}

    def step_metadata(self, source, details, plan_issue):
        return details

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
            print("已收到评分。回位结束、场景就绪后，按回车开始下一回合。", flush=True)
        else:
            print("已收到评分，正在保存回合。", flush=True)

    def _resolve_command(self, identity, result, now):
        status = result["status"]
        targets = result.get("targets", {})
        if status == "applied" and not self.receipt_fields <= targets.keys():
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
        self.relative.engage(sample[1], snapshot, now)
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
            print(f"正在回到本回合受控关节起点，预计 {duration:.1f} 秒。", flush=True)
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
                        "确认场景就绪后按回车开始下一回合。"
                        if self.reward_submitted
                        else "确认场景就绪后输入 0 或 1，按回车保存评分并开始下一回合。"
                    )
                    print(f"受控关节已回到起点。{prompt}", flush=True)
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
        if value == "h" and self.leader is None:
            return
        generation = self.state.generation
        self.state.key(value)
        if self.state.generation == generation:
            return
        if self.state.mode in ("policy", "paused"):
            self.invalidate_plan(now)
        self.actions.clear()
        self.urgent_hold = value in (" ", "right", "t", "h")
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
                truncated=value == "t",
                **self.episode_metrics(now),
                training_valid=True,
            )
            self.terminal_pending = True
            self.end_started_s = now
            self.end_fence = None
            self.return_pending = self.config.return_to_start
            self.return_paused = False
            prompt = "请在场景就绪后输入 0 或 1，按回车保存评分并开始下一回合。"
            if self.return_pending:
                prompt = (
                    "请等受控关节回位并确认场景就绪，再输入 0 或 1，按回车保存评分并开始下一回合。"
                )
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
            or self.state.mode != "policy"
            or (self.plan_id is not None and result["id"] <= self.plan_id)
        ):
            return "cancelled_plan"
        sampled = result["sampled_s"]
        age = now - sampled
        if not 0 <= age < self.config.prediction_max_age_s or sampled < self.feedback_after_s:
            return "expired_plan"
        origin = now if result.get("single_action") else sampled
        first = math.floor((now - origin) * self.fps)
        actions = result["actions"]
        if first >= len(actions):
            return "expired_plan"
        rows = self.plan_rows(result, sampled, now)
        if isinstance(rows, str):
            return rows
        self.inference_s = result["inference_s"]
        self.actions.clear()
        self.actions.extend(
            (origin + index / self.fps, action, details)
            for index, (action, details) in enumerate(rows)
            if index >= first
        )
        self.plan_id = result["id"]
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
            self.observe(snapshot, now)
        if (
            (
                self.state.mode in ("policy", "human")
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
            self.state.mode == "policy"
            and self.last_control_s is not None
            and now - self.last_control_s > 2.5 / self.fps
        ):
            self.actions.clear()
            self.state.generation += 1
            self.invalidate_plan(now)
        if (
            self.return_motion is not None
            and self.last_control_s is not None
            and now - self.last_control_s > 2.5 / self.fps
            and self.return_motion.fence is None
        ):
            self.pause(snapshot, now)
        plan_issue = self._accept_plan(self.inference.poll(), observation_ready, now)
        self.handle_plan_issue(plan_issue, snapshot, now)
        if observation_ready and self.state.mode == "policy":
            if len(self.actions) <= math.ceil(self.inference_s * self.fps) + 1:
                self.request_id += 1
                self.inference.submit(
                    dict(
                        id=self.request_id,
                        generation=self.state.generation,
                        **self.inference_options(),
                        snapshot=snapshot,
                    )
                )
        source, details = "HOLD", {}
        targets = self._hold(snapshot) if self.needs_hold_latch else dict(self.hold)
        while self.actions and self.actions[0][0] < now - 1 / self.fps:
            self.actions.popleft()
        if self.state.mode == "policy" and not self.actions and not self.needs_hold_latch:
            self.hold = self._hold(snapshot)
            targets = dict(self.hold)
        if self.return_pending and not self.terminal_pending:
            targets = self._return_targets(snapshot, now, observation_ready)
            if targets is None:
                return
        elif priority_hold:
            targets = dict(self.hold)
        elif self.state.mode == "policy" and self.actions:
            _, action, plan_details = self.actions.popleft()
            selected, source, details = self.policy_command(action, plan_details)
            targets = dict(zip(self.names, map(float, selected), strict=True))
        elif self.state.mode == "human" and not self.needs_human_latch:
            sample = self.leader.read(self.config.leader_max_age_s)
            if sample is None:
                self.pause(snapshot, now)
                targets = dict(self.hold)
                print("主臂数据已过期，保持当前位置，按 h 重新接管。", flush=True)
            else:
                targets = self._hold(snapshot)
                targets.update(self.relative.action(sample[1], snapshot, now))
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
                    **self.step_metadata(source, details, plan_issue),
                    requested=[targets[n] for n in self.names],
                    plan_id=self.plan_id if source not in ("HUMAN", "HOLD") else None,
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


def run(
    config, *, worker, session_type=ControlSession, resume=False, publisher=collection_publisher
):
    root = Path(config.output_dir)
    with ExitStack() as stack:
        stack.enter_context(RunLock(root / "control.lock"))
        history = recording_history(root, resume=resume)
        pending = sum(path.is_dir() for path in (root / "pending").glob("*"))
        print(
            f"记录目录 {root.resolve()}，已评分回合 {len(history)}，待处理回合 {pending}。",
            flush=True,
        )
        console = stack.enter_context(ConsoleInput(hotkeys=session_type.hotkeys))
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
        contract_path = root / "run-contract.json"
        if resume and contract_path.is_file():
            saved = json.loads(contract_path.read_bytes())
            if {k: v for k, v in saved.items() if k != "base"} != session_type.run_metadata:
                raise ValueError("Directory belongs to another collection workflow")
            config = session_type.resume_config(config, saved)
        inference = worker(config, initial)
        stack.callback(inference.close)
        print("正在加载策略，按 q 可退出。", flush=True)
        ready = None
        while ready is None:
            if any(value == "q" for _, value in console.poll()):
                return
            ready = inference.poll()
            time.sleep(0.02)
        if ready["kind"] != "ready":
            raise RuntimeError("Inference worker did not initialize")
        canonical_base = json.loads(json.dumps(ready["base_contract"]))
        if any(row.get("base_contract") != canonical_base for row in history):
            raise ValueError("Saved episode belongs to another base model or action contract")
        arm_fields = session_type.handover_fields(ready)
        if (not arm_fields and config.intervention) or not set(arm_fields) <= units.keys():
            raise ValueError("Passive-leader takeover requires named arm position channels")
        session_type.check_setup(config, ready, initial)
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != {"base": canonical_base, **session_type.run_metadata}:
                raise ValueError(
                    "output_dir belongs to another base model, task or action contract"
                )
        else:
            atomic_json(
                contract_path,
                {"base": ready["base_contract"], **session_type.run_metadata},
            )
        atomic_json(root / "run-config.json", asdict(config))
        stack.enter_context(publisher(root))
        leader = None
        if config.intervention:
            leader = LeaderReader(config, units, leader_sides(arm_fields))
            stack.callback(leader.close)
            while leader.read(config.leader_max_age_s) is None:
                if leader.error is not None:
                    raise RuntimeError(f"主臂读取失败。{leader.error}") from leader.error
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
        session = session_type(client, inference, leader, config, ready, initial)
        session.finished = [
            {key: row[key] for key in session.summary_fields if key in row}
            for row in history
            if not row.get("evaluation")
        ]
        print(f"就绪后自动开始首回合 | {session.help}", flush=True)
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
