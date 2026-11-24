"""Human-gated corrections using the platform's Host and dataset interfaces."""

import argparse
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from alohamini.apps.replay import TargetRanges
from alohamini.apps.teleoperation import KeyboardInput
from alohamini.datasets.record import state_names
from alohamini.runtime.handover import (
    AlignedLeader,
    AlignmentConfig,
    RelativeLeader,
    arm_position_units,
    joint_motion_scale,
    measured_hold_targets,
)


@dataclass(frozen=True)
class DAggerConfig(AlignmentConfig):
    leader_id: str | None = None
    calibration_dir: str | None = None
    left_port: str = "/dev/am_arm_leader_left"
    right_port: str = "/dev/am_arm_leader_right"

    handover_mode: str = "aligned"

    def __post_init__(self):
        super().__post_init__()
        if self.handover_mode not in ("aligned", "relative"):
            raise ValueError("handover_mode must be aligned or relative")

    def leader_options(self):
        return {
            key: getattr(self, key)
            for key in ("leader_id", "calibration_dir", "left_port", "right_port")
        }


def add_dagger_arguments(parser):
    parser.add_argument(
        "--dagger.handover_mode",
        choices=("aligned", "relative"),
        default=argparse.SUPPRESS,
        help="接管方式：aligned 手动对齐（默认），relative 从当前位置相对控制",
    )
    for name, help_text in (
        ("alignment_tolerance_deg", "关节对齐误差，默认 3 度"),
        ("gripper_tolerance", "夹爪对齐误差，默认 3 个开度百分点"),
        ("settle_time_s", "关节稳定后使能保持，默认 0.3 秒"),
        ("hold_output_limit", "主臂保持输出上限比例，默认 0.2"),
        ("alignment_timeout_s", "对齐及等待接管超时，默认 60 秒"),
        ("max_joint_velocity_deg_s", "人工关节目标变化速度上限，默认 60 度/秒"),
    ):
        parser.add_argument(
            f"--dagger.{name}", type=float, default=argparse.SUPPRESS, help=help_text
        )
    for flag, name in (
        ("--teleop.id", "leader_id"),
        ("--calibration_dir", "calibration_dir"),
        ("--left_port", "left_port"),
        ("--right_port", "right_port"),
    ):
        parser.add_argument(flag, dest=f"dagger.{name}", default=argparse.SUPPRESS)


def pop_dagger_options(options):
    settings = {
        key.removeprefix("dagger."): options.pop(key)
        for key in list(options)
        if key.startswith("dagger.")
    }
    return DAggerConfig(**settings)


class DAggerKeyboard(KeyboardInput):
    """Edge-triggered controls; held keys never advance multiple phases."""

    def __init__(self):
        super().__init__()
        self._events = deque()

    def update(self, name, pressed):
        with self._lock:
            if name in ("esc", "q") and pressed:
                self._quit = True
            if name not in ("space", "tab"):
                return
            if pressed and name not in self._pressed:
                # Keep at most one operator request pending. A sequence typed
                # while inference blocks must not skip the alignment gate.
                if not self._events:
                    self._events.append(name)
            if pressed:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)

    def events(self):
        if self.read() is None:
            return ["quit"]
        with self._lock:
            result = list(self._events)
            self._events.clear()
            return result


class ManualHandover:
    """Pause policy execution and record each human correction as a separate episode."""

    def __init__(self, leader, keyboard, config=None):
        self.keyboard = keyboard
        self.config = config or DAggerConfig()
        self.leader = leader
        self.aligned = (
            AlignedLeader(leader, self.config) if self.config.handover_mode == "aligned" else None
        )
        self.relative = None
        self.phase = "policy"
        self.stopped = False
        self.recording = False
        self.save_pending = False
        self._hold = None
        self._last_notice = None
        self._save_future = self._save_executor = None

    def _notice(self, message):
        if message != self._last_notice:
            logging.warning("[DAgger] %s", message)
            self._last_notice = message

    @staticmethod
    def hold_targets(snapshot):
        return measured_hold_targets(
            snapshot, [*arm_position_units(snapshot), "lift_axis.height_mm"]
        )

    _scale = staticmethod(joint_motion_scale)

    def _release(self):
        if self.aligned is not None:
            self.aligned.release()
        self.relative = None

    def frame_metadata(self):
        return {"intervention": True, "control_source": "human", "dagger": asdict(self.config)}

    def _begin_correction(self, dataset):
        self.phase = "human"
        if dataset is not None:
            dataset.begin_episode()
            dataset.event({"type": "sequence_boundary", "reason": "human_correction_start"})
            self.recording = True
        self._notice("人工纠错中；Tab 结束并保存。")

    def _pause(self, snapshot, policy, dataset):
        # Clearing the inference generation invalidates outstanding cloud/RTC
        # results as well as local ACT/chunk history.
        policy.reset()
        self.phase = "paused"
        self._hold = self.hold_targets(snapshot)
        self._release()
        if self.recording:
            dataset.event({"type": "sequence_boundary", "reason": "human_correction_end"})
            self.recording = False
            self.save_pending = True
        instruction = "对齐主臂" if self.aligned is not None else "开始相对接管"
        self._notice(f"已暂停；Tab {instruction}，Space 恢复策略，Esc 退出。")

    def handle_events(self, snapshot, policy, dataset, *, now, refresh=None):
        self.finish_save(dataset)
        events = self.keyboard.events()
        if events and refresh is not None:
            fresh = refresh()
            if fresh is None:
                self.feedback_lost(snapshot, policy, dataset)
                return
            snapshot = fresh
        for event in events:
            if event == "quit":
                self.stopped = True
                return
            if event == "space":
                if self.phase == "policy":
                    self._pause(snapshot, policy, dataset)
                elif self.phase in ("aligning", "ready"):
                    self._pause(snapshot, policy, dataset)
                elif self.phase == "paused" and not self.save_pending:
                    self._release()
                    policy.reset()
                    self.phase = "policy"
                    self._notice("策略恢复；Space 暂停。")
                elif self.phase == "paused":
                    self._notice("正在保存纠错；保存完成后按 Space 交还策略。")
            elif event == "tab":
                if self.phase == "paused" and not self.save_pending:
                    if self.aligned is None:
                        ranges = TargetRanges(state_names(snapshot.robot_model), snapshot)
                        bounds = ranges.arm_bounds
                        self.relative = RelativeLeader(
                            names=bounds.keys(),
                            bounds=bounds,
                            max_joint_velocity_deg_s=self.config.max_joint_velocity_deg_s,
                        )
                        self._begin_correction(dataset)
                        continue
                    self.phase = "aligning"
                    self.aligned.start(now)
                    self._notice("手动对齐主臂，关节到位后逐个保持；从臂保持不动。")
                elif self.phase == "ready":
                    # Check every joint again immediately before release.
                    if not self.aligned.confirm(snapshot, now):
                        self._pause(snapshot, policy, dataset)
                        self._notice("主臂位置已偏移；Tab 重新对齐。")
                        continue
                    self._begin_correction(dataset)
                elif self.phase == "human":
                    self._pause(snapshot, policy, dataset)
                elif self.phase in ("aligning", "ready"):
                    self._pause(snapshot, policy, dataset)

    def action(self, snapshot, *, now):
        if self.phase == "policy":
            return None
        if self.phase == "paused":
            if self._hold is None:
                self._hold = self.hold_targets(snapshot)
            return dict(self._hold)
        if self.phase in ("aligning", "ready"):
            joint, held, total = self.aligned.align(snapshot, now)
            if joint is not None:
                self._notice(f"已对齐并保持 {held}/{total}：{joint}")
            if held == total:
                self.phase = "ready"
                self._notice("全部到位；扶稳主臂后按 Tab 释放保持并开始纠错。")
            return dict(self._hold)
        if self.relative is not None:
            pose = self.leader.read(arm_position_units(snapshot))
            if self.relative.origin is None:
                self.relative.engage(pose, snapshot, now)
                self._hold = self.hold_targets(snapshot)
            return {**self._hold, **self.relative.action(pose, snapshot, now)}
        return self.aligned.action(snapshot, self._hold, now)

    def feedback_lost(self, snapshot, policy, dataset):
        """Never keep latched leaders across a watchdog or missing robot feedback."""
        self._pause(snapshot, policy, dataset)
        self._hold = None  # Relatch measured positions only after fresh feedback returns.

    def poll_disconnected(self):
        # A disconnected session may exit, but cannot queue a future takeover.
        if "quit" in self.keyboard.events():
            self.stopped = True

    def finish_save(self, dataset):
        if self.save_pending:
            if self._save_future is None:
                self._save_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="DAggerSave"
                )
                self._save_future = self._save_executor.submit(dataset.save_episode)
            elif self._save_future.done():
                self._save_future.result()
                self._save_executor.shutdown()
                self._save_future = self._save_executor = None
                self.save_pending = False
                self._notice("纠错已保存；Tab 再次接管，Space 交还策略。")

    def close(self):
        try:
            self._release()
        finally:
            if self._save_executor is not None:
                self._save_executor.shutdown(wait=True)
                self._save_future.result()
