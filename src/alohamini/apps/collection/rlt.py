"""Residual policy phases, reference actions and labels for scored collection."""

import math
from dataclasses import replace
from functools import partial

from alohamini.apps.collection import scored_session
from alohamini.apps.collection.control import ControlState as CollectionState
from alohamini.apps.collection.workers import InferenceWorker, serve
from alohamini.policies.rlt.gate import GateLatch
from alohamini.rl.rlt.episodes import RUN_METADATA, EpisodeRecorder


class ControlState(CollectionState):
    critical = False

    def start(self):
        super().start()
        self.critical = False

    def key(self, key):
        if key == "c" and self.active:
            self.critical = not self.critical
            self.generation += 1
        else:
            super().key(key)


class GripperRelease:
    """Detect a measured closed-to-open transition within one active phase."""

    @staticmethod
    def selected_field(config, learned, motors):
        if not config.gripper_exit:
            return None
        field = config.gripper_exit_field
        if field is None:
            grippers = [n for n in learned if n.endswith("_gripper.pos")]
            if len(grippers) != 1:
                raise ValueError(
                    "Select --gripper-exit-field unless exactly one gripper is learned"
                )
            field = grippers[0]
        motor = motors.get(field.removesuffix(".pos"), {})
        if motor.get("normalization") != "range_0_100":
            raise ValueError("Gripper exit needs a calibrated gripper position in percent")
        return field

    @classmethod
    def for_collection(cls, config, learned, motors):
        field = cls.selected_field(config, learned, motors)
        if field is None:
            return None
        return cls(
            config.gripper_exit_closed_pct,
            config.gripper_exit_open_pct,
            config.gripper_exit_confirmations,
            field,
        )

    def __init__(self, closed, opened, confirmations, field=None):
        self.field = field
        self.closed = closed
        self.opened = opened
        self.confirmations = confirmations
        self.last_sample = None
        self.last_time = None
        self.block_reentry = False
        self.new_phase()

    def new_phase(self):
        self.armed = False
        self.votes = 0

    def observe(self, opening, sample, now, max_gap, *, critical):
        if sample is None or not math.isfinite(opening) or not 0 <= opening <= 100:
            self.votes = 0
            return False
        if self.last_sample is not None and sample <= self.last_sample:
            return False
        if self.last_time is not None and now - self.last_time > max_gap:
            self.votes = 0
        self.last_sample, self.last_time = sample, now
        if opening <= self.closed:
            self.block_reentry = False
            self.armed = critical
        self.votes = self.votes + 1 if critical and self.armed and opening >= self.opened else 0
        if self.votes < self.confirmations:
            return False
        self.new_phase()
        self.block_reentry = True
        return True


class ControlSession(scored_session.ControlSession):
    hotkeys = ("h", " ", "c")
    state_type = ControlState
    receipt_field_key = "learned_names"
    recorder_type = EpisodeRecorder
    run_metadata = RUN_METADATA
    summary_fields = (
        *scored_session.ControlSession.summary_fields,
        "critical_entries",
        "critical_seconds",
    )

    @staticmethod
    def resume_config(config, saved):
        if config.learned_action_names is None:
            return replace(config, learned_action_names=saved["base"]["learned_names"])
        return config

    @staticmethod
    def check_setup(config, ready, initial):
        GripperRelease.selected_field(
            config,
            ready["base_contract"]["learned_names"],
            initial.payload["_robot_metadata"]["motors"],
        )

    @property
    def help(self):
        phase = "分类器自动进入/退出残差控制阶段" if self.gate else "c 残差控制阶段开/关"
        if self.release:
            phase += f" | 夹持后实测张开至 {self.release.opened:g}% 自动退出"
        return (
            f"{phase} | {scored_session.HANDOVER_HELP}{scored_session.CONTROL_HELP}"
            f" | {scored_session.SCORE_HELP}"
        )

    def __init__(self, client, inference, leader, config, ready, initial):
        super().__init__(client, inference, leader, config, ready, initial)
        self.gate = (
            GateLatch(
                config.gate_enter_threshold, config.gate_exit_threshold, config.gate_confirmations
            )
            if config.gate_checkpoint
            else None
        )
        self.gate_probability = None
        self.release = GripperRelease.for_collection(
            config, self.relative.names, self.metadata["motors"]
        )
        self.phase_source = "classifier" if self.gate else "manual"
        self.release_gate_after_s = -math.inf
        self.gripper_exits = []
        self.gate_sampled_s = -math.inf
        self.gate_sequence = None
        self.critical_since = None
        self.critical_seconds = 0.0
        self.critical_entries = 0
        self.plan_actor_version = -1

    def episode_metadata(self):
        return dict(
            collection_format="rlt",
            actor=self.ready.get("actor_identity"),
            gate=self.ready.get("gate_identity"),
            gripper_exit_rule=(
                dict(
                    field=self.release.field,
                    closed_pct=self.release.closed,
                    open_pct=self.release.opened,
                    confirmations=self.release.confirmations,
                )
                if self.release
                else None
            ),
            phase_labels="classifier-output-v1" if self.gate else "optional-operator-v2",
        )

    def _start_episode(self, now):
        super()._start_episode(now)
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
        self.critical_since = None
        self.critical_seconds = 0.0
        self.critical_entries = 0

    def pause(self, snapshot, now, *, feedback_lost=False):
        if self.release:
            self.release.votes = 0
        super().pause(snapshot, now, feedback_lost=feedback_lost)

    def event(self, kind, value, snapshot, now, *, defer_feedback=False):
        if value == "t" or (value == "c" and self.gate is not None):
            return
        phase, mode = self.state.critical, self.state.mode
        super().event(kind, value, snapshot, now, defer_feedback=defer_feedback)
        if self.release and (mode == "paused" or self.state.mode == "paused"):
            self.release.votes = 0
        if phase != self.state.critical and value == "c":
            self._phase_changed(now, "manual")

    def episode_metrics(self, now):
        return dict(
            critical_seconds=self.critical_seconds
            + (0.0 if self.critical_since is None else now - self.critical_since),
            critical_entries=self.critical_entries,
        )

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
        policy = "RLT" if self.state.critical and self.ready.get("actor_identity") else "基座策略"
        print(f"残差控制阶段已{phase}，自动控制时使用 {policy}。", flush=True)

    def invalidate_plan(self, now):
        if self.gate is None:
            return
        self.gate.reset()
        self.gate_probability = None
        if self.state.critical and self.state.active:
            self.state.critical = False
            self._phase_changed(now, "classifier")

    def observe(self, snapshot, now):
        if self.release is None or self.state.mode not in ("policy", "human"):
            return
        opening = snapshot.payload.get(self.release.field)
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
                field=self.release.field,
                opening_pct=opening,
                generation=self.state.generation,
            )
        )
        self.record.put("summary", dict(gripper_exits=list(self.gripper_exits)))
        print(f"{self.release.field} 实测开度 {opening:.1f}%，已退出 RLT。", flush=True)

    def inference_options(self):
        return {"critical": self.state.critical}

    def plan_rows(self, result, sampled, now):
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
        self.plan_actor_version = result.get("actor_version", -1)
        return zip(result["actions"], result["reference"], strict=True)

    def handle_plan_issue(self, issue, snapshot, now):
        if issue == "invalid_gate":
            self.pause(snapshot, now)
            print("阶段分类器缺少有效概率，已暂停。", flush=True)

    def policy_command(self, action, ref):
        use_actor = (
            self.state.critical
            and self.ready.get("actor_identity") is not None
            and self.plan_actor_version >= 0
        )
        return (
            action if use_actor else ref,
            "RLT" if use_actor else "VLA",
            {"reference": list(map(float, ref))},
        )

    def step_metadata(self, source, details, plan_issue):
        return dict(
            critical=self.state.critical,
            critical_label=self.state.critical if self.phase_source == "manual" else None,
            critical_source=self.phase_source,
            gate_probability=self.gate_probability,
            reference=details.get("reference"),
            actor_version=self.plan_actor_version if source == "RLT" else -1,
            hold_reason=plan_issue if source == "HOLD" and plan_issue == "invalid_gate" else None,
        )


def load(config, initial):
    from alohamini.policies.rlt.policy import RLTPolicy

    policy = RLTPolicy(config)
    policy.bind(initial)
    native = policy.native
    ready = dict(
        names=native.names,
        fps=native.fps,
        cameras=native.manifest["cameras"],
        fixed=native.fixed_dimensions,
        metadata=native.robot_metadata,
        base_contract=policy.contract,
        actor_identity=policy.actor_identity,
        gate_identity=policy.gate_identity,
    )

    def select(job):
        return policy.predict(
            job["snapshot"], observation_id=job["id"], apply_actor=job.get("critical", False)
        )

    return ready, native.reset, select, None


worker = partial(InferenceWorker, target=partial(serve, load=load))


def run(config, *, resume=False):
    scored_session.run(
        config,
        worker=worker,
        session_type=ControlSession,
        resume=resume,
    )
