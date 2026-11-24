"""Scored collection using the native checkpoint action interface."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from alohamini.apps.collection.workers import InferenceWorker, serve
from alohamini.datasets.experience.storage import digest_file
from alohamini.fixed import dimensions


@dataclass
class CollectionConfig:
    checkpoint: str
    task: str
    output_dir: str
    robot_model: str = "alohamini2pro"
    host: str = "127.0.0.1"
    device: str = "cuda"
    calibration_mode: str = "strict"
    goal_images: dict[str, str] | None = None
    intervention: str | None = None
    left_port: str = "/dev/am_arm_leader_left"
    right_port: str = "/dev/am_arm_leader_right"
    leader_id: str | None = None
    calibration_dir: str | None = None
    return_to_start: bool = False
    max_joint_velocity_deg_s: float = 60.0
    leader_max_age_s: float = 0.15
    prediction_max_age_s: float = 1.0
    ack_timeout_s: float = 0.5
    command_settle_timeout_s: float = 2.0
    return_duration_s: float = 3.0


def load(config, initial):
    from alohamini.learning.policy import NativePolicy
    from alohamini.learning.trajectory_execution import prepare_execution

    native = NativePolicy(
        config.checkpoint,
        device=config.device,
        task=config.task,
        robot_model=config.robot_model,
        calibration_mode=config.calibration_mode,
        goal_images=config.goal_images,
    )
    native.bind_robot(initial)
    policy = prepare_execution(native, control_fps=native.fps)
    fixed = (native.fixed_dimensions or {}).get("targets", {})
    arm = set(dimensions(["arm_left", "arm_right"], config.robot_model))
    arm_fields = [n for n in native.names if n in arm and n not in fixed]
    handover_fields = arm_fields
    if config.intervention in ("left", "right"):
        selected = set(dimensions([f"arm_{config.intervention}"], config.robot_model))
        handover_fields = [name for name in arm_fields if name in selected]
    contract = dict(
        names=native.names,
        receipt_fields=arm_fields,
        task=config.task,
        policy=native.manifest["kind"],
        policy_sha256=native.checkpoint_manifest_sha256,
        weights_sha256=digest_file(native.checkpoint_path / "model.safetensors"),
        robot_metadata=native.robot_metadata,
        calibration_mode=config.calibration_mode,
        fps=native.fps,
        fixed_dimensions=native.fixed_dimensions,
        goal_images={
            name: digest_file(Path(path).expanduser())
            for name, path in (config.goal_images or {}).items()
        },
    )
    ready = dict(
        names=native.names,
        fps=native.fps,
        fixed=native.fixed_dimensions,
        cameras=native.manifest["cameras"],
        metadata=native.robot_metadata,
        base_contract=contract,
        handover_fields=handover_fields,
    )

    def select(job):
        action = policy.select_action(job["snapshot"])
        return {
            "actions": [] if action is None else [[action[n] for n in native.names]],
            "single_action": True,
        }

    return ready, policy.reset, select, getattr(policy, "close", None)


worker = partial(InferenceWorker, target=partial(serve, load=load))
