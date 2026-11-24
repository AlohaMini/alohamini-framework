"""Configuration for passive-leader episode collection and RL token training."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Older run-config.json files also recorded learner and powered-leader defaults.
# Explicitly active options that change how demonstrations are made fail closed.
_LEGACY_FIELDS = frozenset(
    (
        "actor_warmup_updates",
        "auto_critical",
        "auto_critical_min_off_chunks",
        "auto_critical_min_on_chunks",
        "auto_critical_once",
        "auto_critical_smooth_steps",
        "auto_critical_start_delay_s",
        "auto_critical_threshold_off",
        "auto_critical_threshold_on",
        "batch_size",
        "bc_weight",
        "buffer_capacity",
        "critic_per_actor",
        "critical_annotations",
        "critical_batch_size",
        "critical_classifier",
        "critical_epochs",
        "critical_hidden_dim",
        "critical_layers",
        "critical_lr",
        "critical_sample_stride",
        "fixed_std",
        "gamma",
        "hidden_dim",
        "hidden_layers",
        "leader_follow_speed_deg_s",
        "leader_guide_output_limits",
        "leader_guide_smoothing_s",
        "leader_output_limits",
        "leader_position_pid",
        "leader_preserve_lower_ram",
        "leader_yield_deg",
        "learner_checkpoint",
        "reference_dropout",
        "tau",
        "utd",
        "warmup_episodes",
    )
)


@dataclass
class RLTConfig:
    checkpoint: str
    output_dir: str
    task: str
    host: str = "127.0.0.1"
    robot_model: str = "alohamini2pro"
    device: str = "cuda"
    token_checkpoint: str = ""
    actor_checkpoint: str = ""
    gate_checkpoint: str = ""
    gate_enter_threshold: float = 0.75
    gate_exit_threshold: float = 0.35
    gate_confirmations: int = 2
    gripper_exit: bool | None = None
    gripper_exit_closed_pct: float = 18.0
    gripper_exit_open_pct: float = 24.0
    gripper_exit_confirmations: int = 3
    gate_steps: int = 1000
    gate_batch_size: int = 128
    gate_sample_stride: int = 6
    gate_eval_stride: int | None = None
    dataset: str = ""
    leader_id: str | None = None
    calibration_dir: str | None = None
    left_port: str = "/dev/am_arm_leader_left"
    calibration_mode: str = "strict"
    fixed_dimensions: list[str] | None = None
    chunk_size: int = 10
    replay_stride: int = 2
    token_dim: int = 2048
    token_layers: int = 4
    token_heads: int = 8
    lr: float = 0.0003
    token_steps: int = 5000
    token_batch_size: int = 4
    token_validation_frames_per_episode: int = 32
    validation_fraction: float = 0.1
    ac_steps: int = 5000
    ac_batch_size: int = 128
    ac_gamma: float = 0.999
    ac_tau: float = 0.005
    ac_bc_weight: float = 1.0
    ac_residual_limit: float = 0.25
    seed: int = 42
    prediction_max_age_s: float = 1.0
    leader_max_age_s: float = 0.15
    ack_timeout_s: float = 0.5
    command_settle_timeout_s: float = 2.0
    fps: float | None = None
    return_to_start: bool = True
    return_duration_s: float = 3.0

    def __post_init__(self):
        if self.gate_eval_stride is not None and (
            type(self.gate_eval_stride) is not int or self.gate_eval_stride < 1
        ):
            raise ValueError("gate_eval_stride must be positive")
        for name in (
            "checkpoint",
            "output_dir",
            "task",
            "host",
            "robot_model",
            "device",
            "left_port",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "token_checkpoint",
            "actor_checkpoint",
            "gate_checkpoint",
            "dataset",
            "leader_id",
            "calibration_dir",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        for name in (
            "chunk_size",
            "replay_stride",
            "token_dim",
            "token_layers",
            "token_heads",
            "token_steps",
            "token_batch_size",
            "token_validation_frames_per_episode",
            "ac_steps",
            "ac_batch_size",
            "gate_steps",
            "gate_batch_size",
            "gate_sample_stride",
            "gate_confirmations",
            "gripper_exit_confirmations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.token_dim % self.token_heads:
            raise ValueError("token_dim must be divisible by token_heads")
        for name in (
            "lr",
            "prediction_max_age_s",
            "leader_max_age_s",
            "ack_timeout_s",
            "command_settle_timeout_s",
            "return_duration_s",
            "validation_fraction",
            "ac_gamma",
            "ac_tau",
            "ac_bc_weight",
            "ac_residual_limit",
        ):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.ac_gamma > 1 or self.ac_tau > 1:
            raise ValueError("ac_gamma and ac_tau must be in (0, 1]")
        thresholds = (self.gate_exit_threshold, self.gate_enter_threshold)
        if (
            any(type(v) not in (int, float) for v in thresholds)
            or not 0 <= thresholds[0] < thresholds[1] <= 1
        ):
            raise ValueError("gate thresholds must satisfy 0 <= exit < enter <= 1")
        if self.gate_checkpoint and not self.token_checkpoint:
            raise ValueError("gate_checkpoint requires its matching token_checkpoint")
        if self.gripper_exit is not None and type(self.gripper_exit) is not bool:
            raise ValueError("gripper_exit must be a boolean or null")
        closed, opened = self.gripper_exit_closed_pct, self.gripper_exit_open_pct
        if (
            any(type(v) not in (int, float) for v in (closed, opened))
            or not 0 <= closed < opened <= 100
        ):
            raise ValueError("Gripper exit thresholds must satisfy 0 <= closed < open <= 100")
        if self.actor_checkpoint and not self.token_checkpoint:
            raise ValueError("actor_checkpoint requires its matching token_checkpoint")
        if self.fps is not None and (type(self.fps) not in (float, int) or not 1 <= self.fps <= 30):
            raise ValueError("fps must be in [1, 30]")
        if type(self.return_to_start) is not bool:
            raise ValueError("return_to_start must be a boolean")
        if self.calibration_mode not in ("strict", "normalized", "legacy_normalized"):
            raise ValueError("Unsupported calibration_mode")
        if self.fixed_dimensions is not None and (
            not isinstance(self.fixed_dimensions, list)
            or not self.fixed_dimensions
            or any(not isinstance(name, str) or not name.strip() for name in self.fixed_dimensions)
        ):
            raise ValueError("fixed_dimensions must be a nonempty list of dimension names")

    @classmethod
    def read(cls, path=None, **overrides):
        """Merge optional JSON defaults and explicit command-line settings."""
        directory = Path.cwd()
        data = {}
        if path is not None:
            path = Path(path).expanduser().resolve()
            directory = path.parent
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("RLT configuration must be a JSON object")
        data.update(overrides)
        for name in (
            "auto_critical",
            "leader_output_limits",
            "leader_guide_output_limits",
            "leader_position_pid",
            "leader_endpoint",
        ):
            if data.get(name):
                raise ValueError(f"{name} is incompatible with passive local-leader collection")
        for name in _LEGACY_FIELDS | {"leader_endpoint"}:
            data.pop(name, None)
        unknown = data.keys() - {field.name for field in fields(cls)}
        missing = {"checkpoint", "output_dir", "task"} - data.keys()
        if unknown or missing:
            raise ValueError(
                f"Invalid RLT settings; unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        for name in (
            "checkpoint", "output_dir", "token_checkpoint", "actor_checkpoint", "gate_checkpoint",
            "dataset", "calibration_dir",
        ):
            if data.get(name):
                if not isinstance(data[name], str):
                    raise ValueError(f"{name} must be a path string")
                value = Path(data[name]).expanduser()
                data[name] = str(value.resolve() if name in overrides else directory / value)
        return cls(**data)

    @property
    def leader_output_limits(self):
        """Existing passive-leader preflight checks this value before opening its bus."""
        return None

    def to_dict(self):
        return asdict(self)
