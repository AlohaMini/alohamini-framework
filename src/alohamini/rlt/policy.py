"""Frozen native PI0.5 references with identical offline/live action transforms."""

import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from alohamini.learning.processor import scale_action
from alohamini.policies.pi05.configuration_pi05 import get_config
from alohamini.rlt.actor_critic import AC_FORMAT, ChunkActor
from alohamini.rlt.gate_classifier import GATE_FORMAT, StageClassifier
from alohamini.rlt.networks import RLTokenEncoder
from alohamini.rlt.storage import digest_file, load_hashed_weights


def load_native(config, *, offline=False):
    from alohamini.learning.policy import NativePolicy

    native = NativePolicy(
        config.checkpoint,
        device=config.device,
        task=config.task,
        n_action_steps=config.chunk_size,
        calibration_mode=config.calibration_mode,
        fixed_dimensions=config.fixed_dimensions,
        robot_model=config.robot_model,
        offline=offline,
    )
    if native.manifest["kind"] != "pi05" or native.field_transform is not None:
        raise ValueError("RLT requires a native PI0.5 checkpoint in its original 18-D coordinates")
    if config.chunk_size > native.config.network["action_horizon"]:
        raise ValueError("RL chunk exceeds the VLA horizon")
    if config.fps is not None and config.fps != native.fps:
        raise ValueError("RLT control FPS must match the checkpoint's action time scale")
    native.model.requires_grad_(False).eval()
    native.model.network.gradient_checkpointing_disable()
    return native


def base_contract(native, config):
    path = native.checkpoint_path
    names = [name for name in native.names if name.startswith("arm_left_")]
    if len(names) != 7:
        raise ValueError("RLT expects six left-arm joints and a gripper")
    if set(names) & set((native.fixed_dimensions or {}).get("targets", {})):
        raise ValueError("The learned left arm contains fixed output coordinates")
    return dict(
        version=1,
        base={
            name: digest_file(path / name)
            for name in ("policy.json", "model.safetensors", "tokenizer.model")
        },
        task=config.task,
        names=native.names,
        learned_names=names,
        fps=native.fps,
        chunk_size=config.chunk_size,
        fixed_state_references=native.fixed_state_references,
        fixed_dimensions=native.fixed_dimensions,
        token_spec=dict(
            input_dim=get_config(native.config.network["paligemma_variant"]).width,
            dim=config.token_dim,
            layers=config.token_layers,
            heads=config.token_heads,
        ),
    )


def require_contract(actual, expected):
    # JSON roundtrip normalizes integer dictionary keys from persisted metadata.
    actual = json.loads(json.dumps(actual))
    expected = json.loads(json.dumps(expected))
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("RLT artifact differs from the base model, task or action/token contract")


def encode_physical(native, actions, context):
    actions = torch.as_tensor(actions, dtype=torch.float32, device=context.device)
    actions = scale_action(
        actions,
        getattr(native.config, "inference_action_scale_dims", ()),
        1 / getattr(native.config, "inference_action_scale", 1.0),
    )
    return native.processor.action_prefix(actions, context=context)


@dataclass
class RLTReference:
    observation_id: int
    token: torch.Tensor | None
    proprio: torch.Tensor
    reference: torch.Tensor
    context: torch.Tensor
    physical: torch.Tensor


class RLTPolicy:
    def __init__(self, config, *, offline=False):
        self.config = config
        self.native = load_native(config, offline=offline)
        self.contract = base_contract(self.native, config)
        self.indices = [self.native.names.index(n) for n in self.contract["learned_names"]]
        self.encoder = self.gate = self.gate_identity = None
        self.actor = self.actor_identity = self.token_sha256 = None
        if config.token_checkpoint:
            checkpoint, self.token_sha256 = load_hashed_weights(config.token_checkpoint)
            require_contract(checkpoint["contract"], self.contract)
            self.encoder = RLTokenEncoder(**self.contract["token_spec"]).to(config.device)
            self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
            self.encoder.requires_grad_(False).eval()
        if config.actor_checkpoint:
            self.load_actor(config.actor_checkpoint)
        if config.gate_checkpoint:
            self.load_gate(config.gate_checkpoint)

    def load_actor(self, path):
        saved, sha256 = load_hashed_weights(path)
        if saved.get("format") != AC_FORMAT:
            raise ValueError("Unsupported RLT Actor checkpoint format")
        require_contract(saved["contract"], self.contract)
        if self.encoder is None or saved["token_sha256"] != self.token_sha256:
            raise ValueError("RLT Actor requires the exact token checkpoint used for AC training")
        limit, version = saved["residual_limit"], saved["actor_updates"]
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("Invalid Actor residual limit")
        if type(version) is not int or version < 1:
            raise ValueError("RLT Actor checkpoint has no completed updates")
        actor = ChunkActor(self.contract, limit)
        actor.load_state_dict(saved["actor"], strict=True)
        if not all(torch.isfinite(p).all() for p in actor.parameters()):
            raise ValueError("Non-finite RLT Actor weights")
        self.actor = actor.to(self.config.device).requires_grad_(False).eval()
        self.actor_identity = dict(sha256=sha256, token_sha256=self.token_sha256, version=version)

    def load_gate(self, path):
        checkpoint, sha = load_hashed_weights(path)
        if checkpoint.get("format") != GATE_FORMAT:
            raise ValueError("Unsupported RLT stage classifier format")
        require_contract(checkpoint["contract"], self.contract)
        if self.encoder is None or checkpoint["token_sha256"] != self.token_sha256:
            raise ValueError("Gate requires the exact token checkpoint used during training")
        model = StageClassifier(self.contract)
        model.load_state_dict(checkpoint["classifier"], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Non-finite gate weights")
        self.gate = model.to(self.config.device).requires_grad_(False).eval()
        self.gate_identity = {"sha256": sha, "token_sha256": self.token_sha256}

    def bind(self, snapshot):
        self.native.bind_robot(snapshot)

    @torch.no_grad()
    def features(self, batch):
        with self.native.inference_context():
            _, _, _, features, mask = self.native.model.network.encode_prefix(batch["observation"])
        return features.detach().float(), mask

    def _snapshot_batch(self, snapshot, rgb_images):
        native = self.native
        if not native._robot_bound:
            self.bind(snapshot)
        if rgb_images is None:
            return native._observation_batch(snapshot)
        return native._observation_batch(snapshot, rgb_images=rgb_images)

    @torch.no_grad()
    def gate_features(self, snapshot, *, rgb_images=None):
        """Compute only the frozen prefix used by the live gate, without sampling actions."""
        batch = self._snapshot_batch(snapshot, rgb_images)
        features, mask = self.features(batch)
        return self.encoder(features, mask), batch["observation"].state[:, self.indices].float()

    @torch.no_grad()
    def reference(self, snapshot, observation_id=0, *, rgb_images=None, encode_token=True):
        native = self.native
        batch = self._snapshot_batch(snapshot, rgb_images)
        with native.inference_context():
            raw, features, mask = native.model.predict_action_chunk(batch, return_features=True)
        raw = raw[:, : self.config.chunk_size].float()
        token = None
        if encode_token and self.encoder is not None:
            token = self.encoder(features.float(), mask)
        proprio = batch["observation"].state[:, self.indices].float()
        physical = native.execution_action(raw, context=batch["_action_context"])
        if not torch.isfinite(physical).all() or not torch.isfinite(proprio).all():
            raise ValueError("Non-finite VLA reference")
        return RLTReference(observation_id, token, proprio, raw, batch["_action_context"], physical)

    @torch.no_grad()
    def predict(self, snapshot, *, observation_id=0, apply_actor=False):
        auto_gate = bool(self.config.gate_checkpoint)
        use_actor = (apply_actor or auto_gate) and self.actor is not None
        reference = self.reference(snapshot, observation_id, encode_token=use_actor or auto_gate)
        gate_probability = None
        if auto_gate:
            score = self.gate(reference.token, reference.proprio)
            if not torch.isfinite(score).all():
                raise ValueError("Non-finite RLT stage classifier output")
            gate_probability = score.sigmoid().item()
        physical = reference.physical
        version = -1
        if use_actor:
            corrected = reference.reference.clone()
            corrected[..., self.indices] = self.actor(
                reference.token, reference.proprio, corrected[..., self.indices]
            )
            physical = self.native.execution_action(corrected, context=reference.context)
            if not torch.isfinite(physical).all():
                raise ValueError("Non-finite RLT Actor output")
            version = self.actor_identity["version"]
        return dict(
            actions=physical[0].detach().cpu().numpy(),
            reference=reference.physical[0].detach().cpu().numpy(),
            actor_version=version,
            gate_probability=gate_probability,
        )


def pad_actions(actions, chunk_size):
    result = np.zeros((chunk_size, actions.shape[-1]), dtype=np.float32)
    result[: len(actions)] = actions
    return result
