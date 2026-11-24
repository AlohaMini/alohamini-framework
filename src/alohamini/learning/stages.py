"""Single-device training stages with native optimizer, execution and checkpoint layout."""

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from alohamini.learning.checkpoint import save_weights
from alohamini.learning.execution import Execution
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization
from alohamini.learning.training_state import _link, rng_state


def stage_optimizer(model, *, lr, steps, kind, clip, decay):
    cfg = resolve_optimization(
        {"steps": steps, "optimizer": {"type": kind}, "scheduler": {"type": "none"}},
        SimpleNamespace(
            optimizer_lr=lr, optimizer_grad_clip_norm=clip, optimizer_weight_decay=decay
        ),
    )
    optimizer, _ = make_optimizer_and_scheduler(cfg, model)
    return optimizer


class _Objective(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        return self.loss(self.model, batch) * batch["_weight"], {}


@contextmanager
def supervised_stage(model, objective, *, device, lr, steps, clip, precision):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Prepared supervised stages currently support one process")
    execution = Execution(device, precision)
    try:
        model = _Objective(model, objective)
        optimizer = stage_optimizer(model, lr=lr, steps=steps, kind="adamw", clip=clip, decay=0.01)
        model, optimizer = execution.prepare(model, optimizer)

        def update(batch):
            for _ in range(20):
                result = execution.update(
                    model, [batch], optimizer, clip, dict, reduction=lambda _: {"_weight": 1}
                )
                if result["optimizer_step"]:
                    return result["loss"]
            raise RuntimeError("20 consecutive AMP overflows in training stage")

        yield update
    finally:
        execution.close()


def save_stage_checkpoint(root, step, models, manifest, config, *, training=None, aliases=()):
    """Publish all component weights and optional restart state as one immutable step."""
    root = Path(root)
    parent = root / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{step:06d}"
    if path.exists():
        raise FileExistsError(path)
    stage = Path(tempfile.mkdtemp(prefix=f".{step}-", dir=parent))
    try:
        portable = stage / "pretrained_model"
        portable.mkdir()
        save_weights(portable / "model.safetensors", nn.ModuleDict(models))
        meta = {**manifest, "components": list(models)}
        (portable / "stage.json").write_text(json.dumps(meta, allow_nan=False))
        (portable / "train_config.json").write_text(json.dumps(config, allow_nan=False))
        if training is not None:
            (stage / "training_state").mkdir()
            torch.save(
                dict(training, step=step, rng=rng_state()), stage / "training_state/state.pt"
            )
        stage.rename(path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    _link(path, parent / "last")
    for name in ("checkpoint", *aliases):
        _link(path / "pretrained_model", root / name)
    return path


def read_stage_weights(path):
    from safetensors.torch import load_file

    path = Path(path).resolve()
    if path.is_file():  # Earlier standalone training artifacts.
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            return torch.load(stream, map_location="cpu", weights_only=True), digest
    manifest = (path / "stage.json").read_bytes()
    weight = path / "model.safetensors"
    with weight.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    digest.update(manifest)
    data = json.loads(manifest)
    tensors = load_file(weight)
    for name in data.pop("components"):
        data[name] = {k[len(name) + 1 :]: v for k, v in tensors.items() if k.startswith(name + ".")}
    return data, digest.hexdigest()
