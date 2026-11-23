import json
import subprocess
from pathlib import Path

import pytest
import torch
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.evaluate import evaluate_checkpoint
from alohamini.learning.policy import NativePolicy
from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


@pytest.mark.parametrize(
    "fps, stride, anchors, first",
    [
        (None, 2, [0, 2], [0, 1, 2]),
        (10, 1, [0, 3], [0, 3, 3]),
        (10, 2, [0], [0, 3, 3]),
        (20, 1, [0, 2, 3], [0, 2, 3]),
    ],
)
def test_episode_clock_and_anchor_stride(recording, tmp_path, fps, stride, anchors, first):
    root = tmp_path / "v3"
    export_lerobot(recording, root)
    data = AlohaMiniDataset(
        root,
        episodes=[1, 0],
        cameras=[],
        chunk_size=3,
        fps=fps,
        downsample_rate=stride,
        annotation_fields={},
    )
    assert data.source_fps == 30 and data.info["fps"] == (fps or 30)
    assert len(data) == 2 * len(anchors)
    for ep, bias in enumerate((10, 0)):
        idx = ep * len(anchors)
        assert data[idx]["action"][:, 0].tolist() == [bias + x for x in first]
        assert [
            data[i]["_annotations"]["frame_index"] for i in range(idx, idx + len(anchors))
        ] == anchors
        assert data[idx]["action_is_pad"].tolist() == [False, False, fps == 10]


@pytest.mark.parametrize(
    "args",
    [
        ["--num_epochs=1", "--steps=3"],
        ["--fps=0"],
        ["--downsample_rate=0"],
        ["--num_epochs=-1"],
    ],
)
def test_invalid_temporal_cli(args):
    with pytest.raises(SystemExit):
        parse_training_args(["--run_name=a", "--dataset.root=a", *args])


@pytest.mark.parametrize("processes_count, batches, updates", [(1, 3, 2), (2, 2, 1)])
def test_epoch_training_resume_and_offline_clock(
    recording, tmp_path, monkeypatch, processes_count, batches, updates
):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "work"))
    processes = []
    popen = subprocess.Popen

    def start(*args, **kwargs):
        child = popen(*args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", start)
    run_dir = tmp_path / "run"
    settings = dict(
        dataset=str(recording),
        output_dir=str(run_dir),
        policy="act",
        device="cpu",
        num_processes=processes_count,
        dataset_streaming=processes_count > 1,
        num_epochs=1,
        fps=20,
        downsample_rate=1,
        batch_size=1,
        seed=91,
        cameras=["forward"],
        lr=2e-4,
        lr_backbone=1e-5,
        backbone="resnet18",
        chunk_size=3,
        nheads=4,
        enc_layers=1,
        dec_layers=1,
        kl_weight=0.25,
        cpu_threads=1,
        gradient_accumulation_steps=2,
        num_workers=0,
        train_episodes=[0],
        state="none",
        image_size=[32, 32],
        model=model_options(state=False),
        save_freq=1,
    )

    def run(cfg, step, consumed):
        job = launch_training(cfg)
        assert processes[-1].wait(timeout=120) == 0, Path(job["log"]).read_text()
        checkpoint = run_dir / "checkpoints" / f"{step:06d}"
        saved = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
        assert saved["steps"] == step and saved["num_epochs"] == cfg["num_epochs"]
        assert saved["model"]["kl_weight"] == 0.25 and saved["model"]["n_decoder_layers"] == 1
        state = torch.load(checkpoint / "training_state/state.pt", weights_only=True)
        assert saved["optimizer"]["lr_backbone"] == 1e-5
        assert {g["lr"] for g in state["optimizer"]["param_groups"]} == {2e-4, 1e-5}
        assert state["consumed_batches"] == consumed
        return checkpoint / "pretrained_model"

    checkpoint = run(settings, updates, batches)
    cfg, _ = parse_training_args(
        [
            "--config_path",
            str(checkpoint / "train_config.json"),
            "--resume=true",
            "--num_epochs=2",
        ]
    )
    checkpoint = run(cfg, 2 * updates, 2 * batches)
    policy = NativePolicy(checkpoint, device="cpu", offline=True)
    assert policy.fps == 20 and policy.source["fps"] == 30
    assert evaluate_checkpoint(checkpoint, recording, episodes=[1])["samples"] == 3
