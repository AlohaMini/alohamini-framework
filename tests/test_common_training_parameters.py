import json
from types import SimpleNamespace

import pytest
from test_cdp_dit_policy import small_options
from test_multi_task_dit import data_and_model, dit_options
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.learning.checkpoint import resolve_pretrained
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization
from alohamini.learning.train_config import parse_training_args
from alohamini.policies.registry import ALGORITHMS, algorithm


def parse(*args):
    return parse_training_args(["--dataset.root=data", "--run_name=test", *args])[0]


def test_all_sixteen_public_flags():
    cfg = parse(
        "--seed=19",
        "--batch_size=2",
        "--lr=.0002",
        "--lr_backbone=.00001",
        "--num_epochs=3",
        "--num_workers=0",
        "--fps=10",
        '--cameras=["forward"]',
        "--chunk_size=4",
        "--backbone=resnet18",
        "--downsample_rate=2",
        "--nheads=4",
        "--enc_layers=3",
        "--dec_layers=1",
        "--kl_weight=.5",
        "--save_freq=2",
    )
    assert cfg["optimizer"] == {"lr": 0.0002, "lr_backbone": 0.00001}
    assert cfg["model"] == dict(
        chunk_size=4,
        vision_backbone="resnet18",
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=1,
        kl_weight=0.5,
    )
    assert cfg["cameras"] == ["forward"] and cfg["seed"] == 19
    assert [
        cfg[key]
        for key in (
            "batch_size",
            "num_epochs",
            "num_workers",
            "fps",
            "downsample_rate",
            "save_freq",
        )
    ] == [2, 3, 0, 10, 2, 2]


@pytest.mark.parametrize("kind", ALGORITHMS)
def test_chunk_size_targets_a_native_field_for_every_registration(kind):
    size = 30 if kind == "lewm" else 32
    settings = resolve_pretrained({"policy": kind, "chunk_size": size})
    assert settings["model"] and "chunk_size" not in settings
    component = algorithm(kind)
    if kind == "lewm":
        component.prepare_options(settings["model"], None)
        assert settings["model"]["planning_horizon"] == 6
        assert component.evaluation_sample_spec(settings["model"])["delta_indices"][
            "action"
        ] == list(range(size))
        return
    if kind in ("bspline", "vqbet"):
        assert component.action_horizon(SimpleNamespace(**settings["model"])) == size
        return
    windows = component.sample_spec(
        settings["model"], cameras=["forward", "wrist_left", "wrist_right"]
    )
    if "action" in windows.get("delta_indices", {}):
        assert len([t for t in windows["delta_indices"]["action"] if t >= 0]) == size


def test_architecture_constraints_and_cli_precedence(tmp_path):
    path = tmp_path / "args.json"
    path.write_text(
        json.dumps(
            {
                "policy": "cdp",
                "model": {"num_heads": 4, "n_obs_steps": 8},
                "optimizer": {"lr": 0.001},
            }
        )
    )
    cfg = parse("--config", str(path), "--nheads=2", "--lr=.0002", "--chunk_size=12")
    assert cfg["model"]["horizon"] == 20 and cfg["model"]["num_heads"] == 2
    assert cfg["optimizer"]["lr"] == 0.0002
    with pytest.raises(ValueError, match="ties"):
        parse("--policy.type=dit_policy", "--enc_layers=3", "--dec_layers=4")
    path.write_text(json.dumps({"policy": "dit_policy", "enc_layers": 3}))
    assert parse("--config", str(path), "--dec_layers=4")["model"]["num_layers"] == 4
    with pytest.raises(ValueError, match="kl_weight"):
        parse("--policy.type=diffusion", "--kl_weight=1")
    with pytest.raises(ValueError, match="Conflicting"):
        parse("--nheads=1", "--policy.n_heads=2")
    assert parse("--policy.type=pi05", "--chunk_size=5")["model"]["network"]["action_horizon"] == 5
    assert (
        parse("--policy.type=acdit", "--policy.stage=mobility", "--dec_layers=4")["model"][
            "mobility_num_layers"
        ]
        == 4
    )


@pytest.mark.parametrize("kind", ["act", "am_act", "cdp", "dit_policy", "multi_task_dit"])
def test_absolute_backbone_lr_preserves_native_group_options(kind, tmp_path, recording):
    component = algorithm(kind)
    if kind == "multi_task_dit":
        _, _, model, _, _ = data_and_model(recording, dit_options(tmp_path / "clip"))
        vision = model.observation_encoder.vision_encoder
    elif kind in ("act", "am_act"):
        model = component.build(model_options())
        vision = model.model.backbone
    else:
        model = component.build(small_options(kind))
        vision = model.vision
    visual = {id(p) for p in vision.parameters() if p.requires_grad}
    assert visual

    def groups(main, backbone=None):
        opts = {"lr": main}
        if backbone is not None:
            opts["lr_backbone"] = backbone
        cfg = resolve_optimization(
            {"optimizer": opts, "scheduler": {"type": "none"}, "steps": 2}, model.config
        )
        optim, _ = make_optimizer_and_scheduler(cfg, model)
        return {
            id(p): (g["lr"], g["weight_decay"]) for g in optim.param_groups for p in g["params"]
        }

    native = groups(0.001)
    if kind == "multi_task_dit":
        assert all(
            native[i][0] == pytest.approx(0.001 * model.config.vision_encoder_lr_multiplier)
            for i in visual
        )
    for main in (0.001, 0.0002):
        actual = groups(main, 0.00001)
        assert actual.keys() == native.keys()
        for i, (lr, decay) in actual.items():
            assert decay == native[i][1]
            assert lr == (0.00001 if i in visual else main)
    vision.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable visual"):
        groups(0.001, 0.00001)
