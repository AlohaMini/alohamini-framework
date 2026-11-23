"""Training resources fail before allocation and retain their selected files."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from test_hyvla_openwam import tokenizer_assets

from alohamini.learning.resources import ResourceAlgorithm, prepare_training
from alohamini.learning.weights import inspect_weights, load_weights
from alohamini.policies.chunk import local_assets
from alohamini.policies.registry import ALGORITHMS, algorithm


def weights(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.full((1, 1), float(value)), "bias": torch.zeros(1)}, path)
    return path


@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_asset_directory_never_selects_working_directory(value):
    with pytest.raises(ValueError, match="specified"):
        local_assets(value)


def test_explicit_current_directory_remains_valid():
    assert local_assets(".") == Path.cwd()


def test_weight_inspection_validates_headers_and_pins_shards(tmp_path):
    model = torch.nn.Linear(1, 1)
    weights(tmp_path / "first.safetensors", 2)
    index = tmp_path / "model.safetensors.index.json"
    mapping = {name: "first.safetensors" for name in ("weight", "bias")}
    index.write_text(json.dumps({"weight_map": mapping}))
    selected = inspect_weights(tmp_path)
    weights(tmp_path / "later.safetensors", 9)
    index.write_text(json.dumps({"weight_map": {name: "later.safetensors" for name in mapping}}))
    load_weights(model, selected)
    assert model.weight.item() == 2
    # The prepared source, rather than the changed index, is consumed.
    assert selected.files[0][0].name == "first.safetensors"


def test_header_failure_precedes_any_tensor_copy(tmp_path):
    empty = tmp_path / "model.safetensors"
    empty.touch()
    with pytest.raises(Exception, match="header"):
        inspect_weights(tmp_path)


def test_every_registered_algorithm_has_the_training_resource_contract():
    for name in ALGORITHMS:
        assert isinstance(algorithm(name), ResourceAlgorithm), name


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_missing_explicit_training_sources_fail_during_preparation(name, tmp_path):
    component = algorithm(name)
    cfg = {
        "policy": name,
        "device": "cpu",
        "pretrained_path": str(tmp_path / "absent"),
        "model": {
            key: str(tmp_path / ("absent_" + key)) for key in getattr(component, "asset_fields", ())
        },
    }
    with pytest.raises((OSError, ValueError, KeyError, TypeError)):
        prepare_training(cfg)


def test_openwam_numeric_selection_explicit_file_and_frozen_load(tmp_path):
    assets = tmp_path / "tokenizer/google/umt5-xxl"
    tokenizer_assets(assets)
    first = weights(tmp_path / "checkpoint_step_2.safetensors", 2)
    latest = weights(tmp_path / "checkpoint_step_10.safetensors", 10)
    (tmp_path / "checkpoint_step_bad.safetensors").touch()
    cfg = {"policy": "openwam", "device": "cpu", "pretrained_path": str(tmp_path)}
    original = deepcopy(cfg)
    prepared = prepare_training(cfg)
    assert cfg == original
    assert prepared.settings["pretrained_path"] == str(latest)
    assert prepared.options["assets_path"] == str(assets)
    weights(tmp_path / "checkpoint_step_100.safetensors", 100)
    model = SimpleNamespace(
        config=SimpleNamespace(action_representation="alpha80"), network=torch.nn.Linear(1, 1)
    )
    prepared.component.initialize(model, prepared.settings)
    assert model.network.weight.item() == 10
    explicit = prepare_training({**cfg, "pretrained_path": str(first)})
    assert explicit.settings["pretrained_path"] == str(first)
    assert explicit.options["assets_path"] == str(assets)


def test_detached_launch_serializes_selected_weight_file(tmp_path, monkeypatch):
    from alohamini.learning.train import launch_training

    tokenizer_assets(tmp_path / "base/tokenizer/google/umt5-xxl")
    weights(tmp_path / "base/checkpoint_step_9.safetensors")
    latest = weights(tmp_path / "base/checkpoint_step_12.safetensors")
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(pid=123))
    job = launch_training(
        {
            "policy": "openwam",
            "device": "cpu",
            "run_name": "pin",
            "pretrained_path": str(tmp_path / "base"),
        }
    )
    saved = json.loads(Path(job["config"]).read_text())
    assert saved["pretrained_path"] == str(latest)
    assert saved["model"]["assets_path"] == str(tmp_path / "base/tokenizer/google/umt5-xxl")


@pytest.mark.parametrize("name", ["diffusion", "multi_task_dit"])
def test_single_weight_initializers_use_the_prepared_shard(name, tmp_path):
    options = {}
    if name == "multi_task_dit":
        from test_multi_task_dit import dit_options

        options = dit_options(tmp_path / "assets")
    base = tmp_path / "base"
    weights(base / "selected.safetensors", 2)
    index = base / "model.safetensors.index.json"
    mapping = {key: "selected.safetensors" for key in ("weight", "bias")}
    index.write_text(json.dumps({"weight_map": mapping}))
    prepared = prepare_training(
        {"policy": name, "device": "cpu", "pretrained_path": str(base), "model": options}
    )
    weights(base / "later.safetensors", 9)
    index.write_text(json.dumps({"weight_map": {key: "later.safetensors" for key in mapping}}))
    model = torch.nn.Linear(1, 1)
    prepared.component.initialize(model, prepared.settings)
    assert model.weight.item() == 2


def test_smolvla_requires_full_base_metadata_before_model_allocation(tmp_path):
    base = tmp_path / "base"
    weights(base / "model.safetensors")
    cfg = {"policy": "smolvla", "device": "cpu", "pretrained_path": str(base)}
    with pytest.raises(FileNotFoundError, match="config.json"):
        prepare_training(cfg)
    (base / "config.json").write_text('{"type":"smolvla","use_peft":true}')
    with pytest.raises(ValueError, match="PEFT"):
        prepare_training(cfg)


def test_fineart_validates_existing_codecs_and_defers_automatic_fitting(tmp_path, monkeypatch):
    from test_fineart_vla import tiny_options

    options = tiny_options(tmp_path / "assets", monkeypatch)
    base = tmp_path / "base"
    weights(base / "model.safetensors")
    cfg = {
        "policy": "fineart_vla",
        "device": "cpu",
        "pretrained_path": str(base),
        "model": options,
    }
    assert prepare_training(cfg).component is not None
    (Path(options["fast_tokenizer_path"]) / "processor_config.json").write_text("invalid")
    with pytest.raises(ValueError):
        prepare_training(cfg)
    # The generated codec replaces the optional input when automatic fitting is enabled.
    options["auto_fit_fast_tokenizer"] = True
    assert prepare_training(cfg).component is not None


def test_fastwam_validates_frozen_encoders_without_constructing_them(tmp_path):
    base = tmp_path / "base"
    weights(base / "model.safetensors")
    frozen = tmp_path / "wan"
    frozen.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer_assets(tokenizer)
    cfg = {
        "policy": "fastwam",
        "device": "cpu",
        "pretrained_path": str(base),
        "model": {
            "vae_model_id": str(frozen),
            "text_encoder_model_id": str(frozen),
            "tokenizer_model_id": str(tokenizer),
        },
    }
    with pytest.raises(FileNotFoundError):
        prepare_training(cfg)
    for name, filename in (
        ("vae", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
    ):
        weights(frozen / name / filename)
        (frozen / name / "config.json").write_text("{}")
    assert prepare_training(cfg).component is not None


def test_lingbot_keeps_separate_backbone_tokenizer_and_image_assets(tmp_path):
    import shutil

    from test_xiaomi_lingbot import tiny_assets

    complete = tmp_path / "complete"
    tiny_assets(complete)
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    shutil.copyfile(complete / "config.json", backbone / "config.json")
    base = tmp_path / "base"
    weights(base / "model.safetensors")
    (base / "config.json").write_text("{}")
    prepared = prepare_training(
        {
            "policy": "lingbot_vla_v2",
            "device": "cpu",
            "pretrained_path": str(base),
            "model": {
                "tokenizer_path": str(backbone),
                "text_tokenizer_path": str(complete),
                "image_processor_path": str(complete),
            },
        }
    )
    assert prepared.options["tokenizer_path"] == str(backbone)


@pytest.mark.parametrize("native", [False, True])
def test_invalid_resources_fail_before_execution_or_job_files(tmp_path, monkeypatch, native):
    import alohamini.learning.train as training

    root = tmp_path / "base"
    root.mkdir()
    if native:
        (root / "policy.json").write_text(
            json.dumps(
                {
                    "format": "alohamini-policy",
                    "version": 1,
                    "kind": "act",
                    "config": {},
                    "state": "none",
                    "cameras": [],
                    "image_size": [32, 32],
                }
            )
        )
        (root / "model.safetensors").touch()
        policy = "act"
    else:
        tokenizer_assets(root / "tokenizer/google/umt5-xxl")
        policy = "openwam"
    cfg = {"policy": policy, "device": "cpu", "pretrained_path": str(root), "run_name": "bad"}
    calls = []
    monkeypatch.setattr(training, "Execution", lambda *args: calls.append(args))
    monkeypatch.setattr(training.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    for run in (training.train, training.launch_training):
        with pytest.raises((OSError, ValueError)):
            run(cfg)
    assert calls == []
    assert not (tmp_path / "workspace/logs/training").exists()


def test_resume_uses_checkpoint_weights_without_the_old_base(tmp_path):
    root = tmp_path / "checkpoint"
    model = root / "pretrained_model"
    weights(model / "model.safetensors")
    tokenizer_assets(model / "assets")
    (model / "policy.json").write_text(
        json.dumps(
            {
                "format": "alohamini-policy",
                "version": 1,
                "kind": "openwam",
                "config": {},
            }
        )
    )
    (root / "training_state").mkdir()
    (root / "training_state/state.pt").write_bytes(b"state checked by the resume loader")
    cfg = {
        "policy": "openwam",
        "resume": True,
        "_checkpoint": str(root),
        "pretrained_path": str(tmp_path / "removed-base"),
    }
    assert prepare_training(cfg).options is None
    with pytest.raises(ValueError, match="policy type"):
        prepare_training({**cfg, "policy": "act"})


def test_scratch_and_conditional_modes_do_not_require_a_policy_base():
    for name in ("act", "am_act", "diffusion", "bspline", "cdp", "dit_policy", "vqbet", "lewm"):
        assert prepare_training({"policy": name, "device": "cpu"}).component is not None
