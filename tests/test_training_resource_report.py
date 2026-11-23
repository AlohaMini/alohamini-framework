"""Resource reports use the same selections and validators as actual training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_hyvla_openwam import tokenizer_assets
from test_training_resources import weights

from alohamini.learning.resource_checks import ResourceCheck, ResourceChecks
from alohamini.learning.resources import (
    ResourceAlgorithm,
    check_training_resources,
    prepare_training,
)
from alohamini.policies.registry import ALGORITHMS, algorithm


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_every_algorithm_returns_serializable_failure_details(name, tmp_path, monkeypatch):
    component = algorithm(name)
    monkeypatch.setattr(
        type(component), "build", lambda *args, **kwargs: pytest.fail("model built")
    )
    settings = {
        "policy": name,
        "device": "cpu",
        "pretrained_path": str(tmp_path / "missing"),
        "model": {field: str(tmp_path / field) for field in getattr(component, "asset_fields", ())},
    }
    report = json.loads(json.dumps(check_training_resources(settings), allow_nan=False))
    assert report["schema_version"] == 1
    assert report["policy"] == name
    assert report["ok"] is False
    assert report["resolved_settings"] is None
    assert report["errors"] or any(row["status"] == "failed" for row in report["resources"])
    for row in report["resources"]:
        assert {"name", "kind", "paths", "available", "error", "status"} <= row.keys()


def test_openwam_reports_multiple_failures_without_starting_training(tmp_path):
    (tmp_path / "tokenizer/google/umt5-xxl").mkdir(parents=True)
    report = check_training_resources(
        {"policy": "openwam", "device": "cpu", "pretrained_path": str(tmp_path)}
    )
    assert not report["ok"] and report["complete"]
    failed = {row["name"]: row for row in report["resources"] if row["status"] == "failed"}
    assert failed.keys() == {"policy", "assets_path"}
    assert all(row["available"] is False and row["error"]["message"] for row in failed.values())


def test_json_roundtrip_keeps_weight_shards_after_the_index_changes(tmp_path):
    base = tmp_path / "base"
    weights(base / "first.safetensors", 2)
    index = base / "model.safetensors.index.json"
    mapping = {name: "first.safetensors" for name in ("weight", "bias")}
    index.write_text(json.dumps({"weight_map": mapping}))
    report = check_training_resources(
        {
            "policy": "diffusion",
            "device": "cpu",
            "pretrained_path": str(base),
            "model": {"pretrained_backbone_weights": None},
        }
    )
    assert report["ok"] and report["complete"]
    selected = next(row for row in report["resources"] if row["name"] == "policy")
    assert selected["paths"] == [str(base / "first.safetensors")]
    settings = json.loads(json.dumps(report["resolved_settings"]))
    weights(base / "newer.safetensors", 9)
    index.write_text(json.dumps({"weight_map": {name: "newer.safetensors" for name in mapping}}))
    prepared = prepare_training(settings)
    model = torch.nn.Linear(1, 1)
    prepared.component.initialize(model, prepared.settings)
    assert model.weight.item() == 2
    assert prepared.report["resources"][0]["paths"] == selected["paths"]
    # A changed source configuration deliberately resolves a new selection.
    settings["pretrained_path"] = str(tmp_path / "missing")
    assert not check_training_resources(settings)["ok"]


def test_selected_files_are_revalidated_instead_of_switching_sources(tmp_path):
    tokenizer_assets(tmp_path / "tokenizer/google/umt5-xxl")
    first = weights(tmp_path / "checkpoint_step_10.safetensors")
    report = check_training_resources(
        {"policy": "openwam", "device": "cpu", "pretrained_path": str(tmp_path)}
    )
    first.write_bytes(b"damaged")
    weights(tmp_path / "checkpoint_step_20.safetensors")
    failed = check_training_resources(report["resolved_settings"])
    row = next(row for row in failed["resources"] if row["name"] == "policy")
    assert not failed["ok"] and row["paths"] == [str(first)]
    assert "Invalid weights" in row["error"]["message"]


def test_dependent_checks_are_blocked_when_their_input_fails(tmp_path):
    base = tmp_path / "base"
    weights(base / "model.safetensors")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"not sentencepiece")
    report = check_training_resources(
        {
            "policy": "pi05",
            "device": "cpu",
            "pretrained_path": str(base),
            "model": {
                "training_recipe": "ki",
                "tokenizer_path": str(tokenizer),
                "fast_tokenizer_path": str(tmp_path / "fast"),
            },
        }
    )
    rows = {row["name"]: row for row in report["resources"]}
    assert not report["ok"] and not report["complete"]
    assert rows["tokenizer_path"]["status"] == "failed"
    assert rows["fast_tokenizer_path"]["status"] == "blocked"
    assert rows["fast_tokenizer_path"]["available"] is None


def test_remote_and_generated_resources_are_never_reported_as_verified():
    report = check_training_resources(
        {
            "policy": "multi_task_dit",
            "device": "cpu",
            "model": {"vision_encoder_name": "example/clip", "text_encoder_name": "example/clip"},
        }
    )
    assert report["ok"] and not report["complete"]
    remote = [row for row in report["resources"] if row["status"] == "unverified"]
    assert len(remote) == 2
    assert all(row["paths"] == ["example/clip"] and row["available"] is None for row in remote)
    checks = ResourceChecks("example", strict=False)
    checks.run(ResourceCheck("codec", "tokenizer", None, status="deferred", reason="Fit from data"))
    assert checks.report()["ok"] and not checks.report()["complete"]


@pytest.mark.parametrize("content", ["invalid JSON", "[]", '{"policy":"unknown"}'])
def test_cli_returns_json_for_configuration_failures(tmp_path, capsys, content):
    from alohamini.cli import main

    path = tmp_path / "training.json"
    path.write_text(content)
    assert main(["train", "check", "--config", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1 and not report["ok"]
    assert report["errors"]


def test_cli_json_can_be_reused_in_a_new_process(tmp_path):
    tokenizer_assets(tmp_path / "base/tokenizer/google/umt5-xxl")
    first = weights(tmp_path / "base/checkpoint_step_10.safetensors")
    config = tmp_path / "training.json"
    config.write_text(
        json.dumps(
            {
                "policy": "openwam",
                "device": "cpu",
                "pretrained_path": str(tmp_path / "base"),
            }
        )
    )
    command = [sys.executable, "-m", "alohamini", "train", "check", "--config", str(config)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    initial = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    report = json.loads(initial.stdout)
    assert report["ok"]
    weights(tmp_path / "base/checkpoint_step_20.safetensors")
    config.write_text(json.dumps(report["resolved_settings"]))
    repeated = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    selected = next(
        row for row in json.loads(repeated.stdout)["resources"] if row["name"] == "policy"
    )
    assert selected["paths"] == [str(first)]
    assert not (tmp_path / "runs").exists()


def test_training_command_dispatch_preserves_arguments(monkeypatch):
    import alohamini.learning.train as training
    from alohamini.cli import _main

    args = []
    monkeypatch.setattr(training, "main", lambda argv: args.extend(argv) or 0)
    assert _main(["train", "--config", "training.json", "--foreground"]) == 0
    assert args == ["--config", "training.json", "--foreground"]


def test_a_new_adapter_needs_no_separate_reporting_rules(tmp_path, monkeypatch):
    import alohamini.policies.registry as registry

    class ExampleAlgorithm(ResourceAlgorithm):
        asset_fields = ("tokens",)
        tokenizer_fields = ("tokens",)

        def apply_preset(self, settings):
            return settings

        def options(self, settings, device):
            return dict(settings["model"])

    tokenizer_assets(tmp_path / "tokens")
    weights(tmp_path / "base/model.safetensors")
    monkeypatch.setattr(registry, "algorithm", lambda name: ExampleAlgorithm())
    report = check_training_resources(
        {
            "policy": "new_algorithm",
            "device": "cpu",
            "pretrained_path": str(tmp_path / "base"),
            "model": {"tokens": str(tmp_path / "tokens")},
        }
    )
    assert report["ok"] and report["complete"]
    assert {row["name"] for row in report["resources"]} == {"policy", "tokens"}


def test_cli_keeps_loader_output_out_of_the_json_stream(tmp_path, capsys, monkeypatch):
    from alohamini.learning import resource_cli, resources

    config = tmp_path / "training.json"
    config.write_text("{}")

    def inspect(settings):
        print("Resource loader diagnostic")
        return ResourceChecks("example", strict=False).report()

    monkeypatch.setattr(resources, "check_training_resources", inspect)
    assert resource_cli.main(["--config", str(config)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"]
    assert "Resource loader diagnostic" in captured.err


@pytest.mark.parametrize("policy", ["act", "diffusion", "dit_policy"])
def test_torchvision_backbone_cache_is_reported_without_downloading(tmp_path, monkeypatch, policy):
    monkeypatch.setattr("torch.hub.get_dir", lambda: str(tmp_path / "torch"))
    settings = {"policy": policy, "device": "cpu"}
    report = check_training_resources(settings)
    backbone = next(row for row in report["resources"] if row["name"] == "vision_backbone")
    assert report["ok"] and not report["complete"]
    assert backbone["status"] == "unverified" and backbone["available"] is None
    assert not (tmp_path / "torch").exists()
    path = Path(backbone["paths"][0])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"Cache availability is checked; tensor schema is checked by the loader")
    ready = check_training_resources(settings)
    cached = next(row for row in ready["resources"] if row["name"] == "vision_backbone")
    assert cached["available"] is True and cached["validation"] == "nonempty_file"


def test_missing_optional_dependency_is_a_serializable_resource_error(tmp_path, monkeypatch):
    import builtins

    tokenizer_assets(tmp_path / "tokenizer/google/umt5-xxl")
    weights(tmp_path / "checkpoint_step_1.safetensors")
    original = builtins.__import__

    def imported(name, *args, **kwargs):
        if name == "safetensors":
            raise ModuleNotFoundError("No module named 'safetensors'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", imported)
    report = check_training_resources(
        {
            "policy": "openwam",
            "pretrained_path": str(tmp_path),
            "device": "cpu",
        }
    )
    assert not report["ok"]
    policy = next(row for row in report["resources"] if row["name"] == "policy")
    assert policy["error"]["type"] == "ModuleNotFoundError"
    json.dumps(report, allow_nan=False)


def test_public_report_interface_import_does_not_import_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from alohamini.learning.resources import check_training_resources; "
            "assert 'torch' not in sys.modules; assert 'safetensors' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr
