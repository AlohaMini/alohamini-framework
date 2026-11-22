import io
import json

import numpy as np
import pytest
from safetensors.numpy import save_file
from sentencepiece import SentencePieceTrainer

from alohamini.learning.preflight import inspect_training, training_catalog
from alohamini.policies.registry import ALGORITHMS


def test_catalog_and_act_default():
    catalog = {entry["id"]: entry["label"] for entry in training_catalog()}
    assert catalog.keys() == ALGORITHMS.keys()
    assert catalog["act"] == "ACT"
    assert catalog["pi05"] == "π0.5"
    assert inspect_training("act", "cpu", "", {}) == {"assets": [], "issues": []}


def test_reports_device_and_missing_pi05_resources_together():
    result = inspect_training("pi05", "cuda:1000000", "", {})
    assert "tokenizer_path" in result["assets"]
    assert {issue["code"] for issue in result["issues"]} == {
        "device", "missing_weights", "missing_tokenizer",
    }


def test_pi05_checks_local_files_and_recovers_after_replacement(tmp_path):
    weights = tmp_path / "model.safetensors"
    tokenizer = tmp_path / "paligemma_tokenizer.model"
    weights.write_bytes(b"invalid")
    tokenizer.write_bytes(b"invalid")
    paths = {"tokenizer_path": str(tokenizer)}
    result = inspect_training("pi05", "cpu", str(weights), paths)
    assert [issue["code"] for issue in result["issues"]] == ["weights", "tokenizer"]
    save_file({"test": np.ones(1, dtype=np.float32)}, weights)
    blob = io.BytesIO()
    SentencePieceTrainer.train(
        sentence_iterator=iter(["pick and place", "move the cube"]),
        model_writer=blob,
        vocab_size=24,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    tokenizer.write_bytes(blob.getvalue())
    assert inspect_training("pi05", "cpu", str(tmp_path), paths)["issues"] == []


def test_rejects_mismatched_native_checkpoint(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps({
        "format": "alohamini-policy", "version": 1, "kind": "act",
    }))
    save_file({"test": np.ones(1, dtype=np.float32)}, tmp_path / "model.safetensors")
    result = inspect_training("pi05", "cpu", str(tmp_path), {})
    assert result["issues"][0]["code"] == "weights"
    assert "not a pi05 checkpoint" in result["issues"][0]["detail"]


@pytest.mark.parametrize("policy", ["act", "diffusion"])
def test_only_accepts_native_checkpoints_for_native_only_policies(tmp_path, policy):
    save_file({"test": np.ones(1, dtype=np.float32)}, tmp_path / "model.safetensors")
    result = inspect_training(policy, "cpu", str(tmp_path), {})
    assert result["issues"][0]["code"] == "native_checkpoint"
