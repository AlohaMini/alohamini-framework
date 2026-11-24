"""Full-episode token validation selection, feature caching and training integration."""

import json
from types import SimpleNamespace

import pytest
import torch

from alohamini.policies.rlt.networks import RLTokenEncoder
from alohamini.rl.rlt import token_training as training
from alohamini.rl.rlt.config import RLTConfig
from alohamini.rl.rlt.token_training import RLTokenDecoder


def test_validation_covers_all_ten_episodes_and_endpoints():
    data = object.__new__(training.TokenData)
    data.val_episodes = [57, 21, 75, 18, 33, 40, 93, 27, 51, 39]
    data.groups = {}
    locations = []
    for episode in data.val_episodes:
        start = len(locations)
        locations.extend((episode, frame) for frame in range(100 + episode))
        data.groups[episode] = list(range(start, len(locations)))
    data.dataset = SimpleNamespace(sample_indices=list(range(len(locations))), locations=locations)
    samples = data.validation_samples(32)
    assert len(samples) == 320
    assert samples == data.validation_samples(32)
    for episode in data.val_episodes:
        frames = [sample["frame_index"] for sample in samples if sample["episode"] == episode]
        assert len(frames) == len(set(frames)) == 32
        assert frames[0] == 0 and frames[-1] == 99 + episode
        gaps = [b - a for a, b in zip(frames, frames[1:], strict=False)]
        assert max(gaps) - min(gaps) <= 1


def test_short_rollout_has_unique_paths():
    data = object.__new__(training.TokenData)
    data.val_episodes = ["heldout"]
    data.groups = {"heldout": ["obs/0", "obs/1", "obs/2"]}
    data.dataset = None
    assert [sample["index"] for sample in data.validation_samples(32)] == data.groups["heldout"]


def test_cached_validation_ignores_padding_and_averages_per_frame():
    class Data:
        dataset = object()
        calls = 0

        def validation_samples(self, count):
            return [dict(episode=5, index=0), dict(episode=8, index=1)]

        def features(self, indices):
            self.calls += 1
            if indices == [0]:
                return torch.tensor([[[1.0], [1000.0]]]), torch.tensor([[True, False]])
            return torch.full((1, 2, 1), 3.0), torch.ones(1, 2, dtype=torch.bool)

    class Encoder(torch.nn.Module):
        def forward(self, features, mask):
            return features[:, 0]

    class Decoder(torch.nn.Module):
        def forward(self, token, features, mask):
            return torch.zeros_like(features)

    data = Data()
    cache = training.TokenValidation(data, 32)
    for batch_size in (1, 2):
        metrics = cache.evaluate(Encoder(), Decoder(), "cpu", batch_size)
        assert metrics["validation_loss"] == 5.0
        assert metrics["validation_loss_by_episode"] == {"5": 1.0, "8": 9.0}
        assert metrics["validation_frames"] == 2
    assert data.calls == 2


def test_training_baseline_and_optimizer_use_separate_samples(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    spec = dict(input_dim=4, dim=4, layers=1, heads=1, max_tokens=4)
    contract = dict(token_spec=spec)
    calls = []

    class Data(training.TokenData):
        def __init__(self, config, policy):
            self.dataset = SimpleNamespace(
                sample_indices=list(range(6)),
                locations=[(0, i) for i in range(2)] + [(1, i) for i in range(4)],
            )
            self.groups = {0: [0, 1], 1: [2, 3, 4, 5]}
            self.train_episodes, self.val_episodes = [0], [1]
            self.training, self.validation = self.groups[0], self.groups[1]

        def features(self, indices):
            calls.extend(indices)
            return torch.ones(len(indices), 3, 4), torch.ones(len(indices), 3, dtype=torch.bool)

    monkeypatch.setattr(training, "RLTPolicy", lambda config: SimpleNamespace(contract=contract))
    monkeypatch.setattr(training, "TokenData", Data)
    baseline = tmp_path / "baseline.pt"
    torch.save(
        dict(
            contract=contract,
            encoder=RLTokenEncoder(**spec).state_dict(),
            decoder=RLTokenDecoder(**spec).state_dict(),
            step=5000,
            validation_loss=0.2,
        ),
        baseline,
    )
    config = RLTConfig(
        checkpoint="unused",
        output_dir=str(tmp_path),
        task="place",
        device="cpu",
        token_steps=3,
        token_batch_size=2,
        token_validation_frames_per_episode=2,
    )
    from alohamini.cli import _main

    settings = tmp_path / "config.json"
    settings.write_text(json.dumps(config.to_dict()))
    assert (
        _main(
            [
                "train",
                "rlt",
                "token",
                "--config",
                str(settings),
                "--baseline-checkpoint",
                str(baseline),
            ]
        )
        == 0
    )
    assert calls[:2] == [2, 5]
    assert len(calls) == 2 + 3 * 2
    assert set(calls[2:]) <= {0, 1}
    metrics = [
        json.loads(line) for line in (tmp_path / "token-metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["step"] == 3 and metrics[-1]["validation_frames"] == 2
    report = json.loads((tmp_path / "token-baseline-validation.json").read_text())
    assert report["checkpoint_step"] == 5000 and report["validation_frames"] == 2
    from alohamini.learning.stages import read_stage_weights

    assert read_stage_weights(tmp_path / "checkpoint")[0]["step"] == 3


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_validation_frame_count_must_be_positive_integer(count):
    with pytest.raises(ValueError, match="token_validation_frames_per_episode"):
        RLTConfig(
            checkpoint="base",
            output_dir="out",
            task="place",
            token_validation_frames_per_episode=count,
        )


def test_token_bottleneck_causal_decoder_and_frozen_targets():
    torch.manual_seed(7)
    encoder = RLTokenEncoder(input_dim=8, dim=8, layers=1, heads=2)
    decoder = RLTokenDecoder(input_dim=8, dim=8, layers=1, heads=2)
    features = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True] * 4])
    token = encoder(features.detach(), mask)
    original = decoder(token, features, mask)
    changed = features.detach().clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(decoder(token, changed, mask)[:, :3], original[:, :3])
    decoder.loss(token, features, mask).backward()
    assert features.grad is None
    assert encoder.token.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


@pytest.mark.parametrize("filename", ["token.pt", "token-metrics.jsonl", "token-split.json"])
def test_existing_training_artifacts_survive_a_new_run(tmp_path, filename):
    artifact = tmp_path / filename
    artifact.write_bytes(b"previous-run")
    config = RLTConfig(checkpoint="unused", output_dir=str(tmp_path), task="place")
    with pytest.raises(FileExistsError, match="--output-dir"):
        training.train_token(config)
    assert artifact.read_bytes() == b"previous-run"
