import math

import pytest
import torch

from alohamini.rl import RLTrainer
from alohamini.rl.replay import UniformReplaySampler


def test_custom_algorithm_and_source_share_step_schedule():
    events = []

    class Source:
        def sample(self, batch_size):
            events.append(("sample", batch_size))
            return ["observation"] * batch_size

    class Algorithm:
        def update(self, batch):
            assert batch == ["observation", "observation"]
            events.append(("update",))
            return {"items": len(batch)}

    trainer = RLTrainer(Algorithm(), Source(), batch_size=2)
    trainer.fit(
        5,
        start_step=2,
        eval_freq=2,
        evaluate=lambda: {"evaluated": 1},
        on_step=lambda step, metrics: events.append(("log", step, dict(metrics))),
        save_freq=3,
        save_checkpoint=lambda step, metrics: events.append(("save", step, dict(metrics))),
    )
    train = {"items": 2}
    evaluated = {**train, "evaluated": 1}
    assert events == [
        ("sample", 2),
        ("update",),
        ("log", 3, train),
        ("save", 3, train),
        ("sample", 2),
        ("update",),
        ("log", 4, evaluated),
        ("sample", 2),
        ("update",),
        ("log", 5, evaluated),
        ("save", 5, evaluated),
    ]


def test_sampler_with_replacement_and_restore_are_independent_of_global_rng():
    sampler = UniformReplaySampler(3, lambda indices: indices, seed=17)
    expected = torch.Generator(device="cpu").manual_seed(17)
    torch.testing.assert_close(sampler.sample(8), torch.randint(3, (8,), generator=expected))
    state = sampler.get_rng_state()
    expected_next = sampler.sample(8)
    restored = UniformReplaySampler(3, lambda indices: indices, seed=18)
    restored.set_rng_state(state)
    torch.testing.assert_close(restored.sample(8), expected_next)


def test_failed_update_is_never_published():
    class Source:
        def sample(self, batch_size):
            return math.nan

    class Algorithm:
        def update(self, batch):
            return {"loss": batch}

    trainer = RLTrainer(Algorithm(), Source(), batch_size=1)
    with pytest.raises(ValueError, match="update 1"):
        trainer.fit(
            1,
            on_step=lambda *_: pytest.fail("Logged a failed update"),
            save_checkpoint=lambda *_: pytest.fail("Saved a failed update"),
        )
