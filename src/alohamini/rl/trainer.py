"""Step-based RL training for algorithms and batch sources supplied by the caller."""

import math
from collections.abc import Callable, Mapping
from typing import Generic, Protocol, TypeVar

Batch = TypeVar("Batch")
BatchOut = TypeVar("BatchOut", covariant=True)
BatchIn = TypeVar("BatchIn", contravariant=True)
Metrics = Mapping[str, float]
StepCallback = Callable[[int, Metrics], None]


class BatchSource(Protocol[BatchOut]):
    def sample(self, batch_size: int, /) -> BatchOut: ...


class RLAlgorithm(Protocol[BatchIn]):
    def update(self, batch: BatchIn, /) -> Metrics: ...


class RLTrainer(Generic[Batch]):
    """Run one algorithm update per sampled batch.

    Supply any source with sample(batch_size) and any algorithm with update(batch).
    The algorithm owns its optimizers and returns scalar metrics after the update.

    Example:
        trainer = RLTrainer(algorithm, replay_source, batch_size=64)
        trainer.fit(10_000, on_step=log_metrics, save_checkpoint=save, save_freq=500)
    """

    def __init__(
        self, algorithm: RLAlgorithm[Batch], data_source: BatchSource[Batch], *, batch_size: int
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.algorithm = algorithm
        self.data_source = data_source
        self.batch_size = batch_size

    def fit(
        self,
        steps: int,
        *,
        start_step: int = 0,
        evaluate: Callable[[], Metrics] | None = None,
        eval_freq: int | None = None,
        on_step: StepCallback | None = None,
        save_checkpoint: StepCallback | None = None,
        save_freq: int | None = None,
    ) -> None:
        """Train to an absolute step, starting after the last restored update.

        Evaluation and checkpoint hooks always run on the final step. A frequency
        also schedules them on its multiples. on_step sees the merged metrics
        before checkpointing, so a checkpoint can record the corresponding log offset.
        """
        if type(steps) is not int or type(start_step) is not int or not 0 <= start_step < steps:
            raise ValueError("Expected integer steps with 0 <= start_step < steps")
        for name, freq, hook in (
            ("eval_freq", eval_freq, evaluate),
            ("save_freq", save_freq, save_checkpoint),
        ):
            if freq is not None and (type(freq) is not int or freq < 1 or hook is None):
                raise ValueError(f"{name} requires a positive integer and its callback")

        for step in range(start_step + 1, steps + 1):
            batch = self.data_source.sample(self.batch_size)
            metrics = dict(self.algorithm.update(batch))
            if evaluate is not None and (step == steps or (eval_freq and step % eval_freq == 0)):
                metrics.update(evaluate())
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Non-finite RL metric at update {step}")
            if on_step is not None:
                on_step(step, metrics)
            if save_checkpoint is not None and (
                step == steps or (save_freq and step % save_freq == 0)
            ):
                save_checkpoint(step, metrics)
