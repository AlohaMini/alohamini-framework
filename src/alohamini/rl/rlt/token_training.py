"""Episode-separated offline RL-token training on existing native datasets or rollouts."""

import json
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from alohamini.datasets.experience.storage import RunLock, atomic_json, json_lines, load_snapshot
from alohamini.learning.loading import collate_samples
from alohamini.learning.precision import tf32_context
from alohamini.learning.stages import read_stage_weights, save_stage_checkpoint, supervised_stage
from alohamini.policies.rlt.networks import RLTokenEncoder
from alohamini.policies.rlt.policy import RLTPolicy, require_contract
from alohamini.rl.rlt.episodes import recorded_action_names


class RLTokenDecoder(nn.Module):
    def __init__(self, input_dim=2048, dim=2048, layers=4, heads=8, max_tokens=2048):
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)
        self.position = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )
        self.output = nn.Linear(dim, input_dim)

    def forward(self, token, targets, mask):
        length = targets.shape[1]
        if length > self.position.shape[1]:
            raise ValueError("Decoder sequence exceeds max_tokens")
        x = torch.cat((token[:, None], self.projection(targets.detach()[:, :-1].float())), dim=1)
        valid = torch.cat((torch.ones_like(mask[:, :1], dtype=torch.bool), mask[:, :-1]), dim=1)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        return self.output(
            self.decoder(x + self.position[:, :length], mask=causal, src_key_padding_mask=~valid)
        )

    def loss(self, token, targets, mask):
        error = (self(token, targets, mask) - targets.detach().float()).square()
        return (error * mask[..., None]).sum() / (mask.sum() * targets.shape[-1]).clamp_min(1)


class TokenData:
    def __init__(self, config, policy):
        self.config, self.policy = config, policy
        root = (
            Path(config.dataset or policy.native.manifest["training"].get("dataset", ""))
            .expanduser()
            .resolve()
        )
        self.root = root
        if not root.is_dir():
            raise ValueError("Set dataset to an accessible demonstration or RLT rollout directory")
        self.dataset = None
        manifest = policy.native.manifest
        if (root / "meta/info.json").is_file():
            from alohamini.datasets.tools import dataset_info
            from alohamini.learning.data import AlohaMiniDataset

            info = dataset_info(root)
            available = (
                list(range(info["total_episodes"]))
                if "total_episodes" in info
                else sorted(
                    int(path.name.removeprefix("episode_"))
                    for path in (root / "episodes").glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")
                    if path.is_dir()
                )
            )
            saved = manifest.get("training", {})
            training = saved.get("train_episodes")
            validation = saved.get("val_episodes")
            # The training machine and the robot often mount the same dataset at different paths.
            same_dataset = root == Path(
                saved.get("dataset", "")
            ).expanduser().resolve() or info == manifest.get("source_info")
            if (
                same_dataset
                and training
                and validation
                and set(training + validation) <= set(available)
            ):
                self.train_episodes, self.val_episodes = list(training), list(validation)
            else:
                self.train_episodes, self.val_episodes = self._split(available)
            self.dataset = AlohaMiniDataset(
                root,
                episodes=self.train_episodes + self.val_episodes,
                include_action=False,
                state=manifest["state"],
                exclude_fixed_state=manifest.get("fixed_state_excluded", False),
                cameras=manifest["cameras"],
                image_size=manifest["image_size"],
                image_resize_mode=manifest.get("image_resize_mode", "stretch"),
                image_pad_value=manifest.get("image_pad_value", 0.0),
                include_task=True,
                return_uint8=True,
                allow_unannotated=saved.get("allow_unannotated", False),
                fps=manifest.get("temporal_sampling", {}).get("fps"),
                downsample_rate=manifest.get("temporal_sampling", {}).get("downsample_rate", 1),
            )
            self.groups = {}
            for index, row_index in enumerate(self.dataset.sample_indices):
                episode = self.dataset.locations[row_index][0]
                self.groups.setdefault(episode, []).append(index)
        else:
            directories = sorted((root / "episodes").glob("*/episode.json"))
            self.groups = {}
            for summary_path in directories:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    summary.get("status") != "committed"
                    or summary.get("evaluation")
                    or not summary.get("training_valid", True)
                ):
                    continue
                directory = summary_path.parent
                paths = [
                    directory / "observations" / str(row["index"])
                    for row in json_lines(directory / "steps.jsonl")
                ]
                if paths:
                    self.groups[directory.name] = paths
            self.train_episodes, self.val_episodes = self._split(list(self.groups))
        if set(self.train_episodes) & set(self.val_episodes):
            raise ValueError("Token train and validation episodes overlap")
        self.training = [index for episode in self.train_episodes for index in self.groups[episode]]
        self.validation = [index for episode in self.val_episodes for index in self.groups[episode]]
        if not self.training or not self.validation:
            raise ValueError("Token training requires nonempty train and validation episodes")

    def _split(self, episodes):
        if len(episodes) < 2:
            raise ValueError("At least two episodes are required for token train/validation")
        episodes = list(episodes)
        np.random.default_rng(self.config.seed).shuffle(episodes)
        count = max(
            1, min(len(episodes) - 1, round(len(episodes) * self.config.validation_fraction))
        )
        return episodes[count:], episodes[:count]

    def batch(self, index):
        native = self.policy.native
        if self.dataset is None:
            snapshot = load_snapshot(index)
            return native.prepare_snapshot(snapshot)
        return native.processor(collate_samples([self._offline_sample(index)]))

    def _offline_sample(self, index):
        native = self.policy.native
        sample = dict(self.dataset[index])
        sample.pop("action", None)
        sample.pop("action_is_pad", None)
        if not sample.get("task"):
            sample["task"] = self.config.task
        sample = native.fixed_input(sample)
        return sample

    def features(self, indices):
        if self.dataset is not None:
            batch = collate_samples([self._offline_sample(i) for i in indices])
            return self.policy.features(self.policy.native.processor(batch))
        examples = [self.policy.features(self.batch(index)) for index in indices]
        return torch.cat([e[0] for e in examples]), torch.cat([e[1] for e in examples])

    def validation_samples(self, frames_per_episode):
        """Select fixed, unique observations across each held-out episode's full timeline."""
        samples = []
        for episode in self.val_episodes:
            indices = self.groups[episode]
            positions = (
                np.linspace(0, len(indices) - 1, min(frames_per_episode, len(indices)))
                .round()
                .astype(int)
            )
            for position in positions:
                index = indices[position]
                sample = dict(episode=episode, position=int(position), episode_frames=len(indices))
                if self.dataset is None:
                    sample.update(index=str(index), observation_path=str(index))
                else:
                    row = self.dataset.sample_indices[index]
                    sample.update(index=int(index), frame_index=int(self.dataset.locations[row][1]))
                samples.append(sample)
        return samples


class TokenValidation:
    """Cache frozen features in CPU memory once; never use them for optimizer updates."""

    def __init__(self, data, frames_per_episode):
        self.samples = data.validation_samples(frames_per_episode)
        self.cached = []
        for sample in self.samples:
            index = Path(sample["index"]) if data.dataset is None else sample["index"]
            features, mask = data.features([index])
            self.cached.append((features.detach().cpu(), mask.detach().cpu()))
        print(f"[RLT token] cached {len(self.samples)} validation frames on CPU", flush=True)

    @torch.no_grad()
    def evaluate(self, encoder, decoder, device, batch_size):
        encoder.eval()
        decoder.eval()
        losses = []
        for start in range(0, len(self.cached), batch_size):
            batch = self.cached[start : start + batch_size]
            features = torch.cat([entry[0] for entry in batch]).to(device)
            mask = torch.cat([entry[1] for entry in batch]).to(device)
            predicted = decoder(encoder(features, mask), features, mask)
            squared = (predicted - features.float()).square() * mask[..., None]
            per_frame = squared.sum((1, 2)) / (mask.sum(1) * features.shape[-1]).clamp_min(1)
            losses.extend(per_frame.cpu().tolist())
        by_episode = {}
        for sample, loss in zip(self.samples, losses, strict=True):
            by_episode.setdefault(str(sample["episode"]), []).append(loss)
        return dict(
            validation_loss=sum(losses) / len(losses),
            validation_frames=len(losses),
            validation_loss_by_episode={
                episode: sum(values) / len(values) for episode, values in by_episode.items()
            },
        )


def train_token(config, *, baseline_checkpoint=None):
    root = Path(config.output_dir)
    with RunLock(root / "token-training.lock"), tf32_context(config.tf32), ExitStack() as stack:
        artifacts = (
            "checkpoints",
            "token.pt",
            "token-last.pt",
            "token-metrics.jsonl",
            "token-split.json",
            "token-validation.json",
            "token-baseline-validation.json",
        )
        if any((root / name).exists() for name in artifacts):
            raise FileExistsError(
                f"Token training results already exist in {root}. "
                "Use --output-dir with a new directory."
            )
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        learned = config.learned_action_names
        if learned is None and config.dataset:
            learned = recorded_action_names(config.dataset)
        policy = RLTPolicy(
            replace(
                config,
                token_checkpoint="",
                actor_checkpoint="",
                gate_checkpoint="",
                learned_action_names=learned,
            )
        )
        data = TokenData(config, policy)
        validation = TokenValidation(data, config.token_validation_frames_per_episode)
        atomic_json(
            root / "token-validation.json",
            dict(
                selection="uniform_per_episode_including_endpoints",
                frames_per_episode=config.token_validation_frames_per_episode,
                total_frames=len(validation.samples),
                samples=validation.samples,
            ),
        )
        if baseline_checkpoint:
            saved, _ = read_stage_weights(baseline_checkpoint)
            require_contract(saved["contract"], policy.contract)
            baseline_encoder = RLTokenEncoder(**policy.contract["token_spec"]).to(config.device)
            baseline_decoder = RLTokenDecoder(**policy.contract["token_spec"]).to(config.device)
            baseline_encoder.load_state_dict(saved["encoder"], strict=True)
            baseline_decoder.load_state_dict(saved["decoder"], strict=True)
            metrics = validation.evaluate(
                baseline_encoder, baseline_decoder, config.device, config.token_batch_size
            )
            atomic_json(
                root / "token-baseline-validation.json",
                dict(
                    checkpoint=str(Path(baseline_checkpoint).resolve()),
                    checkpoint_step=saved["step"],
                    original_validation_loss=saved["validation_loss"],
                    **metrics,
                ),
            )
            print(
                f"[RLT token] baseline step={saved['step']} val={metrics['validation_loss']:.6g}",
                flush=True,
            )
            del baseline_encoder, baseline_decoder, saved
        # Baseline evaluation must not change the new model's random initialization.
        torch.manual_seed(config.seed)
        encoder = RLTokenEncoder(**policy.contract["token_spec"]).to(config.device)
        decoder = RLTokenDecoder(**policy.contract["token_spec"]).to(config.device)
        models = {"encoder": encoder, "decoder": decoder}

        def objective(modules, batch):
            feat, mask = batch["features"], batch["mask"]
            return modules["decoder"].loss(modules["encoder"](feat, mask), feat, mask)

        update = stack.enter_context(
            supervised_stage(
                torch.nn.ModuleDict(models),
                objective,
                device=config.device,
                lr=config.lr,
                steps=config.token_steps,
                clip=1.0,
                precision=config.mixed_precision,
            )
        )
        atomic_json(
            root / "token-split.json",
            dict(
                train_episodes=data.train_episodes,
                val_episodes=data.val_episodes,
                contract=policy.contract,
            ),
        )
        best = float("inf")
        started = time.monotonic()
        with (root / "token-metrics.jsonl").open("a", encoding="utf-8") as log:
            for step in range(1, config.token_steps + 1):
                indices = rng.choice(len(data.training), config.token_batch_size)
                features, mask = data.features([data.training[i] for i in indices])
                loss = update({"features": features, "mask": mask})
                if step == 1 or step % config.save_freq == 0 or step == config.token_steps:
                    metrics = dict(
                        step=step,
                        loss=float(loss),
                        **validation.evaluate(
                            encoder, decoder, config.device, config.token_batch_size
                        ),
                        elapsed_s=time.monotonic() - started,
                    )
                    val = metrics["validation_loss"]
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    print(
                        f"[RLT token] step={step} loss={metrics['loss']:.6g} val={val:.6g}",
                        flush=True,
                    )
                    saved = dict(
                        format="alohamini-rlt-token-v1",
                        contract=policy.contract,
                        step=step,
                        validation_loss=val,
                        config=config.to_dict(),
                    )
                    save_stage_checkpoint(
                        root,
                        step,
                        models,
                        saved,
                        config.to_dict(),
                        aliases=("token",) if val < best else (),
                    )
                    best = min(best, val)
