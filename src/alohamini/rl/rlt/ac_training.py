"""Train chunk Q functions and a residual Actor from scored RLT episodes."""

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from alohamini.datasets.experience.storage import RunLock
from alohamini.learning.precision import tf32_context
from alohamini.learning.stages import read_stage_weights, save_stage_checkpoint
from alohamini.learning.training_state import restore_rng
from alohamini.policies.rlt.networks import AC_FORMAT
from alohamini.policies.rlt.policy import RLTPolicy
from alohamini.rl.replay import UniformReplaySampler
from alohamini.rl.rlt.actor_critic import ActorCritic
from alohamini.rl.rlt.episodes import scored_episodes
from alohamini.rl.rlt.replay import materialize_episode
from alohamini.rl.trainer import RLTrainer


def split_episodes(rewards, fraction, seed):
    """Keep each observed outcome in training and hold out whole episodes where possible."""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for reward in (0, 1):
        group = [eid for eid in sorted(rewards) if rewards[eid] == reward]
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * fraction))) if group else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation


def _pack(episodes, ids):
    rows = [row for eid in ids for row in episodes[eid]]
    if not rows:
        return {}
    fields = (
        "z",
        "state",
        "ref",
        "bc_target",
        "action",
        "reward",
        "valid",
        "elapsed_steps",
        "reward_delay",
        "next_z",
        "next_state",
        "next_ref",
        "next_valid",
        "done",
    )
    result = {}
    for field in fields:
        values = np.stack([row[field] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {field} in AC replay")
        result[field] = torch.from_numpy(values)
    return result


def _batch(data, indices, device):
    return {key: value[indices].to(device) for key, value in data.items()}


@torch.no_grad()
def _evaluate(learner, data):
    if not data:
        return {}
    config = learner.config
    count = len(data["done"])
    loss_sum = bc_sum = valid_steps = 0.0
    for start in range(0, count, config.ac_batch_size):
        batch = _batch(data, slice(start, start + config.ac_batch_size), config.device)
        n = len(batch["done"])
        valid = float(batch["valid"].sum())
        loss_sum += n * float(learner.critic_loss(batch))
        bc_sum += valid * float(learner.actor_losses(batch)[0])
        valid_steps += valid
    return dict(validation_critic_loss=loss_sum / count, validation_bc_loss=bc_sum / valid_steps)


def train_ac(config, *, resume=False):
    if not config.dataset or not config.token_checkpoint:
        raise ValueError("RLT actor-critic training requires scored data and token_checkpoint")
    if config.replay_stride > config.chunk_size:
        raise ValueError("replay_stride must be at most chunk_size so terminal rewards are covered")
    dataset, root = Path(config.dataset).resolve(), Path(config.output_dir).resolve()
    if root == dataset:
        raise ValueError("Use a separate AC output directory")
    candidates = scored_episodes(dataset)
    with RunLock(root / "ac.lock"), tf32_context(config.tf32):
        checkpoint = root / "actor"
        last = root / "checkpoints/last"
        if config.mixed_precision != "none":
            raise ValueError("Actor-critic updates currently require FP32")
        if not resume and (checkpoint.exists() or last.exists() or (root / "ac.pt").exists()):
            raise FileExistsError(f"AC weights already exist in {root}; use a new output directory")
        torch.manual_seed(config.seed)
        policy = RLTPolicy(replace(config, actor_checkpoint="", gate_checkpoint=""))
        contract, token_sha = policy.contract, policy.token_sha256
        replay, metadata = {}, {}
        for eid, reward, path in candidates:
            # Flow sampling for one episode stays reproducible when the dataset grows.
            seed = hashlib.sha256(f"{config.seed}/{eid}".encode()).digest()[:8]
            torch.manual_seed(int.from_bytes(seed, "big"))
            policy.native.reset()
            windows = materialize_episode(policy, path)
            if windows:
                replay[eid] = windows
                metadata[eid] = dict(reward=reward, windows=len(windows))
            print(f"AC replay {eid} reward={reward} windows={len(windows)}", flush=True)
        del policy
        if not replay:
            raise ValueError("No acknowledged actions in the selected training episodes")
        train_ids, val_ids = split_episodes(
            {eid: meta["reward"] for eid, meta in metadata.items()},
            config.validation_fraction,
            config.seed,
        )
        train, val = _pack(replay, train_ids), _pack(replay, val_ids)
        del replay
        split = dict(train=train_ids, validation=val_ids, episodes=metadata)
        print(
            f"AC train {len(train_ids)} episodes / {len(train['done'])} windows; "
            f"validation {len(val_ids)} episodes / {len(val.get('done', []))} windows",
            flush=True,
        )
        torch.manual_seed(config.seed)
        sampler = UniformReplaySampler(
            len(train["done"]), partial(_batch, train, device=config.device), seed=config.seed
        )
        learner = ActorCritic(config, contract)
        trainer = RLTrainer(learner, sampler, batch_size=config.ac_batch_size)
        models = {"actor": learner.actor, "critic": learner.critic}
        signature = hashlib.sha256()
        for partition in (train, val):
            for key, value in sorted(partition.items()):
                signature.update(key.encode())
                signature.update(str((value.dtype, value.shape)).encode())
                signature.update(memoryview(value.contiguous().numpy()).cast("B"))
        settings = config.to_dict()
        identity = {
            k: settings[k]
            for k in (
                "seed",
                "lr",
                "ac_batch_size",
                "ac_gamma",
                "ac_tau",
                "ac_bc_weight",
                "ac_residual_limit",
                "replay_stride",
                "validation_fraction",
                "mixed_precision",
                "tf32",
            )
        }
        start = 0
        if resume:
            path = last.resolve(strict=True)
            saved, _ = read_stage_weights(path / "pretrained_model")
            state = torch.load(
                path / "training_state/state.pt", map_location="cpu", weights_only=True
            )
            if (
                saved["contract"] != contract
                or saved["token_sha256"] != token_sha
                or state["identity"] != identity
                or saved["split"] != split
                or state["replay_sha256"] != signature.hexdigest()
            ):
                raise ValueError(
                    "Resume requires the original replay, base, token and training settings"
                )
            start = state["step"]
            if config.ac_steps <= start:
                raise ValueError("Resume requires ac_steps greater than the saved step")
            for name, module in models.items():
                module.load_state_dict(saved[name])
                getattr(learner, "target_" + name).load_state_dict(state["target_" + name])
                getattr(learner, name + "_optimizer").load_state_dict(state[name + "_optimizer"])
            learner.updates, learner.actor_updates = start, saved["actor_updates"]
            learner.sampled_terminals = saved["metrics"]["sampled_terminals"]
            learner.sampled_successes = saved["metrics"]["sampled_successes"]
            sampler.set_rng_state(state["sampler_rng"])
            restore_rng(state["rng"])
        log_path = root / "ac-metrics.jsonl"
        if resume:
            with log_path.open("rb+") as stream:
                stream.truncate(state["log_bytes"])
        with log_path.open("a" if resume else "w", encoding="utf-8") as log:

            def log_metrics(step, metrics):
                log.write(json.dumps(dict(step=step, **metrics), allow_nan=False) + "\n")
                if step == 1 or step % 100 == 0 or step == config.ac_steps:
                    log.flush()
                    print(f"AC update {step} {metrics}", flush=True)

            def save(step, metrics):
                log.flush()
                training = dict(
                    identity=identity,
                    replay_sha256=signature.hexdigest(),
                    log_bytes=log.buffer.tell(),
                    sampler_rng=sampler.get_rng_state(),
                )
                for name in models:
                    training["target_" + name] = getattr(learner, "target_" + name).state_dict()
                    training[name + "_optimizer"] = getattr(
                        learner, name + "_optimizer"
                    ).state_dict()
                save_stage_checkpoint(
                    root,
                    step,
                    models,
                    dict(
                        format=AC_FORMAT,
                        contract=contract,
                        token_sha256=token_sha,
                        residual_limit=config.ac_residual_limit,
                        actor_updates=learner.actor_updates,
                        critic_updates=learner.updates,
                        split=split,
                        metrics=metrics,
                    ),
                    settings,
                    training=training,
                    aliases=("actor",),
                )

            trainer.fit(
                config.ac_steps,
                start_step=start,
                evaluate=partial(_evaluate, learner, val),
                on_step=log_metrics,
                save_checkpoint=save,
                save_freq=config.save_freq,
            )
        return checkpoint
