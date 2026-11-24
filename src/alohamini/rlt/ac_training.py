"""Train chunk Q functions and a residual Actor from scored RLT episodes."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from alohamini.rlt.actor_critic import AC_FORMAT, ActorCritic
from alohamini.rlt.policy import RLTPolicy
from alohamini.rlt.replay import materialize_episode
from alohamini.rlt.storage import RunLock, atomic_torch


def scored_episodes(root):
    if (root / "episodes").is_dir():
        candidates = [(p.name, p) for p in sorted((root / "episodes").iterdir()) if p.is_dir()]
    else:
        if (root / "lerobot_v3/meta/info.json").is_file():
            root = root / "lerobot_v3"
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            raise ValueError(f"No RLT episodes or published LeRobot v3 metadata in {root}")
        info = json.loads(info_path.read_bytes())
        ids = info.get("alohamini_rlt", {}).get("episode_ids", [])
        if len(ids) != info["total_episodes"] or len(set(ids)) != len(ids):
            raise ValueError("Invalid RLT episode index in v3 metadata")
        candidates = [(eid, root / "meta/rlt" / f"episode_{i:06d}") for i, eid in enumerate(ids)]
    result = []
    for eid, path in candidates:
        summary = json.loads((path / "episode.json").read_bytes())
        if summary["episode_id"] != eid:
            raise ValueError(f"Episode ID differs from its dataset index in {path}")
        if (
            summary.get("status") != "committed"
            or not summary.get("ended")
            or not summary.get("training_valid", True)
            or summary.get("evaluation")
            or summary.get("writer_error")
        ):
            continue
        reward = summary.get("reward")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError(f"Invalid operator reward in {path}")
        result.append((eid, reward, path))
    if not result:
        raise ValueError("No completed, scored, training-valid RLT episodes")
    return result


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


def train_ac(config):
    if not config.dataset or not config.token_checkpoint:
        raise ValueError("train-ac requires a scored RLT dataset and a trained token_checkpoint")
    if config.replay_stride > config.chunk_size:
        raise ValueError("replay_stride must be at most chunk_size so terminal rewards are covered")
    dataset, root = Path(config.dataset).resolve(), Path(config.output_dir).resolve()
    if root == dataset:
        raise ValueError("Use a separate AC output directory")
    candidates = scored_episodes(dataset)
    with RunLock(root / "ac.lock"):
        checkpoint = root / "ac.pt"
        if checkpoint.exists():
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
        rng = torch.Generator().manual_seed(config.seed)
        learner = ActorCritic(config, contract)
        sampled_terminals = sampled_successes = 0
        with (root / "ac-metrics.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, config.ac_steps + 1):
                indices = torch.randint(len(train["done"]), (config.ac_batch_size,), generator=rng)
                metrics = learner.update(_batch(train, indices, config.device))
                sampled_terminals += int(train["done"][indices].sum())
                sampled_successes += int(train["reward"][indices].any(dim=1).sum())
                metrics.update(
                    sampled_terminals=sampled_terminals, sampled_successes=sampled_successes
                )
                if step == config.ac_steps:
                    metrics.update(_evaluate(learner, val))
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError(f"Non-finite AC metric at update {step}")
                log.write(json.dumps(dict(step=step, **metrics), allow_nan=False) + "\n")
                if step == 1 or step % 100 == 0 or step == config.ac_steps:
                    log.flush()
                    print(f"AC update {step} {metrics}", flush=True)
        atomic_torch(
            checkpoint,
            dict(
                format=AC_FORMAT,
                contract=contract,
                token_sha256=token_sha,
                residual_limit=config.ac_residual_limit,
                actor_updates=learner.actor_updates,
                critic_updates=learner.updates,
                actor=learner.actor.state_dict(),
                critic=learner.critic.state_dict(),
                split=split,
                metrics=metrics,
                config=config.to_dict(),
            ),
        )
        print(f"Saved {checkpoint}", flush=True)
        return checkpoint
