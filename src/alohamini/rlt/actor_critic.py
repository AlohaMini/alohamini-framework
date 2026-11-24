"""Offline chunk actor and twin Q functions in the frozen policy's action coordinates."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

AC_FORMAT = "alohamini-rlt-residual-ac-v1"


def _mlp(inputs, outputs):
    return nn.Sequential(
        nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, outputs)
    )


class ChunkActor(nn.Module):
    def __init__(self, contract, residual_limit):
        super().__init__()
        axes = len(contract["learned_names"])
        size = contract["chunk_size"] * axes
        self.net = _mlp(contract["token_spec"]["dim"] + axes + size, size)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.residual_limit = residual_limit

    def forward(self, token, state, reference):
        inputs = torch.cat((token, state, reference.flatten(1)), dim=1)
        residual = self.net(inputs).reshape_as(reference).tanh()
        return reference + self.residual_limit * residual


class TwinCritic(nn.Module):
    def __init__(self, contract):
        super().__init__()
        axes = len(contract["learned_names"])
        size = contract["token_spec"]["dim"] + axes + contract["chunk_size"] * (axes + 1)
        self.q1 = _mlp(size, 1)
        self.q2 = _mlp(size, 1)

    def forward(self, token, state, actions, valid):
        x = torch.cat((token, state, (actions * valid[..., None]).flatten(1), valid), dim=1)
        return self.q1(x), self.q2(x)


class ActorCritic:
    def __init__(self, config, contract):
        self.config = config
        self.actor = ChunkActor(contract, config.ac_residual_limit).to(config.device)
        self.critic = TwinCritic(contract).to(config.device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.updates = self.actor_updates = 0

    @torch.no_grad()
    def target(self, batch):
        gamma = self.config.ac_gamma
        valid = batch["valid"]
        offsets = torch.arange(valid.shape[1], device=valid.device)
        reward = (batch["reward"] * valid * gamma**offsets).sum(1, keepdim=True)
        reward = reward * gamma ** batch["reward_delay"]
        action = self.target_actor(batch["next_z"], batch["next_state"], batch["next_ref"])
        q1, q2 = self.target_critic(
            batch["next_z"], batch["next_state"], action, batch["next_valid"]
        )
        bootstrap = gamma ** batch["elapsed_steps"] * torch.minimum(q1, q2)
        return reward + torch.where(batch["done"].bool(), 0.0, bootstrap)

    def critic_loss(self, batch):
        target = self.target(batch)
        q1, q2 = self.critic(batch["z"], batch["state"], batch["action"], batch["valid"])
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def actor_losses(self, batch):
        actions = self.actor(batch["z"], batch["state"], batch["ref"])
        valid = batch["valid"]
        bc = ((actions - batch["bc_target"]).square().mean(-1) * valid).sum() / valid.sum()
        q1, q2 = self.critic(batch["z"], batch["state"], actions, valid)
        return bc, torch.minimum(q1, q2).mean()

    def update(self, batch):
        loss = self.critic_loss(batch)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10, error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.updates += 1
        metrics = {"critic_loss": float(loss.detach())}
        if self.updates % 2 == 1:
            self.critic.requires_grad_(False)
            try:
                bc, q = self.actor_losses(batch)
                objective = self.config.ac_bc_weight * bc - q
                self.actor_optimizer.zero_grad(set_to_none=True)
                objective.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10, error_if_nonfinite=True)
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(True)
            self.actor_updates += 1
            with torch.no_grad():
                for online, target in (
                    (self.actor, self.target_actor),
                    (self.critic, self.target_critic),
                ):
                    for src, dst in zip(online.parameters(), target.parameters(), strict=True):
                        dst.lerp_(src, self.config.ac_tau)
            metrics.update(
                actor_loss=float(objective.detach()),
                bc_loss=float(bc.detach()),
                actor_q=float(q.detach()),
            )
        return metrics
