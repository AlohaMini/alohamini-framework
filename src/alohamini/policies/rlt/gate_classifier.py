"""Binary stage head on a frozen token and normalized left-arm proprioception."""

import torch
from torch import nn

GATE_FORMAT = "alohamini-rlt-stage-gate-v1"


class StageClassifier(nn.Module):
    def __init__(self, contract, mode="token_state"):
        super().__init__()
        if mode not in ("token_state", "token_only", "state_only"):
            raise ValueError("Unsupported gate feature selection")
        self.mode = mode
        token_dim = contract["token_spec"]["dim"]
        state_dim = len(contract["learned_names"])
        self.token = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 64), nn.ReLU())
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.head = nn.Linear(128 if mode == "token_state" else 64, 1)

    def forward(self, token, state):
        if self.mode == "state_only":
            x = self.state(state)
        elif self.mode == "token_only":
            x = self.token(token)
        else:
            x = torch.cat((self.token(token), self.state(state)), dim=-1)
        return self.head(nn.functional.dropout(x, p=0.1, training=self.training)).squeeze(-1)
