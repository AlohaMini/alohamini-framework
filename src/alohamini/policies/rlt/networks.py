"""Networks used for RLT inference."""

import torch
from torch import nn


class RLTokenEncoder(nn.Module):
    def __init__(self, input_dim=2048, dim=2048, layers=4, heads=8, max_tokens=2048):
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.position = nn.Parameter(torch.randn(1, max_tokens + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )

    def forward(self, features, mask):
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError("Expected features [B,L,D] and a matching validity mask")
        length = features.shape[1] + 1
        if length > self.position.shape[1] or not mask.any(dim=1).all():
            raise ValueError("Token sequence is too long or has no valid image tokens")
        x = torch.cat(
            (self.projection(features.float()), self.token.expand(features.shape[0], -1, -1)), dim=1
        )
        valid = torch.cat((mask.bool(), torch.ones_like(mask[:, :1], dtype=torch.bool)), dim=1)
        return self.encoder(x + self.position[:, :length], src_key_padding_mask=~valid)[:, -1]


AC_FORMAT = "alohamini-rlt-residual-ac-v1"


def chunk_mlp(inputs, outputs):
    return nn.Sequential(
        nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, outputs)
    )


class ChunkActor(nn.Module):
    def __init__(self, contract, residual_limit):
        super().__init__()
        axes = len(contract["learned_names"])
        size = contract["chunk_size"] * axes
        self.net = chunk_mlp(contract["token_spec"]["dim"] + axes + size, size)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.residual_limit = residual_limit

    def forward(self, token, state, reference):
        inputs = torch.cat((token, state, reference.flatten(1)), dim=1)
        residual = self.net(inputs).reshape_as(reference).tanh()
        return reference + self.residual_limit * residual


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
