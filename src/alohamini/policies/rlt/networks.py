"""RL-token encoder and causal feature-reconstruction decoder."""

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
