from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values + self.net(values))


class ResidualTCN(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: int = 48,
        blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(int(input_features), int(channels), 1)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(int(channels), int(kernel_size), 2**index, float(dropout))
                for index in range(int(blocks))
            ]
        )
        self.output = nn.Conv1d(int(channels), 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = features.permute(0, 2, 1)
        return self.output(self.blocks(self.input(values))).squeeze(1)

