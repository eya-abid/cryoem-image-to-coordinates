from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        hidden = self.dropout(hidden)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + self.skip(inputs), inplace=True)


class StrongCNNToCoordinates(nn.Module):
    """Shared residual architecture instantiated with each retained system's atom count."""

    def __init__(
        self,
        atom_count: int = 1489,
        hidden_dimension: int = 2048,
        mlp_layers: int = 4,
        dropout: float = 0.15,
        bottleneck_dimension: int = 512,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            ResidualConvBlock(32, 64, stride=2, dropout=dropout * 0.5),
            ResidualConvBlock(64, 128, stride=2, dropout=dropout * 0.5),
            ResidualConvBlock(128, 256, stride=2, dropout=dropout),
            ResidualConvBlock(256, 256, stride=1, dropout=dropout),
            ResidualConvBlock(256, 512, stride=2, dropout=dropout),
            ResidualConvBlock(512, 512, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = (
            nn.Identity()
            if bottleneck_dimension == 512
            else nn.Sequential(
                nn.Linear(512, bottleneck_dimension),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        )
        head: list[nn.Module] = []
        input_dimension = bottleneck_dimension
        for _ in range(max(mlp_layers - 1, 1)):
            head.extend(
                [
                    nn.Linear(input_dimension, hidden_dimension),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            input_dimension = hidden_dimension
        head.append(nn.Linear(input_dimension, atom_count * 3))
        self.head = nn.Sequential(*head)
        self.atom_count = atom_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(inputs)
        hidden = self.layers(hidden)
        hidden = self.pool(hidden).flatten(1)
        hidden = self.bottleneck(hidden)
        output = self.head(hidden)
        return output.view(inputs.size(0), self.atom_count, 3)
