from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Convolutional encoder used by the retained staged models."""

    def __init__(self, latent_dim: int, channels: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(output_channels),
                ]
            )
            input_channels = output_channels
        self.encoder = nn.Sequential(*layers)
        downsample = 2 ** len(channels)
        flattened = (128 // downsample) ** 2 * channels[-1]
        self.fc = nn.Linear(flattened, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(image)
        return self.fc(torch.flatten(encoded, start_dim=1))


class ConvDecoder(nn.Module):
    """Included so complete autoencoder checkpoints load without key filtering."""

    def __init__(self, latent_dim: int, channels: Sequence[int], output_activation: str):
        super().__init__()
        downsample = 2 ** len(channels)
        bottleneck_size = 128 // downsample
        self.channels = tuple(channels)
        self.bottleneck_size = bottleneck_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels[-1] * bottleneck_size * bottleneck_size),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        reversed_channels = list(channels[::-1])
        for index in range(len(reversed_channels) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        reversed_channels[index], reversed_channels[index + 1], 3, 2, 1, 1
                    ),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(reversed_channels[index + 1]),
                ]
            )
        layers.append(nn.ConvTranspose2d(reversed_channels[-1], 1, 3, 2, 1, 1))
        layers.append(nn.Identity() if output_activation == "identity" else nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.fc(latent)
        decoded = decoded.view(
            -1, self.channels[-1], self.bottleneck_size, self.bottleneck_size
        )
        return self.decoder(decoded)


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: Sequence[int] = (16, 32, 64),
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, channels)
        self.decoder = ConvDecoder(latent_dim, channels, output_activation)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.pool = nn.MaxPool1d(2, 2, ceil_mode=True)
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(values))


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(input_channels, input_channels // 2, 2, 2)
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        difference = skip.size(-1) - values.size(-1)
        if difference > 0:
            start = difference // 2
            skip = skip[..., start : start + values.size(-1)]
        elif difference < 0:
            difference = -difference
            start = difference // 2
            values = values[..., start : start + skip.size(-1)]
        return self.conv(torch.cat([skip, values], dim=1))


class UNet1D(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.inc = DoubleConv(1, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)
        self.outc = nn.Module()
        self.outc.conv = nn.Conv1d(base_channels, 3, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level1 = self.inc(values)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        values = self.up1(level4, level3)
        values = self.up2(values, level2)
        values = self.up3(values, level1)
        return self.outc.conv(values)


class StagedCoordinateDecoder(nn.Module):
    """Learned latent expansion followed by the retained 1D U-Net."""

    def __init__(
        self,
        latent_dim: int,
        atom_count: int,
        hidden_dim: int = 1024,
        expansion_layers: int = 5,
        dropout: float = 0.1,
        progressive: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dimension = latent_dim
        if progressive:
            raw_dimensions = np.geomspace(
                float(latent_dim), float(hidden_dim), num=expansion_layers, endpoint=True
            )
            hidden_dimensions = [
                max(8, int(round(value / 8.0) * 8)) for value in raw_dimensions[1:]
            ]
            for index in range(1, len(hidden_dimensions)):
                hidden_dimensions[index] = max(
                    hidden_dimensions[index], hidden_dimensions[index - 1] + 8
                )
        else:
            hidden_dimensions = [hidden_dim] * (expansion_layers - 1)
        for dimension in hidden_dimensions:
            layers.extend([nn.Linear(input_dimension, dimension), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dimension = dimension
        layers.append(nn.Linear(input_dimension, atom_count))
        self.expand = nn.Sequential(*layers)
        self.unet = UNet1D(base_channels=32)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(latent).unsqueeze(1)
        return self.unet(expanded).transpose(1, 2)


class StagedImageToCoordinates(nn.Module):
    def __init__(self, encoder: ConvAutoencoder, decoder: StagedCoordinateDecoder):
        super().__init__()
        self.encoder = encoder.encoder
        self.coordinate_decoder = decoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.coordinate_decoder(self.encoder(image))
