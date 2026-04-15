from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True) -> None:
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeasonalGenerator(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, aug_param_dim: int) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels, down=True)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, down=True)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, down=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, down=False)
        self.dec2 = ConvBlock(base_channels * 8, base_channels * 2, down=False)
        self.dec1 = ConvBlock(base_channels * 4, base_channels, down=False)
        self.to_image = nn.Conv2d(base_channels * 2, in_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_params = nn.Linear(base_channels * 8, aug_param_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.bottleneck(e3)

        d3 = self.dec3(bottleneck)
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        translated = torch.tanh(self.to_image(torch.cat([d1, e1], dim=1)))

        params = self.pool(bottleneck).flatten(1)
        aug_params = self.to_params(params)
        return translated, aug_params
