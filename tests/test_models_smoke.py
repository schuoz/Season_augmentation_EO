from __future__ import annotations

import torch

from models.discriminator import PatchDiscriminator
from models.generator import SeasonalGenerator
from models.losses import SeasonalLoss
from train.engine import apply_aug_params


def test_generator_and_discriminator_shapes() -> None:
    model = SeasonalGenerator(in_channels=4, base_channels=32, aug_param_dim=8)
    disc = PatchDiscriminator(in_channels=4, base_channels=32)
    x = torch.randn(2, 4, 128, 128)
    translated, params = model(x)
    logits = disc(translated)
    assert translated.shape == x.shape
    assert params.shape == (2, 8)
    assert logits.shape[0] == 2


def test_loss_and_param_apply() -> None:
    criterion = SeasonalLoss(10.0, 1.0, 2.0)
    src = torch.randn(2, 4, 64, 64)
    tgt = torch.randn(2, 4, 64, 64)
    pred = torch.randn(2, 4, 64, 64)
    params = torch.randn(2, 8)
    transformed = apply_aug_params(src, params)
    assert transformed.shape == src.shape
    recon = criterion.reconstruction(pred, tgt)
    assert recon.item() >= 0
