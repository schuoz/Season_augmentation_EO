from __future__ import annotations

import torch
from torch import nn


class SeasonalLoss(nn.Module):
    def __init__(self, lambda_recon: float, lambda_gan: float, lambda_param: float) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_gan = lambda_gan
        self.lambda_param = lambda_param
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def reconstruction(self, translated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(translated, target)

    def gan_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        labels = torch.ones_like(fake_logits)
        return self.bce(fake_logits, labels)

    def gan_discriminator(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        return self.bce(real_logits, real_labels) + self.bce(fake_logits, fake_labels)

    def parameter_consistency(
        self, predicted_params: torch.Tensor, reapplied_params: torch.Tensor
    ) -> torch.Tensor:
        return self.l1(predicted_params, reapplied_params)

    def generator_total(
        self, recon: torch.Tensor, gan_g: torch.Tensor, param_consistency: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.lambda_recon * recon
            + self.lambda_gan * gan_g
            + self.lambda_param * param_consistency
        )
