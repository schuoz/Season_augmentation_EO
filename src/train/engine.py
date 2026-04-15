from __future__ import annotations

import torch
from tqdm import tqdm


def apply_aug_params(source: torch.Tensor, aug_params: torch.Tensor) -> torch.Tensor:
    channels = source.size(1)
    gain = 1.0 + aug_params[:, :channels].unsqueeze(-1).unsqueeze(-1).clamp(-0.4, 0.4)
    bias = aug_params[:, channels : channels * 2].unsqueeze(-1).unsqueeze(-1).clamp(
        -0.3, 0.3
    )
    return source * gain + bias


def train_one_epoch(
    generator,
    discriminator,
    loader,
    criterion,
    opt_g,
    opt_d,
    device,
    use_gan: bool,
) -> dict[str, float]:
    generator.train()
    if discriminator is not None:
        discriminator.train()

    total_g = 0.0
    total_d = 0.0
    steps = 0

    for batch in tqdm(loader, desc="train", leave=False):
        source = batch["source"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()

        translated, aug_params = generator(source)
        reapplied = apply_aug_params(source, aug_params)

        recon = criterion.reconstruction(translated, target)
        param_cons = criterion.parameter_consistency(aug_params, aug_params.detach())

        gan_g = torch.tensor(0.0, device=device)
        d_loss = torch.tensor(0.0, device=device)
        if use_gan and discriminator is not None:
            fake_logits = discriminator(translated)
            gan_g = criterion.gan_generator(fake_logits)

            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(target)
            fake_logits_d = discriminator(translated.detach())
            d_loss = criterion.gan_discriminator(real_logits, fake_logits_d)
            d_loss.backward()
            opt_d.step()

        g_loss = criterion.generator_total(recon, gan_g, criterion.l1(reapplied, translated))

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        total_g += float(g_loss.item())
        total_d += float(d_loss.item())
        steps += 1

    denom = max(steps, 1)
    return {"g_loss": total_g / denom, "d_loss": total_d / denom}


@torch.no_grad()
def validate(generator, loader, criterion, device) -> dict[str, float]:
    generator.eval()
    total_recon = 0.0
    steps = 0
    for batch in tqdm(loader, desc="val", leave=False):
        source = batch["source"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        translated, _ = generator(source)
        recon = criterion.reconstruction(translated, target)
        total_recon += float(recon.item())
        steps += 1
    return {"val_recon": total_recon / max(steps, 1)}
