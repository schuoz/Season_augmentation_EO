from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    generator,
    discriminator,
    opt_g,
    opt_d,
    best: bool = False,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
    }
    torch.save(payload, path / "model_full.pt")
    if best:
        torch.save(generator.state_dict(), path / "generator_best.pt")


def save_param_head(checkpoint_dir: str, state_dict: dict) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path / "param_head_best.pt")
