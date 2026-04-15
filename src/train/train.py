from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataset import PairedSeasonDataset
from data.sampler import build_dataloader
from models.discriminator import PatchDiscriminator
from models.losses import SeasonalLoss
from models.generator import SeasonalGenerator
from train.checkpoint import save_checkpoint, save_param_head
from train.engine import train_one_epoch, validate
from utils.config import load_config
from utils.distributed import cleanup_distributed, is_main_process, setup_distributed
from utils.logging import create_writer
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    distributed, rank, _ = setup_distributed(cfg["distributed"]["backend"])
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"].copy()
    train_csv = data_cfg.pop("train_csv")
    val_csv = data_cfg.pop("val_csv")
    train_ds = PairedSeasonDataset(csv_path=train_csv, **data_cfg)
    val_ds = PairedSeasonDataset(csv_path=val_csv, **data_cfg)

    train_loader, train_sampler = build_dataloader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        distributed=distributed,
        shuffle=True,
    )
    val_loader, _ = build_dataloader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        distributed=distributed,
        shuffle=False,
    )

    generator = SeasonalGenerator(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        aug_param_dim=cfg["model"]["aug_param_dim"],
    ).to(device)
    discriminator = None
    if cfg["model"]["use_gan"]:
        discriminator = PatchDiscriminator(cfg["model"]["in_channels"]).to(device)

    if distributed:
        generator = DDP(generator, device_ids=[local_rank])
        if discriminator is not None:
            discriminator = DDP(discriminator, device_ids=[local_rank])

    criterion = SeasonalLoss(
        lambda_recon=cfg["training"]["lambda_recon"],
        lambda_gan=cfg["training"]["lambda_gan"],
        lambda_param=cfg["training"]["lambda_param"],
    )
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["training"]["lr"], betas=(0.5, 0.999))
    opt_d = None
    if discriminator is not None:
        opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=cfg["training"]["lr"],
            betas=(0.5, 0.999),
        )

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    writer = create_writer(str(output_dir)) if is_main_process(rank) else None
    best_val = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            criterion=criterion,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
            use_gan=cfg["model"]["use_gan"],
        )
        val_stats = validate(generator=generator, loader=val_loader, criterion=criterion, device=device)

        if writer is not None:
            writer.add_scalar("train/g_loss", train_stats["g_loss"], epoch)
            writer.add_scalar("train/d_loss", train_stats["d_loss"], epoch)
            writer.add_scalar("val/recon", val_stats["val_recon"], epoch)

        if is_main_process(rank):
            val_score = val_stats["val_recon"]
            is_best = val_score < best_val
            if is_best:
                best_val = val_score
            save_checkpoint(
                checkpoint_dir=str(checkpoint_dir),
                epoch=epoch,
                generator=generator.module if isinstance(generator, DDP) else generator,
                discriminator=(
                    discriminator.module
                    if discriminator is not None and isinstance(discriminator, DDP)
                    else discriminator
                ),
                opt_g=opt_g,
                opt_d=opt_d,
                best=is_best,
            )
            gen_obj = generator.module if isinstance(generator, DDP) else generator
            save_param_head(str(checkpoint_dir), gen_obj.to_params.state_dict())

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
