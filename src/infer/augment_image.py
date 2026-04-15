from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.io import denormalize, normalize, read_geotiff, write_geotiff
from models.generator import SeasonalGenerator
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--mode", choices=["translator"], default="translator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sample = read_geotiff(cfg["input_path"], cfg["bands"])
    x = normalize(sample.array, cfg["normalize_mean"], cfg["normalize_std"])
    x_t = torch.from_numpy(x).unsqueeze(0).float()

    model = SeasonalGenerator(
        in_channels=len(cfg["bands"]),
        base_channels=64,
        aug_param_dim=cfg["aug_param_dim"],
    )
    state = torch.load(cfg["checkpoint_generator"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        translated, _ = model(x_t)
    out = translated.squeeze(0).cpu().numpy()
    out = denormalize(out, cfg["normalize_mean"], cfg["normalize_std"]).astype(sample.array.dtype)

    Path(cfg["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(cfg["output_path"], out, sample.profile)


if __name__ == "__main__":
    main()
