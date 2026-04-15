from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.io import denormalize, normalize, read_geotiff, write_geotiff
from models.generator import SeasonalGenerator
from train.engine import apply_aug_params
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
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
    gen_state = torch.load(cfg["checkpoint_generator"], map_location="cpu")
    model.load_state_dict(gen_state)
    model.eval()

    with torch.no_grad():
        _, aug_params = model(x_t)
        augmented = apply_aug_params(x_t, aug_params)

    out = augmented.squeeze(0).cpu().numpy()
    out = denormalize(out, cfg["normalize_mean"], cfg["normalize_std"]).astype(sample.array.dtype)
    Path(cfg["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(cfg["output_path"], out, sample.profile)

    param_path = Path(cfg["save_params_json"])
    param_path.parent.mkdir(parents=True, exist_ok=True)
    with param_path.open("w", encoding="utf-8") as f:
        json.dump({"aug_params": aug_params.squeeze(0).cpu().tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
