from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.io import normalize, read_geotiff


class PairedSeasonDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        bands: list[int],
        patch_size: int,
        normalize_mean: list[float],
        normalize_std: list[float],
        cloud_mask_value: float | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.bands = bands
        self.patch_size = patch_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.cloud_mask_value = cloud_mask_value

        required = {"source_path", "target_path"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in dataset CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _sample_patch(self, arr: np.ndarray) -> np.ndarray:
        _, h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image is smaller than patch size ({self.patch_size}): {arr.shape}"
            )
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        return arr[:, top : top + self.patch_size, left : left + self.patch_size]

    def _mask_clouds(self, arr: np.ndarray) -> np.ndarray:
        if self.cloud_mask_value is None:
            return arr
        arr = arr.copy()
        arr[arr == self.cloud_mask_value] = 0.0
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        src = read_geotiff(row["source_path"], self.bands).array
        tgt = read_geotiff(row["target_path"], self.bands).array

        src = self._mask_clouds(src)
        tgt = self._mask_clouds(tgt)

        src = self._sample_patch(src)
        tgt = self._sample_patch(tgt)

        src = normalize(src, self.normalize_mean, self.normalize_std)
        tgt = normalize(tgt, self.normalize_mean, self.normalize_std)

        sample = {
            "source": torch.from_numpy(src),
            "target": torch.from_numpy(tgt),
        }
        if "source_season" in row and "target_season" in row:
            sample["season_pair"] = f"{row['source_season']}_{row['target_season']}"
        return sample
