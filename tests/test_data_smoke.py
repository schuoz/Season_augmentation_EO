from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from data.dataset import PairedSeasonDataset


def _write_tif(path: Path, arr: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))


def test_paired_dataset_reads_patch(tmp_path: Path) -> None:
    src = tmp_path / "src.tif"
    tgt = tmp_path / "tgt.tif"
    arr = np.random.rand(4, 64, 64).astype(np.float32)
    _write_tif(src, arr)
    _write_tif(tgt, arr)

    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame([{"source_path": str(src), "target_path": str(tgt)}]).to_csv(
        csv_path, index=False
    )
    ds = PairedSeasonDataset(
        csv_path=str(csv_path),
        bands=[1, 2, 3, 4],
        patch_size=32,
        normalize_mean=[0.5, 0.5, 0.5, 0.5],
        normalize_std=[0.2, 0.2, 0.2, 0.2],
    )
    sample = ds[0]
    assert sample["source"].shape == (4, 32, 32)
    assert sample["target"].shape == (4, 32, 32)
