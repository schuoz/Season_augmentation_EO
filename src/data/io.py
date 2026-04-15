from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio


@dataclass
class RasterSample:
    array: np.ndarray
    profile: dict


def read_geotiff(path: str, bands: list[int]) -> RasterSample:
    with rasterio.open(path) as src:
        array = src.read(indexes=bands).astype(np.float32)
        profile = src.profile.copy()
    return RasterSample(array=array, profile=profile)


def normalize(array: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)[:, None, None]
    std_arr = np.asarray(std, dtype=np.float32)[:, None, None]
    return (array - mean_arr) / np.clip(std_arr, 1e-6, None)


def denormalize(array: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)[:, None, None]
    std_arr = np.asarray(std, dtype=np.float32)[:, None, None]
    return array * std_arr + mean_arr


def write_geotiff(path: str, array: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=int(array.shape[0]),
        dtype=str(array.dtype),
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array)
