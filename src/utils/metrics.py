from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=2.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c, _, _ = pred.shape
    scores = []
    for band in range(c):
        scores.append(
            structural_similarity(target[band], pred[band], data_range=2.0)
        )
    return float(np.mean(scores))
