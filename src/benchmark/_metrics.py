"""Pure image-similarity metrics shared across benchmarks."""

from typing import Tuple

import numpy as np


def mse(original: np.ndarray, decoded: np.ndarray) -> float:
    """Mean squared error. ``inf`` if shapes differ."""
    if original.shape != decoded.shape:
        return float("inf")
    diff = original.astype(np.float64) - decoded.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(mse_value: float, max_pixel_value: float = 255.0) -> float:
    if mse_value == 0:
        return float("inf")
    if mse_value == float("inf"):
        return 0.0
    return float(20 * np.log10(max_pixel_value / np.sqrt(mse_value)))


def ssim_global(
    original: np.ndarray, decoded: np.ndarray, max_pixel_value: float = 255.0
) -> float:
    """Global (image-wide, non-windowed) SSIM — cheap approximation.

    Not the same as Wang et al. window-based SSIM; kept here to preserve the
    behavior the original ``ReproducibilityBenchmark`` already relied on.
    """
    if original.shape != decoded.shape:
        return 0.0
    a = original.astype(np.float64).flatten()
    b = decoded.astype(np.float64).flatten()
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a)), float(np.var(b))
    covar = float(np.mean((a - mean_a) * (b - mean_b)))
    c1 = (0.01 * max_pixel_value) ** 2
    c2 = (0.03 * max_pixel_value) ** 2
    num = (2 * mean_a * mean_b + c1) * (2 * covar + c2)
    den = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den)


def max_pixel_value_for(arr: np.ndarray) -> float:
    if arr.dtype == np.uint8:
        return 255.0
    if arr.dtype == np.uint16:
        return 65535.0
    if arr.dtype in (np.float32, np.float64):
        return 1.0 if arr.max() <= 1.0 else 255.0
    return 255.0


def all_metrics(
    original: np.ndarray, decoded: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """Return (mse, psnr, rmse, mae, ssim, max_pixel_value)."""
    max_pixel = max_pixel_value_for(original)
    m = mse(original, decoded)
    p = psnr(m, max_pixel)
    if original.shape == decoded.shape:
        rmse = float(np.sqrt(m))
        mae = float(
            np.mean(np.abs(original.astype(np.float64) - decoded.astype(np.float64)))
        )
        s = ssim_global(original, decoded, max_pixel)
    else:
        rmse = float("inf")
        mae = float("inf")
        s = 0.0
    return m, p, rmse, mae, s, max_pixel
