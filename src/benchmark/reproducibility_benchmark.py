"""Similarity benchmark for compression engines."""

from typing import Any, Dict, Optional

import numpy as np

from benchmark._metrics import all_metrics
from benchmark.base import Benchmark


class ReproducibilityBenchmark(Benchmark):
    """Measures similarity between original and decoded images.

    Metrics: MSE, PSNR, RMSE, MAE, a cheap global SSIM, and a
    perfect-reconstruction flag.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(
        self,
        original_image,
        decoded_image,
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        original = self._image_to_array(original_image)
        decoded = self._image_to_array(decoded_image)

        m, p, rmse, mae, s, max_pixel = all_metrics(original, decoded)

        result = {
            "mse": m,
            "psnr": p,
            "rmse": rmse,
            "mae": mae,
            "ssim": s,
            "perfect_reconstruction": m == 0.0,
            "max_pixel_value": max_pixel,
            "original_shape": original.shape,
            "decoded_shape": decoded.shape,
        }
        self._add_result(result)
        return result

    def summarize(self) -> Dict[str, Any]:
        if not self._results:
            return {"error": "No results available for summary"}

        perfect_list = self._results["perfect_reconstruction"]
        perfect_count = sum(perfect_list)
        perfect_rate = perfect_count / len(perfect_list)

        def _stats(values):
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "std": 0, "median": 0, "count": 0}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "count": len(values),
            }

        summary = {
            "perfect_reconstruction_rate": perfect_rate,
            "perfect_reconstructions": perfect_count,
            "mse": _stats(self._results["mse"]),
            "psnr": _stats(self._results["psnr"]),
            "rmse": _stats(self._results["rmse"]),
            "mae": _stats(self._results["mae"]),
            "ssim": _stats(self._results["ssim"]),
            "shape_mismatches": sum(
                1
                for i in range(len(self._results["original_shape"]))
                if self._results["original_shape"][i] != self._results["decoded_shape"][i]
            ),
        }

        psnr_values = self._results["psnr"]
        if psnr_values:
            avg_psnr = float(np.mean(psnr_values))
            if avg_psnr >= 40:
                quality = "excellent"
            elif avg_psnr >= 30:
                quality = "good"
            elif avg_psnr >= 20:
                quality = "fair"
            else:
                quality = "poor"
            summary["quality_assessment"] = quality
            summary["average_psnr"] = avg_psnr

        return summary
