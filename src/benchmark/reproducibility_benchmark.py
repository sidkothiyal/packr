"""Similarity benchmark for compression engines."""

from typing import Dict, Any, Union
from pathlib import Path
from PIL import Image

import numpy as np

from benchmark.base import Benchmark


class ReproducibilityBenchmark(Benchmark):
    """Benchmark focused on measuring similarity between original and decoded images.

    This benchmark calculates similarity metrics including:
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Perfect reconstruction detection
    """

    def __init__(self, name: str = None):
        """Initialize the ReproducibilityBenchmark.

        Args:
            name: Optional name for the benchmark. If None, uses 'ReproducibilityBenchmark'.
        """
        super().__init__(name)

    def _calculate_mse(self, original: np.ndarray, decoded: np.ndarray) -> float:
        """Calculate Mean Squared Error between two images.

        Args:
            original: Original image array.
            decoded: Decoded image array.

        Returns:
            MSE value.
        """
        if original.shape != decoded.shape:
            return float("inf")

        # Convert to float for precision
        original_float = original.astype(np.float64)
        decoded_float = decoded.astype(np.float64)

        mse = np.mean((original_float - decoded_float) ** 2)
        return float(mse)

    def _calculate_psnr(self, mse: float, max_pixel_value: float = 255.0) -> float:
        """Calculate Peak Signal-to-Noise Ratio from MSE.

        Args:
            mse: Mean Squared Error value.
            max_pixel_value: Maximum possible pixel value.

        Returns:
            PSNR value in dB.
        """
        if mse == 0:
            return float("inf")  # Perfect reconstruction
        if mse == float("inf"):
            return 0.0

        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return float(psnr)

    def __call__(
        self,
        original_image: Union[np.ndarray, Path, Image],
        decoded_image: Union[np.ndarray, Path, Image],
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        """Calculate similarity metrics between original and decoded images.

        Args:
            original_image: Original image data.
            decoded_image: Decoded/reconstructed image data.
            compressed_data: Compressed representation from the compression engine (unused).
            encoding_time: Time taken to encode the image (unused).
            decoding_time: Time taken to decode the image (unused).

        Returns:
            Dict containing similarity metrics.
        """
        # Convert images to arrays for numerical comparison
        original_array = self._image_to_array(original_image)
        decoded_array = self._image_to_array(decoded_image)

        # Calculate MSE
        mse = self._calculate_mse(original_array, decoded_array)

        # Determine max pixel value based on data type
        if original_array.dtype in [np.uint8]:
            max_pixel_value = 255.0
        elif original_array.dtype in [np.uint16]:
            max_pixel_value = 65535.0
        elif original_array.dtype in [np.float32, np.float64]:
            max_pixel_value = 1.0 if original_array.max() <= 1.0 else 255.0
        else:
            max_pixel_value = 255.0  # Default assumption

        # Calculate PSNR
        psnr = self._calculate_psnr(mse, max_pixel_value)

        # Check for perfect reconstruction
        perfect_reconstruction = mse == 0.0

        # Calculate additional similarity metrics
        if original_array.shape == decoded_array.shape:
            # Calculate normalized metrics
            original_flat = original_array.flatten().astype(np.float64)
            decoded_flat = decoded_array.flatten().astype(np.float64)

            # Root Mean Square Error
            rmse = np.sqrt(mse)

            # Mean Absolute Error
            mae = np.mean(np.abs(original_flat - decoded_flat))

            # Structural Similarity (simplified)
            mean_orig = np.mean(original_flat)
            mean_dec = np.mean(decoded_flat)
            var_orig = np.var(original_flat)
            var_dec = np.var(decoded_flat)
            covar = np.mean((original_flat - mean_orig) * (decoded_flat - mean_dec))

            # Constants for SSIM calculation
            c1 = (0.01 * max_pixel_value) ** 2
            c2 = (0.03 * max_pixel_value) ** 2

            ssim = ((2 * mean_orig * mean_dec + c1) * (2 * covar + c2)) / (
                (mean_orig**2 + mean_dec**2 + c1) * (var_orig + var_dec + c2)
            )
        else:
            rmse = float("inf")
            mae = float("inf")
            ssim = 0.0

        result = {
            "mse": mse,
            "psnr": psnr,
            "rmse": rmse,
            "mae": mae,
            "ssim": ssim,
            "perfect_reconstruction": perfect_reconstruction,
            "max_pixel_value": max_pixel_value,
            "original_shape": original_array.shape,
            "decoded_shape": decoded_array.shape,
        }

        # Store results for summary statistics
        self._add_result(result)

        return result

    def summarize(self) -> Dict[str, Any]:
        """Get summary statistics for all similarity measurements.

        Returns:
            Dict containing summary statistics for similarity metrics.
        """
        if not self._results:
            return {"error": "No results available for summary"}

        # Overall statistics
        # Perfect reconstruction statistics
        perfect_reconstructions = sum(self._results["perfect_reconstruction"])
        perfect_reconstruction_rate = perfect_reconstructions / len(
            self._results["perfect_reconstruction"]
        )

        # Collect metrics for statistical analysis
        mse_values = self._results["mse"]
        psnr_values = self._results["psnr"]
        rmse_values = self._results["rmse"]
        mae_values = self._results["mae"]
        ssim_values = self._results["ssim"]

        def _stats(values, name):
            """Calculate basic statistics for a list of values."""
            if not values:
                return {
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "std": 0,
                    "count": 0,
                    "median": 0,
                }
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
            "perfect_reconstruction_rate": perfect_reconstruction_rate,
            "perfect_reconstructions": perfect_reconstructions,
            "mse": _stats(mse_values, "mse"),
            "psnr": _stats(psnr_values, "psnr"),
            "rmse": _stats(rmse_values, "rmse"),
            "mae": _stats(mae_values, "mae"),
            "ssim": _stats(ssim_values, "ssim"),
            "shape_mismatches": sum(
                1
                for res_idx in range(len(self._results["original_shape"]))
                if self._results["original_shape"][res_idx]
                != self._results["decoded_shape"][res_idx]
            ),
        }

        # Add quality assessment
        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            if avg_psnr >= 40:
                quality_assessment = "excellent"
            elif avg_psnr >= 30:
                quality_assessment = "good"
            elif avg_psnr >= 20:
                quality_assessment = "fair"
            else:
                quality_assessment = "poor"

            summary["quality_assessment"] = quality_assessment
            summary["average_psnr"] = avg_psnr

        return summary
