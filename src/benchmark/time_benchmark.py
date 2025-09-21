"""Time-focused benchmark for compression engines."""

from typing import Dict, Any, Union
from pathlib import Path
from PIL import Image

import numpy as np

from benchmark.base import Benchmark


class TimeBenchmark(Benchmark):
    """Benchmark focused on timing metrics for compression engines.

    This benchmark measures and tracks timing-related metrics including:
    - Encoding time
    - Decoding time
    - Total processing time
    - Throughput (pixels per second)
    """

    def __init__(self, name: str = None):
        """Initialize the TimeBenchmark.

        Args:
            name: Optional name for the benchmark. If None, uses 'TimeBenchmark'.
        """
        super().__init__(name)

    def __call__(
        self,
        original_image: Union[np.ndarray, Path, Image],
        decoded_image: Union[np.ndarray, Path, Image],
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        """Calculate timing metrics for a single compression test.

        Args:
            original_image: Original image data.
            decoded_image: Decoded/reconstructed image data.
            compressed_data: Compressed representation from the compression engine.
            encoding_time: Time taken to encode the image (in seconds).
            decoding_time: Time taken to decode the image (in seconds).

        Returns:
            Dict containing timing metrics.
        """
        # Get image dimensions for throughput calculation
        image_shape = self._get_image_size(original_image)
        total_pixels = self._calculate_pixels(image_shape)

        # Calculate metrics
        total_time = encoding_time + decoding_time
        encoding_throughput = total_pixels / encoding_time if encoding_time > 0 else float("inf")
        decoding_throughput = total_pixels / decoding_time if decoding_time > 0 else float("inf")
        overall_throughput = total_pixels / total_time if total_time > 0 else float("inf")

        # Calculate processing efficiency (lower is better)
        processing_efficiency = total_time / total_pixels if total_pixels > 0 else float("inf")

        metrics = {
            "encoding_time": encoding_time,
            "decoding_time": decoding_time,
            "total_time": total_time,
            "total_pixels": total_pixels,
            "encoding_throughput_pps": encoding_throughput,  # pixels per second
            "decoding_throughput_pps": decoding_throughput,
            "overall_throughput_pps": overall_throughput,
            "processing_efficiency_spp": processing_efficiency,  # seconds per pixel
            "encoding_time_ratio": encoding_time / total_time if total_time > 0 else 0.0,
            "decoding_time_ratio": decoding_time / total_time if total_time > 0 else 0.0,
        }

        # Store results for summary statistics
        self._add_result(metrics)

        return metrics

    def summarize(self) -> Dict[str, Any]:
        """Get summary statistics for all timing measurements.

        Returns:
            Dict containing summary statistics for timing metrics.
        """
        if not self._results:
            return {"error": "No results available for summary"}

        # Collect all metrics
        encoding_times = self._results["encoding_time"]
        decoding_times = self._results["decoding_time"]
        total_times = self._results["total_time"]
        throughputs = self._results["overall_throughput_pps"]
        encoding_throughputs = self._results["encoding_throughput_pps"]
        decoding_throughputs = self._results["decoding_throughput_pps"]

        def _stats(values):
            """Calculate basic statistics for a list of values."""
            if not values:
                return {"mean": 0, "min": 0, "max": 0, "std": 0}
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "count": len(values),
            }

        summary = {
            "total_tests": len(self._results),
            "encoding_time": _stats(encoding_times),
            "decoding_time": _stats(decoding_times),
            "total_time": _stats(total_times),
            "overall_throughput_pps": _stats(throughputs),
            "encoding_throughput_pps": _stats(encoding_throughputs),
            "decoding_throughput_pps": _stats(decoding_throughputs),
            "total_processing_time": sum(total_times),
            "average_encoding_ratio": (
                np.mean(self._results["encoding_time_ratio"]) if self._results else 0.0
            ),
            "average_decoding_ratio": (
                np.mean(self._results["decoding_time_ratio"]) if self._results else 0.0
            ),
        }

        return summary
