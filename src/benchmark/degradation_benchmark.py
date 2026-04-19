"""Degradation benchmark — measures quality loss across repeated encode/decode cycles."""

from typing import Any, Dict, List, Optional

import numpy as np

from benchmark._metrics import all_metrics
from benchmark.base import Benchmark


class DegradationBenchmark(Benchmark):
    """Re-encode/decode N times and track per-cycle quality vs. the original.

    For each image the benchmark stores a curve of {mse, psnr, ssim} at each
    sampled iteration plus the final-cycle metrics and the delta-PSNR
    (cycle 1 vs. cycle N).

    The compression engine is injected post-hoc by the runner via
    :meth:`Benchmarks.bind_engine` so configs do not need to reference the
    engine twice.
    """

    def __init__(
        self,
        num_cycles: int = 100,
        sample_iterations: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        if num_cycles < 1:
            raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")
        self.num_cycles = num_cycles
        self.sample_iterations = sample_iterations
        self.compression_engine = None  # bound by Benchmarks.bind_engine

    def _should_sample(self, iteration: int) -> bool:
        if self.sample_iterations is None:
            return True
        return iteration in self.sample_iterations

    def __call__(
        self,
        original_image,
        decoded_image,
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        if self.compression_engine is None:
            raise RuntimeError(
                "DegradationBenchmark requires a compression engine. "
                "Call Benchmarks.bind_engine(engine) before running."
            )

        original = self._image_to_array(original_image)
        # The first decoded image is the runner's initial pass; treat it as cycle 1.
        current = self._image_to_array(decoded_image)

        curve: List[Dict[str, float]] = []

        def _sample(iteration: int, image: np.ndarray) -> Dict[str, float]:
            m, p, rmse, mae, s, _ = all_metrics(original, image)
            return {"iter": iteration, "mse": m, "psnr": p, "rmse": rmse, "mae": mae, "ssim": s}

        if self._should_sample(1):
            curve.append(_sample(1, current))

        for i in range(2, self.num_cycles + 1):
            data, _ = self.compression_engine.encode(current)
            current, _ = self.compression_engine.decode(data)
            current = self._image_to_array(current)
            if self._should_sample(i):
                curve.append(_sample(i, current))

        first = curve[0]
        last = curve[-1]

        result = {
            "curve": curve,
            "final": {k: last[k] for k in ("mse", "psnr", "rmse", "mae", "ssim")},
            "delta_psnr": first["psnr"] - last["psnr"],
            "num_cycles": self.num_cycles,
        }
        self._add_result(result)
        return result

    def summarize(self) -> Dict[str, Any]:
        if not self._results:
            return {"error": "No results available for summary"}

        curves = self._results["curve"]  # list of per-image curves
        # Collect iteration indices actually sampled (assumed consistent across images).
        iterations = [point["iter"] for point in curves[0]]

        def _mean_at(metric: str, idx: int) -> float:
            return float(np.mean([curve[idx][metric] for curve in curves]))

        mean_curve = [
            {
                "iter": iterations[idx],
                "mse": _mean_at("mse", idx),
                "psnr": _mean_at("psnr", idx),
                "ssim": _mean_at("ssim", idx),
            }
            for idx in range(len(iterations))
        ]

        final_psnrs = [r["psnr"] for r in self._results["final"]]
        final_ssims = [r["ssim"] for r in self._results["final"]]
        final_mses = [r["mse"] for r in self._results["final"]]
        delta_psnrs = self._results["delta_psnr"]

        def _stats(values):
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "count": int(arr.size),
            }

        return {
            "num_cycles": self.num_cycles,
            "mean_curve_across_images": mean_curve,
            "final_psnr": _stats(final_psnrs),
            "final_ssim": _stats(final_ssims),
            "final_mse": _stats(final_mses),
            "delta_psnr": _stats(delta_psnrs),
        }
