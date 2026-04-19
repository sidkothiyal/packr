# benchmark

Metric collectors run by `runner.py` against every (original, decoded, compressed) triple produced by a `CompressionEngine`. A `Benchmarks` container fans each per-image call out to every registered `Benchmark` and aggregates summaries.

## Available Benchmarks

| Benchmark | Measures |
|---|---|
| `TimeBenchmark` | Encoding time, decoding time, total time, throughput (pixels/s), encode/decode time ratios. |
| `ReproducibilityBenchmark` | MSE, PSNR, RMSE, MAE, global SSIM, perfect-reconstruction rate, shape-mismatch count, coarse quality assessment from average PSNR. |
| `DegradationBenchmark` | Runs N encode/decode cycles; records PSNR/SSIM/MSE per sampled cycle; reports final metrics and ΔPSNR (cycle 1 vs cycle N). Requires engine injection. |

## Configs

Benchmark groups live in `config/benchmark/`:

| File | Contents |
|---|---|
| `default.yaml` | `TimeBenchmark` + `ReproducibilityBenchmark`. |
| `timing_only.yaml` | `TimeBenchmark` only. |
| `similarity_only.yaml` | `ReproducibilityBenchmark` only. |
| `degradation.yaml` | Includes `DegradationBenchmark`. |

Select via `benchmark=<group>` on the runner.

## Writing a Benchmark

```python
from benchmark.base import Benchmark

class MyBenchmark(Benchmark):
    def __call__(self, original_image, decoded_image, compressed_data,
                 encoding_time, decoding_time):
        orig = self._image_to_array(original_image)
        dec  = self._image_to_array(decoded_image)
        metrics = {"my_metric": float(...)}
        self._add_result(metrics)     # stored column-wise in self._results
        return metrics

    def summarize(self):
        if not self._results:
            return {"error": "No results available for summary"}
        import numpy as np
        return {"my_metric_mean": float(np.mean(self._results["my_metric"]))}
```

If your benchmark needs the compression engine (like `DegradationBenchmark`), set `self.compression_engine = None` in `__init__`. The runner calls `Benchmarks.bind_engine(engine)` which fills it in.

## Metric Helpers

`_metrics.py` exposes `all_metrics(original, decoded) -> (mse, psnr, rmse, mae, ssim, max_pixel)`. Reuse it instead of reimplementing the similarity math.
