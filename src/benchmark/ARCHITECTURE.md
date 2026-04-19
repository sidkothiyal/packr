# benchmark — Architecture

## Module Layout

| File | Purpose |
|---|---|
| `base.py` | `Benchmark` ABC (accumulator + `__call__`/`summarize` contract) and `Benchmarks` container. |
| `_metrics.py` | Shared similarity math: MSE, PSNR, RMSE, MAE, global SSIM. Returns `(mse, psnr, rmse, mae, ssim, max_pixel)`. |
| `time_benchmark.py` | `TimeBenchmark` — timing + throughput stats. |
| `reproducibility_benchmark.py` | `ReproducibilityBenchmark` — similarity + perfect-reconstruction rate. |
| `degradation_benchmark.py` | `DegradationBenchmark` — multi-cycle quality curve; requires engine injection. |

## Benchmark Contract

```python
class Benchmark(ABC):
    name: str
    _results: Dict[str, List[Any]]   # column-wise per-image results

    def __call__(original_image, decoded_image, compressed_data,
                 encoding_time, decoding_time) -> Dict[str, Any]: ...
    def summarize() -> Dict[str, Any]: ...

    # helpers
    _add_result(metrics: Dict[str, Any])   # append each key to self._results lists
    _image_to_array(img)                   # -> HWC uint8 RGB via utils.image.ensure_array
    _get_image_size(img) -> tuple
    _calculate_pixels(shape) -> int
```

Every implementation must guard `summarize()` against empty `_results` (return `{"error": ...}` or equivalent — existing benchmarks follow this pattern).

## Benchmarks Container

```python
Benchmarks(benchmarks: List[Benchmark])          # empty list raises ValueError
bench(...)         # fans call out, returns {name: per-bench metrics}
bench.summarize()  # {name: summary}
bench.results      # {name: _results dict}  (used by runner for JSON output)
bench.bind_engine(engine)   # sets engine on any benchmark exposing compression_engine=None
```

`bind_engine` uses `getattr(bench, "compression_engine", "sentinel")` — benchmarks that never declare the attribute are untouched, so this is opt-in.

## TimeBenchmark

Per-image: encoding / decoding / total time; `total_pixels = prod(shape)`; three throughputs (pps) and two time ratios. Summary: mean/min/max/std/count for each plus the cumulative `total_processing_time`.

## ReproducibilityBenchmark

Per-image: full metric tuple plus `perfect_reconstruction = (mse == 0)`, and original vs decoded shape. Summary: percentiles for each metric, `perfect_reconstruction_rate`, `shape_mismatches`, and a coarse quality band derived from average PSNR (`excellent >= 40`, `good >= 30`, `fair >= 20`, else `poor`).

## DegradationBenchmark

```
__init__(num_cycles=100, sample_iterations=None)
```

Treats the runner's first encode/decode pass as **cycle 1**. For cycles 2..N it re-encodes `current` through the bound engine and samples metrics at each `sample_iterations` index (or every cycle if `None`). Per-image result contains:

- `curve` — list of `{iter, mse, psnr, rmse, mae, ssim}` points.
- `final` — metrics at the last sampled cycle.
- `delta_psnr` — cycle 1 PSNR minus final PSNR.
- `num_cycles`.

Summary computes a mean curve across images and stats for final PSNR/SSIM/MSE and `delta_psnr`. Assumes sampled iteration indices are consistent across images (true when `sample_iterations` is fixed).

## Runner Integration

`runner.create_benchmarks_from_config(cfg.benchmark)` instantiates every entry in `cfg.benchmark.benchmarks` via `hydra.utils.instantiate` and wraps them in `Benchmarks`. The runner then calls `bind_engine` before dispatch.
