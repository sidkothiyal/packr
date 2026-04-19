# packr

**A library (and hopefully a set of models) for running ML based compression models specifically for images (to start with).**

## Why?

Because why not?!

But more realistically, legacy compression engines have theoretical limits on compression, our goal is to push the boundaries of image compression by exploring novel ML-based approaches and providing robust benchmarking tools for comparative analysis.

## Included Contents

### Compression Engines

Every engine is composed of an `Encoder` and a `Decoder` that exchange self-describing `bytes`. The split keeps the two sides independently instantiable, so the same code can later back a streaming use-case where encoding and decoding happen on different machines.

| Engine | Lossy | Notes |
|---|---|---|
| `lz4` | no | Pickle + LZ4 frame. Baseline. |
| `tar` | no | Pickle inside a TAR (gz/bz2/xz/uncompressed). Baseline. |
| `jpeg` | yes | Always re-encodes to JPEG at configurable `quality` (default 90). |
| `compression_ae` | yes | ONNX autoencoder (encoder + decoder pair). Loads models exported from [elemental](../elemental). Device `auto`/`cuda`/`cpu`. Latent serialization is pluggable (`float32` / `float16` / per-tensor-quantized `int8`). |

### Benchmarks

| Benchmark | Measures |
|---|---|
| `TimeBenchmark` | encoding/decoding time, throughput (pixels/s), time ratios. |
| `ReproducibilityBenchmark` | MSE, PSNR, RMSE, MAE, global SSIM, perfect-reconstruction rate. |
| `DegradationBenchmark` | Runs N encode/decode cycles and records PSNR/SSIM/MSE at each iteration vs. the original. Reports the full degradation curve, final metrics, and ΔPSNR. |

## Running

```bash
uv sync --extra dev                                        # install + dev extras
uv run python src/runner.py                                # uses default (lz4 + default benchmarks)
uv run python src/runner.py compression_engine=jpeg        # switch engine via hydra
uv run python src/runner.py compression_engine=compression_ae \
    ++compression_engine.encoder.model_path=/path/to/encoder.onnx \
    ++compression_engine.decoder.model_path=/path/to/decoder.onnx
uv run python src/runner.py benchmark=degradation          # include degradation benchmark
```

Run the tests with:

```bash
uv run pytest tests/
```

ONNX integration tests auto-skip if `onnxruntime` is missing or the exported model files are not present.

## Roadmap

- [x] Benchmarking framework (time, reproducibility, degradation)
- [x] JPEG baseline
- [x] ONNX autoencoder inference
- [ ] Finetune ONNX ML models on custom datasets from within packr
- [ ] Streaming mode — compress → network → decompress across two processes
- [ ] Video compression

## License

This project is licensed under the MIT License - see the LICENSE file for details.
