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
| `webp` | yes | WebP at configurable quality. ~30% smaller than JPEG at equal quality; essential modern baseline. **Must-have.** |
| `avif` | yes | AV1-based codec; current state-of-the-art lossy baseline the ML model must beat. **Must-have.** |
| `png` | no | Image-aware lossless baseline (distinct from byte-level LZ4/TAR). **Must-have.** |
| `jpeg_xl` | yes/no | JPEG successor; both lossy and lossless modes; upper bound for classical codec quality. **Good to have.** |
| `bpg` | yes | HEVC-based; often 2× smaller than JPEG at equal PSNR; strong lossy upper bound. **Good to have.** |

### Benchmarks

| Benchmark | Measures |
|---|---|
| `TimeBenchmark` | encoding/decoding time, throughput (pixels/s), time ratios. |
| `ReproducibilityBenchmark` | MSE, PSNR, RMSE, MAE, global SSIM, perfect-reconstruction rate. |
| `DegradationBenchmark` | Runs N encode/decode cycles and records PSNR/SSIM/MSE at each iteration vs. the original. Reports the full degradation curve, final metrics, and ΔPSNR. |
| `CompressionRatioBenchmark` | Compressed bytes / original bytes. Fundamental metric for comparing engines. **Must-have.** |
| `PerceptualBenchmark` | LPIPS — perceptual similarity metric that correlates with human judgment; essential for evaluating ML codecs where MSE/PSNR can be misleading. **Must-have.** |
| `FrequencyBenchmark` | PSNR in the DCT/wavelet frequency domain; catches blocking and ringing artifacts that pixel-domain PSNR misses. **Must-have.** |
| `LatencyProfileBenchmark` | Breaks timing into encode / decode / serialization phases; diagnoses where the ONNX pipeline is slow. **Good to have.** |
| `MemoryBenchmark` | Peak RSS during encode/decode; ML models can balloon memory, important for deployment sizing. **Good to have.** |

## Running

```bash
uv sync --extra dev                                        # install + dev extras
uv run python src/runner.py                                # uses default (compression_ae + default benchmarks)
uv run python src/runner.py compression_engine=jpeg        # switch engine via hydra
uv run python src/runner.py compression_engine=lz4         # classical baseline
uv run python src/runner.py compression_engine=compression_ae \
    ++compression_engine.encoder.model_path=/path/to/encoder.onnx \
    ++compression_engine.decoder.model_path=/path/to/decoder.onnx
uv run python src/runner.py benchmark=degradation          # include degradation benchmark
uv run python src/runner.py --config-name=compare          # run every engine and write a consolidated summary.json
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
