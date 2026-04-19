# compression_engine

Pluggable image compression backends. Every engine is a `CompressionEngine` composed of an `Encoder` and a `Decoder` that exchange self-describing `bytes`. This lets the same class back both in-process benchmarking today and a future streaming mode where the two sides run on different machines.

## Available Engines

| Engine | Lossy | Config | Notes |
|---|---|---|---|
| `LZ4Encoder` / `LZ4Decoder` | no | `config/compression_engine/lz4.yaml` | Pickle + LZ4 frame. Fast lossless baseline. `compression_level: 0` is default. |
| `TarEncoder` / `TarDecoder` | no | `config/compression_engine/tar.yaml` | Pickle inside a TAR archive. `compression_mode` picks `w`, `w:gz`, `w:bz2`, or `w:xz`. |
| `JPEGEncoder` / `JPEGDecoder` | yes | `config/compression_engine/jpeg.yaml` | Re-encodes as JPEG at configurable `quality` (default 90). |
| `ONNXEncoder` / `ONNXDecoder` | yes | `config/compression_engine/compression_ae.yaml` | Loads an encoder/decoder ONNX pair (e.g. exported from [elemental](../../../elemental)). `device: auto\|cuda\|cpu`. Latent serialized via `LatentSerializer` in `float32`, `float16`, or per-tensor quantized `int8`. |

## Usage

```python
from hydra.utils import instantiate
engine = instantiate(cfg.compression_engine)     # CompressionEngine

compressed, enc_t = engine.encode("path/to/img.png")   # or np.ndarray / PIL
decoded, dec_t    = engine.decode(compressed)
```

Any supported input type (`np.ndarray`, `PIL.Image`, `str`, `pathlib.Path`) is accepted — the engine normalizes to HWC uint8 RGB before encoding.

## Selecting via Hydra

```bash
uv run python src/runner.py compression_engine=jpeg
uv run python src/runner.py compression_engine=jpeg ++compression_engine.encoder.quality=75
uv run python src/runner.py compression_engine=compression_ae \
    ++compression_engine.encoder.model_path=/path/encoder.onnx \
    ++compression_engine.decoder.model_path=/path/decoder.onnx \
    ++compression_engine.encoder.latent_serializer.dtype=int8 \
    ++compression_engine.decoder.latent_serializer.dtype=int8
```

For ONNX engines, encoder and decoder must agree on `latent_serializer.dtype`, `mean`/`std`, and `size_multiple`.

## Adding a New Engine

See `ARCHITECTURE.md` in this directory and the root `CLAUDE.md`.
