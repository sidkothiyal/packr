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

## Batch compression

Byte-stream codecs (LZ4, TAR) gain compression ratio when multiple images are compressed jointly — the compressor sees cross-image redundancy (pickle framing, repeated pixel patterns, dictionary entries) that is invisible when each image is processed alone.

Engines opt into this by setting `supports_batch = True` on both the encoder and decoder and implementing `encode_batch(images)` / `decode_batch(data)`. `LZ4` and `TAR` do; `JPEG` and `ONNX` do not (per-image DCT and per-image latents respectively have no cross-image signal).

Enable globally with the top-level `batch_size` config knob:

```bash
uv run python src/runner.py compression_engine=lz4 batch_size=16
uv run python src/runner.py --config-name=compare batch_size=16
```

Engines without batch support silently use the per-image path, so a single `batch_size` value is safe across a compare run. In batched mode `CompressionEngine.benchmark` amortizes `compressed_data` length and timing evenly across the images in each chunk so per-image benchmark metrics remain meaningful.

## Adding a New Engine

See `ARCHITECTURE.md` in this directory and the root `CLAUDE.md`.
