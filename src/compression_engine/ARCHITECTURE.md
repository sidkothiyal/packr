# compression_engine — Architecture

## Module Layout

| File | Purpose |
|---|---|
| `base.py` | `Encoder` / `Decoder` ABCs and `CompressionEngine` composition. |
| `lz4_engine.py` | `LZ4Encoder` / `LZ4Decoder`. Pickle → `lz4.frame`. |
| `tar_engine.py` | `TarEncoder` / `TarDecoder`. Pickle → in-memory TAR. Validates mode against `_VALID_WRITE_MODES`. |
| `jpeg_engine.py` | `JPEGEncoder` / `JPEGDecoder`. PIL JPEG round-trip. |
| `onnx_engine.py` | `ONNXEncoder` / `ONNXDecoder`, `LatentSerializer`, and provider resolution. |

## Encoder / Decoder Contracts

```python
class Encoder(ABC):
    supports_batch: bool = False
    def encode(self, image: np.ndarray) -> Tuple[bytes, float]: ...                    # HWC uint8 RGB -> bytes + enc_time_s
    def encode_batch(self, images: List[np.ndarray]) -> Tuple[bytes, float]: ...       # default raises NotImplementedError

class Decoder(ABC):
    supports_batch: bool = False
    def decode(self, data: bytes) -> Tuple[np.ndarray, float]: ...                     # bytes -> HWC uint8 RGB + dec_time_s
    def decode_batch(self, data: bytes) -> Tuple[List[np.ndarray], float]: ...         # default raises NotImplementedError
```

Both define a `name` property (defaults to class name; overridable in the constructor). Timing uses `time.perf_counter()` and spans the full (de)serialization.

Subclasses that can compress multiple images jointly set `supports_batch = True` on both sides and override the batch methods. Byte-stream codecs (`LZ4`, `TAR`) do; per-image codecs (`JPEG`, `ONNX`) inherit the raising defaults.

## CompressionEngine

```python
engine = CompressionEngine(encoder, decoder, name=None)
engine.encode(image_like)  -> (bytes, float)                 # funnels input through utils.image.ensure_array
engine.decode(bytes)       -> (np.ndarray, float)
engine.encode_batch(images)-> (bytes, float)                 # delegates to encoder.encode_batch after ensure_array
engine.decode_batch(bytes) -> (List[np.ndarray], float)
engine.supports_batch                                        # True iff both encoder and decoder opt in
engine.benchmark(image_paths, benchmarks, batch_size=1)      # per-image when batch_size==1 or engine doesn't support batch; otherwise chunked
```

`engine.benchmark` is the single loop used by `runner.run_compression_benchmark`. It always produces the five arguments every `Benchmark` expects. In the batched path the chunk is encoded/decoded once; then per-image `compressed_data` is reported as a bytes object of length `len(blob) // N` and `encoding_time` / `decoding_time` as `total / N`, so summary statistics remain per-image even though compression was joint.

`DegradationBenchmark` is bound to `engine` directly (via `Benchmarks.bind_engine`) and continues to use single-image `encode`/`decode` in both modes — degradation is inherently a per-image cycle study.

## Self-Describing Bytes

Decoders must not depend on shared in-memory state with their encoder. Each engine embeds whatever metadata it needs:

- **LZ4** — pickle preserves numpy shape/dtype inside the payload. Batch mode pickles a `list[np.ndarray]` before LZ4-compressing the stream.
- **TAR** — pickle inside `image_data.pkl` entry (single-image) or `image_batch.pkl` (batched — pickled `list[np.ndarray]`); `compression_mode` must match on both sides (config-level constraint, not embedded).
- **JPEG** — JPEG stream is fully self-describing.
- **ONNX** — `LatentSerializer` writes a custom binary header.

## LatentSerializer Header

```
magic       4B    b"PKRL"
version     1B    currently 1
dtype_code  1B    0=f32, 1=f16, 2=i8
shape_len   1B    number of latent dims
dims        shape_len * 4B   u32 big-endian
orig_H      4B    u32
orig_W      4B    u32
[scale      4B    f32]     only when dtype_code == 2
[zp         4B    i32]     only when dtype_code == 2
payload     raw contiguous tensor bytes
```

All multi-byte values are big-endian so bytes transfer portably across host endianness. int8 quantization is per-tensor: `scale = (max - min) / 255`, `zp = round(-min/scale) - 128`. Bump `_VERSION` if the layout changes.

## ONNX Pipeline

`ONNXEncoder.encode`:
1. `to_chw_float01` — HWC uint8 → CHW float32 [0, 1].
2. Normalize with `mean` / `std` (default ImageNet-ish 0.5/0.5).
3. Batch to NCHW.
4. `_pad_to_multiple(..., size_multiple)` — reflection pad; record original (H, W).
5. `session.run` → latent.
6. `LatentSerializer.serialize(latent, orig_hw)`.

`ONNXDecoder.decode` inverts: deserialize → `session.run` → denormalize → crop to original (H, W) → `from_chw_float01`.

`_resolve_providers` selects ONNX Runtime providers:
- `cpu` → `[CPUExecutionProvider]`.
- `cuda` → `[CUDAExecutionProvider, CPUExecutionProvider]`; raises if CUDA EP unavailable.
- `auto` → CUDA+CPU if CUDA is available, else CPU only.

`session` and `model_path` are kept as public attributes so a future finetuning path can reload the same weights in Torch without duplicating config.
