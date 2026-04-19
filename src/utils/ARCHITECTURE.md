# utils — Architecture

## Module Layout

| File | Purpose |
|---|---|
| `image.py` | Image I/O (`load_image`/`save_image`), type normalization (`ensure_array`), and CHW/float conversions for ML engines. |

## Canonical Format

HWC uint8 RGB numpy arrays are the lingua franca of packr. The rule is: **normalize at the boundary, never branch downstream.** Every `Encoder.encode`, `Decoder.decode`, and `Benchmark.__call__` either receives arrays already normalized (via `CompressionEngine.encode` / `Benchmark._image_to_array`) or calls `ensure_array` itself.

## ensure_array

Accepts four input classes:

| Input | Handling |
|---|---|
| `np.ndarray` | Cast to uint8 (clip `[0, 255]`) if needed; promote 2D grayscale to 3-channel by stacking. |
| `PIL.Image.Image` | `convert("RGB")` → `np.array`. |
| `str` / `pathlib.Path` | Dispatch to `load_image`. |
| anything else | `TypeError`. |

Note: there is no explicit RGBA handling branch — PIL's `convert("RGB")` drops alpha. For numpy inputs with 4 channels, behavior is determined by whatever downstream code does with the extra channel; add handling here if a use-case arises.

## Float/CHW Helpers

`to_chw_float01` / `from_chw_float01` are used only by ONNX engines:

- `to_chw_float01`: `astype(float32).transpose(2, 0, 1) / 255.0`.
- `from_chw_float01`: `clip(0, 1) * 255`, `transpose(1, 2, 0)`, `astype(uint8)`.

The ONNX engines additionally apply `mean`/`std` normalization *after* `to_chw_float01`; that step is local to `onnx_engine.py` and not part of the utility layer.
