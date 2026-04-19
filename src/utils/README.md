# utils

Shared helpers. Currently just image I/O and format conversions.

## image.py

Canonical internal representation: **HWC uint8 RGB numpy array**. Every engine and benchmark funnels its public inputs through `ensure_array` so downstream code never has to branch on input type.

| Symbol | Use |
|---|---|
| `ImageLike` | Type alias: `np.ndarray \| PIL.Image.Image \| str \| pathlib.Path`. |
| `load_image(path)` | Load any PIL-readable file as HWC uint8 RGB. |
| `save_image(arr, path)` | Save HWC uint8 RGB to disk; format inferred from suffix. Clips non-uint8 input. |
| `ensure_array(image)` | Normalize any `ImageLike` to HWC uint8 RGB. Promotes 2D grayscale to 3-channel. |
| `to_chw_float01(arr)` | HWC uint8 [0, 255] → CHW float32 [0, 1]. |
| `from_chw_float01(arr)` | CHW float32 [0, 1] → HWC uint8 [0, 255], clamped. |

## Usage

```python
from utils.image import ensure_array, to_chw_float01, from_chw_float01

arr = ensure_array(anything)                  # HWC uint8 RGB
chw_f = to_chw_float01(arr)                    # for ML models
arr_back = from_chw_float01(chw_f)             # back to display format
```

Only ONNX engines need the CHW/float path — classical engines operate on HWC uint8 throughout.
