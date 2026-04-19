"""Unified image I/O for packr.

All internal representations are HWC uint8 RGB numpy arrays. Engines and
benchmarks should call :func:`ensure_array` at their public boundary so they
never need to branch on input type.
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageLike = Union[np.ndarray, Image.Image, str, Path]


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as HWC uint8 RGB."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_image(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Save an HWC uint8 RGB array to disk. Format inferred from suffix."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def ensure_array(image: ImageLike) -> np.ndarray:
    """Normalize any supported image input to HWC uint8 RGB numpy array."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, (str, Path)):
        return load_image(image)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def to_chw_float01(arr: np.ndarray) -> np.ndarray:
    """HWC uint8 [0,255] -> CHW float32 [0,1]."""
    return arr.astype(np.float32).transpose(2, 0, 1) / 255.0


def from_chw_float01(arr: np.ndarray) -> np.ndarray:
    """CHW float32 [0,1] -> HWC uint8 [0,255], clamped."""
    out = np.clip(arr, 0.0, 1.0) * 255.0
    return out.transpose(1, 2, 0).astype(np.uint8)
