"""Passthrough baseline: re-encode as PNG.

PNG is the standard lossless image format — every image loader can read
it, and its DEFLATE layer is a strong entropy coder for natural-image
pixel data. This engine reports what "just save it as a PNG" costs,
giving every other lossless engine (LZ4, TAR) a sanity baseline: if a
codec can't beat PNG, it isn't worth shipping.

Not batch-capable: PNG's entropy coder is per-image, and PNG byte
streams are near-random so cross-image redundancy is negligible.
"""

import io
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from compression_engine.base import Decoder, Encoder


class PassthroughEncoder(Encoder):
    """Encode an HWC uint8 RGB array as PNG bytes."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "passthrough_encoder")

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        buf = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buf, format="PNG")
        return buf.getvalue(), time.perf_counter() - start


class PassthroughDecoder(Decoder):
    """Decode PNG bytes back to HWC uint8 RGB."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "passthrough_decoder")

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        with Image.open(io.BytesIO(data)) as img:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr, time.perf_counter() - start
