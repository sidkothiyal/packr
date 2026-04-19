"""JPEG compression engine.

Always re-encodes to JPEG regardless of input format. Useful as a lossy
baseline for comparing other engines.
"""

import io
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from compression_engine.base import Decoder, Encoder


class JPEGEncoder(Encoder):
    """Re-encode any HWC uint8 RGB image as JPEG."""

    def __init__(self, quality: int = 90, name: Optional[str] = None):
        super().__init__(name or f"jpeg_encoder_q{quality}")
        if not 1 <= quality <= 100:
            raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
        self.quality = quality

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        buf = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buf, format="JPEG", quality=self.quality)
        return buf.getvalue(), time.perf_counter() - start


class JPEGDecoder(Decoder):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "jpeg_decoder")

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        with Image.open(io.BytesIO(data)) as img:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr, time.perf_counter() - start
