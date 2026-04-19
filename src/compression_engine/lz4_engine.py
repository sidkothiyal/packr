"""LZ4 compression engine (lossless)."""

import pickle
import time
from typing import Optional, Tuple

import lz4.frame
import numpy as np

from compression_engine.base import Decoder, Encoder


class LZ4Encoder(Encoder):
    """Pickle the numpy array, then LZ4-frame compress the bytes."""

    def __init__(self, compression_level: int = 0, name: Optional[str] = None):
        super().__init__(name or "lz4_encoder")
        self.compression_level = compression_level

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        serialized = pickle.dumps(image, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(serialized, compression_level=self.compression_level)
        return compressed, time.perf_counter() - start


class LZ4Decoder(Decoder):
    """Inverse of :class:`LZ4Encoder`."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "lz4_decoder")

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        decompressed = lz4.frame.decompress(data)
        image = pickle.loads(decompressed)
        return image, time.perf_counter() - start
