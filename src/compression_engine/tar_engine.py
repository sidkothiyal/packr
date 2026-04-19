"""TAR archive compression engine (lossless, optional gz/bz2/xz)."""

import io
import pickle
import tarfile
import time
from typing import Optional, Tuple

import numpy as np

from compression_engine.base import Decoder, Encoder

_VALID_WRITE_MODES = {"w", "w:", "w:gz", "w:bz2", "w:xz"}


def _read_mode(write_mode: str) -> str:
    return write_mode.replace("w", "r", 1)


class TarEncoder(Encoder):
    """Serialize the array with pickle, archive in an in-memory TAR."""

    def __init__(self, compression_mode: str = "w:gz", name: Optional[str] = None):
        super().__init__(name or "tar_encoder")
        if compression_mode not in _VALID_WRITE_MODES:
            raise ValueError(
                f"Invalid compression mode: {compression_mode}. "
                f"Valid modes: {sorted(_VALID_WRITE_MODES)}"
            )
        self.compression_mode = compression_mode

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        serialized = pickle.dumps(image, protocol=pickle.HIGHEST_PROTOCOL)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode=self.compression_mode) as tar:
            info = tarfile.TarInfo(name="image_data.pkl")
            info.size = len(serialized)
            tar.addfile(info, io.BytesIO(serialized))
        return buf.getvalue(), time.perf_counter() - start


class TarDecoder(Decoder):
    def __init__(self, compression_mode: str = "w:gz", name: Optional[str] = None):
        super().__init__(name or "tar_decoder")
        if compression_mode not in _VALID_WRITE_MODES:
            raise ValueError(
                f"Invalid compression mode: {compression_mode}. "
                f"Valid modes: {sorted(_VALID_WRITE_MODES)}"
            )
        self.compression_mode = compression_mode

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode=_read_mode(self.compression_mode)) as tar:
            extracted = tar.extractfile("image_data.pkl")
            if extracted is None:
                raise ValueError("Could not extract image_data.pkl from TAR archive")
            serialized = extracted.read()
        image = pickle.loads(serialized)
        return image, time.perf_counter() - start
