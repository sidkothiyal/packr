"""Tests for compression_engine.base."""

import time
from typing import Tuple

import numpy as np
import pytest

from compression_engine.base import CompressionEngine, Decoder, Encoder


class _IdentityEncoder(Encoder):
    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        return image.tobytes() + b"|" + str(image.shape).encode(), time.perf_counter() - start


class _IdentityDecoder(Decoder):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        raw, _ = data.rsplit(b"|", 1)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.shape)
        return arr.copy(), time.perf_counter() - start


def test_engine_composes_encoder_and_decoder(rgb_image_small):
    engine = CompressionEngine(
        encoder=_IdentityEncoder(),
        decoder=_IdentityDecoder(shape=rgb_image_small.shape),
        name="identity",
    )
    data, enc_time = engine.encode(rgb_image_small)
    assert isinstance(data, bytes)
    assert enc_time >= 0
    decoded, dec_time = engine.decode(data)
    assert dec_time >= 0
    assert np.array_equal(decoded, rgb_image_small)
    assert engine.name == "identity"


def test_engine_normalizes_path_input(rgb_image_tmp_path):
    arr, path = rgb_image_tmp_path
    engine = CompressionEngine(
        encoder=_IdentityEncoder(),
        decoder=_IdentityDecoder(shape=arr.shape),
    )
    data, _ = engine.encode(path)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, arr)


def test_encoder_is_abstract():
    with pytest.raises(TypeError):
        Encoder()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Decoder()  # type: ignore[abstract]
