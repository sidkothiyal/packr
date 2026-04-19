"""Tests for compression_engine.lz4_engine."""

import numpy as np

from compression_engine.base import CompressionEngine
from compression_engine.lz4_engine import LZ4Decoder, LZ4Encoder


def test_lz4_lossless_roundtrip(rgb_image_small):
    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder(), name="lz4")
    data, _ = engine.encode(rgb_image_small)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, rgb_image_small)


def test_lz4_accepts_path(rgb_image_tmp_path):
    arr, path = rgb_image_tmp_path
    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder())
    data, _ = engine.encode(path)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, arr)
