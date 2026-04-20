"""Tests for compression_engine.passthrough_engine (PNG baseline)."""

import numpy as np
import pytest

from compression_engine.base import CompressionEngine
from compression_engine.passthrough_engine import PassthroughDecoder, PassthroughEncoder


def test_passthrough_lossless_roundtrip(rgb_image_small):
    engine = CompressionEngine(PassthroughEncoder(), PassthroughDecoder(), name="passthrough")
    data, _ = engine.encode(rgb_image_small)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, rgb_image_small)
    # PNG magic bytes
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_passthrough_accepts_path(rgb_image_tmp_path):
    arr, path = rgb_image_tmp_path
    engine = CompressionEngine(PassthroughEncoder(), PassthroughDecoder())
    data, _ = engine.encode(path)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, arr)


def test_passthrough_rejects_batch(rgb_image_small):
    encoder = PassthroughEncoder()
    decoder = PassthroughDecoder()
    assert encoder.supports_batch is False
    assert decoder.supports_batch is False
    with pytest.raises(NotImplementedError):
        encoder.encode_batch([rgb_image_small, rgb_image_small])
    with pytest.raises(NotImplementedError):
        decoder.decode_batch(b"")
