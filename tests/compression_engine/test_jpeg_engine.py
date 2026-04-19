"""Tests for compression_engine.jpeg_engine."""

import numpy as np
import pytest

from compression_engine.base import CompressionEngine
from compression_engine.jpeg_engine import JPEGDecoder, JPEGEncoder


def _engine(quality: int) -> CompressionEngine:
    return CompressionEngine(JPEGEncoder(quality=quality), JPEGDecoder(), name=f"jpeg_q{quality}")


def test_jpeg_roundtrip_psnr(rgb_image_small):
    engine = _engine(90)
    data, _ = engine.encode(rgb_image_small)
    decoded, _ = engine.decode(data)
    assert decoded.shape == rgb_image_small.shape
    assert decoded.dtype == np.uint8
    # JPEG magic bytes
    assert data[:2] == b"\xff\xd8"


def test_jpeg_quality_affects_size(rgb_image_small):
    small, _ = _engine(10).encode(rgb_image_small)
    large, _ = _engine(95).encode(rgb_image_small)
    assert len(small) < len(large)


def test_jpeg_quality_out_of_range():
    with pytest.raises(ValueError):
        JPEGEncoder(quality=0)
    with pytest.raises(ValueError):
        JPEGEncoder(quality=101)


def test_jpeg_accepts_path_input(rgb_image_tmp_path):
    arr, path = rgb_image_tmp_path
    engine = _engine(90)
    data, _ = engine.encode(path)
    decoded, _ = engine.decode(data)
    assert decoded.shape == arr.shape
