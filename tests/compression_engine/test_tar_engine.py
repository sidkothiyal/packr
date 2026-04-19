"""Tests for compression_engine.tar_engine."""

import numpy as np
import pytest

from compression_engine.base import CompressionEngine
from compression_engine.tar_engine import TarDecoder, TarEncoder


@pytest.mark.parametrize("mode", ["w:gz", "w:bz2", "w:xz", "w"])
def test_tar_lossless_roundtrip(rgb_image_small, mode):
    engine = CompressionEngine(
        TarEncoder(compression_mode=mode),
        TarDecoder(compression_mode=mode),
        name=f"tar_{mode}",
    )
    data, _ = engine.encode(rgb_image_small)
    decoded, _ = engine.decode(data)
    assert np.array_equal(decoded, rgb_image_small)


def test_tar_invalid_mode():
    with pytest.raises(ValueError):
        TarEncoder(compression_mode="w:bogus")
    with pytest.raises(ValueError):
        TarDecoder(compression_mode="w:bogus")
