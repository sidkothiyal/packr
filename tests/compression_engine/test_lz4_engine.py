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


def test_lz4_supports_batch_flag():
    assert LZ4Encoder().supports_batch is True
    assert LZ4Decoder().supports_batch is True


def test_lz4_batch_roundtrip():
    rng = np.random.default_rng(seed=42)
    images = [rng.integers(0, 256, size=(48, 72, 3), dtype=np.uint8) for _ in range(4)]
    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder(), name="lz4")

    blob, _ = engine.encode_batch(images)
    decoded, _ = engine.decode_batch(blob)

    assert len(decoded) == len(images)
    for original, recovered in zip(images, decoded):
        assert np.array_equal(original, recovered)


def test_lz4_benchmark_batch_size_minus_one(tmp_path):
    """batch_size=-1 must compress every image in a single batch."""
    from PIL import Image

    rng = np.random.default_rng(seed=3)
    arrays = [rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(5)]
    paths = []
    for i, arr in enumerate(arrays):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(p)

    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder(), name="lz4")

    calls = {"n": 0}

    def fake_benchmarks(**kwargs):
        calls["n"] += 1

    original_encode_batch = engine.encode_batch
    batch_sizes_seen = []

    def tracking_encode_batch(images):
        batch_sizes_seen.append(len(images))
        return original_encode_batch(images)

    engine.encode_batch = tracking_encode_batch
    engine.benchmark(paths, fake_benchmarks, batch_size=-1)

    assert batch_sizes_seen == [len(paths)]
    assert calls["n"] == len(paths)
