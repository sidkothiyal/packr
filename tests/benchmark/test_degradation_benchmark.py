"""Tests for benchmark.degradation_benchmark."""

import time
from typing import Tuple

import numpy as np
import pytest

from benchmark.base import Benchmarks
from benchmark.degradation_benchmark import DegradationBenchmark
from compression_engine.base import CompressionEngine, Decoder, Encoder
from compression_engine.lz4_engine import LZ4Decoder, LZ4Encoder


class _LossyEncoder(Encoder):
    """Adds +1 to every pixel on each encode (with wrap-around via uint8)."""

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        shifted = (image.astype(np.int16) + 1).clip(0, 255).astype(np.uint8)
        header = np.array(shifted.shape, dtype=np.int32).tobytes()
        return header + shifted.tobytes(), time.perf_counter() - start


class _LossyDecoder(Decoder):
    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        shape = np.frombuffer(data[:12], dtype=np.int32)
        arr = np.frombuffer(data[12:], dtype=np.uint8).reshape(tuple(int(x) for x in shape))
        return arr.copy(), time.perf_counter() - start


def _run(benchmark: DegradationBenchmark, engine: CompressionEngine, image: np.ndarray):
    data, enc_time = engine.encode(image)
    decoded, dec_time = engine.decode(data)
    return benchmark(
        original_image=image,
        decoded_image=decoded,
        compressed_data=data,
        encoding_time=enc_time,
        decoding_time=dec_time,
    )


def test_lossless_engine_keeps_perfect_psnr(rgb_image_small):
    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder())
    b = DegradationBenchmark(num_cycles=5)
    b.compression_engine = engine

    result = _run(b, engine, rgb_image_small)
    assert len(result["curve"]) == 5
    for point in result["curve"]:
        assert point["psnr"] == float("inf")
    assert result["delta_psnr"] == 0.0 or np.isnan(result["delta_psnr"])


def test_lossy_engine_psnr_monotonically_decreases(rgb_image_small):
    engine = CompressionEngine(_LossyEncoder(), _LossyDecoder())
    b = DegradationBenchmark(num_cycles=5)
    b.compression_engine = engine

    result = _run(b, engine, rgb_image_small)
    psnrs = [point["psnr"] for point in result["curve"]]
    # PSNR should be non-increasing as error accumulates
    for prev, curr in zip(psnrs, psnrs[1:]):
        assert curr <= prev + 1e-9
    assert result["final"]["mse"] > 0
    assert result["delta_psnr"] > 0


def test_bind_engine_injects_engine(rgb_image_small):
    engine = CompressionEngine(LZ4Encoder(), LZ4Decoder())
    b = DegradationBenchmark(num_cycles=2)
    container = Benchmarks([b])
    assert b.compression_engine is None
    container.bind_engine(engine)
    assert b.compression_engine is engine


def test_unbound_engine_raises(rgb_image_small):
    b = DegradationBenchmark(num_cycles=2)
    with pytest.raises(RuntimeError, match="requires a compression engine"):
        b(rgb_image_small, rgb_image_small, b"", 0.0, 0.0)


def test_sample_iterations_filters_curve(rgb_image_small):
    engine = CompressionEngine(_LossyEncoder(), _LossyDecoder())
    b = DegradationBenchmark(num_cycles=10, sample_iterations=[1, 5, 10])
    b.compression_engine = engine
    result = _run(b, engine, rgb_image_small)
    iters = [point["iter"] for point in result["curve"]]
    assert iters == [1, 5, 10]


def test_summarize_includes_mean_curve(rgb_image_small):
    engine = CompressionEngine(_LossyEncoder(), _LossyDecoder())
    b = DegradationBenchmark(num_cycles=3)
    b.compression_engine = engine
    for _ in range(2):
        _run(b, engine, rgb_image_small)
    summary = b.summarize()
    assert "mean_curve_across_images" in summary
    assert len(summary["mean_curve_across_images"]) == 3
    assert summary["num_cycles"] == 3
