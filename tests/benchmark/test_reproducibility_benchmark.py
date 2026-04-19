"""Tests for benchmark.reproducibility_benchmark."""

import numpy as np

from benchmark.reproducibility_benchmark import ReproducibilityBenchmark


def test_identical_images_perfect_reconstruction(rgb_image_small):
    b = ReproducibilityBenchmark()
    result = b(
        original_image=rgb_image_small,
        decoded_image=rgb_image_small,
        compressed_data=b"",
        encoding_time=0.0,
        decoding_time=0.0,
    )
    assert result["perfect_reconstruction"]
    assert result["mse"] == 0.0
    assert result["psnr"] == float("inf")
    assert 0.0 <= result["ssim"] <= 1.0 + 1e-6


def test_nonidentical_images_finite_psnr(rgb_image_small):
    decoded = rgb_image_small.copy()
    decoded[..., 0] = np.clip(decoded[..., 0].astype(int) + 10, 0, 255)
    b = ReproducibilityBenchmark()
    result = b(rgb_image_small, decoded, b"", 0.0, 0.0)
    assert result["mse"] > 0
    assert 0 < result["psnr"] < float("inf")


def test_summarize_shape(rgb_image_small):
    b = ReproducibilityBenchmark()
    for _ in range(3):
        b(rgb_image_small, rgb_image_small, b"", 0.0, 0.0)
    summary = b.summarize()
    assert summary["perfect_reconstruction_rate"] == 1.0
    assert "psnr" in summary
    assert summary["psnr"]["count"] == 3
