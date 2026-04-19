"""Tests for compression_engine.onnx_engine.

Most tests exercise :class:`LatentSerializer` (no onnxruntime needed).
End-to-end tests using the exported compression_ae ONNX models are
conditional on both onnxruntime and the model files being present.
"""

from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from compression_engine.base import CompressionEngine  # noqa: E402
from compression_engine.onnx_engine import (  # noqa: E402
    LatentSerializer,
    ONNXDecoder,
    ONNXEncoder,
    _pad_to_multiple,
)

COMPRESSION_AE_DIR = Path("/home/siddharth/xyz/elemental/exports/compression_ae")
ENCODER_ONNX = COMPRESSION_AE_DIR / "compression_ae_encoder.onnx"
DECODER_ONNX = COMPRESSION_AE_DIR / "compression_ae_decoder.onnx"

onnx_models_missing = pytest.mark.skipif(
    not (ENCODER_ONNX.exists() and DECODER_ONNX.exists()),
    reason="compression_ae ONNX exports not present",
)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_latent_serializer_roundtrip(dtype):
    rng = np.random.default_rng(42)
    latent = rng.standard_normal((1, 256, 8, 12)).astype(np.float32)
    serializer = LatentSerializer(dtype=dtype)
    data = serializer.serialize(latent, orig_hw=(64, 96))
    restored, orig_hw = serializer.deserialize(data)
    assert restored.shape == latent.shape
    assert orig_hw == (64, 96)
    tolerance = {"float32": 1e-6, "float16": 1e-2, "int8": 5e-2}[dtype]
    assert np.allclose(restored, latent, atol=tolerance * (latent.max() - latent.min()))


def test_latent_serializer_int8_smaller_than_float32():
    latent = np.random.default_rng(0).standard_normal((1, 256, 8, 12)).astype(np.float32)
    f32 = LatentSerializer("float32").serialize(latent, (64, 96))
    i8 = LatentSerializer("int8").serialize(latent, (64, 96))
    assert len(i8) < len(f32)


def test_latent_serializer_invalid_dtype():
    with pytest.raises(ValueError):
        LatentSerializer(dtype="bogus")


def test_pad_to_multiple():
    arr = np.zeros((1, 3, 7, 11), dtype=np.float32)
    padded, orig = _pad_to_multiple(arr, multiple=8)
    assert padded.shape == (1, 3, 8, 16)
    assert orig == (7, 11)

    aligned = np.zeros((1, 3, 8, 16), dtype=np.float32)
    padded2, orig2 = _pad_to_multiple(aligned, multiple=8)
    assert padded2 is aligned
    assert orig2 == (8, 16)


@onnx_models_missing
def test_onnx_roundtrip_shape_preserved(rgb_image_large):
    encoder = ONNXEncoder(model_path=str(ENCODER_ONNX), device="cpu")
    decoder = ONNXDecoder(model_path=str(DECODER_ONNX), device="cpu")
    engine = CompressionEngine(encoder, decoder, name="compression_ae")

    data, _ = engine.encode(rgb_image_large)
    decoded, _ = engine.decode(data)
    assert decoded.shape == rgb_image_large.shape
    assert decoded.dtype == np.uint8


@onnx_models_missing
def test_onnx_roundtrip_padded_size():
    """Non-multiple-of-8 size within the model's input tolerance is padded and cropped back."""
    # 636x957 reflection-pads to 640x960 — matches the compression_ae fixed spatial input.
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(636, 957, 3), dtype=np.uint8)
    encoder = ONNXEncoder(model_path=str(ENCODER_ONNX), device="cpu")
    decoder = ONNXDecoder(model_path=str(DECODER_ONNX), device="cpu")
    engine = CompressionEngine(encoder, decoder)
    data, _ = engine.encode(img)
    decoded, _ = engine.decode(data)
    assert decoded.shape == img.shape
