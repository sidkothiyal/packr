"""Shared pytest fixtures for packr tests."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image_small() -> np.ndarray:
    """Deterministic 64x64 uint8 RGB image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def rgb_image_large() -> np.ndarray:
    """Deterministic 640x960 uint8 RGB image sized for the compression_ae ONNX model."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 256, size=(640, 960, 3), dtype=np.uint8)


@pytest.fixture
def rgb_image_tmp_path(tmp_path: Path, rgb_image_small: np.ndarray) -> Tuple[np.ndarray, Path]:
    """Same small image, also written to a tmp PNG. Returns (array, path)."""
    path = tmp_path / "sample.png"
    Image.fromarray(rgb_image_small, mode="RGB").save(path)
    return rgb_image_small, path
