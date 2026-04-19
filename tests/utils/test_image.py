"""Tests for utils.image."""

import numpy as np
from PIL import Image

from utils.image import (
    ensure_array,
    from_chw_float01,
    load_image,
    save_image,
    to_chw_float01,
)


def test_ensure_array_from_ndarray(rgb_image_small):
    out = ensure_array(rgb_image_small)
    assert out is rgb_image_small or np.array_equal(out, rgb_image_small)
    assert out.dtype == np.uint8
    assert out.shape == (64, 64, 3)


def test_ensure_array_from_path(rgb_image_tmp_path):
    arr, path = rgb_image_tmp_path
    out = ensure_array(path)
    assert out.shape == arr.shape
    assert np.array_equal(out, arr)


def test_ensure_array_from_pil(rgb_image_small):
    pil = Image.fromarray(rgb_image_small, mode="RGB")
    out = ensure_array(pil)
    assert np.array_equal(out, rgb_image_small)


def test_ensure_array_grayscale_expanded():
    gray = np.zeros((8, 8), dtype=np.uint8)
    out = ensure_array(gray)
    assert out.shape == (8, 8, 3)


def test_load_save_roundtrip(tmp_path, rgb_image_small):
    path = tmp_path / "out.png"
    save_image(rgb_image_small, path)
    loaded = load_image(path)
    assert np.array_equal(loaded, rgb_image_small)


def test_chw_float_roundtrip(rgb_image_small):
    chw = to_chw_float01(rgb_image_small)
    assert chw.shape == (3, 64, 64)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0
    back = from_chw_float01(chw)
    assert back.shape == rgb_image_small.shape
    assert np.array_equal(back, rgb_image_small)
