"""ONNX-based ML compression engine.

Loads a pair of ONNX encoder/decoder models (e.g. an autoencoder exported
from the ``elemental`` training library) and uses them as a codec. The
latent tensor is serialized with a configurable dtype (float32, float16, or
per-tensor-quantized int8) so the 'compressed' representation can actually
be smaller than the source image.

Streaming note: Encoder and Decoder are fully independent. A consumer that
only needs to decode received bytes can instantiate :class:`ONNXDecoder`
alone. Both sides must agree on :class:`LatentSerializer` configuration —
the serialized header carries enough metadata (shape, dtype, scale/zp) for
the decoder to reconstruct the tensor without external config.

Finetuning note: :attr:`ONNXEncoder.session` / :attr:`model_path` are
exposed so a future finetuning module can reload the same weights into a
torch model without modifying this inference interface.
"""

import struct
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from compression_engine.base import Decoder, Encoder
from utils.image import from_chw_float01, to_chw_float01

_MAGIC = b"PKRL"  # PacKR Latent
_VERSION = 1
_DTYPE_CODES = {"float32": 0, "float16": 1, "int8": 2}
_DTYPE_BY_CODE = {v: k for k, v in _DTYPE_CODES.items()}


class LatentSerializer:
    """Serialize/deserialize latent tensors with a configurable dtype.

    For ``int8`` we apply per-tensor symmetric-ish quantization with a
    stored float32 scale and int32 zero-point, which gives roughly 4x
    shrinkage vs float32 and ~2x vs float16 at the cost of one
    min-range/max-range pass over the tensor.

    Header layout (big-endian, so serialized bytes are portable across
    hosts of different endianness)::

        magic      4B   b"PKRL"
        version    1B   currently 1
        dtype_code 1B   0=f32, 1=f16, 2=i8
        shape_len  1B   number of dims (usually 4: NCHW)
        dims       shape_len * u32
        orig_H     u32
        orig_W     u32
        [scale f32, zp i32]  only when dtype_code == 2 (int8)
        payload    raw contiguous tensor bytes
    """

    def __init__(self, dtype: str = "float16"):
        if dtype not in _DTYPE_CODES:
            raise ValueError(
                f"Unsupported latent dtype: {dtype!r}. "
                f"Choose from {sorted(_DTYPE_CODES)}"
            )
        self.dtype = dtype

    def serialize(self, latent: np.ndarray, orig_hw: Tuple[int, int]) -> bytes:
        header = bytearray()
        header += _MAGIC
        header += struct.pack(">BB", _VERSION, _DTYPE_CODES[self.dtype])
        header += struct.pack(">B", latent.ndim)
        for d in latent.shape:
            header += struct.pack(">I", int(d))
        header += struct.pack(">II", int(orig_hw[0]), int(orig_hw[1]))

        if self.dtype == "float32":
            payload = np.ascontiguousarray(latent, dtype=">f4").tobytes()
        elif self.dtype == "float16":
            payload = np.ascontiguousarray(latent, dtype=">f2").tobytes()
        else:  # int8
            x = latent.astype(np.float32)
            lo, hi = float(x.min()), float(x.max())
            if hi == lo:
                scale, zp = 1.0, 0
                q = np.zeros_like(x, dtype=np.int8)
            else:
                scale = (hi - lo) / 255.0
                zp = int(round(-lo / scale)) - 128
                q = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)
            header += struct.pack(">fi", scale, zp)
            payload = q.tobytes()

        return bytes(header) + payload

    def deserialize(self, data: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
        if data[:4] != _MAGIC:
            raise ValueError("Invalid latent header: magic bytes mismatch")
        offset = 4
        version, dtype_code = struct.unpack_from(">BB", data, offset)
        offset += 2
        if version != _VERSION:
            raise ValueError(f"Unsupported latent version: {version}")
        dtype = _DTYPE_BY_CODE[dtype_code]

        (ndim,) = struct.unpack_from(">B", data, offset)
        offset += 1
        shape = struct.unpack_from(f">{ndim}I", data, offset)
        offset += 4 * ndim
        orig_h, orig_w = struct.unpack_from(">II", data, offset)
        offset += 8

        if dtype == "float32":
            arr = np.frombuffer(data, dtype=">f4", offset=offset).astype(np.float32)
        elif dtype == "float16":
            arr = np.frombuffer(data, dtype=">f2", offset=offset).astype(np.float32)
        else:  # int8
            scale, zp = struct.unpack_from(">fi", data, offset)
            offset += 8
            q = np.frombuffer(data, dtype=np.int8, offset=offset).astype(np.float32)
            arr = (q - zp) * scale

        return arr.reshape(shape), (int(orig_h), int(orig_w))


def _pad_to_multiple(
    arr: np.ndarray, multiple: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Reflection-pad an NCHW array so H and W are multiples of ``multiple``.

    Returns the padded array and the original (H, W) so the decoder can
    crop back.
    """
    _, _, h, w = arr.shape
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph == 0 and pw == 0:
        return arr, (h, w)
    padded = np.pad(arr, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    return padded, (h, w)


def _resolve_providers(device: str) -> List[str]:
    """Pick ONNX Runtime execution providers for ``device`` preference."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider requested but not available. "
                "Install onnxruntime-gpu or set device='cpu'."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ONNXEncoder(Encoder):
    """Run a preprocessed image through an ONNX encoder and serialize the latent."""

    def __init__(
        self,
        model_path: Union[str, Path],
        input_name: str = "input_image",
        output_name: str = "encoded_image",
        device: str = "auto",
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        size_multiple: int = 8,
        latent_serializer: Optional[LatentSerializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "onnx_encoder")
        import onnxruntime as ort

        self.model_path = str(Path(model_path).expanduser())
        self.input_name = input_name
        self.output_name = output_name
        self.device = device
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.size_multiple = size_multiple
        self.serializer = latent_serializer or LatentSerializer()

        self.session = ort.InferenceSession(
            self.model_path, providers=_resolve_providers(device)
        )

    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        start = time.perf_counter()
        chw = to_chw_float01(image)
        normalized = (chw - self.mean) / self.std
        batched = normalized[np.newaxis, ...]
        padded, orig_hw = _pad_to_multiple(batched, self.size_multiple)

        (latent,) = self.session.run([self.output_name], {self.input_name: padded})

        data = self.serializer.serialize(latent, orig_hw)
        return data, time.perf_counter() - start


class ONNXDecoder(Decoder):
    """Deserialize a latent and run it through an ONNX decoder."""

    def __init__(
        self,
        model_path: Union[str, Path],
        input_name: str = "encoded_image",
        output_name: str = "reconstructed_image",
        device: str = "auto",
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
        size_multiple: int = 8,
        latent_serializer: Optional[LatentSerializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "onnx_decoder")
        import onnxruntime as ort

        self.model_path = str(Path(model_path).expanduser())
        self.input_name = input_name
        self.output_name = output_name
        self.device = device
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.size_multiple = size_multiple
        self.serializer = latent_serializer or LatentSerializer()

        self.session = ort.InferenceSession(
            self.model_path, providers=_resolve_providers(device)
        )

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        latent, (orig_h, orig_w) = self.serializer.deserialize(data)
        latent = latent.astype(np.float32)

        (out,) = self.session.run([self.output_name], {self.input_name: latent})
        # NCHW float, still normalized — denormalize, crop, pack to uint8 HWC.
        out = out[0] * self.std + self.mean
        out = out[:, :orig_h, :orig_w]
        image = from_chw_float01(out)
        return image, time.perf_counter() - start
