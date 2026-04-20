"""Base classes for compression engines.

The ``CompressionEngine`` is a composition of an :class:`Encoder` and a
:class:`Decoder`. Splitting these two halves lets the same machinery support
streaming use-cases (encode on machine A, decode on machine B) by letting
consumers instantiate only one side.

Contracts:
    * ``Encoder.encode(image)`` receives HWC uint8 RGB and returns
      ``(bytes, encoding_time_seconds)``. The bytes are self-describing so
      the matching Decoder can reconstruct the image without shared state.
    * ``Decoder.decode(data)`` receives those bytes and returns
      ``(HWC uint8 RGB np.ndarray, decoding_time_seconds)``.

Batch contract (optional):
    Subclasses that can compress multiple images jointly may set
    ``supports_batch = True`` and override ``encode_batch`` / ``decode_batch``.
    Byte-stream codecs (LZ4, TAR) gain compression ratio this way by
    exposing cross-image redundancy to a single compressor run. Engines
    that do not override keep the default ``NotImplementedError``; the
    benchmark loop routes around them automatically.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.image import ImageLike, ensure_array


class Encoder(ABC):
    """Encodes an HWC uint8 RGB image into self-describing bytes."""

    supports_batch: bool = False

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @abstractmethod
    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        """Encode a HWC uint8 RGB image into bytes.

        Returns:
            Tuple of (serialized_bytes, encoding_time_seconds).
        """

    def encode_batch(self, images: List[np.ndarray]) -> Tuple[bytes, float]:
        """Encode a list of HWC uint8 RGB images into a single bytes blob.

        Override on subclasses that set ``supports_batch = True``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch encoding"
        )

    @property
    def name(self) -> str:
        return self._name


class Decoder(ABC):
    """Decodes bytes produced by the paired Encoder back into an image."""

    supports_batch: bool = False

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @abstractmethod
    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        """Decode bytes back into HWC uint8 RGB image.

        Returns:
            Tuple of (image_array, decoding_time_seconds).
        """

    def decode_batch(self, data: bytes) -> Tuple[List[np.ndarray], float]:
        """Decode a bytes blob produced by :meth:`Encoder.encode_batch`.

        Override on subclasses that set ``supports_batch = True``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch decoding"
        )

    @property
    def name(self) -> str:
        return self._name


class CompressionEngine:
    """Pairs an :class:`Encoder` with a :class:`Decoder`.

    Engine classes are normally instantiated via hydra by composing an
    encoder and decoder sub-config. The engine takes care of input
    normalization and the benchmark orchestration loop; the encoder and
    decoder implement the actual codec logic.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        name: Optional[str] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self._name = name or f"{encoder.name}+{decoder.name}"

    @property
    def name(self) -> str:
        return self._name

    def encode(self, image: ImageLike) -> Tuple[bytes, float]:
        return self.encoder.encode(ensure_array(image))

    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        return self.decoder.decode(data)

    @property
    def supports_batch(self) -> bool:
        return self.encoder.supports_batch and self.decoder.supports_batch

    def encode_batch(self, images: List[np.ndarray]) -> Tuple[bytes, float]:
        return self.encoder.encode_batch([ensure_array(im) for im in images])

    def decode_batch(self, data: bytes) -> Tuple[List[np.ndarray], float]:
        return self.decoder.decode_batch(data)

    def benchmark(
        self,
        image_paths: List[Union[str, Path]],
        benchmarks,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Run all benchmarks on each image via this engine.

        When ``batch_size > 1`` and the engine supports batching, images are
        compressed in groups. ``batch_size = -1`` batches every image in one
        shot. Per-image ``compressed_data`` length, encoding time, and
        decoding time are amortized (blob size / N, batch time / N) so
        benchmark metrics remain per-image.
        """
        if batch_size == -1:
            batch_size = len(image_paths)
        if batch_size > 1 and self.supports_batch:
            self._benchmark_batched(image_paths, benchmarks, batch_size)
        else:
            self._benchmark_per_image(image_paths, benchmarks)

    def _benchmark_per_image(
        self,
        image_paths: List[Union[str, Path]],
        benchmarks,
    ) -> None:
        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            print(f"Processing image {i + 1}/{len(image_paths)}: {image_path.name}")

            compressed_data, encoding_time = self.encode(image_path)
            decoded_data, decoding_time = self.decode(compressed_data)

            benchmarks(
                original_image=image_path,
                decoded_image=decoded_data,
                compressed_data=compressed_data,
                encoding_time=encoding_time,
                decoding_time=decoding_time,
            )

    def _benchmark_batched(
        self,
        image_paths: List[Union[str, Path]],
        benchmarks,
        batch_size: int,
    ) -> None:
        paths = [Path(p) for p in image_paths]
        total = len(paths)
        for start in range(0, total, batch_size):
            chunk = paths[start : start + batch_size]
            n = len(chunk)
            print(
                f"Processing batch {start // batch_size + 1} "
                f"({start + 1}-{start + n}/{total})"
            )

            arrays = [ensure_array(p) for p in chunk]
            blob, encoding_time = self.encode_batch(arrays)
            decoded_list, decoding_time = self.decode_batch(blob)

            if len(decoded_list) != n:
                raise RuntimeError(
                    f"Batch decode returned {len(decoded_list)} images, expected {n}"
                )

            amortized_bytes = bytes(len(blob) // n)
            amortized_encode = encoding_time / n
            amortized_decode = decoding_time / n

            for original_path, decoded in zip(chunk, decoded_list):
                benchmarks(
                    original_image=original_path,
                    decoded_image=decoded,
                    compressed_data=amortized_bytes,
                    encoding_time=amortized_encode,
                    decoding_time=amortized_decode,
                )
