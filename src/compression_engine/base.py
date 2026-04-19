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
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.image import ImageLike, ensure_array


class Encoder(ABC):
    """Encodes an HWC uint8 RGB image into self-describing bytes."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @abstractmethod
    def encode(self, image: np.ndarray) -> Tuple[bytes, float]:
        """Encode a HWC uint8 RGB image into bytes.

        Returns:
            Tuple of (serialized_bytes, encoding_time_seconds).
        """

    @property
    def name(self) -> str:
        return self._name


class Decoder(ABC):
    """Decodes bytes produced by the paired Encoder back into an image."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @abstractmethod
    def decode(self, data: bytes) -> Tuple[np.ndarray, float]:
        """Decode bytes back into HWC uint8 RGB image.

        Returns:
            Tuple of (image_array, decoding_time_seconds).
        """

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

    def benchmark(
        self,
        image_paths: List[Union[str, Path]],
        benchmarks,
    ) -> Dict[str, Any]:
        """Run all benchmarks on each image via this engine."""
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
