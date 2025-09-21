"""Base compression engine abstract class."""

from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, List, Dict
from pathlib import Path
from PIL import Image

import numpy as np

from benchmark.base import Benchmarks


class CompressionEngine(ABC):
    """Abstract base class for compression engines.

    This class defines the interface that all compression engines must implement.
    Compression engines should be able to encode (compress) and decode (decompress)
    image data.
    """

    def __init__(self, name: str = None):
        """Initialize the compression engine.

        Args:
            name: Optional name for the compression engine. If None, uses class name.
        """
        self._name = name or self.__class__.__name__.lower()

    @abstractmethod
    def encode(self, image: Union[np.ndarray, str, Path]) -> Tuple[Any, float]:
        """Encode (compress) an image.

        Args:
            image: Input image data. Can be:
                - np.ndarray: Image array (H, W, C) or (H, W)
                - str or Path: Path to image file

        Returns:
            1. Compressed representation of the image. The exact type depends on
            the compression algorithm implementation.
            2. Time taken to encode the image (in seconds).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement encode method")

    @abstractmethod
    def decode(self, compressed_data: Any) -> Tuple[Union[np.ndarray, Path, Image], float]:
        """Decode (decompress) compressed image data.

        Args:
            compressed_data: Compressed image data returned by encode method.

        Returns:
            1. Decoded image array.
            2. Time taken to decode the image (in seconds).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement decode method")

    def compress_and_save(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> None:
        """Convenience method to compress an image file and save the result.

        Args:
            input_path: Path to input image file.
            output_path: Path where compressed data should be saved.
        """
        raise NotImplementedError("Subclasses should implement format-specific saving")

    def load_and_decompress(self, compressed_path: Union[str, Path]) -> np.ndarray:
        """Convenience method to load compressed data and decompress it.

        Args:
            compressed_path: Path to compressed data file.

        Returns:
            np.ndarray: Decompressed image array.
        """
        raise NotImplementedError("Subclasses should implement format-specific loading")

    def benchmark(self, image_paths: List[Union[str, Path]], benchmarks) -> Dict[str, Any]:
        """Run benchmarks on a list of images using this compression engine.

        Args:
            image_paths: List of paths to images to benchmark.
            benchmarks: Benchmarks instance containing the benchmark implementations.

        Returns:
            Dict containing all benchmark results and metadata.

        Raises:
            FileNotFoundError: If any image path doesn't exist.
            Exception: If encoding/decoding fails for any image.
        """
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")

            # Encode the image
            compressed_data, encoding_time = self.encode(image_path)

            # Decode the compressed data
            decoded_data, decoding_time = self.decode(compressed_data)

            # Run all benchmarks on this compression result
            benchmark_results = benchmarks(
                original_image=image_path,
                decoded_image=decoded_data,
                compressed_data=compressed_data,
                encoding_time=encoding_time,
                decoding_time=decoding_time,
            )

    @property
    def name(self) -> str:
        """Get the name of the compression engine.

        Returns:
            str: Name of the compression engine.
        """
        return self._name
