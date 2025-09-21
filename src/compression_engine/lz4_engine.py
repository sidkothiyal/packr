"""LZ4 compression engine for image data."""

import time
import pickle
from typing import Any, Union, Tuple
from pathlib import Path
from PIL import Image

import lz4.frame
import numpy as np


from compression_engine.base import CompressionEngine


class LZ4CompressionEngine(CompressionEngine):
    """LZ4-based compression engine for image data.

    This engine uses LZ4 compression algorithm to compress serialized numpy arrays.
    The compression process:
    1. Convert image to numpy array
    2. Serialize array with pickle
    3. Compress serialized data with LZ4
    """

    def __init__(self, name: str = None, compression_level: int = 0):
        """Initialize the LZ4 compression engine.

        Args:
            name: Optional name for the engine. If None, uses 'lz4'.
            compression_level: LZ4 compression level (0=default, 1-12 for higher compression).
        """
        super().__init__(name or "lz4")
        self.compression_level = compression_level

    def _image_to_array(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Convert image to numpy array.

        Args:
            image: Input image in various formats.

        Returns:
            numpy array representation of the image.
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, (str, Path)):
            return np.array(Image.open(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def encode(self, image: Union[np.ndarray, str, Path]) -> Tuple[bytes, float]:
        """Encode (compress) an image using LZ4.

        Args:
            image: Input image data.

        Returns:
            Tuple of (compressed_data, encoding_time).
        """
        start_time = time.perf_counter()

        # Convert to numpy array
        image_array = self._image_to_array(image)

        # Serialize the numpy array
        serialized_data = pickle.dumps(image_array, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress with LZ4
        compressed_data = lz4.frame.compress(
            serialized_data, compression_level=self.compression_level
        )

        encoding_time = time.perf_counter() - start_time

        return compressed_data, encoding_time

    def decode(self, compressed_data: bytes) -> Tuple[np.ndarray, float]:
        """Decode (decompress) LZ4 compressed image data.

        Args:
            compressed_data: LZ4 compressed image data.

        Returns:
            Tuple of (decompressed_image_array, decoding_time).
        """
        start_time = time.perf_counter()

        # Decompress with LZ4
        decompressed_data = lz4.frame.decompress(compressed_data)

        # Deserialize the numpy array
        image_array = pickle.loads(decompressed_data)

        decoding_time = time.perf_counter() - start_time

        return image_array, decoding_time

    def compress_and_save(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> None:
        """Compress an image file and save the result to disk.

        Args:
            input_path: Path to input image file.
            output_path: Path where compressed data should be saved.
        """
        compressed_data, _ = self.encode(input_path)

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".lz4")

        with open(output_path, "wb") as f:
            f.write(compressed_data)

    def load_and_decompress(self, compressed_path: Union[str, Path]) -> np.ndarray:
        """Load compressed data from disk and decompress it.

        Args:
            compressed_path: Path to LZ4 compressed data file.

        Returns:
            Decompressed image array.
        """
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()

        decompressed_array, _ = self.decode(compressed_data)
        return decompressed_array
