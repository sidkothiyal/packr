"""TAR compression engine for image data."""

import time
import pickle
import tarfile
import io
from typing import Any, Union, Tuple
from pathlib import Path
from PIL import Image

import numpy as np

from compression_engine.base import CompressionEngine


class TarCompressionEngine(CompressionEngine):
    """TAR-based compression engine for image data.

    This engine uses TAR archiving (optionally with compression) to store serialized numpy arrays.
    The compression process:
    1. Convert image to numpy array
    2. Serialize array with pickle
    3. Store serialized data in a TAR archive (optionally compressed)
    """

    VALID_COMPRESSION_MODES = ["w", "w:", "w:gz", "w:bz2", "w:xz"]

    def __init__(self, name: str = None, compression_mode: str = "w:gz"):
        """Initialize the TAR compression engine.

        Args:
            name: Optional name for the engine. If None, uses 'tar'.
            compression_mode: TAR mode for compression:
                - 'w' or 'w:': uncompressed TAR
                - 'w:gz': gzip compression (default)
                - 'w:bz2': bzip2 compression
                - 'w:xz': xz/lzma compression
        """
        super().__init__(name or "tar")
        self.compression_mode = compression_mode
        self._validate_compression_mode()

    def _validate_compression_mode(self):
        """Validate the compression mode."""
        if self.compression_mode not in self.VALID_COMPRESSION_MODES:
            raise ValueError(
                f"Invalid compression mode: {self.compression_mode}. "
                f"Valid modes: {self.VALID_COMPRESSION_MODES}"
            )

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
        """Encode (compress) an image using TAR.

        Args:
            image: Input image data.

        Returns:
            Tuple of (compressed_tar_data, encoding_time).
        """
        start_time = time.perf_counter()

        # Convert to numpy array
        image_array = self._image_to_array(image)

        # Serialize the numpy array
        serialized_data = pickle.dumps(image_array, protocol=pickle.HIGHEST_PROTOCOL)

        # Create TAR archive in memory
        tar_buffer = io.BytesIO()

        with tarfile.open(fileobj=tar_buffer, mode=self.compression_mode) as tar:
            # Create a TarInfo object for the data
            info = tarfile.TarInfo(name="image_data.pkl")
            info.size = len(serialized_data)

            # Add the serialized data to the archive
            tar.addfile(info, io.BytesIO(serialized_data))

        # Get the TAR data
        tar_data = tar_buffer.getvalue()
        tar_buffer.close()

        encoding_time = time.perf_counter() - start_time

        return tar_data, encoding_time

    def decode(self, compressed_data: bytes) -> Tuple[np.ndarray, float]:
        """Decode (decompress) TAR compressed image data.

        Args:
            compressed_data: TAR compressed image data.

        Returns:
            Tuple of (decompressed_image_array, decoding_time).
        """
        start_time = time.perf_counter()

        # Create buffer from compressed data
        tar_buffer = io.BytesIO(compressed_data)

        # Extract data from TAR archive
        read_mode = self.compression_mode.replace("w", "r")
        with tarfile.open(fileobj=tar_buffer, mode=read_mode) as tar:
            # Extract the image data file
            extracted_file = tar.extractfile("image_data.pkl")
            if extracted_file is None:
                raise ValueError("Could not extract image data from TAR archive")

            serialized_data = extracted_file.read()
            extracted_file.close()

        tar_buffer.close()

        # Deserialize the numpy array
        image_array = pickle.loads(serialized_data)

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
            # Choose extension based on compression mode
            if self.compression_mode.endswith(":gz"):
                output_path = output_path.with_suffix(".tar.gz")
            elif self.compression_mode.endswith(":bz2"):
                output_path = output_path.with_suffix(".tar.bz2")
            elif self.compression_mode.endswith(":xz"):
                output_path = output_path.with_suffix(".tar.xz")
            else:
                output_path = output_path.with_suffix(".tar")

        with open(output_path, "wb") as f:
            f.write(compressed_data)

    def load_and_decompress(self, compressed_path: Union[str, Path]) -> np.ndarray:
        """Load compressed data from disk and decompress it.

        Args:
            compressed_path: Path to TAR compressed data file.

        Returns:
            Decompressed image array.
        """
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()

        decompressed_array, _ = self.decode(compressed_data)
        return decompressed_array
