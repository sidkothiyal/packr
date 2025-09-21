"""Base benchmark abstract class for compression engines."""

from abc import ABC
from typing import Dict, Any, List, Union
from pathlib import Path
from PIL import Image

import numpy as np


class Benchmark(ABC):
    """Abstract base class for benchmarking compression engines.

    This class defines the interface for benchmarking compression engines,
    measuring various metrics like compression ratio, encoding/decoding time,
    and image quality metrics.
    """

    def __init__(self, name: str = None):
        """Initialize the benchmark.

        Args:
            name: Optional name for the benchmark. If None, uses class name.
        """
        self.name = name or self.__class__.__name__
        self._results: Dict[str, Union[List[Any], np.ndarray]] = {}

    def _add_result(self, result: Dict[str, Any]):
        """Add a result to the benchmark.

        Args:
            result: Result to add.
        """
        for key, value in result.items():
            if key not in self._results:
                self._results[key] = []
            self._results[key].append(value)

    def _image_to_array(self, image: Union[np.ndarray, Path, Image, str]) -> np.ndarray:
        """Convert image to numpy array for comparison.

        Args:
            image: Image in various formats.

        Returns:
            Numpy array representation.
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, (Path, str)):
            return self._load_image_from_path(Path(image))
        elif isinstance(image, Image):
            return np.array(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _load_image_from_path(self, path: Path) -> np.ndarray:
        """Load image from file path, handling various compression formats.

        Args:
            path: Path to the image file.

        Returns:
            Numpy array representation of the image.
        """
        path = Path(path)

        # Check file extension to determine format
        if path.suffix.lower() == ".lz4":
            return self._load_from_lz4(path)
        elif path.suffix.lower() in [".tar", ".tar.gz", ".tar.bz2", ".tar.xz"]:
            return self._load_from_tar(path)
        else:
            # Standard image file - load with PIL
            try:
                return np.array(Image.open(path))
            except Exception as e:
                raise ValueError(f"Could not load image from {path}: {e}")

    def _load_from_lz4(self, path: Path) -> np.ndarray:
        """Load numpy array from LZ4 compressed file.

        Args:
            path: Path to LZ4 compressed file.

        Returns:
            Decompressed numpy array.
        """
        try:
            from compression_engine.lz4_engine import LZ4CompressionEngine

            engine = LZ4CompressionEngine()
            return engine.load_and_decompress(path)
        except ImportError:
            raise ImportError(
                "LZ4CompressionEngine not available. Make sure compression_engine package is installed."
            )
        except Exception as e:
            raise ValueError(f"Could not load LZ4 compressed image from {path}: {e}")

    def _load_from_tar(self, path: Path) -> np.ndarray:
        """Load numpy array from TAR compressed file.

        Args:
            path: Path to TAR compressed file.

        Returns:
            Decompressed numpy array.
        """
        try:
            from compression_engine.tar_engine import TarCompressionEngine

            # Determine compression mode from file extension
            if path.suffix == ".gz" and path.stem.endswith(".tar"):
                compression_mode = "r:gz"
            elif path.suffix == ".bz2" and path.stem.endswith(".tar"):
                compression_mode = "r:bz2"
            elif path.suffix == ".xz" and path.stem.endswith(".tar"):
                compression_mode = "r:xz"
            else:
                compression_mode = "r"  # Uncompressed TAR

            engine = TarCompressionEngine(compression_mode=compression_mode)
            return engine.load_and_decompress(path)
        except ImportError:
            raise ImportError(
                "TarCompressionEngine not available. Make sure compression_engine package is installed."
            )
        except Exception as e:
            raise ValueError(f"Could not load TAR compressed image from {path}: {e}")

    def _get_image_size(self, image: Union[np.ndarray, Path, Image]) -> tuple:
        """Get the size of an image in pixels.

        Args:
            image: Image data in various formats.

        Returns:
            Tuple of (width, height, channels) or (width, height) for grayscale.
        """
        if isinstance(image, np.ndarray):
            return image.shape
        elif isinstance(image, Path):
            with Image.open(image) as img:
                return (*img.size, len(img.getbands()))
        elif isinstance(image, Image):
            return (*image.size, len(image.getbands()))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _calculate_pixels(self, image_shape: tuple) -> int:
        """Calculate total number of pixels from image shape.

        Args:
            image_shape: Shape tuple from image.

        Returns:
            Total number of pixels.
        """
        return np.prod(image_shape)

    def __call__(
        self,
        original_image: Union[np.ndarray, Path, Image],
        decoded_image: Union[np.ndarray, Path, Image],
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        """Calculate benchmark metrics for a single compression test.

        Args:
            original_image: Original image array.
            decoded_image: Decoded/reconstructed image array.
            compressed_data: Compressed representation from the compression engine.
            encoding_time: Time taken to encode the image (in seconds).
            decoding_time: Time taken to decode the image (in seconds).

        Returns:
            Dict containing calculated metrics.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement calculate_metrics method")

    def summarize(self) -> Dict[str, Any]:
        """Get the summary of the benchmark.

        Returns:
            Dict containing the summary of the benchmark.
        """
        raise NotImplementedError("Subclasses must implement summarize method")


class Benchmarks:
    """Container class for managing multiple benchmark instances.

    This class manages a collection of benchmarks and provides methods to run
    all benchmarks on compression data and aggregate results.
    """

    def __init__(self, benchmarks: List[Benchmark]):
        """Initialize the benchmarks container.

        Args:
            benchmarks: List of benchmark instances to manage.

        Raises:
            ValueError: If benchmarks list is empty.
        """
        if not benchmarks:
            raise ValueError("benchmarks list cannot be empty")

        self.benchmarks = benchmarks

    def __call__(
        self,
        original_image: Union[np.ndarray, Path, Image],
        decoded_image: Union[np.ndarray, Path, Image],
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all benchmarks on the provided compression data.

        Args:
            original_image: Original image data.
            decoded_image: Decoded/reconstructed image data.
            compressed_data: Compressed representation from the compression engine.
            encoding_time: Time taken to encode the image (in seconds).
            decoding_time: Time taken to decode the image (in seconds).

        Returns:
            Dict mapping benchmark names to their calculated metrics.
        """
        all_metrics = {}

        for benchmark in self.benchmarks:
            metrics = benchmark(
                original_image=original_image,
                decoded_image=decoded_image,
                compressed_data=compressed_data,
                encoding_time=encoding_time,
                decoding_time=decoding_time,
            )

            all_metrics[benchmark.name] = metrics
        return all_metrics

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics from all benchmarks.

        Returns:
            Dict mapping benchmark names to their summary statistics.
        """
        summaries = {}

        for benchmark in self.benchmarks:
            summary = benchmark.summarize()
            summaries[benchmark.name] = summary

        return summaries

    def __len__(self) -> int:
        """Return the number of benchmarks."""
        return len(self.benchmarks)

    def __iter__(self):
        """Iterate over benchmarks."""
        return iter(self.benchmarks)

    @property
    def names(self) -> List[str]:
        """Get list of benchmark names."""
        return [benchmark.name for benchmark in self.benchmarks]

    def results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get results for all benchmarks."""
        return {benchmark.name: benchmark._results for benchmark in self.benchmarks}
