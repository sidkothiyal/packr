"""Base classes for benchmarking compression engines."""

from abc import ABC
from typing import Any, Dict, List, Optional, Union

import numpy as np

from utils.image import ImageLike, ensure_array


class Benchmark(ABC):
    """Base class for a single benchmark.

    Subclasses must implement :meth:`__call__` and :meth:`summarize`.
    Intermediate per-image results are accumulated via :meth:`_add_result`,
    which stores them column-wise in ``self._results`` (a dict of lists).
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._results: Dict[str, List[Any]] = {}

    def _add_result(self, result: Dict[str, Any]) -> None:
        for key, value in result.items():
            self._results.setdefault(key, []).append(value)

    def _image_to_array(self, image: ImageLike) -> np.ndarray:
        """Normalize any input to HWC uint8 RGB."""
        return ensure_array(image)

    def _get_image_size(self, image: ImageLike) -> tuple:
        return self._image_to_array(image).shape

    def _calculate_pixels(self, image_shape: tuple) -> int:
        return int(np.prod(image_shape))

    def __call__(
        self,
        original_image: ImageLike,
        decoded_image: ImageLike,
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __call__")

    def summarize(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement summarize")


class Benchmarks:
    """Container that runs a list of :class:`Benchmark` instances."""

    def __init__(self, benchmarks: List[Benchmark]):
        if not benchmarks:
            raise ValueError("benchmarks list cannot be empty")
        self.benchmarks = benchmarks

    def bind_engine(self, engine) -> None:
        """Inject the active compression engine into benchmarks that need it.

        Any benchmark exposing a ``compression_engine`` attribute currently
        set to ``None`` will have it replaced with ``engine``. Benchmarks
        that do not declare the attribute are left untouched.
        """
        for benchmark in self.benchmarks:
            if getattr(benchmark, "compression_engine", "sentinel") is None:
                benchmark.compression_engine = engine

    def __call__(
        self,
        original_image: ImageLike,
        decoded_image: ImageLike,
        compressed_data: Any,
        encoding_time: float,
        decoding_time: float,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            benchmark.name: benchmark(
                original_image=original_image,
                decoded_image=decoded_image,
                compressed_data=compressed_data,
                encoding_time=encoding_time,
                decoding_time=decoding_time,
            )
            for benchmark in self.benchmarks
        }

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        return {benchmark.name: benchmark.summarize() for benchmark in self.benchmarks}

    def __len__(self) -> int:
        return len(self.benchmarks)

    def __iter__(self):
        return iter(self.benchmarks)

    @property
    def names(self) -> List[str]:
        return [benchmark.name for benchmark in self.benchmarks]

    def results(self) -> Dict[str, Dict[str, List[Any]]]:
        return {benchmark.name: benchmark._results for benchmark in self.benchmarks}
