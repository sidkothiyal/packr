"""Benchmark runner for compression engines."""

import json
from typing import List, Union, Dict, Any
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from benchmark.base import Benchmarks
from compression_engine.base import CompressionEngine


def run_compression_benchmark(
    compression_engine: CompressionEngine,
    image_paths: List[Union[str, Path]],
    destination_path: Union[str, Path],
    benchmarks: Benchmarks,
    save_compressed_images: bool = True,
    save_decoded_images: bool = True,
) -> Dict[str, Any]:
    """Run comprehensive benchmarking of compression engine.

    This function takes a list of image paths, runs them through various compression
    engines, and benchmarks their performance using the provided benchmark classes.

    Args:
        image_paths: List of paths to images to be compressed and benchmarked.
        destination_path: Directory path where compressed and decoded images will be saved.
        benchmark_classes: List of benchmark instances to run on the compression results.
        compression_engines: List of compression engines to test.
        save_compressed_images: Whether to save compressed image data to disk.
        save_decoded_images: Whether to save decoded images to disk.
        results_filename: Name of the file to save benchmark results.

    Returns:
        Dict containing all benchmark results and summary statistics.

    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If image paths don't exist.
        OSError: If destination path cannot be created.
    """
    # Validate inputs
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    # Convert to Path objects and validate
    image_paths = [Path(p) for p in image_paths]
    destination_path = Path(destination_path)

    # Check if all image paths exist
    missing_images = [p for p in image_paths if not p.exists()]
    if missing_images:
        raise FileNotFoundError(f"Image files not found: {missing_images}")

    # Create destination directory structure
    destination_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organization
    compressed_dir = destination_path / "compressed"
    decoded_dir = destination_path / "decoded"
    results_path = destination_path / f"results_{compression_engine.name}.json"

    compressed_dir.mkdir(exist_ok=True)
    decoded_dir.mkdir(exist_ok=True)

    # Initialize results structure
    all_results = {
        "metadata": {
            "total_images": len(image_paths),
            "compression_engine": compression_engine.name,
            "total_benchmarks": len(benchmarks),
            "benchmark_names": [benchmark.name for benchmark in benchmarks],
            "destination_path": str(destination_path),
            "image_paths": [str(p) for p in image_paths],
        },
        "detailed_results": {},
        "benchmark_summaries": {},
    }

    print(
        f"Starting benchmark run with {len(image_paths)} images, {compression_engine.name} engine, and {len(benchmarks)} benchmarks..."
    )

    compression_engine.benchmark(image_paths, benchmarks)

    all_results["detailed_results"] = benchmarks.results

    all_results["benchmark_summaries"] = benchmarks.summarize()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nBenchmark completed! Results saved to: {results_path}")

    if not save_compressed_images:
        compressed_dir.rmdir()
    if not save_decoded_images:
        decoded_dir.rmdir()

    return all_results


def get_image_paths(input_dir: Union[str, Path], max_images: Union[int, None] = None) -> List[Path]:
    """Get list of image paths from input directory.

    Args:
        input_dir: Directory containing images.
        max_images: Maximum number of images to process. None for all images.

    Returns:
        List of image file paths.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Common image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

    # Get all image files
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f"**/*{ext}"))
        image_paths.extend(input_dir.glob(f"**/*{ext.upper()}"))

    # Sort for consistent ordering
    image_paths.sort()

    # Limit number of images if specified
    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        print(f"Limited to {max_images} images out of {len(image_paths)} found")

    return image_paths


def create_benchmarks_from_config(benchmark_config: DictConfig) -> Benchmarks:
    """Create benchmarks instance from configuration.

    Args:
        benchmark_config: Benchmark configuration from Hydra.

    Returns:
        Benchmarks instance.
    """
    benchmark_instances = []

    for bench_config in benchmark_config.benchmarks:
        benchmark = instantiate(bench_config)
        benchmark_instances.append(benchmark)

    return Benchmarks(benchmark_instances)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run compression benchmarks using Hydra configuration.

    Args:
        cfg: Hydra configuration object.
    """
    print("=== Compression Benchmark Runner ===")
    print(f"Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Get image paths
    image_paths = get_image_paths(cfg.data.input_dir, cfg.data.max_images)

    if not image_paths:
        print(f"No images found in {cfg.data.input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Create compression engine
    compression_engine = instantiate(cfg.compression_engine)
    print(f"Using compression engine: {compression_engine.name}")

    # Create benchmarks
    benchmarks = create_benchmarks_from_config(cfg.benchmark)
    print(f"Running {len(benchmarks)} benchmarks: {benchmarks.names}")

    # Create output directory
    output_dir = Path(cfg.data.output_dir)

    try:
        # Run benchmarks
        results = run_compression_benchmark(
            compression_engine=compression_engine,
            image_paths=image_paths,
            destination_path=output_dir,
            benchmarks=benchmarks,
            save_compressed_images=cfg.runtime.save_compressed_images,
            save_decoded_images=cfg.runtime.save_decoded_images,
        )

        print("\n=== Benchmark Summary ===")
        for benchmark_name, summary in results["benchmark_summaries"].items():
            print(f"\n{benchmark_name.upper()}:")
            print(summary)
        print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        print(f"Error running benchmarks: {e}")
        raise


if __name__ == "__main__":
    main()
