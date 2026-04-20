# CLAUDE.md — packr

**packr** is a modular image compression benchmarking library. It runs classical codecs (LZ4, TAR, JPEG) and ML-based codecs (ONNX autoencoders exported from [elemental](../elemental)) through a shared `Encoder`/`Decoder` interface, and measures time, reproducibility, and multi-cycle degradation. Config via [Hydra](https://hydra.cc/).

## Tooling

- **Python env**: `uv` (defined in `pyproject.toml`). Always use `uv run` to execute Python.
- **Test runner**: `pytest` (config in `pyproject.toml`; `pythonpath = ["src"]`).
- **GPU**: optional `gpu` extra pulls `onnxruntime-gpu`; base install ships CPU `onnxruntime`.

## Commands

```bash
uv sync --extra dev                                                     # install + dev extras
uv run python src/runner.py                                             # default: lz4 + default benchmarks
uv run python src/runner.py compression_engine=jpeg                     # switch engine
uv run python src/runner.py compression_engine=compression_ae \
    ++compression_engine.encoder.model_path=/path/to/encoder.onnx \
    ++compression_engine.decoder.model_path=/path/to/decoder.onnx       # ONNX autoencoder
uv run python src/runner.py benchmark=degradation                       # include degradation curve
uv run python src/runner.py --config-name=compare                       # run all engines, write summary.json
uv run pytest tests/                                                     # all tests
uv run pytest tests/compression_engine/test_jpeg_engine.py -v           # single test file
```

## Non-Obvious Patterns

### Hydra Configs
All configs live in `config/` (singular, not `configs/`). Entry point: `config/config.yaml`. Sub-configs under `config/compression_engine/` and `config/benchmark/` are composed via `defaults:` lists. Objects are instantiated through `hydra.utils.instantiate` using `_target_`.

### Import Convention
`src/` is on `PYTHONPATH` (set via `pyproject.toml`'s `[tool.pytest.ini_options]` and Hydra's entry point). Import as `from compression_engine.base import Encoder` — **never** `from src.compression_engine...`.

### Encoder/Decoder Split
Every compression engine is a `CompressionEngine` that pairs an `Encoder` with a `Decoder`. Each side is independently instantiable so the same classes can back a streaming mode later (encode on machine A, decode on machine B). Encoded `bytes` must be self-describing — the decoder reconstructs without shared in-memory state.

### Canonical Image Format
Internally, images are **HWC uint8 RGB numpy arrays**. All engines and benchmarks must funnel inputs through `utils.image.ensure_array` at their public boundary and never branch on input type elsewhere. Float/CHW conversions happen inside ONNX engines only, via `to_chw_float01` / `from_chw_float01`.

### Benchmark Callable Contract
A `Benchmark` is a stateful callable: `__call__(original_image, decoded_image, compressed_data, encoding_time, decoding_time)` computes per-image metrics and appends them column-wise into `self._results` via `_add_result`. `summarize()` reduces `_results` to scalars/stats. The `Benchmarks` container fans each call out to all registered benchmarks.

### Multi-Engine Runs
`config/compare.yaml` composes every engine under `compression_engines.<name>` using Hydra's `group@package` directive. The runner (`_resolve_engine_configs`) detects `cfg.compression_engines` and iterates, instantiating a **fresh** `Benchmarks` per engine (because `Benchmark._results` is stateful). After the loop it writes `summary.json` alongside the per-engine `results_{name}.json` files.

### Engine Injection for Degradation
`DegradationBenchmark` re-encodes/decodes N times and needs a reference to the active engine. It is *not* passed in the Hydra config — the runner calls `Benchmarks.bind_engine(engine)`, which sets `compression_engine` on any benchmark that exposes it as `None`. If you add a benchmark that needs the engine, follow this same attribute-based pattern.

### Latent Serialization (ONNX engine)
`LatentSerializer` emits a self-describing binary header (magic `PKRL`, version, dtype code, shape, original HxW, plus int8 scale/zp when used). Encoder and decoder must agree on `dtype` (`float32` / `float16` / `int8`); everything else is recovered from the header. When adding a new dtype, update `_DTYPE_CODES`, both serialize/deserialize branches, and bump `_VERSION` if the header layout changes.

### ONNX Size Multiple
Autoencoders typically require spatial dimensions to be multiples of the stride (default 8). `_pad_to_multiple` reflection-pads the input NCHW tensor and records the original (H, W); the decoder crops back using those values from the latent header.

## Adding New Components

| Component | Steps |
|-----------|-------|
| **Compression engine** | `src/compression_engine/my_engine.py` with `MyEncoder(Encoder)` + `MyDecoder(Decoder)`. Add `config/compression_engine/my_engine.yaml` wiring `_target_: compression_engine.base.CompressionEngine` with nested `encoder`/`decoder` sub-configs. |
| **Benchmark** | Subclass `Benchmark` in `src/benchmark/`. Implement `__call__` (call `self._add_result(metrics)`) and `summarize`. Register in a `config/benchmark/*.yaml` benchmark list. |
| **Image transform** | Add to `src/utils/image.py`. Canonical internal form is HWC uint8 RGB. |

## Documentation

Hierarchical docs in every `src/` subdirectory:
- `README.md` — features and usage (for users)
- `ARCHITECTURE.md` — implementation details (for developers)

Outermost `README.md` is the GitHub project overview. Outermost `ARCHITECTURE.md` (if present) maps directories to features.

**Update both files in every directory you modify.** Also update the outermost `README.md` if changes affect project-level docs. If information is now inaccurate after your change, correct it — outdated docs are worse than no docs.

## Testing

Tests mirror `src/` structure under `tests/`. Fixtures live in `tests/conftest.py`. ONNX integration tests auto-skip if `onnxruntime` or the exported model files are unavailable — preserve that pattern when adding ONNX-dependent tests. Add tests that verify **new** behavior only — don't re-test unchanged code.

## Planning

If a task looks like it will have more than 3 steps, create a plan first and get alignment before executing.
