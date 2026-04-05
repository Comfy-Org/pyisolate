# PyIsolate RPC Benchmarks

This document describes the benchmarking suite for measuring RPC call overhead in pyisolate.

## Runner Scripts

Platform-specific benchmark runner scripts are included in this directory:

- `run_benchmarks_linux.sh` — Linux/macOS benchmark runner
- `run_benchmarks_windows.bat` — Windows CMD benchmark runner
- `run_benchmarks_windows.ps1` — Windows PowerShell benchmark runner
- `run_benchmarks_powershell_launcher.bat` — PowerShell execution policy wrapper
- `INSTRUCTIONS.md` — Step-by-step instructions for running benchmarks

## Overview

The benchmark suite measures the performance overhead of proxied calls compared to local execution, specifically excluding setup costs like virtual environment creation and process startup.

### Key Features

- **Comprehensive error handling**: Gracefully handles CUDA OOM errors and process timeouts
- **Detailed results reporting**: Shows successful, failed, and skipped tests separately
- **Multiple test modes**: Support for standard (no share_torch) and shared (share_torch) configurations
- **Statistical analysis**: Provides mean, standard deviation, min/max times for each benchmark

## Benchmark Categories

### 1. Small Data Benchmarks
**What it measures**: Overhead for calls with minimal data transfer
- Integer values (4-8 bytes)
- Small strings (~110 characters)

**Purpose**: Isolates the pure RPC protocol overhead without data serialization costs.

### 2. Large Data Benchmarks
**What it measures**: Overhead for calls with significant data transfer
- Large numpy arrays (~8MB)
- Large byte arrays (~50MB)

**Purpose**: Measures how RPC performance scales with data size and identifies serialization bottlenecks.

### 3. Torch Tensor Benchmarks
**What it measures**: Performance with PyTorch tensors across various sizes
- Tiny tensors (10x10, ~400B)
- Small tensors (100x100, ~40KB)
- Medium tensors (512x512, ~1MB)
- Large tensors (1024x1024, ~4MB)
- Image-sized tensors (3x8192x8192, ~800MB for 8K RGB)
- Model-sized tensors (40132x40132, ~6GB for modern LLM/diffusion models)
- CPU and GPU tensors
- With and without `share_torch` optimization

**Purpose**: Evaluates tensor serialization scaling, tensor-specific optimizations, and GPU memory transfer costs.

### 4. Complex Call Patterns
**What it measures**: Advanced RPC scenarios
- Recursive calls through host singletons
- Extension → Host → Extension call chains
- Proxied singleton performance

**Purpose**: Tests real-world usage patterns and nested call overhead.

## Running Benchmarks

### Quick Start
```bash
# Install dependencies
uv pip install -e ".[bench]"

# Quick benchmark using existing extensions
python benchmarks/simple_benchmark.py

# Include very large tensors (uses significant memory)
python benchmarks/simple_benchmark.py --large-tensors

# Full benchmark suite with statistical analysis
python benchmarks/benchmark.py

# Quick mode (fewer iterations)
python benchmarks/benchmark.py --quick
```

### Command-Line Options

#### `benchmark.py` Options

- `--quick`: Run with reduced iterations (2 warmup, 100 benchmark runs instead of 5 warmup, 1000 benchmark runs)
- `--no-torch`: Skip PyTorch tensor benchmarks entirely
- `--no-gpu`: Skip GPU benchmarks even if CUDA is available
- `--torch-mode {both,standard,shared}`: Control which torch configurations to test (default: shared)
  - `both`: Test both standard and share_torch configurations
  - `standard`: Test only without share_torch optimization
  - `shared`: Test only with share_torch optimization enabled

#### `simple_benchmark.py` Options

- `--large-tensors`: Include very large tensor benchmarks (6GB+ memory usage)

### Example Commands
```bash
# Test both standard and shared modes
python benchmarks/benchmark.py --torch-mode both

# Quick test without GPU
python benchmarks/benchmark.py --quick --no-gpu

# Test only standard mode without torch optimizations
python benchmarks/benchmark.py --torch-mode standard

# Run via pytest for more verbose output
pytest tests/test_benchmarks.py -v -s
```

## Understanding Results

### Results Format

The benchmark output is organized into three sections:

1. **Successful Benchmarks**: Tests that completed successfully with performance metrics
   - Test name (includes _standard or _shared suffix)
   - Mean time in milliseconds
   - Standard deviation
   - Min/Max times observed

2. **Failed Tests**: Tests that encountered errors
   - Test name
   - Error type (e.g., "CUDA OOM/Timeout")

3. **Skipped Tests**: Tests that couldn't run (e.g., when extension was stopped)
   - Test name
   - Reason for skipping

4. **Summary Statistics**: Overall counts of successful/failed/skipped tests

### Example Output

```
============================================================
RPC BENCHMARK RESULTS
============================================================

Successful Benchmarks:
+--------------------------+-------------+----------------+------------+------------+
| Test                     |   Mean (ms) |   Std Dev (ms) |   Min (ms) |   Max (ms) |
+==========================+=============+================+============+============+
| small_int_shared         |        0.31 |           0.06 |       0.26 |       0.69 |
+--------------------------+-------------+----------------+------------+------------+
| small_string_shared      |        0.34 |           0.07 |       0.26 |       0.86 |
+--------------------------+-------------+----------------+------------+------------+
| medium_tensor_gpu_shared |        1.38 |           0.30 |       1.09 |       2.72 |
+--------------------------+-------------+----------------+------------+------------+

Fastest result: 0.31ms

Failed Tests:
+----------------------+------------------+
| Test                 | Error            |
+======================+==================+
| model_6gb_gpu_shared | CUDA OOM/Timeout |
+----------------------+------------------+

Summary: 15 successful, 1 failed, 0 skipped (Total: 16)
```

### Metrics Reported
- **Mean**: Average time across all runs
- **Std Dev**: Standard deviation (consistency measure)
- **Min/Max**: Range of observed times
- **Fastest result**: The lowest mean time observed (baseline reference)

### Baseline Comparisons
Local execution baselines use the `@local_execution` decorator to run the same operations without RPC overhead. This isolates the pure network/serialization costs.

### Share Torch Configuration
Benchmarks test both configurations for PyTorch tensors:

- **Standard Mode** (`share_torch=False`): Tensors are serialized/deserialized for each call
- **Shared Mode** (`share_torch=True`): Tensors are shared via memory mapping (zero-copy)

The performance difference between these modes can be significant for large tensors.

### Memory Usage
For large data benchmarks, memory usage is tracked to identify potential leaks or excessive copying.

## Expected Performance Characteristics

### Typical Overhead Ranges
- **Small data**: 0.1-1ms overhead (protocol + serialization)
- **Large arrays**: Proportional to data size (serialization dominant)
- **Torch tensors**: Varies with `share_torch` optimization
- **Recursive calls**: Multiplicative overhead per hop

### Performance Factors
1. **Data size**: Larger payloads take longer to serialize/deserialize
2. **Data type**: Torch tensors have optimized paths vs generic Python objects
3. **Call depth**: Nested calls accumulate overhead
4. **Memory pressure**: Large transfers may trigger garbage collection

## Interpreting Results

### Good Performance Indicators
- Low standard deviation (consistent performance)
- Reasonable overhead percentages for use case
- Linear scaling with data size
- Memory usage remains stable

### Performance Issues
- High standard deviation (inconsistent performance)
- Exponential scaling with data size
- Memory usage growing over time
- Excessive overhead for small data

## Troubleshooting

### Common Issues
1. **High variability**: System load, garbage collection, or thermal throttling
2. **Memory growth**: Potential leaks or inefficient serialization
3. **GPU errors**: CUDA availability or memory issues
4. **Import errors**: Missing dependencies (torch, numpy, etc.)
5. **CUDA OOM errors**: Large tensors exceeding GPU memory

### Solutions
- Run benchmarks on idle system
- Use `--quick` mode for faster iteration
- Check `nvidia-smi` for GPU status
- Verify all dependencies installed with `pip install -e .[bench]`
- Use `--no-gpu` to skip GPU tests if encountering CUDA OOM
- Use `--torch-mode standard` to avoid tensor sharing issues

### Error Handling

The benchmark suite includes robust error handling:

- **Timeout detection**: Tests that hang (e.g., due to CUDA OOM during serialization) are detected via 30-second timeout
- **Process cleanup**: Failed extensions are automatically stopped and cleaned up
- **Graceful continuation**: Benchmarks continue running even after individual test failures
- **Clear error reporting**: Failed tests are reported with error reasons in the results table

## Benchmark Implementation

### Design Principles
1. **Setup Exclusion**: Extension creation and process startup excluded from timing
2. **Statistical Rigor**: Multiple runs with warmup to ensure stable results
3. **Baseline Comparison**: Local execution baselines for overhead calculation
4. **Real-world Scenarios**: Tests reflect actual usage patterns
5. **Error Resilience**: Individual test failures don't stop the entire benchmark suite

### Technical Details
- Uses `time.perf_counter()` for high-resolution timing
- Garbage collection forced before measurements
- Memory usage tracked with `psutil`
- Statistical analysis with built-in `statistics` module
- Timeout protection via `asyncio.wait_for()` (30 seconds per test)
- Process management through `ExtensionManager.stop_extension()`

### File Structure
- `benchmarks/benchmark.py`: Main benchmark suite with full statistical analysis
- `benchmarks/simple_benchmark.py`: Quick benchmarks for rapid testing
- `tests/test_benchmarks.py`: Benchmark runner class and test utilities

## Memory Benchmarking

### Overview

The memory benchmarking suite (`benchmarks/memory_benchmark.py`) measures RAM and VRAM usage across host and child processes with varying numbers of extensions and different tensor sharing configurations.

### Running Memory Benchmarks

```bash
# Run full memory benchmark suite
python benchmarks/memory_benchmark.py

# Test with custom extension counts
python benchmarks/memory_benchmark.py --counts 1,5,10,20,50

# Test up to 100 extensions
python benchmarks/memory_benchmark.py --max-extensions 100

# Only test large tensor sharing
python benchmarks/memory_benchmark.py --large-only

# Only test small tensor scaling
python benchmarks/memory_benchmark.py --small-only
```

### Memory Benchmark Features

1. **Process Memory Tracking**: Uses `psutil` to track RAM usage across process trees
2. **GPU Memory Tracking**: Uses `nvidia-ml-py3` to track VRAM usage per process
3. **Extension Scaling**: Tests memory usage with 1-100 extensions
4. **Tensor Sharing Analysis**: Compares memory usage with and without `share_torch`
5. **Large Tensor Tests**: Tests with 2GB tensors to verify memory sharing efficiency

### Memory Benchmark Output

The memory benchmark provides detailed tables showing:
- RAM usage per extension
- Memory overhead for tensor transfers
- VRAM usage for GPU tensors
- Memory savings from `share_torch` optimization

Example output:
```
MEMORY BENCHMARK SUMMARY
================================================================================

Baseline Memory Usage:
  RAM: 150.3 MB
  VRAM: 0.0 MB

CPU NO SHARE Results:
+-------------+----------------+-------------------+-------------+---------+
| Extensions  | RAM/Ext (MB)   | Tensor RAM (MB)   | VRAM (MB)   | Shared  |
+=============+================+===================+=============+=========+
| 1           | 45.2           | 1.1               | 0.0         | No      |
+-------------+----------------+-------------------+-------------+---------+
| 5           | 44.8           | 5.3               | 0.0         | No      |
+-------------+----------------+-------------------+-------------+---------+

2GB TENSOR SHARING TEST:
+--------------------+--------------------+--------------------------+------------------------+
| Config             | Tensor Size (MB)   | Distribution RAM (MB)    | RAM/Extension (MB)     |
+====================+====================+==========================+========================+
| share_torch=False  | 2048.0             | 10240.0                  | 2048.0                 |
+--------------------+--------------------+--------------------------+------------------------+
| share_torch=True   | 2048.0             | 512.0                    | 102.4                  |
+--------------------+--------------------+--------------------------+------------------------+

Memory Sharing Analysis:
  Memory saved with share_torch: 9728.0 MB (95.0%)
```

### Key Metrics

- **RAM/Extension**: Average memory overhead per extension process
- **Tensor RAM**: Additional RAM used for tensor distribution
- **VRAM**: GPU memory usage (if CUDA available)
- **Memory Sharing**: Whether tensors are shared (same memory address) or copied

## Contributing

When adding new benchmarks:
1. Follow the existing pattern in `benchmarks/benchmark.py` or `benchmarks/memory_benchmark.py`
2. Include error handling for potential failures
3. Add appropriate test data sizes
4. Document what the benchmark measures
5. Update this README with new benchmark descriptions
6. Test with various `--torch-mode` options to ensure compatibility
7. For memory benchmarks, ensure proper cleanup to avoid memory leaks
