# PyIsolate Benchmark Instructions

Thank you for helping collect benchmark data! This document explains how to run the benchmarks on your system.

## Overview

The benchmark scripts will:
1. Install necessary tools and dependencies
2. Run performance benchmarks to measure RPC overhead
3. Run memory benchmarks to measure RAM and VRAM usage
4. Collect system information
5. Save all results to a single file

## Prerequisites

- Python 3.9 or higher
- Internet connection (for downloading dependencies)
- At least 16GB of RAM recommended (8GB minimum)
- For GPU benchmarks: NVIDIA GPU with CUDA support (optional)

## Instructions

### Windows Users

1. Download or clone this repository to your local machine
2. Open Command Prompt (cmd) or PowerShell
3. Navigate to the pyisolate directory:
   ```
   cd path\to\pyisolate
   ```
4. Run the benchmark script:
   ```
   run_benchmarks_windows.bat
   ```
5. Follow the on-screen instructions
6. When complete, send back the file named `benchmark_results_COMPUTERNAME_TIMESTAMP.txt`

### Linux/macOS Users

1. Download or clone this repository to your local machine
2. Open Terminal
3. Navigate to the pyisolate directory:
   ```
   cd /path/to/pyisolate
   ```
4. Run the benchmark script:
   ```
   ./run_benchmarks_linux.sh
   ```
5. Follow the on-screen instructions
6. When complete, send back the file named `benchmark_results_hostname_timestamp.txt`

## What to Expect

- **First Run**: The script will prompt you to install `uv` if it's not already installed
- **Installation**: The script will automatically install PyTorch with appropriate CUDA support
- **Duration**: The full benchmark suite takes approximately 10-20 minutes
- **Memory Usage**: Some tests may use significant RAM (up to 6GB) and VRAM
- **Errors**: If tests fail due to out-of-memory errors, this is expected and will be noted in the results

## Troubleshooting

### "uv not found" Error

The script requires `uv` for fast package management. Install it using:

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### PyTorch Installation Issues

If PyTorch installation fails:
1. The script will try to install a CPU-only version automatically
2. You can manually install PyTorch from https://pytorch.org/get-started/locally/
3. The benchmarks will still run (with some GPU tests skipped)

### Out of Memory Errors

If you see "CUDA out of memory" or similar errors:
- This is expected for systems with limited VRAM
- The script will continue and note which tests failed
- Results are still valuable!

### Permission Denied (Linux/macOS)

If you get "permission denied" when running the script:
```bash
chmod +x run_benchmarks_linux.sh
./run_benchmarks_linux.sh
```

## What Data is Collected

The benchmark results file contains:
- System specifications (OS, CPU, RAM, GPU)
- Python and package versions
- Performance benchmark results (RPC call timings)
- Memory usage measurements
- Any errors encountered during testing

No personal data is collected.

## Questions?

If you encounter any issues not covered here, please include:
1. The complete error message
2. Your operating system and version
3. Any steps you tried to resolve the issue

Thank you for your help in benchmarking PyIsolate!
