"""
Benchmarking tests for pyisolate RPC overhead measurement.

This module measures the overhead of proxied calls compared to local execution,
excluding setup costs (venv creation, process startup, etc.).

Benchmark categories:
1. Small arguments/return values (int, small strings)
2. Large arguments/return values (large arrays)
3. Small torch tensors (CPU and GPU)
4. Large torch tensors (CPU and GPU) with share_torch enabled
"""

import asyncio
import gc
import os
import statistics
import time
from typing import Any, Optional

import numpy as np
import psutil
import pytest
from tabulate import tabulate

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    ROCM_AVAILABLE = False
    if CUDA_AVAILABLE and hasattr(torch.version, "hip") and torch.version.hip is not None:
        ROCM_AVAILABLE = True

    # Set CUDA/ROCm device from environment variable if specified
    cuda_env = os.environ.get("PYISOLATE_CUDA_DEVICE")
    if CUDA_AVAILABLE and cuda_env is not None:
        torch.cuda.set_device(int(cuda_env))
        print(f"[PyIsolate] Using CUDA device {cuda_env}: {torch.cuda.get_device_name(int(cuda_env))}")
    elif CUDA_AVAILABLE:
        print(
            f"[PyIsolate] Using default CUDA device {torch.cuda.current_device()}: "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    elif ROCM_AVAILABLE:
        print("[PyIsolate] Using AMD ROCm backend.")
    elif XPU_AVAILABLE:
        print("[PyIsolate] Using Intel XPU backend.")
    else:
        print("[PyIsolate] No supported GPU backend available, exiting.")
        import sys
        sys.exit(1)
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from .test_integration import IntegrationTestBase
from shared import ExampleExtensionBase  # Add this import for type annotations


class BenchmarkResults:
    """Container for benchmark results with statistical analysis."""

    def __init__(self, name: str, times: list[float], memory_usage: Optional[dict[str, float]] = None):
        self.name = name
        self.times = times
        self.memory_usage = memory_usage or {}

        # Statistical measures
        self.mean = statistics.mean(times)
        self.median = statistics.median(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        self.min_time = min(times)
        self.max_time = max(times)

    def __repr__(self):
        return f"BenchmarkResults({self.name}: {self.mean:.4f}±{self.stdev:.4f}s)"


class BenchmarkRunner:
    """Manages benchmark execution and statistical analysis."""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 1000):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: list[BenchmarkResults] = []

    async def run_benchmark(
        self, name: str, benchmark_func, *args, measure_memory: bool = False, **kwargs
    ) -> BenchmarkResults:
        """Run a benchmark with warmup and statistical analysis."""

        print(f"\nRunning benchmark: {name}")

        # Warmup runs (not measured)
        print(f"  Warmup ({self.warmup_runs} runs)...")
        for i in range(self.warmup_runs):
            try:
                # Add timeout to detect stuck processes
                await asyncio.wait_for(benchmark_func(*args, **kwargs), timeout=30.0)
            except asyncio.TimeoutError as err:
                print(f"  Timeout during warmup run {i + 1}/{self.warmup_runs} - process may be stuck")
                raise RuntimeError(
                    f"Timeout during warmup for {name} - process may be stuck due to CUDA OOM"
                ) from err
            except (RuntimeError, Exception) as e:
                error_msg = str(e)
                if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                    print(f"  CUDA OOM during warmup run {i + 1}/{self.warmup_runs}: {error_msg}")
                    raise RuntimeError(f"CUDA out of memory during warmup for {name}: {error_msg}") from e
                else:
                    print(f"  Error during warmup run {i + 1}/{self.warmup_runs}: {e}")
                    raise

        # Force garbage collection before measuring
        gc.collect()

        # Benchmark runs (measured)
        print(f"  Measuring ({self.benchmark_runs} runs, this may take a while)...")
        times = []
        memory_before = None
        memory_after = None

        if measure_memory:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

        for i in range(self.benchmark_runs):
            try:
                start_time = time.perf_counter()
                # Add timeout to detect stuck processes
                await asyncio.wait_for(benchmark_func(*args, **kwargs), timeout=30.0)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except asyncio.TimeoutError as err:
                print(f"  Timeout during benchmark run {i + 1}/{self.benchmark_runs} - process may be stuck")
                raise RuntimeError(
                    f"Timeout during benchmark for {name} - process may be stuck due to CUDA OOM"
                ) from err
            except (RuntimeError, Exception) as e:
                error_msg = str(e)
                if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                    print(f"  CUDA OOM during benchmark run {i + 1}/{self.benchmark_runs}: {error_msg}")
                    raise RuntimeError(f"CUDA out of memory during benchmark for {name}: {error_msg}") from e
                else:
                    print(f"  Error during benchmark run {i + 1}/{self.benchmark_runs}: {e}")
                    raise

        memory_usage = {}
        if measure_memory:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = {
                "before_mb": memory_before,
                "after_mb": memory_after,
                "delta_mb": memory_after - memory_before,
            }

        result = BenchmarkResults(name, times, memory_usage)
        self.results.append(result)
        print(f"  Completed: {result.mean * 1000:.2f}±{result.stdev * 1000:.2f}ms")

        return result

    def print_summary(self):
        """Print a formatted summary of all benchmark results."""

        if not self.results:
            print("No benchmark results to display.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Create table data
        headers = ["Benchmark", "Mean (ms)", "Median (ms)", "Std Dev (ms)", "Min (ms)", "Max (ms)"]
        table_data = []

        for result in self.results:
            table_data.append(
                [
                    result.name,
                    f"{result.mean * 1000:.2f}",
                    f"{result.median * 1000:.2f}",
                    f"{result.stdev * 1000:.2f}",
                    f"{result.min_time * 1000:.2f}",
                    f"{result.max_time * 1000:.2f}",
                ]
            )

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Memory usage summary if available
        memory_results = [r for r in self.results if r.memory_usage]
        if memory_results:
            print("\nMEMORY USAGE")
            print("-" * 40)
            memory_headers = ["Benchmark", "Before (MB)", "After (MB)", "Delta (MB)"]
            memory_data = []

            for result in memory_results:
                memory_data.append(
                    [
                        result.name,
                        f"{result.memory_usage['before_mb']:.1f}",
                        f"{result.memory_usage['after_mb']:.1f}",
                        f"{result.memory_usage['delta_mb']:.1f}",
                    ]
                )

            print(tabulate(memory_data, headers=memory_headers, tablefmt="grid"))


@pytest.mark.asyncio
class TestRPCBenchmarks(IntegrationTestBase):
    """Benchmark tests for RPC call overhead."""

    benchmark_ext_shared: Optional[ExampleExtensionBase] = None
    runner: Optional[BenchmarkRunner] = None

    @pytest.fixture(autouse=True)
    async def setup_benchmark_environment(self):
        """Set up the benchmark environment once for all tests."""
        await self.setup_test_environment("benchmark")

        # Create benchmark extension with all required dependencies

        # self.create_extension(
        #     "benchmark_ext",
        #     benchmark_extension_code,
        #     dependencies=["numpy>=1.26.0", "torch>=2.0.0"] if TORCH_AVAILABLE else ["numpy>=1.26.0"],
        # )

        # Load extensions
        extensions_config: list[dict[str, Any]] = [{"name": "benchmark_ext"}]

        # Add share_torch config if available
        if TORCH_AVAILABLE:
            extensions_config.append({"name": "benchmark_ext_shared", "share_torch": True})

        self.extensions = await self.load_extensions(extensions_config[:1])  # Load one for now
        self.benchmark_ext = self.extensions[0]
        self.benchmark_ext_shared = None
        if TORCH_AVAILABLE and len(extensions_config) > 1:
            shared_exts = await self.load_extensions([extensions_config[1]])
            if shared_exts:
                self.benchmark_ext_shared = shared_exts[0]

        # Initialize benchmark runner
        self.runner = BenchmarkRunner(warmup_runs=3, benchmark_runs=15)

        yield

        # Cleanup
        await self.cleanup()

    async def test_small_data_benchmarks(self):
        """Benchmark small data argument/return value overhead."""

        print("\n" + "=" * 60)
        print("SMALL DATA BENCHMARKS")
        print("=" * 60)

        assert self.runner is not None
        # Integer benchmarks
        test_int = 42
        await self.runner.run_benchmark(
            "Small Int - Local Baseline", lambda: self.benchmark_ext.echo_int_local(test_int)
        )

        await self.runner.run_benchmark("Small Int - RPC Call", lambda: self.benchmark_ext.echo_int(test_int))

        # String benchmarks
        test_string = "hello world" * 10  # ~110 chars
        await self.runner.run_benchmark(
            "Small String - Local Baseline", lambda: self.benchmark_ext.echo_string_local(test_string)
        )

        await self.runner.run_benchmark(
            "Small String - RPC Call", lambda: self.benchmark_ext.echo_string(test_string)
        )

    async def test_large_data_benchmarks(self):
        """Benchmark large data argument/return value overhead."""

        print("\n" + "=" * 60)
        print("LARGE DATA BENCHMARKS")
        print("=" * 60)

        assert self.runner is not None
        # Large numpy array (10MB)
        large_array = np.random.random((1024, 1024))  # ~8MB float64

        await self.runner.run_benchmark(
            "Large Array - Local Baseline",
            lambda: self.benchmark_ext.process_large_array_local(large_array),
            measure_memory=True,
        )

        await self.runner.run_benchmark(
            "Large Array - RPC Call",
            lambda: self.benchmark_ext.process_large_array(large_array),
            measure_memory=True,
        )

        # Large byte data (50MB)
        large_bytes = b"x" * (50 * 1024 * 1024)  # 50MB

        await self.runner.run_benchmark(
            "Large Bytes - RPC Call",
            lambda: self.benchmark_ext.echo_large_bytes(large_bytes),
            measure_memory=True,
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    async def test_torch_tensor_benchmarks(self):
        """Benchmark torch tensor argument/return value overhead."""

        print("\n" + "=" * 60)
        print("TORCH TENSOR BENCHMARKS")
        print("=" * 60)

        assert self.runner is not None
        # Small tensor (CPU)
        with torch.inference_mode():
            small_tensor_cpu = torch.randn(100, 100)  # ~40KB

        await self.runner.run_benchmark(
            "Small Tensor CPU - Local Baseline",
            lambda: self.benchmark_ext.process_small_tensor_local(small_tensor_cpu),
        )

        await self.runner.run_benchmark(
            "Small Tensor CPU - RPC Call", lambda: self.benchmark_ext.process_small_tensor(small_tensor_cpu)
        )

        # Large tensor (CPU)
        with torch.inference_mode():
            large_tensor_cpu = torch.randn(1024, 1024)  # ~4MB

        await self.runner.run_benchmark(
            "Large Tensor CPU - RPC Call",
            lambda: self.benchmark_ext.process_large_tensor(large_tensor_cpu),
            measure_memory=True,
        )

        # GPU tests if available
        if CUDA_AVAILABLE:
            with torch.inference_mode():
                small_tensor_gpu = small_tensor_cpu.cuda()
                large_tensor_gpu = large_tensor_cpu.cuda()

            await self.runner.run_benchmark(
                "Small Tensor GPU - RPC Call",
                lambda: self.benchmark_ext.process_small_tensor(small_tensor_gpu),
            )

            await self.runner.run_benchmark(
                "Large Tensor GPU - RPC Call",
                lambda: self.benchmark_ext.process_large_tensor(large_tensor_gpu),
                measure_memory=True,
            )

    async def test_complex_call_patterns(self):
        """Benchmark complex call patterns (recursive, host calls)."""

        print("\n" + "=" * 60)
        print("COMPLEX CALL PATTERN BENCHMARKS")
        print("=" * 60)

        assert self.runner is not None
        # Recursive calls through host singleton
        await self.runner.run_benchmark(
            "Recursive Host Calls (depth=3)", lambda: self.benchmark_ext.recursive_host_call(3)
        )

        await self.runner.run_benchmark(
            "Recursive Host Calls (depth=5)", lambda: self.benchmark_ext.recursive_host_call(5)
        )

    async def test_print_final_summary(self):
        """Print the final benchmark summary (run last)."""

        # Small delay to ensure this runs last
        await asyncio.sleep(0.1)

        assert self.runner is not None
        self.runner.print_summary()

        # Basic assertions to ensure benchmarks ran
        assert len(self.runner.results) > 0, "No benchmark results found"
        assert self.runner is not None

        # Verify we have both local and RPC results for comparison
        local_results = [r for r in self.runner.results if "local" in r.name.lower()]
        rpc_results = [r for r in self.runner.results if "rpc" in r.name.lower()]

        assert len(local_results) > 0, "No local baseline results found"
        assert len(rpc_results) > 0, "No RPC benchmark results found"

        print("\nBenchmark completed successfully!")
        print(f"Total benchmarks run: {len(self.runner.results)}")
        print(f"Local baselines: {len(local_results)}")
        print(f"RPC benchmarks: {len(rpc_results)}")
