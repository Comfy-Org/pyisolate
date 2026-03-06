#!/usr/bin/env python3
"""
Standalone benchmark script for pyisolate RPC overhead measurement.

Usage:
    python benchmark.py [--quick] [--no-torch] [--no-gpu] [--torch-mode {both,standard,shared}]
"""

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

# Add project root to path for pyisolate imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmark_harness import BenchmarkHarness  # noqa: E402

from pyisolate import ExtensionBase, ExtensionConfig, ProxiedSingleton  # noqa: E402

try:
    from importlib.util import find_spec

    TORCH_AVAILABLE = find_spec("torch") is not None
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


# =============================================================================
# Host-side Classes
# =============================================================================


class DatabaseSingleton(ProxiedSingleton):
    """Simple dictionary-based singleton for testing state."""

    def __init__(self):
        self._db = {}

    async def set_value(self, key: str, value):
        self._db[key] = value

    async def get_value(self, key: str):
        return self._db.get(key)


class BenchmarkExtensionWrapper(ExtensionBase):
    """
    Host-side wrapper that proxies calls to the isolated extension.
    """

    async def on_module_loaded(self, module):
        """Called when the isolated module is loaded."""
        if not getattr(module, "benchmark_entrypoint", None):
            raise RuntimeError(f"Module {module.__name__} missing 'benchmark_entrypoint'")

        # Instantiate the child-side extension object
        self.extension = module.benchmark_entrypoint()
        await self.extension.initialize()

    async def do_stuff(self, value):
        return await self.extension.do_stuff(value)


# =============================================================================
# Child-side Code (Injected via string)
# =============================================================================

BENCHMARK_EXTENSION_CODE = '''
import asyncio
import numpy as np
from pyisolate import ExtensionBase, ProxiedSingleton, local_execution

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Re-define Singleton interface on child side so it knows what to proxy
class DatabaseSingleton(ProxiedSingleton):
    def __init__(self):
        self._db = {}
    async def set_value(self, key, value): pass
    async def get_value(self, key): pass

class BenchmarkExtension:
    """Child-side extension implementation."""

    async def initialize(self):
        pass

    async def prepare_shutdown(self):
        pass

    async def do_stuff(self, value):
        return f"Processed: {value}"

    # ========================================
    # Small Data Benchmarks
    # ========================================

    async def echo_int(self, value: int) -> int:
        return value

    async def echo_string(self, value: str) -> str:
        return value

    @local_execution
    def echo_int_local(self, value: int) -> int:
        return value

    @local_execution
    def echo_string_local(self, value: str) -> str:
        return value

    # ========================================
    # Large Data Benchmarks
    # ========================================

    async def process_large_array(self, array: np.ndarray) -> int:
        return array.size

    async def echo_large_bytes(self, data: bytes) -> int:
        return len(data)

    @local_execution
    def process_large_array_local(self, array: np.ndarray) -> int:
        return array.size

    # ========================================
    # Torch Tensor Benchmarks
    # ========================================

    async def process_small_tensor(self, tensor) -> tuple:
        if not TORCH_AVAILABLE: return (0, "cpu")
        return (tensor.numel(), str(tensor.device))

    async def process_large_tensor(self, tensor) -> tuple:
        if not TORCH_AVAILABLE: return (0, "cpu")
        return (tensor.numel(), str(tensor.device))

    @local_execution
    def process_small_tensor_local(self, tensor) -> tuple:
        if not TORCH_AVAILABLE: return (0, "cpu")
        return (tensor.numel(), str(tensor.device))

    # ========================================
    # Recursive/Complex Call Patterns
    # ========================================

    async def recursive_host_call(self, depth: int) -> int:
        """Make recursive calls through host singleton."""
        if depth <= 0:
            return 0

        db = DatabaseSingleton()
        await db.set_value(f"depth_{depth}", depth)
        value = await db.get_value(f"depth_{depth}")
        return value + await self.recursive_host_call(depth - 1)


def benchmark_entrypoint():
    """Entry point."""
    return BenchmarkExtension()
'''


class BenchmarkResult:
    def __init__(self, mean, stdev, min_time, max_time):
        self.mean = mean
        self.stdev = stdev
        self.min_time = min_time
        self.max_time = max_time


class SimpleRunner:
    """Minimal runner to replace TestRPCBenchmarks.runner."""

    def __init__(self, warmup_runs=5, benchmark_runs=1000):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    async def run_benchmark(self, name, func):
        import time

        times = []

        # Warmup
        for _ in range(self.warmup_runs):
            await func()

        # Benchmark
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            await func()
            end = time.perf_counter()
            times.append(end - start)

        return BenchmarkResult(
            statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0, min(times), max(times)
        )


async def run_benchmarks(
    quick: bool = False, no_torch: bool = False, no_gpu: bool = False, torch_mode: str = "both"
):
    print("PyIsolate RPC Benchmark Suite (Refactored for 1.0)")
    print("=" * 60)

    harness = BenchmarkHarness()
    await harness.setup_test_environment("benchmark")

    runner = SimpleRunner(
        warmup_runs=2 if quick else 5,
        benchmark_runs=100 if quick else 1000,
    )

    try:
        torch_available = TORCH_AVAILABLE and not no_torch

        # Define extensions to create
        extensions_config = []
        if torch_mode in ["both", "standard"]:
            harness.create_extension(
                "benchmark_ext",
                dependencies=["numpy>=1.26.0", "torch>=2.0.0"] if torch_available else ["numpy>=1.26.0"],
                share_torch=False,
                extension_code=BENCHMARK_EXTENSION_CODE,
            )
            extensions_config.append({"name": "benchmark_ext", "share": False})

        if torch_mode in ["both", "shared"] and torch_available:
            harness.create_extension(
                "benchmark_ext_shared",
                dependencies=["numpy>=1.26.0", "torch>=2.0.0"],
                share_torch=True,
                extension_code=BENCHMARK_EXTENSION_CODE,
            )
            extensions_config.append({"name": "benchmark_ext_shared", "share": True})

        # Load Extensions using Manager
        manager = harness.get_manager(BenchmarkExtensionWrapper)

        ext_standard = None
        ext_shared = None

        for cfg in extensions_config:
            name = cfg["name"]
            share_torch = cfg["share"]
            print(f"Loading extension {name} (share_torch={share_torch})...")

            # Reconstruct minimal deps for config (manager uses this for venv check/install)
            deps = ["numpy>=1.26.0"]
            if torch_available:
                deps.append("torch>=2.0.0")

            config = ExtensionConfig(
                name=name,
                module_path=str(harness.test_root / "extensions" / name),
                isolated=True,
                dependencies=deps,
                apis=[DatabaseSingleton],  # Host must allow the singleton
                share_torch=share_torch,
            )

            ext = manager.load_extension(config)
            if name == "benchmark_ext":
                ext_standard = ext
            else:
                ext_shared = ext

        print("Extensions loaded.\n")

        # Define Test Data
        test_data = [
            ("small_int", 42),
            ("small_string", "hello world"),
        ]

        runner_results = {}

        # --- Run Benchmarks ---
        # Note: In a full implementation, we'd replicate the comprehensive test suite.
        # Here we verify core functionality by running the 'do_stuff' generic method.
        # This confirms RPC, Serialization, and Process Isolation are working.

        target_extensions = []
        if ext_standard:
            target_extensions.append(("Standard", ext_standard))
        if ext_shared:
            target_extensions.append(("Shared", ext_shared))

        for mode_name, ext in target_extensions:
            print(f"--- Benchmarking {mode_name} Mode ---")
            for data_name, data_val in test_data:
                bench_name = f"{mode_name}_{data_name}"

                async def func(bound_ext=ext, bound_value=data_val):
                    return await bound_ext.do_stuff(bound_value)

                print(f"Running {bench_name}...")
                try:
                    res = await runner.run_benchmark(bench_name, func)
                    runner_results[bench_name] = res
                except Exception as e:
                    print(f"FAILED: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        headers = ["Test", "Mean (ms)", "Std Dev (ms)"]
        table_data = []
        for name, res in runner_results.items():
            table_data.append([name, f"{res.mean * 1000:.3f}", f"{res.stdev * 1000:.3f}"])

        if TABULATE_AVAILABLE:
            print(tabulate(table_data, headers=headers))
        else:
            for row in table_data:
                print(row)

    finally:
        await harness.cleanup()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PyIsolate 1.0 Benchmark")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-torch", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--torch-mode", default="both")

    args = parser.parse_args()

    if find_spec("numpy") is None or find_spec("psutil") is None:
        print("Please install dependencies: pip install numpy psutil tabulate")
        return 1

    asyncio.run(run_benchmarks(args.quick, args.no_torch, args.no_gpu, args.torch_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
