#!/usr/bin/env python3
"""
Standalone benchmark script for pyisolate RPC overhead measurement.

Usage:
    python benchmark.py [--quick] [--no-torch] [--no-gpu]

Options:
    --quick     Run fewer iterations for faster results
    --no-torch  Skip torch tensor benchmarks
    --no-gpu    Skip GPU benchmarks even if CUDA is available
"""

import argparse
import asyncio
import platform
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from tests.test_benchmarks import TestRPCBenchmarks  # noqa: E402
from example.shared import ExampleExtensionBase  # Corrected import path

# Try to import tabulate globally
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    def tabulate(*args, **kwargs):
        return "[tabulate not available]"

# pyright: reportMissingImports=false


async def run_benchmarks(
    quick: bool = False, no_torch: bool = False, no_gpu: bool = False, torch_mode: str = "both",
):
    """Run all benchmarks with the specified options."""

    print("PyIsolate RPC Benchmark Suite")
    print("=" * 50)
    print(f"Quick mode: {quick}")
    print(f"Skip torch: {no_torch}")
    print(f"Skip GPU: {no_gpu}")
    print(f"Torch mode: {torch_mode}")
    print()

    # Create test instance
    test_instance = TestRPCBenchmarks()

    # Override benchmark runner settings for quick mode
    if quick:
        test_instance.runner = None  # Will be created in setup with different settings

    try:
        # Setup manually (not using pytest fixture)
        print("Setting up benchmark environment...")
        await test_instance.setup_test_environment("benchmark")

        # Create benchmark extension with all required dependencies
        benchmark_extension_code = '''
import asyncio
import numpy as np
from shared import ExampleExtension, DatabaseSingleton
from pyisolate import local_execution
from pyisolate._internal.gpu_utils import (
    maybe_serialize_tensor,
    maybe_deserialize_tensor,
    maybe_to_dlpack,
    maybe_from_dlpack,
)
import os
try:
    import torch
    TORCH_AVAILABLE = True
    # Set CUDA or XPU device if specified in environment
    device_idx = os.environ.get("PYISOLATE_CUDA_DEVICE")
    if device_idx is not None:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(int(device_idx))
            elif torch.cuda.is_available():
                torch.cuda.set_device(int(device_idx))
        except Exception as e:
            print(f"[Extension] Failed to set device: {e}")
except ImportError:
    TORCH_AVAILABLE = False

class BenchmarkExtension(ExampleExtension):
    """Extension with methods for benchmarking RPC overhead."""

    async def initialize(self):
        """Initialize the benchmark extension."""
        pass

    async def prepare_shutdown(self):
        """Clean shutdown of benchmark extension."""
        pass

    async def do_stuff(self, value):
        """Required abstract method from ExampleExtension."""
        return f"Processed: {value}"

    # ========================================
    # Small Data Benchmarks
    # ========================================

    async def echo_int(self, value: int) -> int:
        """Echo an integer value."""
        return value

    async def echo_string(self, value: str) -> str:
        """Echo a string value."""
        return value

    @local_execution
    def echo_int_local(self, value: int) -> int:
        """Local execution baseline for integer echo."""
        return value

    @local_execution
    def echo_string_local(self, value: str) -> str:
        """Local execution baseline for string echo."""
        return value

    # ========================================
    # Large Data Benchmarks
    # ========================================

    async def process_large_array(self, array: np.ndarray) -> int:
        """Process a large numpy array and return its size."""
        return array.size

    async def echo_large_bytes(self, data: bytes) -> int:
        """Echo large byte data and return its length."""
        return len(data)

    @local_execution
    def process_large_array_local(self, array: np.ndarray) -> int:
        """Local execution baseline for large array processing."""
        return array.size

    # ========================================
    # Torch Tensor Benchmarks
    # ========================================

    async def process_small_tensor(self, tensor) -> tuple:
        """Process a small torch tensor."""
        if not TORCH_AVAILABLE:
            return (0, "cpu")
        return (tensor.numel(), str(tensor.device))

    async def process_large_tensor(self, tensor) -> tuple:
        """Process a large torch tensor."""
        if not TORCH_AVAILABLE:
            return (0, "cpu")
        return (tensor.numel(), str(tensor.device))

    @local_execution
    def process_small_tensor_local(self, tensor) -> tuple:
        """Local execution baseline for small tensor processing."""
        if not TORCH_AVAILABLE:
            return (0, "cpu")
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

def example_entrypoint():
    """Entry point for the benchmark extension."""
    return BenchmarkExtension()
'''

        torch_available = not no_torch
        try:
            import torch
        except ImportError:
            torch_available = False

        # Create extensions based on torch_mode parameter
        extensions_to_create = []

        if torch_mode in ["both", "standard"]:
            # Create extension WITHOUT share_torch (standard serialization)
            test_instance.create_extension(
                "benchmark_ext",
                dependencies=["numpy>=1.26.0", "torch>=2.0.0"] if torch_available else ["numpy>=1.26.0"],
                share_torch=False,
                extension_code=benchmark_extension_code,
            )
            extensions_to_create.append({"name": "benchmark_ext"})

        if torch_mode in ["both", "shared"] and torch_available:
            # Create extension WITH share_torch (if torch available)
            test_instance.create_extension(
                "benchmark_ext_shared",
                dependencies=["numpy>=1.26.0", "torch>=2.0.0"],
                share_torch=True,
                extension_code=benchmark_extension_code,
            )
            extensions_to_create.append({"name": "benchmark_ext_shared"})

        # Load extensions
        test_instance.extensions = await test_instance.load_extensions(extensions_to_create)

        # Assign extension references based on what was created
        test_instance.benchmark_ext: Optional[ExampleExtensionBase] = None
        test_instance.benchmark_ext_shared: Optional[ExampleExtensionBase] = None

        for i, ext_config in enumerate(extensions_to_create):
            if ext_config["name"] == "benchmark_ext":
                test_instance.benchmark_ext = test_instance.extensions[i]
            elif ext_config["name"] == "benchmark_ext_shared":
                test_instance.benchmark_ext_shared = test_instance.extensions[i]

        # Initialize benchmark runner
        from tests.test_benchmarks import BenchmarkRunner

        if quick:
            test_instance.runner = BenchmarkRunner(warmup_runs=2, benchmark_runs=100)
            print("Using quick mode: 2 warmup runs, 100 benchmark runs")
        else:
            test_instance.runner = BenchmarkRunner(warmup_runs=5, benchmark_runs=1000)
            print("Using standard mode: 5 warmup runs, 1000 benchmark runs")

        # Run simplified benchmarks using do_stuff method
        print("\n1. Running RPC overhead benchmarks...")
        print(f"   Torch mode: {torch_mode}")
        if torch_mode == "both":
            print("   NOTE: Testing both standard (no share_torch) and shared (share_torch) configurations")
        elif torch_mode == "standard":
            print("   NOTE: Testing only standard configuration (no share_torch)")
        elif torch_mode == "shared":
            print("   NOTE: Testing only shared configuration (share_torch enabled)")
            if not torch_available:
                print("   WARNING: Torch not available, shared mode will be skipped")

        # Simple benchmark data
        test_data = [
            ("small_int", 42),
            ("small_string", "hello world"),
            ("medium_string", "hello world" * 100),
            ("large_string", "x" * 10000),
        ]

        if not no_torch:
            try:
                import torch

                torch_available = True

                # Store tensor specifications instead of actual tensors to avoid memory issues
                tensor_specs = [
                    ("tiny_tensor", (10, 10)),  # 100 elements, ~400B
                    ("small_tensor", (100, 100)),  # 10K elements, ~40KB
                    ("medium_tensor", (512, 512)),  # 262K elements, ~1MB
                    ("large_tensor", (1024, 1024)),  # 1M elements, ~4MB
                    ("image_8k", (3, 8192, 8192)),  # 201M elements, ~800MB (8K RGB image)
                ]

                for name, size in tensor_specs:
                    try:
                        print(f"  Creating {name} tensor {size}...")

                        with torch.inference_mode():
                            tensor = torch.randn(*size)
                        test_data.append((f"{name}_cpu", tensor))

                        size_gb = (tensor.numel() * 4) / (1024**3)
                        print(f"    CPU tensor created successfully ({size_gb:.2f}GB)")

                        # Only create GPU tensor if we have sufficient memory and it's not too large
                        if not no_gpu and torch.cuda.is_available():
                            try:
                                # Skip GPU for very large tensors to avoid OOM
                                if name == "image_8k":
                                    print(f"    Creating GPU version of {name} (may use significant VRAM)...")
                                    with torch.inference_mode():
                                        gpu_tensor = tensor.cuda()
                                    test_data.append((f"{name}_gpu", gpu_tensor))
                                    print("    GPU tensor created successfully")
                                else:
                                    with torch.inference_mode():
                                        gpu_tensor = tensor.cuda()
                                    test_data.append((f"{name}_gpu", gpu_tensor))
                                    print("    GPU tensor created successfully")
                            except RuntimeError as gpu_e:
                                print(f"    GPU tensor failed: {gpu_e}")

                        # --- XPU support: create XPU tensors if available ---
                        if not no_gpu and hasattr(torch, "xpu") and torch.xpu.is_available():
                            try:
                                if name == "image_8k":
                                    print(f"    Creating XPU version of {name} (may use significant VRAM)...")
                                    with torch.inference_mode():
                                        xpu_tensor = tensor.to("xpu")
                                    test_data.append((f"{name}_xpu", xpu_tensor))
                                    print("    XPU tensor created successfully")
                                else:
                                    with torch.inference_mode():
                                        xpu_tensor = tensor.to("xpu")
                                    test_data.append((f"{name}_xpu", xpu_tensor))
                                    print("    XPU tensor created successfully")
                            except RuntimeError as xpu_e:
                                print(f"    XPU tensor failed: {xpu_e}")
                    except RuntimeError as e:
                        print(f"  Skipping {name}: {e}")

            except ImportError:
                torch_available = False
                print("  PyTorch not available, skipping tensor benchmarks")

        # Add numpy arrays of various sizes
        import numpy as np

        array_sizes = [
            ("small_array", (100, 100)),  # 10K elements, ~80KB
            ("medium_array", (512, 512)),  # 262K elements, ~2MB
            ("large_array", (1024, 1024)),  # 1M elements, ~8MB
            ("huge_array", (2048, 2048)),  # 4M elements, ~32MB
        ]

        for name, size in array_sizes:
            try:
                array = np.random.random(size)
                test_data.append((name, array))
            except MemoryError as e:
                print(f"  Skipping {name}: {e}")

        # Add the 6GB model test at the very end if torch is available
        if torch_available and not no_torch:
            try:
                print("  Creating model_6gb tensor (40132, 40132) (WARNING: This will use ~6GB RAM)...")
                with torch.inference_mode():
                    model_6gb_tensor = torch.randn(40132, 40132)
                test_data.append(("model_6gb_cpu", model_6gb_tensor))

                size_gb = (model_6gb_tensor.numel() * 4) / (1024**3)
                print(f"    CPU tensor created successfully ({size_gb:.2f}GB)")

                # Try GPU version if available
                if not no_gpu and torch.cuda.is_available():
                    try:
                        print("    Creating GPU version of model_6gb (may use significant VRAM)...")
                        with torch.inference_mode():
                            gpu_tensor = model_6gb_tensor.cuda()
                        test_data.append(("model_6gb_gpu", gpu_tensor))
                        print("    GPU tensor created successfully")
                    except RuntimeError as gpu_e:
                        print(f"    GPU tensor failed: {gpu_e}")
                # --- XPU support: create XPU tensor for 6GB model if available ---
                if not no_gpu and hasattr(torch, "xpu") and torch.xpu.is_available():
                    try:
                        print("    Creating XPU version of model_6gb (may use significant VRAM)...")
                        with torch.inference_mode():
                            xpu_tensor = model_6gb_tensor.to("xpu")
                        test_data.append(("model_6gb_xpu", xpu_tensor))
                        print("    XPU tensor created successfully")
                    except RuntimeError as xpu_e:
                        print(f"    XPU tensor failed: {xpu_e}")
            except RuntimeError as e:
                print(f"  Skipping model_6gb: {e}")

        from tests.test_benchmarks import BenchmarkRunner

        runner = BenchmarkRunner(warmup_runs=2 if quick else 5, benchmark_runs=100 if quick else 1000)

        print(
            f"  Using {'quick' if quick else 'standard'} mode: {runner.warmup_runs} warmup, "
            f"{runner.benchmark_runs} benchmark runs"
        )

        results = {}
        failed_tests = {}  # Track failed tests with error messages
        skipped_tests = {}  # Track skipped tests when extension is not available
        for name, data in test_data:
            print(f"  Testing {name}...")

            # Test with standard extension (no share_torch) if available
            if test_instance.benchmark_ext is not None:

                async def benchmark_func(data=data):
                    return await test_instance.benchmark_ext.do_stuff(data)

                try:
                    result = await runner.run_benchmark(f"{name} (standard)", benchmark_func)
                    results[f"{name}_standard"] = result
                except (RuntimeError, asyncio.TimeoutError, Exception) as e:
                    error_msg = str(e)
                    test_name = f"{name}_standard"

                    if (
                        "CUDA error: out of memory" in error_msg
                        or "out of memory" in error_msg.lower()
                        or "Timeout" in error_msg
                    ):
                        print(f"    Standard failed with CUDA OOM/timeout: {name}")
                        print(f"    Error details: {error_msg[:200]}...")
                        failed_tests[test_name] = "CUDA OOM/Timeout"

                        # Stop the extension to clean up the stuck process
                        try:
                            test_instance.manager.stop_extension("benchmark_ext")
                            test_instance.benchmark_ext = None
                        except Exception as stop_e:
                            print(f"    Failed to stop extension: {stop_e}")
                    else:
                        print(f"    Standard failed: {e}")
                        failed_tests[test_name] = str(e)[:100]
            elif torch_mode in ["both", "standard"]:
                # Extension should have been tested but was stopped due to previous error
                test_name = f"{name}_standard"
                skipped_tests[test_name] = "Extension stopped"

            # Test with share_torch extension (if available and torch tensor)
            if test_instance.benchmark_ext_shared is not None:
                # For torch tensors, always test shared mode
                # For other data types, test shared mode only if torch_mode includes it
                should_test_shared = torch_mode in ["both", "shared"]

                if should_test_shared:
                    print(f"  Testing {name} with share_torch...")

                    async def benchmark_func_shared(data=data):
                        return await test_instance.benchmark_ext_shared.do_stuff(data)

                    try:
                        result = await runner.run_benchmark(f"{name} (share_torch)", benchmark_func_shared)
                        results[f"{name}_shared"] = result
                    except (RuntimeError, asyncio.TimeoutError, Exception) as e:
                        error_msg = str(e)
                        test_name = f"{name}_shared"

                        if (
                            "CUDA error: out of memory" in error_msg
                            or "out of memory" in error_msg.lower()
                            or "Timeout" in error_msg
                        ):
                            print(f"    Share_torch failed with CUDA OOM/timeout: {name}")
                            print(f"    Error details: {error_msg[:200]}...")
                            failed_tests[test_name] = "CUDA OOM/Timeout"

                            # Stop the extension to clean up the stuck process
                            try:
                                test_instance.manager.stop_extension("benchmark_ext_shared")
                                test_instance.benchmark_ext_shared = None
                            except Exception as stop_e:
                                print(f"    Failed to stop extension: {stop_e}")
                        else:
                            print(f"    Share_torch failed: {e}")
                            failed_tests[test_name] = str(e)[:100]
            else:
                # Extension is None (either not created or was stopped)
                should_test_shared = torch_mode in ["both", "shared"]
                if should_test_shared:
                    test_name = f"{name}_shared"
                    skipped_tests[test_name] = "Extension stopped"

        # Print summary
        print("\n" + "=" * 60)
        print("RPC BENCHMARK RESULTS")
        print("=" * 60)

        # Print successful results
        if results:
            if TABULATE_AVAILABLE:
                print("\nSuccessful Benchmarks:")
                headers = ["Test", "Mean (ms)", "Std Dev (ms)", "Min (ms)", "Max (ms)"]
                table_data = []

                for name, result in results.items():
                    table_data.append(
                        [
                            name,
                            f"{result.mean * 1000:.2f}",
                            f"{result.stdev * 1000:.2f}",
                            f"{result.min_time * 1000:.2f}",
                            f"{result.max_time * 1000:.2f}",
                        ]
                    )

                print(tabulate(table_data, headers=headers, tablefmt="grid"))

                # Show fastest result for reference
                baseline = min(r.mean for r in results.values())
                print(f"\nFastest result: {baseline * 1000:.2f}ms")
            else:
                print("\nSuccessful Benchmarks:")
                for name, result in results.items():
                    print(
                        f"    {name}: Mean={result.mean_time * 1000:.2f}ms, "
                        f"Std={result.stdev * 1000:.2f}ms, Min={result.min_time * 1000:.2f}ms, "
                        f"Max={result.max_time * 1000:.2f}ms"
                    )

        # Print failed tests
        if failed_tests:
            print("\nFailed Tests:")
            failed_headers = ["Test", "Error"]
            failed_data = [[name, error] for name, error in failed_tests.items()]
            if TABULATE_AVAILABLE:
                print(tabulate(failed_data, headers=failed_headers, tablefmt="grid"))
            elif failed_data:
                print("[tabulate not available] Failed tests:")
                for row in failed_data:
                    print(row)

        # Print skipped tests
        if skipped_tests:
            print("\nSkipped Tests:")
            skipped_headers = ["Test", "Reason"]
            skipped_data = [[name, reason] for name, reason in skipped_tests.items()]
            print(tabulate(skipped_data, headers=skipped_headers, tablefmt="grid"))

        # Print summary statistics
        total_tests = len(results) + len(failed_tests) + len(skipped_tests)
        if total_tests > 0:
            print(
                f"\nSummary: {len(results)} successful, {len(failed_tests)} failed, "
                f"{len(skipped_tests)} skipped (Total: {total_tests})"
            )

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        import contextlib

        with contextlib.suppress(Exception):
            await test_instance.cleanup()

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pyisolate RPC benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py                 # Run full benchmark suite
    python benchmark.py --quick         # Quick benchmark with fewer runs
    python benchmark.py --no-torch      # Skip torch benchmarks
    python benchmark.py --quick --no-gpu  # Quick mode without GPU tests
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Run fewer iterations for faster results")

    parser.add_argument("--no-torch", action="store_true", help="Skip torch tensor benchmarks")

    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks even if CUDA is available")

    parser.add_argument(
        "--torch-mode",
        choices=["both", "standard", "shared"],
        default="shared",
        help="Which torch mode to test: both, standard (no share_torch), or shared (share_torch only)",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "xpu"],
        default="auto",
        help="Device backend to use: auto (default), cuda (NVIDIA/AMD ROCm), or xpu (Intel oneAPI)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device index (int) or 'cpu' to force CPU mode",
    )

    args = parser.parse_args()

    # Check dependencies
    try:
        import numpy  # noqa: F401
        import psutil  # noqa: F401
        # import tabulate  # noqa: F401 # This line is now handled globally
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install benchmark dependencies with:")
        print("    pip install -e .[bench]")
        return 1

    # Set device and backend
    backend = args.backend
    device_arg = args.device
    if device_arg is not None and str(device_arg).lower() == "cpu":
        backend = "cpu"
        device_idx = None
        print("[PyIsolate] Forcing CPU mode due to --device=cpu")
        args.no_gpu = True
    elif device_arg is not None:
        try:
            device_idx = int(device_arg)
        except ValueError:
            print(f"Invalid --device value: {device_arg}. Must be integer or 'cpu'.")
            sys.exit(1)
    else:
        device_idx = None

    device_str = "cpu"
    device_name = "cpu"
    backend_used = "cpu"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
        # Auto backend selection
        if backend == "auto":
            if cuda_available:
                backend = "cuda"
            elif xpu_available:
                backend = "xpu"
            else:
                backend = "cpu"

        if backend == "cuda":
            if not cuda_available:
                print("[PyIsolate] CUDA backend requested but not available. Exiting.")
                sys.exit(1)
            # Only check for ROCm on Linux
            if platform.system() == "Linux":
                torch_version = getattr(torch, 'version', None)
                hip_version = getattr(torch_version, 'hip', None) if torch_version else None
                if hip_version is not None:
                    print("[PyIsolate] ROCm (AMD) backend detected on Linux.")
            # On Windows, just use CUDA if available
            if device_idx is not None:
                torch.cuda.set_device(device_idx)
                device_str = f"cuda{device_idx}"
                device_name = torch.cuda.get_device_name(device_idx)
            else:
                device_idx = torch.cuda.current_device()
                device_str = f"cuda{device_idx}"
                device_name = torch.cuda.get_device_name(device_idx)
            backend_used = "cuda"
            print(f"[PyIsolate] Using CUDA device {device_idx}: {device_name}")

        elif backend == "xpu":
            if not xpu_available:
                print("[PyIsolate] XPU backend requested but not available. Exiting.")
                sys.exit(1)
            if device_idx is not None:
                torch.xpu.set_device(device_idx)
                device_str = f"xpu{device_idx}"
                device_name = (
                    torch.xpu.get_device_name(device_idx)
                    if hasattr(torch.xpu, "get_device_name")
                    else "Intel XPU"
                )
            else:
                device_idx = torch.xpu.current_device()
                device_str = f"xpu{device_idx}"
                device_name = (
                    torch.xpu.get_device_name(device_idx)
                    if hasattr(torch.xpu, "get_device_name")
                    else "Intel XPU"
                )
            backend_used = "xpu"
            print(f"[PyIsolate] Using Intel XPU device {device_idx}: {device_name}")

        elif backend == "cpu":
            # CPU fallback
            device_str = "cpu"
            device_name = "cpu"
            backend_used = "cpu"
            print(f"[PyIsolate] Using CPU backend.")
        else:
            print("[PyIsolate] No supported GPU backend available, exiting.")
            sys.exit(1)

    except Exception as e:
        print(f"[PyIsolate] Error setting device/backend: {e}")

    # Generate results filename with backend and device info
    import datetime
    import socket
    computer = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_tag = f"{backend_used}{device_idx if device_idx is not None else 0}"
    if device_str != "cpu":
        safe_device_name = device_name.replace(" ", "").replace("/", "-")
        device_tag = f"{backend_used}{device_idx if device_idx is not None else 0}-{safe_device_name}"
    results_filename = f"benchmark_results_{computer}_{device_tag}_{timestamp}.txt"
    print(f"\n[PyIsolate] Results will be saved to: {results_filename}")

    # In main(), after parsing args and determining device_idx:
    if device_idx is not None:
        import os
        os.environ["PYISOLATE_CUDA_DEVICE"] = str(device_idx)

    # Run benchmarks
    try:
        return asyncio.run(
            run_benchmarks(
                quick=args.quick, no_torch=args.no_torch, no_gpu=args.no_gpu, torch_mode=args.torch_mode,
            )
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
