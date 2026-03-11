#!/usr/bin/env python3
"""
Simple RPC benchmark that reuses existing test infrastructure.

This creates a minimal benchmark to measure RPC overhead by running
the working example and measuring call times.
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

# Add example to path
sys.path.insert(0, str(Path(__file__).parent.parent / "example"))

try:
    import numpy as np
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    import numpy as np

    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


async def measure_rpc_overhead(include_large_tensors=False):
    """Measure RPC overhead using the existing example."""

    print("Simple PyIsolate RPC Benchmark")
    print("=" * 40)
    print("This benchmark measures RPC overhead using the existing example extensions.")
    print()

    import os

    if sys.platform == "linux" and os.environ.get("TMPDIR") != "/dev/shm":
        print("WARNING: TMPDIR is not set to /dev/shm on Linux.")
        print("If extensions use share_torch=True, execution WILL fail in strict sandboxes.")
        print("Recommended: export TMPDIR=/dev/shm")
        print("-" * 40)
        print()

    print("Setting up extensions (this may take a moment)...")

    # Use the same setup as the example
    try:
        import os
        from typing import TypedDict, cast

        import yaml
        from shared import DatabaseSingleton, ExampleExtensionBase

        import pyisolate

        # Setup like the example
        pyisolate_dir = os.path.dirname(os.path.dirname(os.path.realpath(pyisolate.__file__)))
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "example")

        config = pyisolate.ExtensionManagerConfig(venv_root_path=os.path.join(base_path, "extension-venvs"))
        manager = pyisolate.ExtensionManager(ExampleExtensionBase, config)

        extensions = []
        extension_dir = os.path.join(base_path, "extensions")

        class CustomConfig(TypedDict):
            enabled: bool
            isolated: bool
            dependencies: list[str]
            share_torch: bool

        for extension_name in os.listdir(extension_dir):
            if os.path.isdir(os.path.join(extension_dir, extension_name)):
                module_path = os.path.join(extension_dir, extension_name)
                yaml_path = os.path.join(module_path, "manifest.yaml")

                with open(yaml_path) as f:
                    manifest = cast(CustomConfig, yaml.safe_load(f))

                if not manifest.get("enabled", True):
                    continue

                pyisolate_install = ["-e", pyisolate_dir]

                ext_config = pyisolate.ExtensionConfig(
                    name=extension_name,
                    module_path=module_path,
                    isolated=manifest["isolated"],
                    dependencies=manifest["dependencies"] + pyisolate_install,
                    apis=[DatabaseSingleton],
                    share_torch=manifest["share_torch"],
                )

                extension = manager.load_extension(ext_config)
                extensions.append(extension)

                # Only load the first extension for benchmarking
                break

        print(f"Loaded {len(extensions)} extensions")

        # Get first extension for benchmarking
        if not extensions:
            print("No extensions loaded!")
            return

        ext = extensions[0]
        print(f"Using extension: {type(ext).__name__}")

        # Simple benchmark data
        test_data = [
            ("small_int", 42),
            ("small_string", "hello world"),
            ("medium_string", "hello world" * 100),
        ]

        if TORCH_AVAILABLE:
            # Create tensors with proper memory management
            tensor_specs = [
                ("tiny_tensor", (10, 10)),  # ~400B
                ("small_tensor", (100, 100)),  # ~40KB
                ("medium_tensor", (512, 512)),  # ~1MB
                ("large_tensor", (1024, 1024)),  # ~4MB
                ("model_tensor", (40132, 40132)),  # ~6GB (modern LLM/diffusion model)
            ]

            for name, size in tensor_specs:
                try:
                    print(f"  Creating {name} tensor {size}...")
                    with torch.inference_mode():
                        tensor = torch.randn(*size)
                    test_data.append((name, tensor))
                    print(f"    {name} created successfully ({tensor.numel() * 4 / (1024**3):.2f}GB)")
                except RuntimeError as e:
                    print(f"    Skipping {name}: {e}")

            if include_large_tensors:
                print("  Including very large tensors (this will use significant memory)...")
                with torch.inference_mode():
                    test_data.extend(
                        [
                            ("huge_tensor", torch.randn(4096, 4096)),  # ~64MB
                            ("image_4k", torch.randn(3, 4096, 4096)),  # ~200MB (4K RGB image)
                        ]
                    )
                # 8K image would be ~800MB, only add if explicitly requested
                print("  (8K image tensor skipped - would use ~800MB)")
            else:
                print("  (Use --large-tensors to include very large tensors)")

        test_data.append(("large_array", np.random.random((100, 100))))

        # Run benchmarks
        print("\nRunning benchmarks (1000 samples per test, this may take a while)...")
        print("NOTE: These tests run with the default extension configuration (share_torch disabled)")
        results = {}

        for name, data in test_data:
            print(f"  Testing {name}...")

            # Warmup
            for _ in range(3):
                try:
                    await ext.do_stuff(data)
                except Exception as e:
                    print(f"    Warmup failed: {e}")
                    break
            else:
                # Benchmark
                times = []
                total_runs = 1000
                for _ in range(total_runs):
                    start = time.perf_counter()
                    try:
                        await ext.do_stuff(data)
                        end = time.perf_counter()
                        times.append(end - start)
                    except Exception as e:
                        print(f"    Benchmark failed: {e}")
                        break

                if times:
                    mean_time = statistics.mean(times)
                    std_time = statistics.stdev(times) if len(times) > 1 else 0
                    results[name] = (mean_time, std_time, len(times))
                    print(f"    {mean_time * 1000:.2f}±{std_time * 1000:.2f}ms ({len(times)} runs)")

        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)

        if results:
            print(f"{'Test':<15} {'Mean (ms)':<12} {'Std Dev (ms)':<14} {'Runs':<6}")
            print("-" * 50)

            for name, (mean_time, std_time, runs) in results.items():
                print(f"{name:<15} {mean_time * 1000:<12.2f} {std_time * 1000:<14.2f} {runs:<6}")

            # Show fastest result for reference
            baseline = min(mean_time for mean_time, _, _ in results.values())
            print(f"\nFastest result: {baseline * 1000:.2f}ms")
        else:
            print("No successful benchmark results!")

        # Cleanup
        import contextlib

        for ext in extensions:
            with contextlib.suppress(Exception):
                await ext.stop()

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple RPC benchmark")
    parser.add_argument(
        "--large-tensors",
        action="store_true",
        help="Include very large tensors (8K images, may use lots of memory)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(measure_rpc_overhead(include_large_tensors=args.large_tensors)))
