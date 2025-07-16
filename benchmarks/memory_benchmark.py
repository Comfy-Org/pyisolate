#!/usr/bin/env python3
"""
Memory usage benchmarks for pyisolate.

This script measures RAM and VRAM usage across host and child processes
with varying numbers of extensions and different tensor sharing configurations.
"""

import argparse
import asyncio
import gc
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add example directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent / "example"))

# Add benchmarks directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import pynvml as nvml

    NVML_AVAILABLE = True
except ImportError:
    nvml = None
    NVML_AVAILABLE = False

import contextlib

from memory_extension_base import MemoryBenchmarkExtensionBase
from tabulate import tabulate

from pyisolate import ExtensionConfig, ExtensionManager, ExtensionManagerConfig
from tests.test_integration import IntegrationTestBase


# 1. Device detection helpers (add after imports)
def detect_available_backends():
    import torch
    backends = ["cpu"]
    cuda_available = torch.cuda.is_available()
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    rocm_available = False
    if cuda_available:
        torch_version = getattr(torch, 'version', None)
        hip_version = getattr(torch_version, 'hip', None) if torch_version else None
        if hip_version is not None:
            rocm_available = True
    if cuda_available and not rocm_available:
        backends.append("cuda")
    if rocm_available:
        backends.append("rocm")
    if xpu_available:
        backends.append("xpu")
    return backends


class MemoryTracker:
    """Tracks memory usage across host and child processes."""

    def __init__(self):
        self.process = psutil.Process()
        self.nvml_initialized = False
        self.gpu_handle = None
        self.baseline_gpu_memory_mb = 0
        self.platform = platform.system()

        if NVML_AVAILABLE and nvml:
            try:
                nvml.nvmlInit()  # type: ignore[attr-defined]
                self.nvml_initialized = True
                # Get the first GPU
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[attr-defined]
                # Store baseline GPU memory usage
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle) if nvml is not None else None  # type: ignore[attr-defined]
                self.baseline_gpu_memory_mb = (mem_info.used / 1024 / 1024) if mem_info is not None else 0
                print(
                    f"NVML initialized on {self.platform}. "
                    f"Initial GPU memory: {self.baseline_gpu_memory_mb:.1f} MB"
                )
            except Exception as e:
                print(f"Failed to initialize NVML on {self.platform}: {e}")
                self.nvml_initialized = False

        # Try to get baseline GPU memory using nvidia-smi as fallback on Windows
        if not self.nvml_initialized and self.platform == "Windows":
            baseline = self._get_gpu_memory_nvidia_smi()
            if baseline is not None:
                self.baseline_gpu_memory_mb = baseline
                print(f"Using nvidia-smi fallback. Initial GPU memory: {baseline:.1f} MB")

    def _get_gpu_memory_nvidia_smi(self) -> Optional[float]:
        """Get GPU memory usage using nvidia-smi command (Windows fallback)."""
        try:
            import shutil
            import subprocess

            # Find nvidia-smi executable
            nvidia_smi = shutil.which("nvidia-smi")
            if not nvidia_smi:
                return None

            # Query GPU memory usage via nvidia-smi
            result = subprocess.run(  # noqa: S603
                [nvidia_smi, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            # Parse the output (in MB)
            used_mb = float(result.stdout.strip())
            return used_mb
        except Exception:
            return None

    def _get_gpu_memory_windows_fallback(self, memory_info: dict[str, float]) -> dict[str, float]:
        """Fallback method to get GPU memory on Windows using nvidia-smi."""
        current_used = self._get_gpu_memory_nvidia_smi()
        if current_used is not None:
            memory_info["gpu_used_mb"] = float(current_used)
            memory_info["total_vram_mb"] = float(current_used)

            # Calculate delta from baseline
            vram_delta = float(current_used) - float(self.baseline_gpu_memory_mb)
            memory_info["host_vram_mb"] = float(max(0.0, vram_delta))

            # Try to get total GPU memory
            try:
                import shutil
                import subprocess

                nvidia_smi = shutil.which("nvidia-smi")
                if nvidia_smi:
                    result = subprocess.run(  # noqa: S603
                        [nvidia_smi, "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    memory_info["gpu_total_mb"] = float(result.stdout.strip())
            except Exception as e:
                # Log the error for debugging but continue
                if self.platform == "Windows":
                    print(f"Could not get total GPU memory via nvidia-smi: {e}", file=sys.stderr)

        return memory_info

    def get_process_tree_pids(self) -> list[int]:
        """Get all PIDs in the process tree (including children)."""
        pids = [self.process.pid]
        try:
            children = self.process.children(recursive=True)
            pids.extend([child.pid for child in children])
        except psutil.NoSuchProcess:
            pass
        return pids

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage for host and all child processes."""
        memory_info = {
            "host_ram_mb": 0.0,
            "children_ram_mb": 0.0,
            "total_ram_mb": 0.0,
            "host_vram_mb": 0.0,
            "total_vram_mb": 0.0,
            "gpu_used_mb": 0.0,
            "gpu_total_mb": 0.0,
            "num_processes": 1.0,  # float for type consistency
        }

        # Get RAM usage
        try:
            # Host process
            host_info = self.process.memory_info()
            memory_info["host_ram_mb"] = (host_info.rss or 0) / 1024 / 1024

            # Child processes
            children = self.process.children(recursive=True)
            memory_info["num_processes"] = 1.0 + len(children)

            for child in children:
                try:
                    child_info = child.memory_info()
                    memory_info["children_ram_mb"] += (child_info.rss or 0) / 1024 / 1024
                except psutil.NoSuchProcess:
                    pass

            memory_info["total_ram_mb"] = (
                (memory_info["host_ram_mb"] or 0)
                + (memory_info["children_ram_mb"] or 0)
            )

        except Exception as e:
            print(f"Error getting RAM usage: {e}")

        # Get GPU memory usage - use total system VRAM since extensions run in separate processes
        if self.nvml_initialized and self.gpu_handle and nvml is not None:
            try:
                # Get total GPU memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)  # type: ignore[attr-defined]
                current_used_mb = float(mem_info.used or 0) / 1024 / 1024 if mem_info else 0
                memory_info["gpu_used_mb"] = current_used_mb
                memory_info["gpu_total_mb"] = float(mem_info.total or 0) / 1024 / 1024 if mem_info else 0
                memory_info["total_vram_mb"] = current_used_mb

                # Calculate VRAM usage relative to baseline (captures all processes)
                vram_delta = current_used_mb - (self.baseline_gpu_memory_mb or 0)
                memory_info["host_vram_mb"] = max(0.0, vram_delta)
            except Exception as e:
                print(f"Error getting GPU memory usage via NVML: {e}")
                if self.platform == "Windows":
                    # Try alternative method for Windows
                    memory_info = self._get_gpu_memory_windows_fallback(memory_info)

        # Fallback: Try Windows nvidia-smi if NVML not initialized
        elif not self.nvml_initialized and self.platform == "Windows":
            memory_info = self._get_gpu_memory_windows_fallback(memory_info)

        # Fallback: try PyTorch CUDA memory for current process if all else failed
        elif CUDA_AVAILABLE and torch.cuda.is_available():
            try:
                # This only captures current process, but better than nothing
                allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

                memory_info["host_vram_mb"] = allocated_mb
                memory_info["total_vram_mb"] = allocated_mb
                memory_info["pytorch_reserved_mb"] = reserved_mb

                print(
                    "Warning: Using PyTorch CUDA memory (current process only): "
                    + f"{allocated_mb:.1f} MB allocated",
                    file=sys.stderr,
                )

            except Exception as e:
                print(f"Error getting PyTorch CUDA memory: {e}")

        return memory_info

    def reset_baseline(self):
        """Reset the baseline GPU memory measurement."""
        if self.nvml_initialized and self.gpu_handle and nvml is not None:
            try:
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)  # type: ignore[attr-defined]
                old_baseline = self.baseline_gpu_memory_mb
                self.baseline_gpu_memory_mb = float(mem_info.used or 0) / 1024 / 1024 if mem_info else 0
                print(
                    f"[DEBUG {self.platform}] Reset baseline from {old_baseline:.1f} MB "
                    f"to {self.baseline_gpu_memory_mb:.1f} MB",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"Error resetting GPU memory baseline on {self.platform}: {e}")

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self.nvml_initialized and nvml is not None:
            with contextlib.suppress(Exception):
                nvml.nvmlShutdown()  # type: ignore[attr-defined]


async def create_memory_benchmark_extension() -> str:
    """Create the extension code for memory benchmarking."""
    return '''
import torch


class MemoryBenchmarkExtension:
    """Extension for memory usage benchmarking."""

    def __init__(self):
        self.stored_tensors = {}

    async def initialize(self):
        """Initialize the extension."""
        pass

    async def prepare_shutdown(self):
        """Clean up before shutdown."""
        self.stored_tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def do_stuff(self, value):
        """Process a value."""
        return f"Processed: {value}"

    async def store_tensor(self, tensor_id: str, tensor: torch.Tensor) -> dict:
        """Store a tensor and return memory info."""
        # Store the tensor
        self.stored_tensors[tensor_id] = tensor

        # Get memory info
        info = {
            'tensor_id': tensor_id,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'size_mb': tensor.element_size() * tensor.numel() / (1024 * 1024),
            'num_stored': len(self.stored_tensors)
        }

        # Force GPU sync if on CUDA
        if tensor.is_cuda:
            torch.cuda.synchronize()

        return info

    async def clear_tensors(self) -> None:
        """Clear all stored tensors."""
        self.stored_tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def get_tensor_info(self, tensor_id: str) -> dict:
        """Get info about a stored tensor."""
        if tensor_id not in self.stored_tensors:
            return {'exists': False}

        tensor = self.stored_tensors[tensor_id]
        return {
            'exists': True,
            'shape': list(tensor.shape),
            'device': str(tensor.device),
            'data_ptr': tensor.data_ptr()  # Memory address for sharing detection
        }


def memory_benchmark_entrypoint():
    """Entry point for the memory benchmark extension."""
    return MemoryBenchmarkExtension()
'''


class MemoryBenchmarkRunner:
    """Runs memory usage benchmarks with multiple extensions."""

    def __init__(self, test_base: IntegrationTestBase):
        self.test_base = test_base
        if self.test_base.test_root is None:
            raise RuntimeError(
                "test_root is not set on test_base. "
                "Did you await setup_test_environment() successfully?"
            )
        self.memory_tracker = MemoryTracker()
        self.results = []

    async def run_baseline_memory_test(self) -> dict[str, float]:
        """Measure baseline memory usage with no extensions."""
        print("\nMeasuring baseline memory usage...")
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()

        # Wait a bit for memory to settle
        await asyncio.sleep(1)

        baseline = self.memory_tracker.get_memory_usage()
        print(f"Baseline: {baseline['total_ram_mb']:.1f} MB RAM, {baseline['total_vram_mb']:.1f} MB VRAM")
        # Print GPU memory usage if available
        gpu_total = baseline.get("gpu_total_mb", 0.0)
        gpu_used = baseline.get("gpu_used_mb", 0.0)
        try:
            gpu_total = float(gpu_total) if gpu_total not in (None, 0) else 0.0
        except Exception:
            gpu_total = 0.0
        try:
            gpu_used = float(gpu_used) if gpu_used is not None else 0.0
        except Exception:
            gpu_used = 0.0
        if gpu_total > 0:
            gpu_pct = (gpu_used / gpu_total) * 100 if gpu_total else 0
            print(
                f"GPU Memory: {gpu_used:.1f} / "
                f"{gpu_total:.1f} MB ({gpu_pct:.1f}% used)"
            )
        else:
            print("GPU Memory: N/A")
        return baseline

    async def run_scaling_test(
        self,
        num_extensions_list: list[int],
        share_torch: bool = True,
        test_tensor_size: tuple[int, ...] = (512, 512),
        device: str = "cpu",
    ) -> list[dict]:
        """Test memory scaling with different numbers of extensions."""
        import torch
        results = []
        extension_code = await create_memory_benchmark_extension()

        for num_extensions in num_extensions_list:
            print(f"\n{'=' * 60}")
            print(f"Testing with {num_extensions} extensions (share_torch={share_torch}, device={device})")
            print("=" * 60)

            # Create extensions
            extensions = []
            extension_venv_root = getattr(self.test_base, "test_root", None)
            if extension_venv_root is not None:
                extension_venv_root = extension_venv_root / "extension-venvs"
            else:
                extension_venv_root = "extension-venvs"
            manager = ExtensionManager(
                MemoryBenchmarkExtensionBase,
                ExtensionManagerConfig(venv_root_path=str(extension_venv_root)),
            )

            # Clean up and reset baseline before measuring
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete

            # Reset GPU memory baseline for this test
            self.memory_tracker.reset_baseline()

            # Wait a moment for memory to settle
            await asyncio.sleep(1)

            before_memory = self.memory_tracker.get_memory_usage()
            print(
                f"Baseline GPU memory: {before_memory.get('gpu_used_mb', 0):.1f} MB "
                f"(baseline: {self.memory_tracker.baseline_gpu_memory_mb:.1f} MB)"
            )

            # Create and load extensions
            print(f"Creating {num_extensions} extensions...")
            for i in range(num_extensions):
                ext_name = f"memory_ext_{i}"
                self.test_base.create_extension(
                    ext_name,
                    dependencies=["torch>=2.0.0"] if TORCH_AVAILABLE else [],
                    share_torch=share_torch,
                    extension_code=extension_code,
                )

                config = ExtensionConfig(
                    name=ext_name,
                    module_path=str((self.test_base.test_root or Path(".")) / "extensions" / ext_name),
                    isolated=True,
                    dependencies=["torch>=2.0.0"] if TORCH_AVAILABLE else [],
                    apis=[],
                    share_torch=share_torch,
                )

                ext = manager.load_extension(config)
                extensions.append((ext_name, ext))

            # Wait for extensions to fully initialize
            await asyncio.sleep(2)

            # Measure memory after loading extensions
            after_load_memory = self.memory_tracker.get_memory_usage()

            # Create test tensor
            print(f"Creating test tensor {test_tensor_size} on {device}...")
            with torch.inference_mode():
                if device == "cuda" or device == "rocm":
                    test_tensor = torch.randn(*test_tensor_size, device="cuda")
                    torch.cuda.synchronize()
                elif device == "xpu":
                    test_tensor = torch.randn(*test_tensor_size, device="xpu")
                else:
                    test_tensor = torch.randn(*test_tensor_size)

            tensor_size_mb = test_tensor.element_size() * test_tensor.numel() / (1024 * 1024)
            print(f"Tensor size: {tensor_size_mb:.1f} MB on {test_tensor.device}")

            # Check memory after tensor creation
            if device in ("cuda", "rocm"):
                post_tensor_memory = self.memory_tracker.get_memory_usage()
                print(
                    f"GPU memory after tensor creation: {post_tensor_memory.get('gpu_used_mb', 0):.1f} MB "
                    f"(delta: {post_tensor_memory.get('host_vram_mb', 0):.1f} MB)"
                )

            # Send tensor to all extensions
            print(f"Sending tensor to {num_extensions} extensions...")
            send_start = time.time()

            for i, (ext_name, ext) in enumerate(extensions):
                try:
                    info = await ext.store_tensor(f"test_tensor_{i}", test_tensor)
                    if i == 0:
                        print(f"  First extension stored: {info}")
                    # Force GPU sync after each send for accurate memory tracking
                    if device in ("cuda", "rocm"):
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"  Failed to send to {ext_name}: {e}")

            send_time = time.time() - send_start
            print(f"Send completed in {send_time:.2f}s")

            # Force final sync before measuring
            if device in ("cuda", "rocm"):
                torch.cuda.synchronize()

            # Wait for memory to settle
            await asyncio.sleep(2)

            # Measure memory after sending tensors
            after_send_memory = self.memory_tracker.get_memory_usage()

            # Check tensor sharing (get data pointers from a few extensions)
            data_ptrs = []
            for i in range(min(3, num_extensions)):
                try:
                    ext_name, ext = extensions[i]
                    info = await ext.get_tensor_info(f"test_tensor_{i}")
                    if info.get("exists"):
                        data_ptrs.append(info.get("data_ptr"))
                except Exception as e:
                    print(f"  Failed to get tensor info from {ext_name}: {e}")

            shared_memory = len(set(data_ptrs)) == 1 if len(data_ptrs) > 1 else None

            # Calculate memory deltas
            result = {
                "num_extensions": num_extensions,
                "share_torch": share_torch,
                "tensor_size_mb": tensor_size_mb,
                "tensor_device": str(test_tensor.device),
                "before_ram_mb": before_memory["total_ram_mb"],
                "after_load_ram_mb": after_load_memory["total_ram_mb"],
                "after_send_ram_mb": after_send_memory["total_ram_mb"],
                "load_ram_delta_mb": after_load_memory["total_ram_mb"] - before_memory["total_ram_mb"],
                "send_ram_delta_mb": after_send_memory["total_ram_mb"] - after_load_memory["total_ram_mb"],
                "ram_per_extension_mb": (
                    float(after_load_memory["total_ram_mb"] or 0)
                    - float(before_memory["total_ram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                "before_vram_mb": before_memory["total_vram_mb"],
                "after_load_vram_mb": after_load_memory["total_vram_mb"],
                "after_send_vram_mb": after_send_memory["total_vram_mb"],
                "send_vram_delta_mb": after_send_memory["total_vram_mb"] - after_load_memory["total_vram_mb"],
                # Add GPU total memory tracking
                "before_gpu_mb": before_memory.get("gpu_used_mb", 0),
                "after_load_gpu_mb": after_load_memory.get("gpu_used_mb", 0),
                "after_send_gpu_mb": after_send_memory.get("gpu_used_mb", 0),
                "load_gpu_delta_mb": after_load_memory.get("gpu_used_mb", 0)
                - before_memory.get("gpu_used_mb", 0),
                "send_gpu_delta_mb": after_send_memory.get("gpu_used_mb", 0)
                - after_load_memory.get("gpu_used_mb", 0),
                "shared_memory": shared_memory,
                "send_time_s": send_time,
            }

            results.append(result)

            # Print summary
            print("\nMemory Summary:")
            print(f"  RAM per extension: {result['ram_per_extension_mb']:.1f} MB")
            print(f"  RAM for tensor transfer: {result['send_ram_delta_mb']:.1f} MB")

            # Debug GPU memory tracking
            print("\nGPU Memory Details:")
            print(f"  Before: {before_memory.get('gpu_used_mb', 0):.1f} MB")
            print(f"  After Load: {after_load_memory.get('gpu_used_mb', 0):.1f} MB")
            print(f"  After Send: {after_send_memory.get('gpu_used_mb', 0):.1f} MB")
            print(f"  Baseline: {self.memory_tracker.baseline_gpu_memory_mb:.1f} MB")

            # Show GPU memory if this is a GPU test
            if device in ("cuda", "rocm") and result["load_gpu_delta_mb"] > 0:
                print(f"  GPU memory for tensor creation: {result['load_gpu_delta_mb']:.1f} MB")
                print(f"  GPU memory for tensor transfer: {result['send_gpu_delta_mb']:.1f} MB")
            else:
                print(f"  VRAM for tensor transfer: {result['send_vram_delta_mb']:.1f} MB")

            if shared_memory is not None:
                print(f"  Tensor memory shared: {shared_memory}")

            # Cleanup
            print("\nCleaning up extensions...")
            manager.stop_all_extensions()
            del test_tensor
            gc.collect()
            if device in ("cuda", "rocm"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Wait for cleanup
            await asyncio.sleep(2)

        return results

    async def run_large_tensor_sharing_test(
        self,
        num_extensions: int = 50,
        tensor_gb: float = 2.0,
        test_both_modes: bool = False,
        device: str = "cpu",
    ) -> dict:
        """Test memory sharing with a large tensor across multiple extensions."""
        import torch
        print(f"\n{'=' * 60}")
        print(
            f"Large Tensor Sharing Test ({tensor_gb}GB tensor, "
            f"{num_extensions} extensions, device={device})"
        )
        print("=" * 60)

        extension_code = await create_memory_benchmark_extension()
        results = {}

        # Test both CPU and GPU tensors
        device_name = device.upper()
        results[device_name.lower()] = {}
        share_torch_modes = [False, True] if test_both_modes else [True]
        for share_torch in share_torch_modes:
            print(f"\n--- Testing {device_name} with share_torch={share_torch} ---")

            # Create extensions
            extensions = []
            extension_venv_root = getattr(self.test_base, "test_root", None)
            if extension_venv_root is not None:
                extension_venv_root = extension_venv_root / "extension-venvs"
            else:
                extension_venv_root = "extension-venvs"
            manager = ExtensionManager(
                MemoryBenchmarkExtensionBase,
                ExtensionManagerConfig(venv_root_path=str(extension_venv_root)),
            )

            # Measure baseline
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
            baseline = self.memory_tracker.get_memory_usage()

            # Create extensions
            for i in range(num_extensions):
                ext_name = f"large_test_ext_{device_name.lower()}_{i}"
                self.test_base.create_extension(
                    ext_name,
                    dependencies=["torch>=2.0.0"],
                    share_torch=share_torch,
                    extension_code=extension_code,
                )

                config = ExtensionConfig(
                    name=ext_name,
                    module_path=str((self.test_base.test_root or Path(".")) / "extensions" / ext_name),
                    isolated=True,
                    dependencies=["torch>=2.0.0"],
                    apis=[],
                    share_torch=share_torch,
                )

                ext = manager.load_extension(config)
                extensions.append((ext_name, ext))

            # Wait for extensions to initialize
            await asyncio.sleep(2)

            # Create large tensor
            # Calculate size for desired GB (float32 = 4 bytes per element)
            num_elements = int(tensor_gb * 1024 * 1024 * 1024 / 4)
            # Make it a square-ish tensor
            side = int(num_elements**0.5)

            print(f"Creating {tensor_gb}GB tensor ({side}x{side}) on {device_name}...")
            with torch.inference_mode():
                if device == "cuda" or device == "rocm":
                    large_tensor = torch.randn(side, side, device="cuda")
                    torch.cuda.synchronize()
                elif device == "xpu":
                    large_tensor = torch.randn(side, side, device="xpu")
                else:
                    large_tensor = torch.randn(side, side)
            actual_size_mb = large_tensor.element_size() * large_tensor.numel() / (1024 * 1024)
            print(f"Actual tensor size: {actual_size_mb:.1f} MB on {large_tensor.device}")

            # Measure after tensor creation
            after_create = self.memory_tracker.get_memory_usage()

            # Send to all extensions
            print(f"Sending large {device_name} tensor to {num_extensions} extensions...")
            send_start = time.time()

            for _i, (ext_name, ext) in enumerate(extensions):
                try:
                    await ext.store_tensor("large_tensor", large_tensor)
                    print(f"  Sent to {ext_name}")
                except Exception as e:
                    print(f"  Failed to send to {ext_name}: {e}")

            send_time = time.time() - send_start

            # Measure after sending
            await asyncio.sleep(2)
            after_send = self.memory_tracker.get_memory_usage()

            # Store results
            results[device_name.lower()][f"share_torch_{share_torch}"] = {
                "baseline_ram_mb": baseline["total_ram_mb"],
                "after_create_ram_mb": after_create["total_ram_mb"],
                "after_send_ram_mb": after_send["total_ram_mb"],
                "baseline_vram_mb": baseline["total_vram_mb"],
                "after_create_vram_mb": after_create["total_vram_mb"],
                "after_send_vram_mb": after_send["total_vram_mb"],
                "tensor_size_mb": actual_size_mb,
                "tensor_device": str(large_tensor.device),
                "ram_for_tensor_creation_mb": (
                    float(after_create["total_ram_mb"] or 0)
                    - float(baseline["total_ram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                "ram_for_distribution_mb": (
                    float(after_send["total_ram_mb"] or 0)
                    - float(after_create["total_ram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                "ram_per_extension_copy_mb": (
                    float(after_send["total_ram_mb"] or 0)
                    - float(after_create["total_ram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                "vram_for_tensor_creation_mb": (
                    float(after_create["total_vram_mb"] or 0)
                    - float(baseline["total_vram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                "vram_for_distribution_mb": (
                    float(after_send["total_vram_mb"] or 0)
                    - float(after_create["total_vram_mb"] or 0)
                ) / num_extensions if num_extensions else 0,
                # Add GPU total memory tracking
                "baseline_gpu_mb": baseline.get("gpu_used_mb", 0),
                "after_create_gpu_mb": after_create.get("gpu_used_mb", 0),
                "after_send_gpu_mb": after_send.get("gpu_used_mb", 0),
                "gpu_for_tensor_creation_mb": (
                    float(after_create.get("gpu_used_mb", 0) or 0)
                    - float(baseline.get("gpu_used_mb", 0) or 0)
                ) / num_extensions if num_extensions else 0,
                "gpu_for_distribution_mb": (
                    float(after_send.get("gpu_used_mb", 0) or 0)
                    - float(after_create.get("gpu_used_mb", 0) or 0)
                ) / num_extensions if num_extensions else 0,
                "send_time_s": send_time,
            }

            # Cleanup
            manager.stop_all_extensions()
            del large_tensor
            gc.collect()
            if device in ("cuda", "rocm"):
                torch.cuda.empty_cache()
            await asyncio.sleep(2)

        return results


async def run_memory_benchmarks(
    extension_counts: list[int],
    test_small_tensor: bool = True,
    test_large_tensor: bool = True,
    max_extensions_for_large: int = 50,
    test_both_modes: bool = False,
    backend: str = "auto",
):
    """Run the full memory benchmark suite."""
    test_base = IntegrationTestBase()
    await test_base.setup_test_environment("memory_benchmark")

    try:
        runner = MemoryBenchmarkRunner(test_base)
        all_results = {}

        # Baseline measurement
        baseline = await runner.run_baseline_memory_test()
        all_results["baseline"] = baseline

        if backend == "auto":
            available_backends = detect_available_backends()
            if "cpu" not in available_backends:
                available_backends = ["cpu"] + available_backends
            else:
                # Ensure cpu is first
                available_backends = [b for b in ["cpu"] + available_backends if b != "cpu"]
                available_backends = ["cpu"] + available_backends
        else:
            available_backends = [backend]

        if test_small_tensor:
            print("\n" + "=" * 80)
            print("SMALL TENSOR SCALING TESTS")
            print("=" * 80)

            small_tensor_size = (512, 512)  # ~1MB tensor

            for backend_used in available_backends:
                if test_both_modes:
                    print(f"\n--- {backend_used.upper()} Tensor Tests (share_torch=False) ---")
                    results_no_share = await runner.run_scaling_test(
                        extension_counts,
                        share_torch=False,
                        test_tensor_size=small_tensor_size,
                        device=backend_used,
                    )
                    all_results[f"{backend_used}_no_share"] = results_no_share

                print(f"\n--- {backend_used.upper()} Tensor Tests (share_torch=True) ---")
                results_share = await runner.run_scaling_test(
                    extension_counts,
                    share_torch=True,
                    test_tensor_size=small_tensor_size,
                    device=backend_used,
                )
                all_results[f"{backend_used}_share"] = results_share

        if test_large_tensor:
            for backend_used in available_backends:
                large_results = await runner.run_large_tensor_sharing_test(
                    num_extensions=min(max_extensions_for_large, max(extension_counts)),
                    tensor_gb=2.0,
                    test_both_modes=test_both_modes,
                    device=backend_used,
                )
                all_results[f"{backend_used}_large"] = large_results

        print_memory_benchmark_summary(all_results)

    finally:
        await test_base.cleanup()


def print_memory_benchmark_summary(results: dict):
    """Print a comprehensive summary of memory benchmark results."""
    print("\n" + "=" * 80)
    print("MEMORY BENCHMARK SUMMARY")
    print("=" * 80)

    # Baseline
    if "baseline" in results:
        baseline = results["baseline"]
        print("\nBaseline Memory Usage:")
        print(f"  RAM: {baseline['total_ram_mb']:.1f} MB")
        print(f"  VRAM: {baseline['total_vram_mb']:.1f} MB")
        gpu_total = baseline.get("gpu_total_mb", 0.0)
        gpu_used = baseline.get("gpu_used_mb", 0.0)
        try:
            gpu_total = float(gpu_total) if gpu_total not in (None, 0) else 0.0
        except Exception:
            gpu_total = 0.0
        try:
            gpu_used = float(gpu_used) if gpu_used is not None else 0.0
        except Exception:
            gpu_used = 0.0
        if gpu_total > 0:
            gpu_pct = (gpu_used / gpu_total) * 100 if gpu_total else 0
            print(f"  GPU Total: {gpu_used:.1f} / {gpu_total:.1f} MB ({gpu_pct:.1f}% used)")
        else:
            print("  GPU Total: N/A")

    # Dynamically print all *_share and *_no_share results
    share_types = [k for k in results if k.endswith(("_share", "_no_share"))]
    for test_type in share_types:
        if test_type in results:
            backend = test_type.replace("_share", "").replace("_no_share", "").upper()
            share_mode = "SHARE_TORCH=TRUE" if test_type.endswith("_share") else "SHARE_TORCH=FALSE"
            print(f"\n{backend} {share_mode} Results:")
            headers = ["Extensions", "RAM/Ext (MB)", "Tensor RAM (MB)", "GPU (MB)", "Shared"]
            table_data = []
            for result in results[test_type]:
                gpu_memory = result.get("send_gpu_delta_mb", result.get("send_vram_delta_mb", 0))
                table_data.append([
                    result["num_extensions"],
                    f"{result['ram_per_extension_mb']:.1f}",
                    f"{result['send_ram_delta_mb']:.1f}",
                    f"{gpu_memory:.1f}",
                    (
                        "Yes"
                        if result.get("shared_memory")
                        else "No" if result.get("shared_memory") is False else "N/A"
                    ),
                ])
            if table_data:
                print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Large tensor sharing results for all backends
    large_keys = [k for k in results if k.endswith("_large")]
    for large_key in large_keys:
        backend = large_key.replace("_large", "").upper()
        print(f"\n2GB TENSOR SHARING TEST: {backend}")
        large_results = results[large_key]
        for dev in large_results:
            print(f"\n{dev.upper()} Tensor Results:")
            headers = [
                "Config",
                "Tensor Size (MB)",
                "Distribution RAM (MB)",
                "RAM/Extension (MB)",
                "Send Time (s)",
            ]
            table_data = []
            for share_torch in [False, True]:
                key = f"share_torch_{share_torch}"
                if key in large_results[dev]:
                    r = large_results[dev][key]
                    table_data.append([
                        f"share_torch={share_torch}",
                        f"{r['tensor_size_mb']:.1f}",
                        f"{r['ram_for_distribution_mb']:.1f}",
                        f"{r['ram_per_extension_copy_mb']:.1f}",
                        f"{r['send_time_s']:.2f}",
                    ])
            if table_data:
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                # Analysis for this backend
                if "share_torch_False" in large_results[dev] and "share_torch_True" in large_results[dev]:
                    no_share = large_results[dev]["share_torch_False"]
                    share = large_results[dev]["share_torch_True"]
                    savings = no_share["ram_for_distribution_mb"] - share["ram_for_distribution_mb"]
                    savings_pct = (
                        (savings / no_share["ram_for_distribution_mb"] * 100)
                        if no_share["ram_for_distribution_mb"] else 0
                    )
                    print(f"\n{dev.upper()} Memory Sharing Analysis:")
                    print(f"  Memory saved with share_torch: {savings:.1f} MB ({savings_pct:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pyisolate memory usage benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python memory_benchmark.py                    # Run full benchmark suite
    python memory_benchmark.py --max-extensions 20  # Test up to 20 extensions
    python memory_benchmark.py --large-only       # Only test large tensor sharing
    python memory_benchmark.py --counts 1,5,10,20  # Custom extension counts
        """,
    )

    parser.add_argument(
        "--max-extensions", type=int, default=50, help="Maximum number of extensions to test (default: 50)"
    )

    parser.add_argument(
        "--counts", type=str, help="Comma-separated list of extension counts to test (e.g., '1,5,10,20')"
    )

    parser.add_argument("--large-only", action="store_true", help="Only run large tensor sharing test")

    parser.add_argument("--small-only", action="store_true", help="Only run small tensor scaling tests")

    parser.add_argument(
        "--test-both-modes",
        action="store_true",
        help="Test both share_torch=True and share_torch=False (default: only share_torch=True)",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "xpu", "rocm", "cpu"],
        default="auto",
        help=(
            "Device backend to use: auto (default), cuda (NVIDIA/AMD ROCm), "
            "xpu (Intel oneAPI), rocm (AMD ROCm), or cpu"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="(Legacy) Device index (int) or 'cpu' to force CPU mode. Use --backend for multi-backend tests.",
    )

    args = parser.parse_args()

    device_idx = None  # Ensure device_idx is always defined

    # Determine extension counts (move this up before device/backend logic)
    if args.counts:
        extension_counts = [int(x.strip()) for x in args.counts.split(",")]
    else:
        # Default progression: 1, 2, 5, 10, 20
        extension_counts = [1, 2, 5, 10, 20]
        if args.max_extensions >= 50:
            extension_counts.append(50)
        if args.max_extensions >= 100:
            extension_counts.append(100)
        # Filter based on max
        extension_counts = [c for c in extension_counts if c <= args.max_extensions]

    # Set device and backend
    backend = args.backend
    device_str = "cpu"
    device_name = "cpu"
    backend_used = "cpu"
    try:
        import torch  # type: ignore
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
        if backend == "cuda" and cuda_available:
            # Use getattr to avoid linter errors for torch.version.hip
            torch_version = getattr(torch, 'version', None)
            hip_version = getattr(torch_version, 'hip', None) if torch_version else None
            if platform.system() == "Linux" and hip_version is not None:
                print("[PyIsolate] ROCm (AMD) backend detected on Linux.")
            elif platform.system() == "Windows":
                print("[PyIsolate] ROCm is not supported on Windows. Falling back to CPU.")
                backend = "cpu"
            if backend == "cuda":
                if args.device is not None:
                    if str(args.device).lower() == "cpu":
                        backend = "cpu"
                        device_str = "cpu"
                        device_name = "cpu"
                        print("[PyIsolate] Forcing CPU mode due to --device=cpu")
                        args.no_gpu = True
                    else:
                        try:
                            device_idx = int(args.device)
                            torch.cuda.set_device(device_idx)
                            device_str = f"cuda{device_idx}"
                            device_name = torch.cuda.get_device_name(device_idx)
                        except ValueError:
                            print(f"Invalid --device value: {args.device}. Must be integer or 'cpu'.")
                            sys.exit(1)
                else:
                    device_idx = torch.cuda.current_device()
                    device_str = f"cuda{device_idx}"
                    device_name = torch.cuda.get_device_name(device_idx)
                backend_used = "cuda"
                print(f"[PyIsolate] Using CUDA/ROCm device {device_idx}: {device_name}")
        elif backend == "xpu" and xpu_available:
            if args.device is not None:
                if str(args.device).lower() == "cpu":
                    backend = "cpu"
                    device_str = "cpu"
                    device_name = "cpu"
                    print("[PyIsolate] Forcing CPU mode due to --device=cpu")
                    args.no_gpu = True
                else:
                    try:
                        device_idx = int(args.device)
                        torch.xpu.set_device(device_idx)
                        device_str = f"xpu{device_idx}"
                        device_name = (
                            torch.xpu.get_device_name(device_idx)
                            if hasattr(torch.xpu, "get_device_name")
                            else "Intel XPU"
                        )
                    except ValueError:
                        print(f"Invalid --device value: {args.device}. Must be integer or 'cpu'.")
                        sys.exit(1)
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
        else:
            print("[PyIsolate] No supported GPU backend available, using CPU only.")
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
    results_filename = f"memory_benchmark_results_{computer}_{device_tag}_{timestamp}.txt"
    print(f"\n[PyIsolate] Results will be saved to: {results_filename}")

    # For memory tracking, print backend-specific info
    if backend_used == "cuda":
        print("[PyIsolate] Using NVML for CUDA/ROCm memory tracking.")
    elif backend_used == "xpu":
        try:
            import torch  # type: ignore
            mem_alloc = torch.xpu.memory_allocated() if hasattr(torch, "xpu") else 0
            print(f"[PyIsolate] Intel XPU memory allocated: {mem_alloc / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"[PyIsolate] Could not get Intel XPU memory info: {e}")
    # For AMD ROCm, optionally try rocm-smi if available (not implemented here,
    # but can be added with subprocess)

    # Determine what to test
    test_small = not args.large_only
    test_large = not args.small_only

    # Run benchmarks
    try:
        asyncio.run(
            run_memory_benchmarks(
                extension_counts=extension_counts,
                test_small_tensor=test_small,
                test_large_tensor=test_large,
                test_both_modes=args.test_both_modes,
                backend=args.backend,
            )
        )
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
