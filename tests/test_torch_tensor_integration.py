# pyright: reportMissingImports=false
"""
Integration tests for passing torch.Tensor objects between host and extensions.

This test suite covers tensor passing with both share_torch=True and share_torch=False configurations.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

# Import pyisolate components
import pyisolate
from pyisolate import ExtensionConfig, ExtensionManager, ExtensionManagerConfig

# Import shared components from example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example"))
from shared import DatabaseSingleton, ExampleExtensionBase  # type: ignore

# Check torch availability
try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    # Set CUDA device from environment variable if specified
    cuda_env = os.environ.get("PYISOLATE_CUDA_DEVICE")
    if HAS_CUDA and cuda_env is not None:
        torch.cuda.set_device(int(cuda_env))
        print(f"[PyIsolate] Using CUDA device {cuda_env}: {torch.cuda.get_device_name(int(cuda_env))}")
    elif HAS_CUDA:
        print(
            f"[PyIsolate] Using default CUDA device "
            f"{torch.cuda.current_device()}: "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    else:
        print("[PyIsolate] CUDA not available, using CPU only.")
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False


class TorchTestBase:
    """Base class for torch tensor tests providing common setup and utilities."""

    def __init__(self):
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.test_root: Optional[Path] = None
        self.manager: Optional[ExtensionManager] = None
        self.extensions: list[ExampleExtensionBase] = []

    async def setup_test_environment(self, test_name: str) -> Path:
        """Set up a temporary test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name) / test_name
        self.test_root.mkdir(parents=True, exist_ok=True)

        # Create venv root directory
        venv_root = self.test_root / "extension-venvs"
        venv_root.mkdir(parents=True, exist_ok=True)

        # Create extensions directory
        extensions_dir = self.test_root / "extensions"
        extensions_dir.mkdir(parents=True, exist_ok=True)

        return self.test_root

    def create_tensor_extension(
        self,
        name: str,
        share_torch: bool,
        extension_code: str,
    ) -> Path:
        """Create a test extension for tensor operations."""
        if not self.test_root:
            raise RuntimeError("Test environment not set up")

        ext_dir = self.test_root / "extensions" / name
        ext_dir.mkdir(parents=True, exist_ok=True)

        # Dependencies based on share_torch setting
        dependencies = []
        if not share_torch:
            # When share_torch is False, we need to install torch in the extension
            dependencies.append("torch>=2.0.0")

        # Create manifest.yaml
        manifest = {
            "enabled": True,
            "isolated": True,
            "dependencies": dependencies,
            "share_torch": share_torch,
        }

        with open(ext_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f)

        # Create __init__.py with extension code
        with open(ext_dir / "__init__.py", "w") as f:
            f.write(extension_code)

        return ext_dir

    async def load_extensions(self, extension_configs: list[dict[str, Any]]) -> list[ExampleExtensionBase]:
        """Load multiple extensions with given configurations."""
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting to load {len(extension_configs)} extensions")

        if not self.test_root:
            raise RuntimeError("Test environment not set up")

        # Get pyisolate directory for editable install
        pyisolate_dir = os.path.dirname(os.path.dirname(os.path.realpath(pyisolate.__file__)))
        logger.debug(f"Pyisolate directory: {pyisolate_dir}")

        # Create extension manager
        config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
        logger.debug(f"Creating ExtensionManager with venv_root_path: {config['venv_root_path']}")
        self.manager = ExtensionManager(ExampleExtensionBase, config)

        extensions = []

        for idx, ext_config in enumerate(extension_configs):
            name = ext_config["name"]
            logger.debug(f"Loading extension {idx + 1}/{len(extension_configs)}: {name}")
            module_path = str(self.test_root / "extensions" / name)

            # Read manifest
            yaml_path = Path(module_path) / "manifest.yaml"
            logger.debug(f"Reading manifest from: {yaml_path}")
            with open(yaml_path) as f:
                manifest = yaml.safe_load(f)

            if not manifest.get("enabled", True):
                logger.debug(f"Skipping disabled extension: {name}")
                continue

            # Create extension config
            extension_config = ExtensionConfig(
                name=name,
                module_path=module_path,
                isolated=manifest["isolated"],
                dependencies=manifest["dependencies"] + ["-e", pyisolate_dir],
                apis=[DatabaseSingleton],
                share_torch=manifest["share_torch"],
            )

            logger.debug(
                f"Loading extension with config: name={name}, isolated={manifest['isolated']}, "
                f"share_torch={manifest['share_torch']}, dependencies={manifest['dependencies']}"
            )

            extension = self.manager.load_extension(extension_config)
            logger.debug(f"Successfully loaded extension: {name}")
            extensions.append(extension)

        self.extensions = extensions
        logger.debug(f"Finished loading {len(extensions)} extensions")
        return extensions

    async def cleanup(self):
        """Clean up test environment."""
        # Shutdown extensions
        for extension in self.extensions:
            try:
                await extension.stop()
            except Exception as e:
                logging.warning(f"Error stopping extension: {e}")

        # Clean up temp directory
        if self.temp_dir:
            self.temp_dir.cleanup()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTorchTensorPassing:
    """Test passing torch tensors between host and extensions."""

    async def test_cpu_tensor_share_torch_true(self):
        """Test passing CPU tensors with share_torch=True."""
        test_base = TorchTestBase()
        await test_base.setup_test_environment("cpu_tensor_share_true")

        try:
            # Create extension that processes tensors
            test_base.create_tensor_extension(
                "tensor_processor",
                share_torch=True,
                extension_code='''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class TensorProcessor(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("TensorProcessor initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("TensorProcessor preparing for shutdown.")

    @override
    async def do_stuff(self, value):
        """Handle tensor operations through the standard interface."""
        import torch

        # Check if value is a dict with operation type
        if isinstance(value, dict) and "operation" in value:
            operation = value["operation"]

            if operation == "process_tensor":
                tensor = value["tensor"]

                # Verify we received a tensor
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

                # Store tensor properties
                tensor_info = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "is_cuda": tensor.is_cuda,
                    "numel": tensor.numel(),
                    "mean": float(tensor.mean()),
                    "sum": float(tensor.sum()),
                }

                await db.set_value("tensor_info", tensor_info)

                # Create a new tensor based on the input
                result_tensor = tensor * 2 + 1

                return result_tensor

            elif operation == "test_multiple_tensors":
                tensors = value["tensors"]

                results = []
                for i, tensor in enumerate(tensors):
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"Tensor {i} is not a torch.Tensor")

                    # Process each tensor
                    processed = tensor ** 2
                    results.append(processed)

                # Stack results
                stacked = torch.stack(results)

                await db.set_value("multi_tensor_shape", list(stacked.shape))

                return stacked

        # Default behavior
        return f"TensorProcessor processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return TensorProcessor()
''',
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "tensor_processor"}])
            extension = extensions[0]
            db = DatabaseSingleton()

            # Test 1: Simple CPU tensor
            import torch

            with torch.inference_mode():
                cpu_tensor = torch.randn(3, 4)

            # Call extension method
            result_tensor = await extension.do_stuff({"operation": "process_tensor", "tensor": cpu_tensor})

            # Verify result is a tensor
            assert isinstance(result_tensor, torch.Tensor)
            assert result_tensor.shape == cpu_tensor.shape
            assert torch.allclose(result_tensor, cpu_tensor * 2 + 1)

            # Check stored info
            tensor_info = await db.get_value("tensor_info")
            assert tensor_info["shape"] == [3, 4]
            assert tensor_info["device"] == "cpu"
            assert tensor_info["is_cuda"] is False

            # Test 2: Multiple tensors
            with torch.inference_mode():
                tensors = [torch.ones(2, 2), torch.zeros(2, 2), torch.eye(2)]
            stacked_result = await extension.do_stuff(
                {"operation": "test_multiple_tensors", "tensors": tensors}
            )

            assert isinstance(stacked_result, torch.Tensor)
            assert stacked_result.shape == torch.Size([3, 2, 2])

            # Verify computations
            assert torch.allclose(stacked_result[0], torch.ones(2, 2))
            assert torch.allclose(stacked_result[1], torch.zeros(2, 2))
            assert torch.allclose(stacked_result[2], torch.eye(2))

        finally:
            await test_base.cleanup()

    async def test_cpu_tensor_share_torch_false(self):
        """Test passing CPU tensors with share_torch=False."""
        test_base = TorchTestBase()
        await test_base.setup_test_environment("cpu_tensor_share_false")

        try:
            # Create extension with its own torch installation
            test_base.create_tensor_extension(
                "isolated_tensor_processor",
                share_torch=False,
                extension_code='''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class IsolatedTensorProcessor(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("IsolatedTensorProcessor initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("IsolatedTensorProcessor preparing for shutdown.")

    @override
    async def do_stuff(self, value):
        """Handle tensor operations through the standard interface."""
        import torch
        import sys

        if isinstance(value, dict) and "operation" in value:
            operation = value["operation"]

            if operation == "verify_isolated_torch":
                torch_info = {
                    "version": torch.__version__,
                    "file_path": torch.__file__,
                    "cuda_available": torch.cuda.is_available(),
                    "num_threads": torch.get_num_threads(),
                }

                await db.set_value("isolated_torch_info", torch_info)
                return torch_info

            elif operation == "process_tensor_isolated":
                tensor = value["tensor"]

                # Verify tensor type
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

                # Perform operations
                normalized = (tensor - tensor.mean()) / tensor.std()

                result_info = {
                    "input_shape": list(tensor.shape),
                    "input_mean": float(tensor.mean()),
                    "input_std": float(tensor.std()),
                    "output_mean": float(normalized.mean()),
                    "output_std": float(normalized.std()),
                }

                await db.set_value("normalization_info", result_info)

                return normalized

            elif operation == "test_different_dtypes":
                tensors_dict = value["tensors_dict"]

                results = {}
                dtype_info = {}

                for name, tensor in tensors_dict.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"{name} is not a tensor")

                    # Store dtype info
                    dtype_info[name] = {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "min": float(tensor.min()),
                        "max": float(tensor.max()),
                    }

                    # Convert to float32 for processing
                    float_tensor = tensor.float()
                    results[name] = float_tensor.sigmoid()

                await db.set_value("dtype_info", dtype_info)

                return results

        return f"IsolatedTensorProcessor processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return IsolatedTensorProcessor()
''',
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "isolated_tensor_processor"}])
            extension = extensions[0]
            db = DatabaseSingleton()

            # First verify isolated torch
            torch_info = await extension.do_stuff({"operation": "verify_isolated_torch"})
            assert "version" in torch_info
            assert "file_path" in torch_info

            import torch

            # Test 1: Basic tensor processing
            with torch.inference_mode():
                input_tensor = torch.randn(4, 5)
            normalized = await extension.do_stuff(
                {"operation": "process_tensor_isolated", "tensor": input_tensor}
            )

            assert isinstance(normalized, torch.Tensor)
            assert normalized.shape == input_tensor.shape

            norm_info = await db.get_value("normalization_info")
            assert abs(norm_info["output_mean"]) < 1e-6  # Should be close to 0
            assert abs(norm_info["output_std"] - 1.0) < 1e-6  # Should be close to 1

            # Test 2: Different dtypes
            with torch.inference_mode():
                tensors_dict = {
                    "float32": torch.randn(2, 3),
                    "int64": torch.randint(0, 10, (2, 3)),
                    "bool": torch.tensor([[True, False], [False, True]]),
                }

            dtype_results = await extension.do_stuff(
                {"operation": "test_different_dtypes", "tensors_dict": tensors_dict}
            )

            assert len(dtype_results) == 3
            for _name, result in dtype_results.items():
                assert isinstance(result, torch.Tensor)
                assert result.dtype == torch.float32  # All converted to float32

            dtype_info = await db.get_value("dtype_info")
            assert dtype_info["float32"]["dtype"] == "torch.float32"
            assert dtype_info["int64"]["dtype"] == "torch.int64"
            assert dtype_info["bool"]["dtype"] == "torch.bool"

        finally:
            await test_base.cleanup()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    async def test_gpu_tensor_passing(self):
        """Test passing GPU tensors between host and extension."""
        test_base = TorchTestBase()
        await test_base.setup_test_environment("gpu_tensor_test")

        try:
            # Create extension that handles GPU tensors
            test_base.create_tensor_extension(
                "gpu_tensor_processor",
                share_torch=True,
                extension_code='''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class GPUTensorProcessor(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("GPUTensorProcessor initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("GPUTensorProcessor preparing for shutdown.")

    @override
    async def do_stuff(self, value):
        """Handle GPU tensor operations through the standard interface."""
        import torch

        if isinstance(value, dict) and "operation" in value:
            operation = value["operation"]

            if operation == "process_gpu_tensor":
                tensor = value["tensor"]

                # Verify tensor is on GPU
                if not tensor.is_cuda:
                    raise ValueError("Expected CUDA tensor")

                # Perform GPU operations
                result = torch.matmul(tensor, tensor.T)

                # Store GPU info
                gpu_info = {
                    "device": str(tensor.device),
                    "is_cuda": tensor.is_cuda,
                    "cuda_device_index": tensor.get_device(),
                    "result_shape": list(result.shape),
                }

                await db.set_value("gpu_info", gpu_info)

                return result

            elif operation == "transfer_between_devices":
                cpu_tensor = value["tensor"]

                # Move to GPU
                gpu_tensor = cpu_tensor.cuda()

                # Perform operation on GPU
                gpu_result = gpu_tensor * 3

                # Move back to CPU
                cpu_result = gpu_result.cpu()

                await db.set_value("transfer_complete", True)

                return cpu_result

        return f"GPUTensorProcessor processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return GPUTensorProcessor()
''',
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "gpu_tensor_processor"}])
            extension = extensions[0]
            db = DatabaseSingleton()

            import torch

            # Test 1: GPU tensor operations
            with torch.inference_mode():
                gpu_tensor = torch.randn(5, 5).cuda()
            gpu_result = await extension.do_stuff({"operation": "process_gpu_tensor", "tensor": gpu_tensor})

            assert isinstance(gpu_result, torch.Tensor)
            assert gpu_result.is_cuda
            assert gpu_result.shape == torch.Size([5, 5])

            gpu_info = await db.get_value("gpu_info")
            assert gpu_info["is_cuda"] is True
            assert "cuda" in gpu_info["device"]

            # Test 2: CPU to GPU transfer
            with torch.inference_mode():
                cpu_tensor = torch.ones(3, 3)
            transferred_result = await extension.do_stuff(
                {"operation": "transfer_between_devices", "tensor": cpu_tensor}
            )

            assert isinstance(transferred_result, torch.Tensor)
            assert not transferred_result.is_cuda  # Should be back on CPU
            assert torch.allclose(transferred_result, cpu_tensor * 3)

            assert await db.get_value("transfer_complete") is True

        finally:
            await test_base.cleanup()

    @pytest.mark.skip(reason="GPU sharing without a shared torch installation is not yet implemented")
    async def test_gpu_tensor_share_torch_false(self):
        """Test GPU tensors with isolated torch installation."""
        test_base = TorchTestBase()
        await test_base.setup_test_environment("gpu_isolated_test")

        try:
            # Create extension with isolated torch and GPU support
            test_base.create_tensor_extension(
                "isolated_gpu_processor",
                share_torch=False,
                extension_code='''
from shared import ExampleExtension, DatabaseSingleton
from typing_extensions import override
import logging

logger = logging.getLogger(__name__)
db = DatabaseSingleton()

class IsolatedGPUProcessor(ExampleExtension):
    @override
    async def initialize(self):
        logger.debug("IsolatedGPUProcessor initialized.")

    @override
    async def prepare_shutdown(self):
        logger.debug("IsolatedGPUProcessor preparing for shutdown.")

    @override
    async def do_stuff(self, value):
        """Handle GPU operations through the standard interface."""
        import torch

        if isinstance(value, dict) and "operation" in value:
            operation = value["operation"]

            if operation == "process_gpu_operations":
                tensor = value["tensor"]

                # Ensure tensor is on GPU
                if not tensor.is_cuda:
                    tensor = tensor.cuda()

                # Perform some GPU-specific operations
                squared = tensor ** 2

                gpu_stats = {
                    "input_device": str(tensor.device),
                    "squared_sum": float(squared.sum()),
                    "memory_allocated": torch.cuda.memory_allocated(),
                }

                await db.set_value("gpu_stats", gpu_stats)

                return squared

        return f"IsolatedGPUProcessor processed: {value}"

def example_entrypoint() -> ExampleExtension:
    return IsolatedGPUProcessor()
''',
            )

            # Load extension
            extensions = await test_base.load_extensions([{"name": "isolated_gpu_processor"}])
            extension = extensions[0]
            db = DatabaseSingleton()

            import torch

            # Test GPU operations
            with torch.inference_mode():
                gpu_tensor = torch.randn(4, 4).cuda()
            squared_result = await extension.do_stuff(
                {"operation": "process_gpu_operations", "tensor": gpu_tensor}
            )

            assert isinstance(squared_result, torch.Tensor)
            assert squared_result.is_cuda

            assert torch.allclose(squared_result, gpu_tensor**2)

            gpu_stats = await db.get_value("gpu_stats")
            assert "cuda" in gpu_stats["input_device"]

        finally:
            await test_base.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
