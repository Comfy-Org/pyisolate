import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import pytest

# Import the reference package path and class
import tests.harness.test_package as test_package_module
from pyisolate._internal.adapter_registry import AdapterRegistry
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.sandbox_detect import detect_sandbox_capability
from pyisolate.config import ExtensionConfig, SandboxMode
from pyisolate.host import Extension
from pyisolate.interfaces import SerializerRegistryProtocol
from tests.harness.test_package import ReferenceTestExtension

logger = logging.getLogger(__name__)


class TestExtensionProtocol(Protocol):
    async def ping(self) -> str: ...
    async def echo_tensor(self, tensor: Any) -> Any: ...
    async def allocate_cuda(self, size_mb: int) -> dict[str, Any]: ...
    async def write_file(self, path: str, content: str) -> str: ...
    async def read_file(self, path: str) -> str: ...
    async def crash_me(self) -> None: ...
    async def get_env_var(self, key: str) -> str | None: ...


class ReferenceAdapter:
    """
    Minimal adapter for the reference harness.
    """

    @property
    def identifier(self) -> str:
        return "reference_harness"

    def get_path_config(self, module_path: str) -> dict[str, Any] | None:
        # Minimal path config
        return {"preferred_root": os.getcwd(), "additional_paths": []}

    def setup_child_environment(self, snapshot: dict[str, Any]) -> None:
        pass

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        # Register torch serializers if available
        try:
            import torch  # noqa: F401

            from pyisolate._internal.tensor_serializer import deserialize_tensor, serialize_tensor

            registry.register("torch.Tensor", serialize_tensor, deserialize_tensor)
        except Exception:
            pass

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        return []  # TODO: Add singletons when needed

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        pass


class ReferenceHost:
    """
    A verbose host harness for running integration tests.
    """

    def __init__(self, use_temp_dir: bool = True):
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self.root_dir: Path = Path(os.getcwd())
        if use_temp_dir:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="pyisolate_harness_")
            self.root_dir = Path(self.temp_dir.name)

        # Setup shared temp for Torch file_system IPC
        self.shared_tmp = self.root_dir / "ipc_shared"
        self.shared_tmp.mkdir(parents=True, exist_ok=True)
        # Force host process (and children via inherit) to use this TMPDIR
        os.environ["TMPDIR"] = str(self.shared_tmp)

        self.venv_root = self.root_dir / "venvs"
        self.venv_root.mkdir(parents=True, exist_ok=True)

        # Keep a stable uv cache across ephemeral harness dirs so large torch
        # dependency sets are reused instead of repeatedly downloaded.
        shared_uv_cache = Path(tempfile.gettempdir()) / "pyisolate_uv_cache_shared"
        shared_uv_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYISOLATE_UV_CACHE_DIR", str(shared_uv_cache))
        os.environ.setdefault("UV_HTTP_TIMEOUT", "180")

        self.extensions: list[Extension[TestExtensionProtocol]] = []
        self._adapter_registered = False
        self.sandbox_available = True
        if sys.platform == "linux":
            self.sandbox_available = detect_sandbox_capability().available

    def setup(self):
        """Initialize the host environment."""
        # Ensure uv is in PATH
        # Since we run tests with the venv python, uv should be in the same bin dir
        venv_bin = os.path.dirname(sys.executable)
        path = os.environ.get("PATH", "")
        if venv_bin not in path.split(os.pathsep):
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

        # Clean up any existing adapter to ensure fresh state
        AdapterRegistry.unregister()

        # Register our reference adapter
        self.adapter = ReferenceAdapter()
        AdapterRegistry.register(self.adapter)
        self._adapter_registered = True

        # Ensure proper torch multiprocessing setup
        try:
            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")
            # set_start_method might fail if already set, which is fine
            with contextlib.suppress(RuntimeError):
                torch.multiprocessing.set_start_method("spawn", force=True)
        except ImportError:
            pass

    def load_test_extension(
        self,
        name: str = "test_ext",
        isolated: bool = True,
        share_torch: bool = True,
        share_cuda: bool = False,
        extra_deps: list[str] | None = None,
    ) -> Extension[TestExtensionProtocol]:
        """
        Loads the static reference extension.
        """
        package_path = Path(test_package_module.__file__).parent.resolve()

        # We need to inject the pyisolate package itself into dependencies
        # so it can be installed in the isolated venv
        pyisolate_root = Path(__file__).parent.parent.parent.resolve()

        if extra_deps is None:
            extra_deps = []
        deps = [f"-e {pyisolate_root}"] + extra_deps

        if share_torch:
            pass  # We rely on site-packages inheritance for torch usually

        # Sandbox Config for IPC
        sandbox_cfg = {
            "writable_paths": [str(self.shared_tmp)],
            "doc": "Required for Torch/PyTorch file_system IPC strategy",
        }

        ext_config = ExtensionConfig(
            name=name,
            module_path=str(package_path),
            isolated=isolated,
            dependencies=deps,
            apis=[],
            env={},
            share_torch=share_torch,
            share_cuda_ipc=share_cuda,
            sandbox=sandbox_cfg,
            sandbox_mode=SandboxMode.REQUIRED if self.sandbox_available else SandboxMode.DISABLED,
        )

        ext = Extension(
            module_path=str(package_path),
            extension_type=ReferenceTestExtension,  # type: ignore
            config=ext_config,
            venv_root_path=str(self.venv_root),
        )

        ext.ensure_process_started()
        self.extensions.append(ext)
        return ext

    async def cleanup(self):
        """Stop all extensions and cleanup resources."""
        cleanup_errors = []

        # Stop processes
        for ext in self.extensions:
            try:
                ext.stop()
            except Exception as e:
                cleanup_errors.append(str(e))

        if self._adapter_registered:
            AdapterRegistry.unregister()

        if self.temp_dir:
            try:
                self.temp_dir.cleanup()
            except Exception as e:
                cleanup_errors.append(f"temp_dir: {e}")

        if cleanup_errors:
            pass


@pytest.fixture
async def reference_host():
    host = ReferenceHost()
    host.setup()
    yield host
    await host.cleanup()
