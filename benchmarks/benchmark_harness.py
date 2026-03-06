import contextlib
import os
import sys
import tempfile
from pathlib import Path

from pyisolate import ExtensionConfig, ExtensionManager, ExtensionManagerConfig
from pyisolate.config import SandboxMode

try:
    import torch.multiprocessing as torch_mp

    TORCH_AVAILABLE = True
except ImportError:
    torch_mp = None
    TORCH_AVAILABLE = False


class BenchmarkHarness:
    """Harness for running benchmarks without depending on test suite infrastructure."""

    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="pyisolate_bench_")
        self.test_root = Path(self.temp_dir.name)
        (self.test_root / "extensions").mkdir(exist_ok=True)
        (self.test_root / "extension-venvs").mkdir(exist_ok=True)
        self.extensions = []
        self.manager = None

    async def setup_test_environment(self, name: str) -> None:
        """Initialize the benchmark environment."""
        # Ensure uv is in PATH (required for venv creation)
        venv_bin = os.path.dirname(sys.executable)
        path = os.environ.get("PATH", "")
        if venv_bin not in path.split(os.pathsep):
            os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

        # Setup shared temp for Torch file_system IPC
        # This is CRITICAL for share_torch=True to work in sandboxed environments
        shared_tmp = self.test_root / "ipc_shared"
        shared_tmp.mkdir(parents=True, exist_ok=True)
        # Force host process (and children via inherit) to use this TMPDIR
        os.environ["TMPDIR"] = str(shared_tmp)

        print(f"Benchmark Harness initialized at {self.test_root}")
        print(f"IPC Shared Directory: {shared_tmp}")

        # Ensure proper torch multiprocessing setup
        if TORCH_AVAILABLE and torch_mp is not None:
            with contextlib.suppress(ImportError):
                torch_mp.set_sharing_strategy("file_system")

    def create_extension(
        self, name: str, dependencies: list[str], share_torch: bool, extension_code: str
    ) -> None:
        """Create an extension module on disk."""
        ext_dir = self.test_root / "extensions" / name
        ext_dir.mkdir(parents=True, exist_ok=True)
        (ext_dir / "__init__.py").write_text(extension_code)

    async def load_extensions(self, extension_configs: list[dict], extension_base_cls) -> list:
        """Load extensions defined in configs."""
        config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
        self.manager = ExtensionManager(extension_base_cls, config)

        loaded_extensions = []
        for cfg in extension_configs:
            name = cfg["name"]
            config = ExtensionConfig(
                name=name,
                module_path=str(self.test_root / "extensions" / name),
                isolated=cfg.get("isolated", True),
                dependencies=cfg.get("dependencies", []),
                apis=cfg.get("apis", []),
                share_torch=cfg.get("share_torch", False),
                share_cuda_ipc=cfg.get("share_cuda_ipc", False),
                sandbox=cfg.get("sandbox", {}),
                sandbox_mode=cfg.get("sandbox_mode", SandboxMode.REQUIRED),
                env=cfg.get("env", {}),
            )
            loaded_extensions.append(self.manager.load_extension(config))

        return loaded_extensions

    def get_manager(self, extension_base_cls):
        if not self.manager:
            config = ExtensionManagerConfig(venv_root_path=str(self.test_root / "extension-venvs"))
            self.manager = ExtensionManager(extension_base_cls, config)
        return self.manager

    async def cleanup(self):
        """Clean up resources."""
        if self.manager:
            try:
                self.manager.stop_all_extensions()
            except Exception as e:
                print(f"Error stopping extensions: {e}")

        if self.temp_dir:
            self.temp_dir.cleanup()
