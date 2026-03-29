import contextlib
import hashlib
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from ..config import ExtensionConfig, SandboxMode
from ..shared import ExtensionBase
from .environment import (
    build_extension_snapshot,
    create_venv,
    install_dependencies,
    normalize_extension_name,
    validate_backend_config,
    validate_dependency,
    validate_path_within_root,
)
from .environment_conda import (
    _resolve_pixi_python,
    create_conda_env,
)
from .rpc_protocol import AsyncRPC
from .rpc_transports import JSONSocketTransport
from .sandbox import build_bwrap_command
from .sandbox_detect import detect_sandbox_capability
from .tensor_serializer import register_tensor_serializer
from .torch_gate import get_torch_optional
from .torch_utils import probe_cuda_ipc_support

__all__ = [
    "Extension",
    "ExtensionBase",
    "build_extension_snapshot",
    "normalize_extension_name",
    "validate_dependency",
]

logger = logging.getLogger(__name__)


class _DeduplicationFilter(logging.Filter):
    def __init__(self, timeout_seconds: int = 10):
        super().__init__()
        self.timeout = timeout_seconds
        self.last_seen: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        import time

        msg_content = record.getMessage()
        msg_hash = hashlib.sha256(msg_content.encode("utf-8")).hexdigest()
        now = time.time()

        if msg_hash in self.last_seen and now - self.last_seen[msg_hash] < self.timeout:
            return False  # Suppress duplicate

        self.last_seen[msg_hash] = now

        if len(self.last_seen) > 1000:
            cutoff = now - self.timeout
            self.last_seen = {k: v for k, v in self.last_seen.items() if v > cutoff}

        return True


T = TypeVar("T", bound=ExtensionBase)


class Extension(Generic[T]):
    def __init__(
        self,
        module_path: str,
        extension_type: type[T],
        config: ExtensionConfig,
        venv_root_path: str,
    ) -> None:
        force_ipc = os.environ.get("PYISOLATE_FORCE_CUDA_IPC") == "1"

        if "share_cuda_ipc" not in config:
            # Default to True ONLY if supported and sharing torch
            ipc_supported, _ = probe_cuda_ipc_support()
            config["share_cuda_ipc"] = force_ipc or (config.get("share_torch", False) and ipc_supported)
        elif force_ipc:
            config["share_cuda_ipc"] = True

        self.name = config["name"]
        self.normalized_name = normalize_extension_name(self.name)

        for dep in config["dependencies"]:
            validate_dependency(dep)

        venv_root = Path(venv_root_path).resolve()
        self.venv_path = venv_root / self.normalized_name
        validate_path_within_root(self.venv_path, venv_root)

        self.module_path = module_path
        self.config = config
        self.extension_type = extension_type
        self._cuda_ipc_enabled = False
        self._host_rpc_services: list[type[Any]] = []

        # Auto-populate APIs from adapter if not already in config
        if "apis" not in self.config:
            try:
                # v1.0: Check registry
                from .adapter_registry import AdapterRegistry

                adapter = AdapterRegistry.get()

                if adapter:
                    rpc_services = adapter.provide_rpc_services()
                    self.config["apis"] = rpc_services
                else:
                    self.config["apis"] = []
            except Exception as exc:
                logger.warning("[Extension] Could not load adapter RPC services: %s", exc)
                self.config["apis"] = []

        if self.config.get("execution_model") == "sealed_worker":
            try:
                from .adapter_registry import AdapterRegistry

                adapter = AdapterRegistry.get()
                if adapter:
                    self._host_rpc_services = list(adapter.provide_rpc_services())
            except Exception as exc:
                logger.warning("[Extension] Could not load sealed-worker host RPC services: %s", exc)
                self._host_rpc_services = []
        else:
            self._host_rpc_services = list(self.config["apis"])

        self.mp: Any
        if self.config["share_torch"]:
            torch, _ = get_torch_optional()
            if torch is None:
                raise RuntimeError(
                    "share_torch=True requires PyTorch. Install 'torch' to use tensor-sharing features."
                )
            self.mp = torch.multiprocessing
        else:
            import multiprocessing

            self.mp = multiprocessing

        self._process_initialized = False
        self.log_queue: Any | None = None
        self.log_listener: QueueListener | None = None

        # UDS / JSON-RPC resources
        self._uds_listener: Any | None = None
        self._uds_path: str | None = None
        self._client_sock: Any | None = None

        self.extension_proxy: T | None = None

    def _package_manager(self) -> str:
        return self.config.get("package_manager", "uv")

    def _execution_model(self) -> str:
        execution_model = self.config.get("execution_model")
        if execution_model is not None:
            return execution_model
        return "sealed_worker" if self._package_manager() == "conda" else "host-coupled"

    def _is_sealed_worker(self) -> bool:
        return self._execution_model() == "sealed_worker"

    def _tensor_transport_mode(self) -> str:
        if self._is_sealed_worker():
            return "json"
        return "shared_memory"

    def ensure_process_started(self) -> None:
        """Start the isolated process if it has not been initialized."""
        if self._process_initialized:
            return
        self._initialize_process()
        self._process_initialized = True

    def _initialize_process(self) -> None:
        """Initialize queues, RPC, and launch the isolated process."""
        try:
            self.ctx = self.mp.get_context("spawn")
        except ValueError as e:
            raise RuntimeError(f"Failed to get 'spawn' context: {e}") from e

        # Determine CUDA IPC eligibility up front (host side)
        self._cuda_ipc_enabled = False
        want_ipc = bool(self.config.get("share_cuda_ipc", False))
        if want_ipc:
            if not self.config.get("share_torch", False):
                raise RuntimeError("share_cuda_ipc requires share_torch=True")
            supported, reason = probe_cuda_ipc_support()
            if not supported:
                raise RuntimeError(f"CUDA IPC requested but unavailable: {reason}")
            self._cuda_ipc_enabled = True
            logger.debug("CUDA IPC enabled for %s", self.name)

        # Monotonically enable IPC logic. Do not disable if already enabled by another extension.
        if self._cuda_ipc_enabled:
            os.environ["PYISOLATE_ENABLE_CUDA_IPC"] = "1"

        # PYISOLATE_CHILD is set in the child's env dict, NOT in os.environ
        # Setting it in os.environ would affect the HOST process serialization logic

        if os.name == "nt":
            # On Windows, Manager().Queue() spawns a process that re-imports __main__,
            # causing issues when __main__ is ComfyUI's main.py. Use a simple queue
            # from the threading module instead - logs go to stdout anyway.
            import queue

            self.log_queue = queue.Queue()  # type: ignore[assignment]
        else:
            self.log_queue = self.ctx.Queue()

        self.extension_proxy = None

        # Create handler with deduplication filter (industry standard)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.addFilter(_DeduplicationFilter(timeout_seconds=5))

        self.log_listener = QueueListener(self.log_queue, stream_handler)
        self.log_listener.start()

        torch, _ = get_torch_optional()
        tensor_transport = self._tensor_transport_mode()
        if torch is not None:
            # Register tensor serializer for JSON-RPC only when torch is available.
            from .serialization_registry import SerializerRegistry

            register_tensor_serializer(SerializerRegistry.get_instance(), mode=tensor_transport)
            # Ensure file_system strategy for CPU tensors.
            if tensor_transport == "shared_memory":
                torch.multiprocessing.set_sharing_strategy("file_system")
        elif self.config.get("share_torch", False):
            raise RuntimeError(
                "share_torch=True requires PyTorch. Install 'torch' to use tensor-sharing features."
            )

        self.proc = self.__launch()

        for api in self._host_rpc_services:
            api()._register(self.rpc)

        # Register event bridge for child→host event dispatch
        from .event_bridge import _EventBridge

        self._event_bridge = _EventBridge()
        self.rpc.register_callee(self._event_bridge, "_event_bridge")

        self.rpc.run()

    def get_proxy(self) -> T:
        """Return (and memoize) the RPC caller for the remote extension."""
        if self.extension_proxy is None:
            self.extension_proxy = self.rpc.create_caller(self.extension_type, "extension")
        return self.extension_proxy

    def stop(self) -> None:
        """Stop the extension process and clean up queues/listeners."""
        errors: list[str] = []

        if hasattr(self, "rpc") and self.rpc:
            try:
                self.rpc.shutdown()
            except Exception as exc:
                errors.append(f"rpc shutdown: {exc}")

        # Terminate process
        if hasattr(self, "proc") and self.proc:
            try:
                # Attempt graceful exit via RPC closure first
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=3.0)

                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait()
            except Exception as exc:
                errors.append(f"terminate: {exc}")

        if self.log_listener:
            try:
                self.log_listener.stop()
            except Exception as exc:
                errors.append(f"log_listener: {exc}")

        # Clean up UDS resources
        if self._client_sock:
            try:
                self._client_sock.close()
            except Exception as exc:
                errors.append(f"client_sock: {exc}")

        if self._uds_listener:
            try:
                self._uds_listener.close()
            except Exception as exc:
                errors.append(f"uds_listener: {exc}")

        if self._uds_path and os.path.exists(self._uds_path):
            try:
                os.unlink(self._uds_path)
            except Exception as exc:
                errors.append(f"unlink uds: {exc}")

        if self.log_queue:
            try:
                # On Windows/multiprocessing, queue might need closing
                if hasattr(self.log_queue, "close"):
                    self.log_queue.close()
            except Exception as exc:
                errors.append(f"log_queue: {exc}")

        self._process_initialized = False
        self.extension_proxy = None
        if hasattr(self, "rpc"):
            del self.rpc

        if errors:
            raise RuntimeError(f"Errors stopping {self.name}: {'; '.join(errors)}")

    def __launch(self) -> Any:
        """Launch the extension in a separate process after venv + deps are ready."""
        validate_backend_config(self.config)

        if self._package_manager() == "conda":
            # Conda backend: force share_cuda_ipc=False (pixi envs don't support IPC)
            self.config["share_cuda_ipc"] = False
            self._cuda_ipc_enabled = False
            create_conda_env(self.venv_path, self.config, self.name)
        else:
            create_venv(self.venv_path, self.config)
            install_dependencies(self.venv_path, self.config, self.name)

        return self._launch_with_uds()

    def _launch_with_uds(self) -> Any:
        """Launch the extension using UDS or TCP + JSON-RPC (Standard Isolation)."""
        from .socket_utils import ensure_ipc_socket_dir, has_af_unix

        # Determine Python executable
        if self._package_manager() == "conda":
            python_exe = str(_resolve_pixi_python(self.venv_path))
        elif os.name == "nt":
            python_exe = str(self.venv_path / "Scripts" / "python.exe")
        else:
            python_exe = str(self.venv_path / "bin" / "python")

        # Create listener socket - use AF_UNIX if available, otherwise TCP loopback
        if has_af_unix():
            run_dir = ensure_ipc_socket_dir()
            uds_path = tempfile.mktemp(prefix="ext_", suffix=".sock", dir=str(run_dir))
            listener_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
            listener_sock.bind(uds_path)
            if os.name != "nt":
                os.chmod(uds_path, 0o600)
            self._uds_path = uds_path
            ipc_address = uds_path
        else:
            # TCP fallback for Windows without AF_UNIX
            listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener_sock.bind(("127.0.0.1", 0))  # Bind to random available port
            _, port = listener_sock.getsockname()
            self._uds_path = None
            ipc_address = f"tcp://127.0.0.1:{port}"

        listener_sock.listen(1)
        self._uds_listener = listener_sock

        # Prepare environment
        env = os.environ.copy()
        pyisolate_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if "env" in self.config:
            env.update(self.config["env"])

        # Get sandbox mode (default: REQUIRED)
        sandbox_mode = self.config.get("sandbox_mode", SandboxMode.REQUIRED)
        # Handle string values from config files
        if isinstance(sandbox_mode, str):
            sandbox_mode = SandboxMode(sandbox_mode)

        # Check platform for sandbox requirement
        use_sandbox = False
        is_sealed_worker = self._is_sealed_worker()
        use_sealed_worker_bwrap = is_sealed_worker
        if sys.platform == "linux":
            if sandbox_mode == SandboxMode.DISABLED:
                use_sandbox = False
            elif use_sealed_worker_bwrap:
                cap = detect_sandbox_capability()
                if not cap.available:
                    raise RuntimeError(
                        f"Process isolation on Linux REQUIRES bubblewrap.\n"
                        f"Error: {cap.remediation}\n"
                        f"Details: {cap.restriction_model} - {cap.raw_error}\n\n"
                        f"If you understand the security risks and want to proceed without "
                        f"sandbox protection, set sandbox_mode='disabled' in your extension config."
                    )
                use_sandbox = True
            elif is_sealed_worker:
                use_sandbox = False
            else:
                cap = detect_sandbox_capability()
                if not cap.available:
                    # REQUIRED mode (default) but bwrap unavailable - fail loud
                    raise RuntimeError(
                        f"Process isolation on Linux REQUIRES bubblewrap.\n"
                        f"Error: {cap.remediation}\n"
                        f"Details: {cap.restriction_model} - {cap.raw_error}\n\n"
                        f"If you understand the security risks and want to proceed without "
                        f"sandbox protection, set sandbox_mode='disabled' in your extension config."
                    )
                else:
                    use_sandbox = True

            if use_sandbox:
                # Build Bwrap Command
                sandbox_config = self.config.get("sandbox", {})
                if isinstance(sandbox_config, bool):
                    sandbox_config = {}

                adapter = None
                try:
                    from .adapter_registry import AdapterRegistry

                    adapter = AdapterRegistry.get()
                except Exception as exc:
                    logger.warning("[Extension] Could not load adapter for sandbox paths: %s", exc)

                cmd = build_bwrap_command(
                    python_exe=python_exe,
                    module_path=self.module_path,
                    venv_path=str(self.venv_path),
                    uds_address=ipc_address,
                    sandbox_config=cast(dict[str, Any], sandbox_config),
                    allow_gpu=True,  # Default to allowing GPU for ComfyUI nodes
                    restriction_model=cap.restriction_model,
                    env_overrides=self.config.get("env"),
                    adapter=adapter,
                    execution_model=self._execution_model(),
                    sealed_host_ro_paths=cast(list[str] | None, self.config.get("sealed_host_ro_paths")),
                )
            else:
                # Linux without sandbox (DISABLED mode)
                cmd = [python_exe, "-m", "pyisolate._internal.uds_client"]
                env["PYISOLATE_UDS_ADDRESS"] = ipc_address
                env["PYISOLATE_CHILD"] = "1"
                env["PYISOLATE_EXTENSION"] = self.name
                env["PYISOLATE_MODULE_PATH"] = self.module_path
                env["PYISOLATE_ENABLE_CUDA_IPC"] = "1" if self._cuda_ipc_enabled else "0"
                if is_sealed_worker:
                    env["PYTHONPATH"] = pyisolate_root
                    env["PYTHONNOUSERSITE"] = "1"

        else:
            # Non-Linux (Windows/Mac) - Fallback to direct launch
            cmd = [python_exe, "-m", "pyisolate._internal.uds_client"]

            env["PYISOLATE_UDS_ADDRESS"] = ipc_address
            env["PYISOLATE_CHILD"] = "1"
            env["PYISOLATE_EXTENSION"] = self.name
            env["PYISOLATE_MODULE_PATH"] = self.module_path
            env["PYISOLATE_ENABLE_CUDA_IPC"] = "1" if self._cuda_ipc_enabled else "0"
            if is_sealed_worker:
                env["PYTHONPATH"] = pyisolate_root
                env["PYTHONNOUSERSITE"] = "1"

        # Tell the child whether to import torch (controls adapter import chain)
        if not self.config.get("share_torch", False):
            env["PYISOLATE_IMPORT_TORCH"] = "0"
            # Also inject into config env for bwrap path which uses env_overrides
            if "env" not in self.config:
                self.config["env"] = {}
            self.config["env"]["PYISOLATE_IMPORT_TORCH"] = "0"

        # Launch process
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=self.module_path if is_sealed_worker else None,
            stdout=None,  # Inherit stdout/stderr for now so we see logs
            stderr=None,
            close_fds=True,
        )

        # Accept connection
        client_sock = None
        accept_error = None

        def accept_connection() -> None:
            nonlocal client_sock, accept_error
            try:
                client_sock, _ = listener_sock.accept()
            except Exception as e:
                accept_error = e

        accept_thread = threading.Thread(target=accept_connection)
        accept_thread.daemon = True
        accept_thread.start()
        accept_thread.join(timeout=30.0)

        if accept_thread.is_alive():
            proc.terminate()
            raise RuntimeError(f"Child failed to connect within timeout for {self.name}")

        if accept_error:
            proc.terminate()
            raise RuntimeError(f"Child failed to connect for {self.name}: {accept_error}")

        if client_sock is None:
            proc.terminate()
            raise RuntimeError(f"Child connection is None for {self.name}")

        # Setup JSON-RPC
        tensor_transport = self._tensor_transport_mode()
        transport = JSONSocketTransport(client_sock)
        if hasattr(transport, "set_tensor_transport_mode"):
            transport.set_tensor_transport_mode(tensor_transport)
        logger.debug("Child connected, sending bootstrap data")

        # Send bootstrap
        snapshot = build_extension_snapshot(self.module_path)
        if is_sealed_worker:
            snapshot["apply_host_sys_path"] = False
            snapshot["preferred_root"] = None
            snapshot["additional_paths"] = []
            sealed_host_ro_paths = self.config.get("sealed_host_ro_paths") or []
            snapshot["sealed_host_ro_paths"] = list(cast(list[str], sealed_host_ro_paths))
            # If RO paths are configured, preserve adapter_ref so the sealed
            # child can rehydrate the adapter and register serializers.
            # Without RO paths, null out adapter to maintain hermetic boundary.
            if not sealed_host_ro_paths:
                snapshot["adapter_ref"] = None
                snapshot["adapter_name"] = None
        ext_type_ref = f"{self.extension_type.__module__}.{self.extension_type.__name__}"

        # Sanitize config for JSON serialization (convert API classes to string refs)
        safe_config = dict(self.config)  # type: ignore[arg-type]
        if "apis" in safe_config:
            api_list: list[str] = [
                f"{api.__module__}.{api.__name__}"
                for api in self.config["apis"]  # type: ignore[union-attr]
            ]
            safe_config["apis"] = api_list

        bootstrap_data = {
            "snapshot": snapshot,
            "config": safe_config,
            "extension_type_ref": ext_type_ref,
            "tensor_transport": tensor_transport,
        }
        transport.send(bootstrap_data)

        self._client_sock = client_sock
        self.rpc = AsyncRPC(transport=transport)

        return proc

    def join(self) -> None:
        """Join the child process, blocking until it exits."""
        self.proc.join()

    def register_event_handler(self, name: str, handler: Any) -> None:
        """Register a handler for named events emitted by the child process."""
        self._event_bridge.register_handler(name, handler)
