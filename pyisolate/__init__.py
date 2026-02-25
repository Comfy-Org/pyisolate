"""
pyisolate - Run Python extensions in isolated virtual environments with seamless RPC.

pyisolate enables you to run Python extensions with conflicting dependencies in the
same application by automatically creating isolated virtual environments for each
extension. Extensions communicate with the host process through a transparent RPC
system, making the isolation invisible to your code.

Key Features:
    - Automatic virtual environment creation and management
    - Transparent bidirectional RPC communication
    - Zero-copy PyTorch tensor sharing (optional)
    - Shared state through ProxiedSingleton pattern
    - Support for both isolated and non-isolated extensions

Basic Usage:
    >>> import pyisolate
    >>> import asyncio
    >>> async def main():
    ...     config = pyisolate.ExtensionManagerConfig(venv_root_path="./venvs")
    ...     manager = pyisolate.ExtensionManager(pyisolate.ExtensionBase, config)
    ...     extension = await manager.load_extension(
    ...         pyisolate.ExtensionConfig(
    ...             name="my_extension",
    ...             module_path="./extensions/my_extension",
    ...             isolated=True,
    ...             dependencies=["numpy>=2.0.0"],
    ...         )
    ...     )
    ...     result = await extension.process_data([1, 2, 3])
    ...     await extension.stop()
    >>> asyncio.run(main())
"""

from typing import TYPE_CHECKING

from ._internal.rpc_protocol import ProxiedSingleton, local_execution
from ._internal.singleton_context import singleton_scope
from ._internal.tensor_serializer import flush_tensor_keeper, purge_orphan_sender_shm_files
from .config import ExtensionConfig, ExtensionManagerConfig, SandboxMode
from .host import ExtensionBase, ExtensionManager

if TYPE_CHECKING:
    from .interfaces import IsolationAdapter

__version__ = "0.9.1"

__all__ = [
    "ExtensionBase",
    "ExtensionManager",
    "ExtensionManagerConfig",
    "ExtensionConfig",
    "SandboxMode",
    "ProxiedSingleton",
    "local_execution",
    "singleton_scope",
    "flush_tensor_keeper",
    "purge_orphan_sender_shm_files",
    "register_adapter",
    "get_adapter",
]


def register_adapter(adapter: "IsolationAdapter") -> None:
    """Register an adapter instance for pyisolate to use."""
    from ._internal.adapter_registry import AdapterRegistry

    AdapterRegistry.register(adapter)


def get_adapter() -> "IsolationAdapter | None":
    """Get the registered adapter, or None if not registered."""
    from ._internal.adapter_registry import AdapterRegistry

    return AdapterRegistry.get()
