"""Host-side ExtensionManager for PyIsolate.

Manages isolated virtual environments, dependency installation, and RPC lifecycle
for extensions loaded into separate processes.
"""

import logging
from typing import Any, Generic, TypeVar, cast

from ._internal.host import Extension
from .config import ExtensionConfig, ExtensionManagerConfig
from .shared import ExtensionBase, ExtensionLocal

__all__ = ["ExtensionManager", "ExtensionBase", "ExtensionConfig", "ExtensionManagerConfig"]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ExtensionBase)


class ExtensionManager(Generic[T]):
    """Manager for loading and supervising isolated extensions."""

    def __init__(self, extension_type: type[T], config: ExtensionManagerConfig) -> None:
        """Initialize the ExtensionManager.

        Args:
            extension_type: Base class that all managed extensions inherit from.
            config: Manager configuration (e.g., root path for virtualenvs).
        """
        self.config = config
        self.extensions: dict[str, Extension[T]] = {}
        self.extension_type = extension_type

    def load_extension(self, config: ExtensionConfig) -> T:
        """Load an extension with the given configuration.

        Creates the venv (if isolated), installs dependencies, starts the child
        process, and returns a proxy that forwards calls to the isolated extension.
        """
        name = config["name"]
        if name in self.extensions:
            raise ValueError(f"Extension '{name}' is already loaded")

        extension: Extension[T] = Extension(
            module_path=config["module_path"],
            extension_type=self.extension_type,
            config=config,
            venv_root_path=self.config["venv_root_path"],
        )

        self.extensions[name] = extension

        class HostExtension(ExtensionLocal):
            def __init__(self, extension_instance: Extension[T]) -> None:
                super().__init__()
                self._extension = extension_instance
                self._proxy: Any = None
                self._pending_event_handlers: list[tuple[str, Any]] = []

            @property
            def proxy(self) -> Any:
                # Invalidate cached proxy if process was stopped and needs restart
                if self._proxy is not None and not self._extension._process_initialized:
                    self._proxy = None

                if self._proxy is None:
                    if hasattr(self._extension, "ensure_process_started"):
                        self._extension.ensure_process_started()
                    # Re-register event handlers after process restart
                    for name, handler in self._pending_event_handlers:
                        self._extension.register_event_handler(name, handler)
                    self._proxy = self._extension.get_proxy()
                    self._initialize_rpc(self._extension.rpc)
                return self._proxy

            def __getattr__(self, item: str) -> Any:
                if hasattr(self._extension, item):
                    return getattr(self._extension, item)
                return getattr(self.proxy, item)

            def register_event_handler(self, name: str, handler: Any) -> None:
                """Register a handler for named events from the child process."""
                self._pending_event_handlers.append((name, handler))
                if self._extension._process_initialized:
                    self._extension.register_event_handler(name, handler)

        return cast(T, HostExtension(extension))

    def stop_all_extensions(self) -> None:
        """Stop all managed extensions and clean up resources."""
        for name, extension in self.extensions.items():
            try:
                extension.stop()
            except Exception as e:
                logger.error(f"Error stopping extension '{name}': {e}")
        self.extensions.clear()
