from types import ModuleType
from typing import TypeVar, final

from ._internal.shared import AsyncRPC, ProxiedSingleton

proxied_type = TypeVar("proxied_type", bound=object)


class ExtensionLocal:
    """Base class for extension functionality that runs locally in the extension process.

    This class provides the core functionality for extensions, including lifecycle hooks
    and RPC communication setup. Methods in this class always execute in the extension's
    own process, not via RPC.
    """

    async def before_module_loaded(self) -> None:
        """Hook called before the extension module is loaded.

        Override this method to perform any setup required before your extension
        module is imported. This is useful for environment preparation or
        pre-initialization tasks.

        Note:
            This method is called in the extension's process, not the host process.
        """

    async def on_module_loaded(self, module: ModuleType) -> None:
        """Hook called after the extension module is successfully loaded.

        Override this method to perform initialization that requires access to
        the loaded module. This is where you typically set up your extension's
        main functionality.

        Args:
            module: The loaded Python module object for your extension.

        Note:
            This method is called in the extension's process, not the host process.
        """

    @final
    def _initialize_rpc(self, rpc: AsyncRPC) -> None:
        """Initialize the RPC communication system for this extension.

        This method is called internally by the framework and should not be
        overridden or called directly by extension code.

        Args:
            rpc: The AsyncRPC instance for this extension's communication.
        """
        self._rpc = rpc

    @final
    def register_callee(self, object_instance: object, object_id: str) -> None:
        """Register an object that can be called remotely from the host process.

        Use this method to make your extension's functionality available to the
        host process via RPC. The registered object's async methods can then be
        called from the host.

        Args:
            object_instance: The object instance to register for remote calls.
            object_id: A unique identifier for this object. The host will use
                this ID to create a caller for this object.

        Raises:
            ValueError: If an object with the given ID is already registered.

        Example:
            >>> class MyService:
            ...     async def process(self, data: str) -> str:
            ...         return f"Processed: {data}"
            >>>
            >>> # In your extension's on_module_loaded:
            >>> service = MyService()
            >>> self.register_callee(service, "my_service")
        """
        self._rpc.register_callee(object_instance, object_id)

    @final
    def create_caller(self, object_type: type[proxied_type], object_id: str) -> proxied_type:
        """Create a proxy object for calling methods on a remote object.

        Use this method to create a caller for objects that exist in the host
        process. The returned proxy object will forward all async method calls
        via RPC.

        Args:
            object_type: The type/interface of the remote object. This is used
                for type checking and to determine which methods are available.
            object_id: The unique identifier of the remote object to connect to.

        Returns:
            A proxy object that forwards async method calls to the remote object.

        Example:
            >>> # Create a caller for a service in the host
            >>> remote_service = self.create_caller(HostService, "host_service")
            >>> result = await remote_service.do_something("data")
        """
        return self._rpc.create_caller(object_type, object_id)

    @final
    def use_remote(self, proxied_singleton: type[ProxiedSingleton]) -> None:
        """Configure a ProxiedSingleton class to use remote instances by default.

        After calling this method, any instantiation of the singleton class will
        return a proxy to the remote instance instead of creating a local instance.
        This is typically used for shared services that should have a single
        instance across all processes.

        Args:
            proxied_singleton: The ProxiedSingleton class to configure for remote use.

        Example:
            >>> # In your extension's initialization:
            >>> self.use_remote(DatabaseSingleton)
            >>> # Now DatabaseSingleton() returns a proxy to the host's instance
            >>> db = DatabaseSingleton()
            >>> await db.set_value("key", "value")
        """
        proxied_singleton.use_remote(self._rpc)


class ExtensionBase(ExtensionLocal):
    """Base class for all extensions in the pyisolate system.

    This is the main class that extension developers should inherit from when
    creating extensions. It provides the complete extension interface including
    lifecycle management, RPC communication, and cleanup functionality.

    Extensions typically override the lifecycle hooks (`before_module_loaded` and
    `on_module_loaded`) to set up their functionality, and then use the RPC
    methods to communicate with the host process.

    Example:
        >>> class MyExtension(ExtensionBase):
        ...     async def on_module_loaded(self, module: ModuleType) -> None:
        ...         # Set up your extension
        ...         self.service = module.MyService()
        ...         self.register_callee(self.service, "my_service")
        ...
        ...     async def process_data(self, data: list) -> float:
        ...         # Extension method callable from host
        ...         import numpy as np
        ...         from pyisolate._internal.gpu_utils import (
        ...             maybe_to_dlpack,
        ...             maybe_from_dlpack,
        ...             maybe_serialize_tensor,
        ...             maybe_deserialize_tensor,
        ...         )
        ...         return np.array(data).mean()

    Attributes:
        _rpc: The AsyncRPC instance for communication (set internally).
    """

    def __init__(self) -> None:
        """Initialize the extension base class.

        This constructor is called automatically when your extension is instantiated.
        You typically don't need to override this unless you need to perform
        initialization before the RPC system is set up.
        """
        super().__init__()

    async def stop(self) -> None:
        """Stop the extension and clean up resources.

        This method is called by the host when shutting down the extension.
        It ensures proper cleanup of the RPC communication system and any
        other resources.

        Note:
            This method is typically called automatically by the ExtensionManager.
            You should not need to call it directly unless managing extensions
            manually.

        If you need to perform custom cleanup, override `before_module_loaded`
        or create a custom cleanup method that is called before `stop()`.
        """
        await self._rpc.stop()
