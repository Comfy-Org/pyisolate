"""Reference IsolationAdapter implementation for testing.

This module provides a complete, working IsolationAdapter implementation
that serves two purposes:

1. **Testing**: Enables pyisolate unit tests without any host application
2. **Documentation**: Shows host implementers exactly how to build an adapter

Host developers should study this implementation as the canonical example
of how to integrate pyisolate with their application.

See Also:
    - README.md: "Implementing a Host Adapter" section
    - pyisolate/interfaces.py: IsolationAdapter protocol definition
    - tests/test_adapter_contract.py: Contract tests for adapters

Example Usage::

    from tests.fixtures.test_adapter import MockHostAdapter, MockRegistry

    # Create adapter
    adapter = MockHostAdapter(root_path="/tmp/myapp")

    # Get path configuration for an extension
    config = adapter.get_path_config("/tmp/myapp/extensions/myext/__init__.py")
    # Returns: {"preferred_root": "/tmp/myapp", "additional_paths": [...]}

    # Register custom serializers
    from pyisolate._internal.serialization_registry import SerializerRegistry
    adapter.register_serializers(SerializerRegistry.get_instance())

    # Get RPC services to expose
    services = adapter.provide_rpc_services()
    # Returns: [MockRegistry]
"""

from __future__ import annotations

from typing import Any

from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate.interfaces import IsolationAdapter, SerializerRegistryProtocol


class MockTestData:
    """Example custom type for serialization testing.

    Demonstrates how host applications define types that need
    custom serialization for RPC transport.

    Attributes:
        value: The wrapped value to serialize.
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockTestData):
            return False
        return bool(self.value == other.value)

    def __repr__(self) -> str:
        return f"MockTestData({self.value!r})"


class MockRegistry(ProxiedSingleton):
    """Example ProxiedSingleton for RPC testing.

    This demonstrates how to create a service that's accessible
    from isolated extensions via RPC. The singleton pattern ensures
    a single instance exists in the host process.

    In a real application, this might be:
    - A model registry that caches loaded models
    - A path resolution service
    - A progress reporting service

    Example::

        registry = MockRegistry()
        obj_id = registry.register({"key": "value"})
        obj = registry.get(obj_id)  # Returns {"key": "value"}
    """

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[str, Any] = {}
        self._counter = 0

    def register(self, obj: Any) -> str:
        """Register an object and return its ID.

        Args:
            obj: Any object to store in the registry.

        Returns:
            A unique string ID for retrieving the object later.
        """
        obj_id = f"obj_{self._counter}"
        self._counter += 1
        self._store[obj_id] = obj
        return obj_id

    def get(self, obj_id: str) -> Any:
        """Retrieve an object by ID.

        Args:
            obj_id: The ID returned from register().

        Returns:
            The stored object, or None if not found.
        """
        return self._store.get(obj_id)

    def clear(self) -> None:
        """Clear all stored objects."""
        self._store.clear()
        self._counter = 0


class MockHostAdapter(IsolationAdapter):
    """Reference adapter implementation for testing and documentation.

    This adapter demonstrates the complete IsolationAdapter protocol.
    Each method is documented to show:
    - What the method should do
    - What arguments it receives
    - What it should return
    - Common implementation patterns

    Args:
        root_path: The root directory for this host application.
                   Extensions will be loaded relative to this.

    Example::

        adapter = MockHostAdapter("/tmp/myhost")
        assert adapter.identifier == "testhost"

        config = adapter.get_path_config("/tmp/myhost/ext/demo/__init__.py")
        assert config["preferred_root"] == "/tmp/myhost"
    """

    def __init__(self, root_path: str = "/tmp/testhost") -> None:
        self._root = root_path
        self._extensions_dir = f"{root_path}/extensions"

    @property
    def identifier(self) -> str:
        """Return unique adapter identifier.

        This should be a short, lowercase string that identifies your
        host application. It's used in:
        - Logging messages
        - Adapter discovery via entry points
        - Debug output

        Returns:
            A unique identifier string (e.g., "comfyui", "testhost").
        """
        return "testhost"

    def get_path_config(self, module_path: str) -> dict[str, Any]:
        """Compute path configuration for an extension.

        This method tells pyisolate how to configure sys.path for
        isolated extensions. The returned configuration ensures that:
        - Host application modules are importable
        - Extension-specific paths are available

        Args:
            module_path: Absolute path to the extension's __init__.py

        Returns:
            A dict with:
            - ``preferred_root``: Your application's root directory.
              This is prepended to sys.path so host modules can be
              imported from isolated extensions.
            - ``additional_paths``: Extra paths to add to sys.path.
              These come after preferred_root but before the venv.

        Example:
            For ComfyUI, this returns the ComfyUI install directory
            and paths to custom_nodes, comfy, etc.::

                {
                    "preferred_root": "/home/user/ComfyUI",
                    "additional_paths": [
                        "/home/user/ComfyUI/custom_nodes",
                    ]
                }
        """
        return {
            "preferred_root": self._root,
            "additional_paths": [self._extensions_dir],
        }

    def setup_child_environment(self, snapshot: dict[str, Any]) -> None:
        """Configure child process after sys.path reconstruction.

        This is called in the child (isolated) process after sys.path
        is configured but before extension code runs. Use this to:

        - Set environment variables specific to isolated processes
        - Initialize logging with child-specific handlers
        - Configure application state that differs in isolation
        - Set up profiling or debugging for isolated code

        Args:
            snapshot: The host snapshot dict containing:
                - ``sys_path``: The configured sys.path
                - ``context_data``: Custom data from the host
                - ``adapter_name``: This adapter's identifier
                - ``preferred_root``: The host's root path

        Note:
            This runs in the CHILD process, not the host. Any state
            set here is isolated from the host process.
        """
        # Example: Could set up child-specific logging
        # import logging
        # logging.getLogger("myhost.isolated").setLevel(logging.DEBUG)

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        """Register custom type serializers for RPC transport.

        PyIsolate handles serialization of basic types automatically:
        - Primitives (int, str, float, bool, None)
        - Collections (list, dict, tuple)
        - torch.Tensor (with special handling for CUDA)

        For application-specific types that need to cross the process
        boundary, register serializers here.

        Args:
            registry: The serializer registry to register with.
                      Call registry.register() for each custom type.

        Example::

            # Serialize MyModel by extracting an ID, deserialize by
            # creating a proxy that forwards method calls via RPC
            registry.register(
                "MyModel",
                serializer=lambda m: {"id": m.model_id},
                deserializer=lambda d: ModelProxy(d["id"])
            )

        Note:
            The type_name should match the class __name__ exactly.
            Serializers must return JSON-serializable data.
        """
        # Register MockTestData for serialization testing
        registry.register(
            "MockTestData",
            serializer=lambda d: {"__testdata__": True, "value": d.value},
            deserializer=lambda d: MockTestData(d["value"]) if d.get("__testdata__") else d,
        )

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        """Return ProxiedSingleton classes to expose via RPC.

        These singletons live in the host process and are accessible
        from isolated extensions via RPC calls. Common uses:

        - Model registries: Cache loaded models, return handles
        - Path services: Resolve paths in the host filesystem
        - Progress reporting: Send progress updates to UI
        - Resource management: Coordinate GPU memory, file handles

        Returns:
            A list of ProxiedSingleton subclasses. Each class will be
            instantiated once in the host and made available to
            isolated extensions.

        Example::

            def provide_rpc_services(self):
                return [ModelRegistry, ProgressReporter, PathService]
        """
        return [MockRegistry]

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        """Post-registration hook for API-specific setup.

        Called after each ProxiedSingleton is registered with RPC.
        Use this for initialization that requires the RPC instance,
        such as:

        - Setting up bidirectional callbacks
        - Registering additional methods dynamically
        - Connecting to external services

        Args:
            api: The singleton instance that was just registered.
            rpc: The AsyncRPC instance for this extension.

        Note:
            This is optional. Many adapters leave this empty.
        """
        return None

    def setup_web_directory(self, module: Any) -> None:
        return None

    def setup_child_event_hooks(self, rpc: AsyncRPC) -> None:
        return None

    def get_sandbox_system_paths(self) -> list[str]:
        return []

    def get_sandbox_gpu_patterns(self) -> list[str]:
        return []
