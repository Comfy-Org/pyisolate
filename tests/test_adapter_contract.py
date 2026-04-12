"""Tests for IsolationAdapter protocol compliance.

These tests verify that adapters implementing IsolationAdapter
behave correctly according to the protocol contract. They test
at the boundary (adapter interface), not internal implementation.

The MockHostAdapter from fixtures serves as the reference implementation
and is used to demonstrate expected behavior for each protocol method.
"""

from typing import cast

from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.serialization_registry import SerializerRegistry
from pyisolate.interfaces import IsolationAdapter

from .fixtures.test_adapter import MockHostAdapter, MockRegistry, MockTestData


class TestAdapterIdentifier:
    """Tests for the identifier property."""

    def test_adapter_has_identifier(self) -> None:
        """Adapter must have a non-empty identifier."""
        adapter = MockHostAdapter()
        assert adapter.identifier
        assert isinstance(adapter.identifier, str)

    def test_identifier_is_lowercase(self) -> None:
        """Identifier should be lowercase for consistency."""
        adapter = MockHostAdapter()
        assert adapter.identifier == adapter.identifier.lower()

    def test_identifier_no_spaces(self) -> None:
        """Identifier should not contain spaces."""
        adapter = MockHostAdapter()
        assert " " not in adapter.identifier


class TestAdapterPathConfig:
    """Tests for get_path_config method."""

    def test_get_path_config_returns_dict(self) -> None:
        """get_path_config must return dict with required keys."""
        adapter = MockHostAdapter()
        config = adapter.get_path_config("/some/module/__init__.py")

        assert isinstance(config, dict)
        assert "preferred_root" in config
        assert "additional_paths" in config

    def test_preferred_root_is_string(self) -> None:
        """preferred_root must be a string path."""
        adapter = MockHostAdapter("/tmp/myapp")
        config = adapter.get_path_config("/tmp/myapp/ext/__init__.py")

        assert isinstance(config["preferred_root"], str)
        assert config["preferred_root"] == "/tmp/myapp"

    def test_additional_paths_is_list(self) -> None:
        """additional_paths must be a list of strings."""
        adapter = MockHostAdapter()
        config = adapter.get_path_config("/some/path")

        assert isinstance(config["additional_paths"], list)
        for path in config["additional_paths"]:
            assert isinstance(path, str)


class TestAdapterSerializers:
    """Tests for register_serializers method."""

    def test_register_serializers_accepts_registry(self) -> None:
        """register_serializers must accept SerializerRegistryProtocol."""
        adapter = MockHostAdapter()
        registry = SerializerRegistry.get_instance()
        registry.clear()  # Start fresh

        # Should not raise
        adapter.register_serializers(registry)

    def test_registered_serializer_is_callable(self) -> None:
        """Registered serializers must be callable."""
        adapter = MockHostAdapter()
        registry = SerializerRegistry.get_instance()
        registry.clear()

        adapter.register_serializers(registry)

        serializer = registry.get_serializer("MockTestData")
        assert serializer is not None
        assert callable(serializer)

    def test_serializer_produces_json_compatible(self) -> None:
        """Serializer output must be JSON-compatible."""
        import json

        adapter = MockHostAdapter()
        registry = SerializerRegistry.get_instance()
        registry.clear()

        adapter.register_serializers(registry)

        serializer = registry.get_serializer("MockTestData")
        assert serializer is not None
        test_obj = MockTestData("hello")
        result = serializer(test_obj)

        # Must be JSON serializable
        json_str = json.dumps(result)
        assert json_str


class TestAdapterRpcServices:
    """Tests for provide_rpc_services method."""

    def test_provide_rpc_services_returns_list(self) -> None:
        """provide_rpc_services must return a list."""
        adapter = MockHostAdapter()
        services = adapter.provide_rpc_services()

        assert isinstance(services, list)

    def test_services_are_proxied_singleton_subclasses(self) -> None:
        """Each service must be a ProxiedSingleton subclass."""
        adapter = MockHostAdapter()
        services = adapter.provide_rpc_services()

        for svc in services:
            assert isinstance(svc, type), f"{svc} is not a class"
            assert issubclass(svc, ProxiedSingleton), f"{svc} is not a ProxiedSingleton"

    def test_services_are_instantiable(self) -> None:
        """Each service class must be instantiable with no args."""
        adapter = MockHostAdapter()
        services = adapter.provide_rpc_services()

        for svc_cls in services:
            # Should not raise
            instance = svc_cls()
            assert instance is not None


class TestAdapterApiRegistration:
    """Tests for handle_api_registration method."""

    def test_handle_api_registration_accepts_args(self) -> None:
        """handle_api_registration must accept api and rpc args."""
        adapter = MockHostAdapter()

        # Create mock api and rpc
        api = MockRegistry()
        rpc = cast(AsyncRPC, object())

        # Should not raise
        adapter.handle_api_registration(api, rpc)


class TestAdapterProtocolCompliance:
    """Tests that verify full protocol compliance."""

    def test_adapter_implements_protocol(self) -> None:
        """MockHostAdapter must implement IsolationAdapter protocol."""
        adapter = MockHostAdapter()

        # Protocol check (structural typing)
        assert isinstance(adapter, IsolationAdapter)

    def test_adapter_is_runtime_checkable(self) -> None:
        """IsolationAdapter must be runtime checkable."""
        # This verifies the @runtime_checkable decorator is present
        adapter = MockHostAdapter()
        assert isinstance(adapter, IsolationAdapter)


class TestMockRegistryBehavior:
    """Tests for MockRegistry ProxiedSingleton example."""

    def test_registry_register_returns_id(self) -> None:
        """register() must return a string ID."""
        registry = MockRegistry()
        obj_id = registry.register({"key": "value"})

        assert isinstance(obj_id, str)
        assert obj_id.startswith("obj_")

    def test_registry_get_retrieves_object(self) -> None:
        """get() must retrieve the registered object."""
        registry = MockRegistry()
        original = {"key": "value"}
        obj_id = registry.register(original)

        retrieved = registry.get(obj_id)
        assert retrieved == original

    def test_registry_get_unknown_returns_none(self) -> None:
        """get() with unknown ID must return None."""
        registry = MockRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_registry_clear_removes_all(self) -> None:
        """clear() must remove all stored objects."""
        registry = MockRegistry()
        registry.register("obj1")
        registry.register("obj2")

        registry.clear()

        assert registry.get("obj_0") is None
        assert registry.get("obj_1") is None


class TestTestDataSerialization:
    """Tests for TestData custom type serialization."""

    def test_testdata_equality(self) -> None:
        """TestData equality must compare values."""
        a = MockTestData("hello")
        b = MockTestData("hello")
        c = MockTestData("world")

        assert a == b
        assert a != c

    def test_testdata_serialization_roundtrip(self) -> None:
        """TestData must survive serialization roundtrip."""
        adapter = MockHostAdapter()
        registry = SerializerRegistry.get_instance()
        registry.clear()

        adapter.register_serializers(registry)

        original = MockTestData("test_value")
        serializer = registry.get_serializer("MockTestData")
        deserializer = registry.get_deserializer("MockTestData")
        assert serializer is not None
        assert deserializer is not None

        serialized = serializer(original)
        deserialized = deserializer(serialized)

        assert deserialized == original
