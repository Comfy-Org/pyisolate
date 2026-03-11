import inspect

import pytest

from pyisolate._internal.serialization_registry import SerializerRegistry
from pyisolate.interfaces import SerializerRegistryProtocol


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    SerializerRegistry.get_instance().clear()


def test_singleton_identity() -> None:
    r1 = SerializerRegistry.get_instance()
    r2 = SerializerRegistry.get_instance()
    assert r1 is r2


def test_register_and_lookup() -> None:
    registry = SerializerRegistry.get_instance()
    registry.register("Foo", lambda x: {"v": x}, lambda x: x["v"])

    assert registry.has_handler("Foo")
    serializer = registry.get_serializer("Foo")
    deserializer = registry.get_deserializer("Foo")

    payload = serializer(123) if serializer else None
    assert payload == {"v": 123}
    assert deserializer(payload) == 123 if deserializer else False


def test_clear_resets_handlers() -> None:
    registry = SerializerRegistry.get_instance()
    registry.register("Bar", lambda x: x)
    assert registry.has_handler("Bar")

    registry.clear()
    assert not registry.has_handler("Bar")


class TestDataTypeFlag:
    def test_register_with_data_type_flag(self) -> None:
        registry = SerializerRegistry.get_instance()
        registry.register("MyData", lambda x: x, data_type=True)
        assert registry.is_data_type("MyData")

    def test_register_without_data_type_flag(self) -> None:
        registry = SerializerRegistry.get_instance()
        registry.register("MyData", lambda x: x)
        assert not registry.is_data_type("MyData")

    def test_is_data_type_unregistered(self) -> None:
        registry = SerializerRegistry.get_instance()
        assert not registry.is_data_type("NeverRegistered")

    def test_clear_resets_data_types(self) -> None:
        registry = SerializerRegistry.get_instance()
        registry.register("MyData", lambda x: x, data_type=True)
        assert registry.is_data_type("MyData")
        registry.clear()
        assert not registry.is_data_type("MyData")

    def test_overwrite_preserves_data_type(self) -> None:
        # Register with data_type=True, then re-register without — set is additive
        registry = SerializerRegistry.get_instance()
        registry.register("MyData", lambda x: x, data_type=True)
        registry.register("MyData", lambda x: x)  # no data_type kwarg
        assert registry.is_data_type("MyData")

    def test_overwrite_adds_data_type(self) -> None:
        # Register without, then re-register with data_type=True
        registry = SerializerRegistry.get_instance()
        registry.register("MyData", lambda x: x)
        assert not registry.is_data_type("MyData")
        registry.register("MyData", lambda x: x, data_type=True)
        assert registry.is_data_type("MyData")

    def test_data_type_idempotent(self) -> None:
        # Repeated register with data_type=True has no side effects
        registry = SerializerRegistry.get_instance()
        for _ in range(3):
            registry.register("MyData", lambda x: x, data_type=True)
        assert registry.is_data_type("MyData")
        registry.clear()
        assert not registry.is_data_type("MyData")

    def test_data_type_cross_type_isolation(self) -> None:
        # Setting type A as data_type does not affect type B
        registry = SerializerRegistry.get_instance()
        registry.register("TypeA", lambda x: x, data_type=True)
        registry.register("TypeB", lambda x: x)
        assert registry.is_data_type("TypeA")
        assert not registry.is_data_type("TypeB")


class TestProtocolCompliance:
    def test_protocol_isinstance_check(self) -> None:
        # SerializerRegistryProtocol is @runtime_checkable
        registry = SerializerRegistry.get_instance()
        assert isinstance(registry, SerializerRegistryProtocol)

    def test_protocol_is_data_type_callable_with_correct_signature(self) -> None:
        # Verify is_data_type exists on the protocol and has the expected signature
        sig = inspect.signature(SerializerRegistryProtocol.is_data_type)
        params = list(sig.parameters)
        assert "self" in params
        assert "type_name" in params
        # Invoke via protocol-typed reference to confirm structural contract
        registry: SerializerRegistryProtocol = SerializerRegistry.get_instance()
        registry.register("SigTest", lambda x: x, data_type=True)
        result = registry.is_data_type("SigTest")
        assert result is True

    def test_protocol_register_accepts_data_type_kwarg(self) -> None:
        registry: SerializerRegistryProtocol = SerializerRegistry.get_instance()
        # Must not raise — data_type is a keyword-only arg on the protocol
        registry.register("ProtoTest", lambda x: x, data_type=True)
        assert registry.is_data_type("ProtoTest")
