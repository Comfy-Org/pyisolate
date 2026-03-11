"""Tests for model_serialization.py — dict guard on deserialize_from_isolation.

The isinstance(data, dict) guard at line 104 prevents adapter-registered
deserializers from being applied to non-dict objects (e.g., already-
materialized PLY/File3D instances reconstructed by _json_object_hook).
Without the guard, passing a materialized object to its own deserializer
(which expects a dict) causes a double-deserialization bug.
"""

import pytest

from pyisolate._internal.model_serialization import deserialize_from_isolation
from pyisolate._internal.serialization_registry import SerializerRegistry


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    SerializerRegistry.get_instance().clear()


class TestDictGuard:
    async def test_dict_with_registered_handler_calls_deserializer(self) -> None:
        # dict + matching handler + deserializer → deserializer IS invoked
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("dict", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({})
        assert result is sentinel

    async def test_non_dict_object_skips_deserializer(self) -> None:
        # Non-dict + matching handler → guard blocks deserializer, object passes through
        class Foo:
            pass

        called = False

        def bad_deserializer(x: object) -> object:
            nonlocal called
            called = True
            return x

        registry = SerializerRegistry.get_instance()
        registry.register("Foo", lambda x: x, bad_deserializer)

        foo = Foo()
        result = await deserialize_from_isolation(foo)
        assert result is foo
        assert not called

    async def test_already_materialized_passthrough(self) -> None:
        # Core bug scenario: a PLY-like object already reconstructed by
        # _json_object_hook has a registered handler. The dict guard must
        # prevent re-deserialization — the object passes through unchanged.
        class PLY:
            def __init__(self, raw_data: bytes) -> None:
                self.raw_data = raw_data

        def ply_deserializer(d: object) -> PLY:
            # If called on a PLY instance (not a dict), this would raise or corrupt
            raise AssertionError("deserializer must not be called on already-materialized PLY")

        registry = SerializerRegistry.get_instance()
        registry.register("PLY", lambda x: x, ply_deserializer)

        materialized = PLY(raw_data=b"\x70\x6c\x79")  # already reconstructed
        result = await deserialize_from_isolation(materialized)
        assert result is materialized
        assert result.raw_data == b"\x70\x6c\x79"


class TestRefTypeDeserialization:
    async def test_dict_ref_type_uses_registered_deserializer(self) -> None:
        # {"__type__": "MyRef"} with registered handler → deserializer called
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({"__type__": "MyRef", "id": "abc"})
        assert result is sentinel

    async def test_dict_ref_type_unknown_returns_dict(self) -> None:
        # Unknown __type__ with no handler → dict returned as-is (recursively deserialized)
        result = await deserialize_from_isolation({"__type__": "Unknown", "val": 42})
        assert isinstance(result, dict)
        assert result["val"] == 42

    async def test_nested_dict_ref_deserialization(self) -> None:
        # {"a": {"__type__": "MyRef"}} — inner ref must be recursively deserialized
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation({"a": {"__type__": "MyRef", "id": "xyz"}})
        assert isinstance(result, dict)
        assert result["a"] is sentinel


class TestContainerPassthrough:
    async def test_list_items_deserialized(self) -> None:
        registry = SerializerRegistry.get_instance()
        sentinel = object()
        registry.register("MyRef", lambda x: x, lambda x: sentinel)
        result = await deserialize_from_isolation([{"__type__": "MyRef"}, 1, "str"])
        assert result[0] is sentinel
        assert result[1] == 1
        assert result[2] == "str"

    async def test_tuple_preserved_as_tuple(self) -> None:
        result = await deserialize_from_isolation((1, 2, 3))
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    async def test_string_passthrough(self) -> None:
        result = await deserialize_from_isolation("hello")
        assert result == "hello"

    async def test_int_passthrough(self) -> None:
        result = await deserialize_from_isolation(42)
        assert result == 42

    async def test_none_passthrough(self) -> None:
        result = await deserialize_from_isolation(None)
        assert result is None
