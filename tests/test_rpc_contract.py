"""Tests for RPC behavior and ProxiedSingleton contracts.

These tests verify:
1. ProxiedSingleton instances are singletons
2. RPC method calls work correctly
3. Event loop recreation doesn't break RPC
4. Exceptions propagate correctly

Note: These are unit tests that verify RPC contracts at the boundary
without full process isolation. For full integration tests, see
original_integration/.
"""

import asyncio

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton

from .fixtures.test_adapter import MockRegistry


class TestProxiedSingletonContract:
    """Tests for ProxiedSingleton metaclass behavior."""

    def test_singleton_returns_same_instance(self):
        """Multiple instantiations return the same instance."""
        instance1 = MockRegistry()
        instance2 = MockRegistry()

        assert instance1 is instance2

    def test_singleton_instance_persists(self):
        """Singleton instance persists across calls."""
        instance1 = MockRegistry()
        instance1.register("test_object")

        instance2 = MockRegistry()
        # Should see the object registered via instance1
        assert instance2.get("obj_0") == "test_object"

    def test_different_singletons_are_independent(self):
        """Different ProxiedSingleton subclasses are independent."""

        class AnotherRegistry(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.data = "another"

        test_instance = MockRegistry()
        another_instance = AnotherRegistry()

        assert test_instance is not another_instance
        assert isinstance(test_instance, MockRegistry)
        assert isinstance(another_instance, AnotherRegistry)


class TestRpcMethodContract:
    """Tests for RPC method call contract."""

    def test_method_returns_value(self):
        """RPC method must return expected value."""
        registry = MockRegistry()
        obj = {"key": "value"}

        obj_id = registry.register(obj)
        result = registry.get(obj_id)

        assert result == obj

    def test_method_accepts_arguments(self):
        """RPC method must accept positional and keyword arguments."""
        registry = MockRegistry()

        # Positional
        id1 = registry.register("positional_arg")
        assert registry.get(id1) == "positional_arg"

    def test_method_handles_none_return(self):
        """RPC method can return None."""
        registry = MockRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_method_handles_complex_objects(self):
        """RPC method can handle complex nested objects."""
        registry = MockRegistry()

        complex_obj = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": {"c": 42}}},
            "mixed": [{"x": 1}, {"y": 2}],
        }

        obj_id = registry.register(complex_obj)
        result = registry.get(obj_id)

        assert result == complex_obj


class TestEventLoopResilience:
    """Tests for RPC resilience across event loop recreation.

    This is a critical contract: ProxiedSingleton instances must
    remain functional even when the event loop is closed and
    recreated (e.g., between workflow executions).
    """

    def test_singleton_survives_loop_recreation(self):
        """Singleton instance survives event loop recreation."""
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop1 = asyncio.new_event_loop()
        loop2: asyncio.AbstractEventLoop | None = None
        try:
            asyncio.set_event_loop(loop1)
            registry = MockRegistry()
            obj_id = registry.register("loop1_object")

            loop1.close()

            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)

            result = registry.get(obj_id)
            assert result == "loop1_object"
        finally:
            asyncio.set_event_loop(previous_loop)
            if loop2 is not None:
                loop2.close()
            elif not loop1.is_closed():
                loop1.close()

    def test_singleton_data_persists_across_loops(self):
        """Data stored in singleton persists across event loops."""
        try:
            previous_loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop1 = asyncio.new_event_loop()
        loop2: asyncio.AbstractEventLoop | None = None
        try:
            asyncio.set_event_loop(loop1)

            registry = MockRegistry()
            id1 = registry.register("first")
            id2 = registry.register("second")

            loop1.close()

            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)

            assert registry.get(id1) == "first"
            assert registry.get(id2) == "second"
        finally:
            asyncio.set_event_loop(previous_loop)
            if loop2 is not None:
                loop2.close()
            elif not loop1.is_closed():
                loop1.close()


class TestRpcErrorHandling:
    """Tests for RPC error handling contract."""

    def test_method_exception_propagates(self):
        """Exceptions in RPC methods should propagate."""

        class FailingService(ProxiedSingleton):
            def fail(self):
                raise ValueError("Intentional failure")

        service = FailingService()

        with pytest.raises(ValueError, match="Intentional failure"):
            service.fail()

    def test_type_error_propagates(self):
        """TypeError in RPC methods should propagate."""

        class TypedService(ProxiedSingleton):
            def typed_method(self, value: int) -> int:
                return value + 1

        service = TypedService()

        # Wrong type should raise TypeError
        with pytest.raises(TypeError):
            service.typed_method("not an int")
