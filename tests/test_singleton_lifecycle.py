"""Singleton lifecycle tests.

These tests explicitly verify singleton injection/cleanup lifecycle behavior,
particularly the singleton_scope context manager and use_remote() injection.
"""

from typing import Any, cast

import pytest

from pyisolate._internal.rpc_protocol import (
    ProxiedSingleton,
    SingletonMetaclass,
)
from pyisolate._internal.singleton_context import singleton_scope


class TestSingletonScopeIsolation:
    """Tests for singleton_scope context manager isolation.

    singleton_scope behavior:
    1. Saves current state at entry
    2. Does NOT clear state at entry (state persists into scope)
    3. On exit: clears current state and restores saved state

    This is designed for test isolation: any modifications during the scope
    are undone when the scope exits.
    """

    def test_scope_restores_state_on_exit(self) -> None:
        """Verify singleton_scope restores previous state on exit."""

        class RestoreService(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                self.value = "original"

        # Create instance before scope
        before = RestoreService()
        before.value = "modified"

        with singleton_scope():
            # Inside scope, same instance exists (state persists into scope)
            inside = RestoreService()
            assert inside is before
            assert inside.value == "modified"

            # Delete and recreate to test restoration
            del SingletonMetaclass._instances[RestoreService]
            new_instance = RestoreService()
            new_instance.value = "new"

        # After scope exits, original state is restored
        after = RestoreService()
        assert after is before
        assert after.value == "modified"

    def test_scope_removes_new_singletons_on_exit(self) -> None:
        """Verify singletons created in scope are removed on exit."""

        class NewService(ProxiedSingleton):
            pass

        # Ensure NewService doesn't exist before scope
        assert NewService not in SingletonMetaclass._instances

        with singleton_scope():
            # Create new singleton in scope
            _new_instance = NewService()
            assert NewService in SingletonMetaclass._instances

        # After scope, NewService is removed (restored to pre-scope state)
        assert NewService not in SingletonMetaclass._instances

    def test_nested_scopes_restore_registry_correctly(self) -> None:
        """Verify nested singleton_scope contexts restore registry correctly.

        Note: singleton_scope restores REGISTRY state (which instances exist),
        not the internal state of instances themselves. Instance mutations
        persist across scope boundaries.
        """

        class OuterService(ProxiedSingleton):
            pass

        class InnerService(ProxiedSingleton):
            pass

        class DeepService(ProxiedSingleton):
            pass

        # Outer scope (from conftest)
        _outer = OuterService()

        with singleton_scope():
            # First nested scope - add InnerService
            _inner = InnerService()
            assert InnerService in SingletonMetaclass._instances

            with singleton_scope():
                # Second nested scope - add DeepService
                _deep = DeepService()
                assert DeepService in SingletonMetaclass._instances

            # After inner scope, DeepService removed
            assert DeepService not in SingletonMetaclass._instances
            # InnerService still exists
            assert InnerService in SingletonMetaclass._instances

        # After outer scope, both InnerService and DeepService removed
        assert InnerService not in SingletonMetaclass._instances
        assert DeepService not in SingletonMetaclass._instances
        # OuterService still exists
        assert OuterService in SingletonMetaclass._instances

    def test_scope_cleanup_on_exception(self) -> None:
        """Verify singleton_scope cleans up registry even on exception."""

        class BeforeService(ProxiedSingleton):
            pass

        class InsideService(ProxiedSingleton):
            pass

        # Create before nested scope
        before = BeforeService()
        assert BeforeService in SingletonMetaclass._instances

        try:
            with singleton_scope():
                # Create new service in scope
                _inside = InsideService()
                assert InsideService in SingletonMetaclass._instances
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # Registry should be restored after exception
        # BeforeService should exist
        assert BeforeService in SingletonMetaclass._instances
        after = BeforeService()
        assert after is before

        # InsideService should be removed
        assert InsideService not in SingletonMetaclass._instances

    def test_scope_isolates_new_registrations(self) -> None:
        """Verify new singletons created in scope don't leak out."""

        class ScopedServiceA(ProxiedSingleton):
            pass

        class ScopedServiceB(ProxiedSingleton):
            pass

        # Create A outside scope
        a_outside = ScopedServiceA()

        with singleton_scope():
            # A exists in scope (persisted)
            assert ScopedServiceA in SingletonMetaclass._instances

            # Create B only in scope
            _b_inside = ScopedServiceB()
            assert ScopedServiceB in SingletonMetaclass._instances

        # After scope: A restored, B removed
        assert ScopedServiceA in SingletonMetaclass._instances
        a_restored = ScopedServiceA()
        assert a_restored is a_outside

        # B was only in scope, now it's gone
        assert ScopedServiceB not in SingletonMetaclass._instances


class TestUseRemoteInjection:
    """Tests for use_remote() proxy injection."""

    def test_use_remote_injects_proxy(self) -> Any:
        """Verify use_remote() injects caller as singleton instance."""

        class RemoteService(ProxiedSingleton):
            async def remote_method(self) -> Any:
                return "remote"

        class FakeRPC:
            def __init__(self) -> None:
                self.callers: list[Any] = []

            def create_caller(self, cls: Any, object_id: Any) -> Any:
                caller = type("FakeCaller", (), {"cls": cls, "object_id": object_id})()
                self.callers.append(caller)
                return caller

        rpc = FakeRPC()
        RemoteService.use_remote(cast(Any, rpc))

        # Instance should be the injected proxy
        instance = RemoteService()
        assert instance is SingletonMetaclass._instances[RemoteService]
        assert instance.cls is RemoteService
        assert instance.object_id == "RemoteService"

    def test_use_remote_requires_proxied_singleton(self) -> Any:
        """Verify use_remote() only works with ProxiedSingleton subclasses."""

        class NotProxied(metaclass=SingletonMetaclass):
            pass

        class FakeRPC:
            def create_caller(self, cls: Any, object_id: Any) -> Any:
                return object()

        rpc = FakeRPC()

        with pytest.raises(AssertionError, match="must inherit from ProxiedSingleton"):
            NotProxied.use_remote(cast(Any, rpc))


class TestNestedSingletonRegistration:
    """Tests for nested ProxiedSingleton registration."""

    def test_nested_singleton_attributes_get_proxies(self) -> Any:
        """Verify type-hinted singleton attributes receive caller proxies."""

        class ChildService(ProxiedSingleton):
            async def child_method(self) -> Any:
                return "child"

        class ParentService(ProxiedSingleton):
            child: ChildService  # Type-hinted attribute

            async def parent_method(self) -> Any:
                return "parent"

        class FakeRPC:
            def __init__(self) -> None:
                self.callers: dict[str, Any] = {}

            def create_caller(self, cls: Any, object_id: Any) -> Any:
                caller = type("FakeCaller", (), {"cls": cls, "object_id": object_id})()
                self.callers[object_id] = caller
                return caller

        rpc = FakeRPC()
        ParentService.use_remote(cast(Any, rpc))

        # Both parent and child should have callers
        assert "ParentService" in rpc.callers
        assert "ChildService" in rpc.callers

        # Parent's child attribute should be the child caller
        parent = ParentService()
        assert hasattr(parent, "child")
        assert parent.child is rpc.callers["ChildService"]

    def test_register_callee_for_nested_singletons(self) -> None:
        """Verify _register() recursively registers nested singletons."""

        class InnerService(ProxiedSingleton):
            pass

        class OuterService(ProxiedSingleton):
            inner = InnerService()

        registered: list[tuple[Any, Any]] = []

        class FakeRPC:
            def register_callee(self, obj: Any, object_id: Any) -> None:
                registered.append((obj, object_id))

        rpc = FakeRPC()
        outer = OuterService()
        outer._register(cast(Any, rpc))

        # Both outer and inner should be registered
        object_ids = [obj_id for _, obj_id in registered]
        assert "OuterService" in object_ids
        assert "InnerService" in object_ids


class TestSingletonEdgeCases:
    """Tests for edge cases in singleton lifecycle."""

    def test_inject_before_instantiation(self) -> None:
        """Verify inject_instance() must be called before instantiation."""

        class LateInjection(ProxiedSingleton):
            pass

        # First instantiation
        LateInjection()

        # Injection after should fail
        with pytest.raises(AssertionError, match="singleton already exists"):
            SingletonMetaclass.inject_instance(LateInjection, object())

    def test_get_instance_creates_if_missing(self) -> None:
        """Verify get_instance() creates instance if not exists."""

        class LazyService(ProxiedSingleton):
            def __init__(self) -> None:
                super().__init__()
                self.initialized = True

        # Should not exist yet
        assert LazyService not in SingletonMetaclass._instances

        # get_instance should create it
        instance = LazyService.get_instance()
        assert instance.initialized is True
        assert LazyService in SingletonMetaclass._instances

        # Should return same instance
        assert LazyService.get_instance() is instance

    def test_get_remote_id_uses_class_name(self) -> None:
        """Verify get_remote_id() returns class name by default."""

        class CustomNameService(ProxiedSingleton):
            pass

        assert CustomNameService.get_remote_id() == "CustomNameService"

    def test_custom_get_remote_id(self) -> Any:
        """Verify get_remote_id() can be overridden."""

        class CustomIdService(ProxiedSingleton):
            @classmethod
            def get_remote_id(cls) -> Any:
                return "custom_service_id"

        assert CustomIdService.get_remote_id() == "custom_service_id"
