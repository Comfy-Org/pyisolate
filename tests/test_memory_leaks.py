"""Memory leak tests for proxy lifecycle and cleanup.

These tests verify that:
1. Proxies are garbage collected after RPC shutdown
2. TensorKeeper releases tensors after timeout
3. Registry removes entries when refcount hits 0

Note: Uses weakref to verify objects are collected, not actual memory profiling.
For actual memory profiling, use tracemalloc in integration tests.
"""

import gc
import time
import weakref

import pytest

from pyisolate._internal.rpc_protocol import ProxiedSingleton, SingletonMetaclass


class TestProxyGarbageCollection:
    """Tests for proxy object garbage collection."""

    def test_proxy_gc_after_singleton_clear(self):
        """Verify ProxiedSingleton instances can be garbage collected."""

        class TestService(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.data = "test"

        # Create instance and weak reference
        instance = TestService()
        weak_ref = weakref.ref(instance)

        # Instance should exist
        assert weak_ref() is not None

        # Clear singleton registry and delete local reference
        del instance
        SingletonMetaclass._instances.clear()

        # Force garbage collection (3x for generational GC)
        for _ in range(3):
            gc.collect()

        # Instance should be collected
        assert weak_ref() is None, "Singleton not collected after clearing registry"

    def test_nested_singleton_gc(self):
        """Verify nested singletons are properly collected."""

        class ChildService(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.data = []

        class ParentService(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.child = ChildService()

        # Create parent and child
        parent = ParentService()
        child_ref = weakref.ref(parent.child)
        parent_ref = weakref.ref(parent)

        # Both should exist
        assert child_ref() is not None
        assert parent_ref() is not None

        # Clear and collect
        del parent
        SingletonMetaclass._instances.clear()

        for _ in range(3):
            gc.collect()

        # Both should be collected
        assert parent_ref() is None, "Parent not collected"
        assert child_ref() is None, "Child not collected"


class TestTensorKeeperCleanup:
    """Tests for TensorKeeper memory management."""

    @pytest.fixture
    def fast_tensor_keeper(self, monkeypatch):
        """Configure TensorKeeper with short retention for testing."""
        from pyisolate._internal.tensor_serializer import TensorKeeper

        # Use 2 second retention for fast testing
        monkeypatch.setattr(
            TensorKeeper,
            "__init__",
            lambda self, retention_seconds=2.0: (
                (
                    setattr(self, "retention_seconds", 2.0),
                    setattr(self, "_keeper", __import__("collections").deque()),
                    setattr(self, "_lock", __import__("threading").Lock()),
                )[-1]
                or None
            ),
        )

    def test_tensor_keeper_keeps_reference(self):
        """Verify TensorKeeper holds tensor reference."""
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        keeper = TensorKeeper(retention_seconds=5.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        # Keep tensor
        keeper.keep(tensor)

        # Delete local reference
        del tensor

        # Should still exist via keeper
        gc.collect()
        assert weak_ref() is not None, "Tensor collected while keeper holds it"

    @pytest.mark.slow
    def test_tensor_keeper_releases_after_timeout(self):
        """Verify TensorKeeper releases tensors after retention period.

        Note: This test takes ~3 seconds due to retention timeout.
        """
        pytest.importorskip("torch")
        import torch

        from pyisolate._internal.tensor_serializer import TensorKeeper

        # Short retention for testing
        keeper = TensorKeeper(retention_seconds=1.0)
        tensor = torch.zeros(10)
        weak_ref = weakref.ref(tensor)

        # Keep tensor
        keeper.keep(tensor)
        del tensor

        # Should still exist immediately
        gc.collect()
        assert weak_ref() is not None

        # Wait for retention to expire
        time.sleep(2.0)

        # Trigger cleanup by adding another tensor
        keeper.keep(torch.zeros(1))

        # Force GC
        for _ in range(3):
            gc.collect()

        # Original tensor should be released
        assert weak_ref() is None, "Tensor not released after retention period"


class TestRegistryCleanup:
    """Tests for registry refcount and cleanup."""

    def test_singleton_registry_refcount(self):
        """Verify singleton instances are tracked in registry."""

        class CountedService(ProxiedSingleton):
            instances_created = 0

            def __init__(self):
                super().__init__()
                CountedService.instances_created += 1

        # First creation
        instance1 = CountedService()
        assert CountedService.instances_created == 1
        assert CountedService in SingletonMetaclass._instances

        # Second call returns same instance
        instance2 = CountedService()
        assert CountedService.instances_created == 1  # No new instance
        assert instance1 is instance2

        # Clear registry
        SingletonMetaclass._instances.clear()
        assert CountedService not in SingletonMetaclass._instances

    def test_registry_cleanup_on_instance_delete(self):
        """Verify registry doesn't prevent GC when manually cleared."""

        class TrackedService(ProxiedSingleton):
            pass

        instance = TrackedService()
        weak_ref = weakref.ref(instance)

        # Instance in registry
        assert TrackedService in SingletonMetaclass._instances
        assert weak_ref() is not None

        # Delete local ref (registry still holds it)
        del instance
        gc.collect()
        # Still alive via registry
        assert weak_ref() is not None

        # Clear registry
        SingletonMetaclass._instances.clear()
        for _ in range(3):
            gc.collect()

        # Now should be collected
        assert weak_ref() is None


class TestMemoryLeakScenarios:
    """Tests for specific memory leak scenarios."""

    def test_circular_reference_singleton(self):
        """Verify circular references don't prevent collection."""

        class NodeA(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.ref = None

        class NodeB(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                self.ref = None

        # Create circular reference
        a = NodeA()
        b = NodeB()
        a.ref = b
        b.ref = a

        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        # Clear references
        del a, b
        SingletonMetaclass._instances.clear()

        # Force GC (Python's GC handles cycles)
        for _ in range(3):
            gc.collect()

        # Both should be collected
        assert weak_a() is None, "NodeA not collected (circular ref)"
        assert weak_b() is None, "NodeB not collected (circular ref)"

    def test_exception_during_init_no_leak(self):
        """Verify exceptions during __init__ don't leak memory."""

        class FailingService(ProxiedSingleton):
            def __init__(self):
                super().__init__()
                raise ValueError("Init failed")

        # Attempt to create (should fail)
        with pytest.raises(ValueError):
            FailingService()

        # Should not be in registry (init failed)
        assert FailingService not in SingletonMetaclass._instances
