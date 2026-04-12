"""Tests for SealedNodeExtension proxy handle mechanism (issue #58 Slice 2).

Proves that sealed workers can wrap unregistered objects as
RemoteObjectHandle, store them in a child-local registry, and resolve
incoming handles back to the original objects by identity.
"""

import numpy as np
import pytest

from pyisolate._internal.remote_handle import RemoteObjectHandle
from pyisolate._internal.serialization_registry import SerializerRegistry
from pyisolate.sealed import SealedNodeExtension


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    SerializerRegistry.get_instance().clear()


class _FakeWidget:
    """Unregistered type used for proxy handle tests."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_sealed_wraps_unregistered_object_as_handle() -> None:
    ext = SealedNodeExtension()
    widget = _FakeWidget(42)

    result = ext._wrap_for_transport(widget)

    assert isinstance(result, RemoteObjectHandle)
    assert result.type_name == "_FakeWidget"
    assert len(ext.remote_objects) == 1
    assert ext.remote_objects[result.object_id] is widget


def test_sealed_resolves_handle_to_original_object() -> None:
    ext = SealedNodeExtension()
    original = _FakeWidget(99)
    ext.remote_objects["id-1"] = original

    handle = RemoteObjectHandle("id-1", "Foo")
    result = ext._resolve_handles(handle)

    assert result is original


def test_sealed_stale_handle_raises_keyerror() -> None:
    ext = SealedNodeExtension()

    handle = RemoteObjectHandle("nonexistent", "Foo")
    with pytest.raises(KeyError, match="nonexistent"):
        ext._resolve_handles(handle)


def test_sealed_ndarray_roundtrip_via_handle() -> None:
    ext = SealedNodeExtension()
    original = np.ones((100, 100), dtype=np.float32)

    # ndarray is a registered data_type serializer, so it stays inline.
    wrapped = ext._wrap_for_transport(original)
    assert isinstance(wrapped, np.ndarray)

    # Inline data passes through unchanged.
    resolved = ext._resolve_handles(wrapped)
    assert resolved is original
    assert np.array_equal(original, resolved)
