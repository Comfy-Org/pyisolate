"""Test that SealedNodeExtension registers ndarray as data_type serializer."""

from __future__ import annotations

import numpy as np
import pytest

from pyisolate._internal.serialization_registry import SerializerRegistry
from pyisolate._internal.singleton_context import singleton_scope
from pyisolate.sealed import SealedNodeExtension


class TestSealedNdarrayTransport:
    def test_ndarray_registered_as_data_type(self) -> None:
        """After SealedNodeExtension init, ndarray is a registered data_type."""
        with singleton_scope():
            SealedNodeExtension()
            registry = SerializerRegistry.get_instance()
            assert registry.has_handler("ndarray")
            assert registry.is_data_type("ndarray")

    def test_ndarray_serializes_as_tensor_value(self) -> None:
        """ndarray serializer produces TensorValue dict, not RemoteObjectHandle."""
        with singleton_scope():
            SealedNodeExtension()
            registry = SerializerRegistry.get_instance()
            serializer = registry.get_serializer("ndarray")
            assert serializer is not None

            arr = np.random.rand(1, 64, 64, 3).astype(np.float32)
            result = serializer(arr)

            assert isinstance(result, dict)
            assert result["__type__"] == "TensorValue"
            assert result["dtype"] == "torch.float32"
            assert result["tensor_size"] == [1, 64, 64, 3]
            assert result["requires_grad"] is False
            assert isinstance(result["data"], list)

    def test_wrap_for_transport_passes_ndarray_inline(self) -> None:
        """_wrap_for_transport does NOT wrap ndarray as RemoteObjectHandle."""
        with singleton_scope():
            ext = SealedNodeExtension()
            arr = np.random.rand(1, 64, 64, 3).astype(np.float32)
            wrapped = ext._wrap_for_transport(arr)

            # Should NOT be RemoteObjectHandle
            from pyisolate._internal.remote_handle import RemoteObjectHandle

            assert not isinstance(wrapped, RemoteObjectHandle)
            # Should still be ndarray (serializer runs at JSON encode time, not at wrap time)
            assert isinstance(wrapped, np.ndarray)

    def test_unsupported_ndarray_dtype_raises(self) -> None:
        """Unsupported ndarray dtypes fail fast instead of silently downcasting."""
        with singleton_scope():
            SealedNodeExtension()
            registry = SerializerRegistry.get_instance()
            serializer = registry.get_serializer("ndarray")
            assert serializer is not None

            arr = np.array(["x", "y"], dtype=np.str_)
            with pytest.raises(TypeError, match="Unsupported ndarray dtype"):
                serializer(arr)
