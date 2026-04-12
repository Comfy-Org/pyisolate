"""Tests for RemoteObjectHandle proxy pattern.

These tests verify the RemoteObjectHandle behavior without RPC.
"""

from pyisolate._internal.remote_handle import RemoteObjectHandle


class TestRemoteObjectHandleContract:
    """Tests for RemoteObjectHandle proxy pattern."""

    def test_remote_handle_stores_id(self) -> None:
        """RemoteObjectHandle stores object ID."""
        handle = RemoteObjectHandle("model_123", "ModelType")

        assert handle.object_id == "model_123"

    def test_remote_handle_stores_type_name(self) -> None:
        """RemoteObjectHandle stores type name."""
        handle = RemoteObjectHandle("model_123", "ModelType")

        assert handle.type_name == "ModelType"

    def test_remote_handle_repr(self) -> None:
        """RemoteObjectHandle has informative repr."""
        handle = RemoteObjectHandle("my_object", "MyClass")

        repr_str = repr(handle)
        assert "RemoteObject" in repr_str
        assert "my_object" in repr_str
        assert "MyClass" in repr_str
