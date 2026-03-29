"""
Generic serialization helpers for PyIsolate.

These helpers let PyIsolate transparently move tensors and adapter-registered
objects across process boundaries. CUDA tensors stay on-device when CUDA IPC is
enabled; otherwise they fall back to CPU shared memory for transport.

Adapter-specific types are handled via the
SerializerRegistry, which allows adapters to register custom serializers without
coupling pyisolate to any specific framework.
"""

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from .serialization_registry import SerializerRegistry
from .torch_gate import get_torch_optional

_cuda_ipc_enabled = sys.platform == "linux" and os.environ.get("PYISOLATE_ENABLE_CUDA_IPC") == "1"

if TYPE_CHECKING:  # pragma: no cover - typing aids
    pass  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _serialize_for_isolation_impl(
    data: Any,
    *,
    registry: SerializerRegistry,
    torch_module: Any,
    remote_handle_type: type[Any],
) -> Any:
    type_name = type(data).__name__

    handle = getattr(data, "_pyisolate_remote_handle", None)
    if isinstance(handle, remote_handle_type):
        return handle

    serializer = registry.get_serializer(type_name)
    if serializer is not None:
        return serializer(data)

    if torch_module is not None and isinstance(data, torch_module.Tensor):
        if data.is_cuda:
            if _cuda_ipc_enabled:
                return data
            return data.cpu()
        return data

    if isinstance(data, dict):
        return {
            k: _serialize_for_isolation_impl(
                v,
                registry=registry,
                torch_module=torch_module,
                remote_handle_type=remote_handle_type,
            )
            for k, v in data.items()
        }

    if isinstance(data, (list, tuple)):
        result = [
            _serialize_for_isolation_impl(
                item,
                registry=registry,
                torch_module=torch_module,
                remote_handle_type=remote_handle_type,
            )
            for item in data
        ]
        return type(data)(result)

    return data


def serialize_for_isolation(data: Any) -> Any:
    """Serialize data for transmission to an isolated process (host side).

    Adapter-registered objects are converted to reference dictionaries so the
    isolated process can fetch them lazily. RemoteObjectHandle instances are passed
    through to preserve identity without pickling heavyweight objects.
    """
    registry = SerializerRegistry.get_instance()
    from .remote_handle import RemoteObjectHandle

    torch, _ = get_torch_optional()
    return _serialize_for_isolation_impl(
        data,
        registry=registry,
        torch_module=torch,
        remote_handle_type=RemoteObjectHandle,
    )


async def deserialize_from_isolation(data: Any, extension: Any = None, _nested: bool = False) -> Any:
    """Deserialize data received from an isolated process (host side).

    Top-level ``RemoteObjectHandle`` values are resolved to concrete objects when an
    extension proxy is available. Nested handles stay opaque so they can be returned
    back to the child without forcing unnecessary pickling/unpickling.
    """
    from .remote_handle import RemoteObjectHandle

    type_name = type(data).__name__

    registry = SerializerRegistry.get_instance()

    if isinstance(data, RemoteObjectHandle):
        # Handles with a registered handler are returned opaque for the caller
        # to process.  Handles with NO registered handler are pack-local proxy
        # handles — keep them opaque so they round-trip back to the originating
        # child without a wasteful (and doomed) RPC resolution attempt.
        return data

    # Check for adapter-registered deserializers by type name.
    # Only apply to dicts (serialized form). Objects already deserialized by the
    # JSON transport layer (e.g., PLY reconstructed via _json_object_hook) are
    # passed through as-is.
    if isinstance(data, dict) and registry.has_handler(type_name):
        deserializer = registry.get_deserializer(type_name)
        if deserializer:
            result = deserializer(data)
            if hasattr(result, "__await__"):
                return await result
            return result

    if isinstance(data, dict):
        ref_type = data.get("__type__")

        # Adapter-registered deserializers for reference dicts
        if ref_type and registry.has_handler(ref_type):
            deserializer = registry.get_deserializer(ref_type)
            if deserializer:
                return deserializer(data)

        deserialized: dict[str, Any] = {}
        for k, v in data.items():
            # Dict entries are considered nested to preserve handles inside
            # structured payloads (e.g., da_model['model']).
            deserialized[k] = await deserialize_from_isolation(v, extension, _nested=True)
        return deserialized

    if isinstance(data, (list, tuple)):
        # For list/tuple, propagate the current nesting flag. Top-level tuples
        # (e.g., node outputs) stay `_nested=False`, allowing handles to resolve
        # to concrete objects when appropriate. Deeper levels inherit `_nested`
        # to avoid over-resolving nested handles.
        result = [await deserialize_from_isolation(item, extension, _nested=_nested) for item in data]
        return type(data)(result)

    return data


def deserialize_proxy_result(data: Any) -> Any:
    """Deserialize RPC results in the isolated process (child side).

    Reference dictionaries emitted by the host are converted into the appropriate
    proxy instances via adapter-registered deserializers while preserving
    container structure.
    """
    if isinstance(data, dict):
        ref_type = data.get("__type__")

        # Adapter-registered deserializers for proxy-bound references
        registry = SerializerRegistry.get_instance()
        if ref_type and registry.has_handler(ref_type):
            deserializer = registry.get_deserializer(ref_type)
            if deserializer:
                return deserializer(data)

        return {k: deserialize_proxy_result(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [deserialize_proxy_result(item) for item in data]
        return type(data)(result)

    return data
