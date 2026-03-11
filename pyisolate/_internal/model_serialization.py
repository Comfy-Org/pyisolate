"""
Generic serialization helpers for PyIsolate.

These helpers let PyIsolate transparently move tensors and adapter-registered
objects across process boundaries. CUDA tensors stay on-device when CUDA IPC is
enabled; otherwise they fall back to CPU shared memory for transport.

Adapter-specific types are handled via the
SerializerRegistry, which allows adapters to register custom serializers without
coupling pyisolate to any specific framework.
"""

import contextlib
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


def serialize_for_isolation(data: Any) -> Any:
    """Serialize data for transmission to an isolated process (host side).

    Adapter-registered objects are converted to reference dictionaries so the
    isolated process can fetch them lazily. RemoteObjectHandle instances are passed
    through to preserve identity without pickling heavyweight objects.
    """
    type_name = type(data).__name__

    # Adapter-registered serializers take precedence over built-in handlers
    registry = SerializerRegistry.get_instance()
    if registry.has_handler(type_name):
        serializer = registry.get_serializer(type_name)
        if serializer:
            return serializer(data)

    # If this object originated as a RemoteObjectHandle, send the original
    # handle only when no adapter serializer is available for this type.
    # This avoids cross-extension stale handle reuse for serializer-backed
    # objects (e.g. CLIP/ModelPatcher/VAE refs).
    from .remote_handle import RemoteObjectHandle

    handle = getattr(data, "_pyisolate_remote_handle", None)
    if isinstance(handle, RemoteObjectHandle):
        return handle

    torch, _ = get_torch_optional()
    if torch is not None and isinstance(data, torch.Tensor):
        if data.is_cuda:
            if _cuda_ipc_enabled:
                return data
            return data.cpu()
        return data

    if isinstance(data, dict):
        return {k: serialize_for_isolation(v) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        result = [serialize_for_isolation(item) for item in data]
        return type(data)(result)

    return data


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
        if _nested or extension is None:
            return data
        if registry.has_handler(data.type_name):
            return data
        try:
            resolved = await extension.get_remote_object(data.object_id)
            with contextlib.suppress(Exception):
                resolved._pyisolate_remote_handle = data
            return resolved
        except Exception:
            return data

    # Check for adapter-registered deserializers by type name (e.g., NodeOutput).
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
