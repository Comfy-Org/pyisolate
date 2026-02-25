"""
RPC Transport Layer.

This module contains:
- RPCTransport Protocol
- QueueTransport
- ConnectionTransport
- JSONSocketTransport
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import socket
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

# We only import this to get type hinting working. It can also be a torch.multiprocessing
if TYPE_CHECKING:
    import multiprocessing as typehint_mp
    from multiprocessing.connection import Connection
else:
    typehint_mp = None  # Resolved at runtime in methods if needed, or by user

logger = logging.getLogger(__name__)


@runtime_checkable
class RPCTransport(Protocol):
    """Protocol for RPC transport mechanisms.

    Implementations must provide thread-safe send/recv operations.
    """

    def send(self, obj: Any) -> None:
        """Send an object to the remote endpoint."""
        ...

    def recv(self) -> Any:
        """Receive an object from the remote endpoint. Blocks until available."""
        ...

    def close(self) -> None:
        """Close the transport. Further send/recv calls may fail."""
        ...


class QueueTransport:
    """Transport using multiprocessing.Queue pairs (standard IPC)."""

    def __init__(
        self,
        send_queue: typehint_mp.Queue[Any],  # type: ignore
        recv_queue: typehint_mp.Queue[Any],  # type: ignore
    ) -> None:
        self._send_queue = send_queue
        self._recv_queue = recv_queue

    def send(self, obj: Any) -> None:
        self._send_queue.put(obj)

    def recv(self) -> Any:
        return self._recv_queue.get()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._send_queue.close()
        with contextlib.suppress(Exception):
            self._recv_queue.close()


class ConnectionTransport:
    """Transport using multiprocessing.connection.Connection (Unix Domain Sockets).

    Used for bwrap sandbox isolation where Queue-based IPC is not available.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, obj: Any) -> None:
        with self._lock:
            self._conn.send(obj)

    def recv(self) -> Any:
        return self._conn.recv()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


class JSONSocketTransport:
    """Transport using raw sockets + JSON-RPC (pickle-safe).

    This transport uses JSON serialization instead of pickle to prevent
    RCE attacks via __reduce__ exploits from sandboxed child processes.

    Used for ALL Linux isolation modes (sandbox and non-sandbox).
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, obj: Any) -> None:
        """Serialize to JSON with length prefix."""
        import json
        import struct

        try:
            data = json.dumps(obj, default=self._json_default).encode("utf-8")
        except TypeError as e:
            type_name = type(obj).__name__
            logger.error(
                "Cannot serialize object:\n"
                "  Type: %s\n"
                "  Error: %s\n"
                "  Resolution: Register a custom serializer via SerializerRegistry",
                type_name,
                e,
            )
            raise TypeError(f"Cannot JSON-serialize {type_name}: {e}") from e

        msg = struct.pack(">I", len(data)) + data
        with self._lock:
            self._sock.sendall(msg)

    def recv(self) -> Any:
        """Receive length-prefixed JSON message."""
        import json
        import struct

        with self._recv_lock:
            raw_len = self._recvall(4)
            if not raw_len or len(raw_len) < 4:
                raise ConnectionError("Socket closed or incomplete length header")
            msg_len = struct.unpack(">I", raw_len)[0]
            if msg_len > 100 * 1024 * 1024:  # 100MB sanity limit
                raise ValueError(f"Message too large: {msg_len} bytes")
            data = self._recvall(msg_len)
            if len(data) < msg_len:
                raise ConnectionError(f"Incomplete message: got {len(data)}/{msg_len} bytes")
            return json.loads(data.decode("utf-8"), object_hook=self._json_object_hook)

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the underlying socket."""
        with contextlib.suppress(Exception):
            self._sock.close()

    def _json_default(self, obj: Any) -> Any:
        """Handle non-JSON types during serialization."""
        import traceback as tb_module
        from enum import Enum
        from types import FunctionType, MethodType

        # Skip callables/methods - they can't be serialized and are typically not needed
        # Introspection Support: We serialize signature metadata so the other side
        # can construct a proxy that passes inspect.signature() checks.
        if isinstance(obj, (MethodType, FunctionType)) or callable(obj) and not isinstance(obj, type):
            sig_metadata = []
            try:
                sig = inspect.signature(obj)
                for param in sig.parameters.values():
                    # Serialize (name, kind, has_default)
                    # We can't easily serialize arbitrary default values, so just boolean flag
                    has_default = param.default is not inspect.Parameter.empty
                    sig_metadata.append((param.name, int(param.kind), has_default))
            except Exception:
                # Some callables (e.g. builtins) might not have signature
                pass

            return {
                "__pyisolate_callable__": True,
                "type": type(obj).__name__,
                "name": getattr(obj, "__name__", str(obj)),
                "signature": sig_metadata,
            }

        # Handle exceptions explicitly
        if isinstance(obj, BaseException):
            return {
                "__pyisolate_exception__": True,
                "type": type(obj).__name__,
                "module": type(obj).__module__,
                "args": [str(a) for a in obj.args],  # Convert args to strings for JSON
                "message": str(obj),
                "traceback": tb_module.format_exc() if tb_module.format_exc() != "NoneType: None\n" else "",
            }

        # Handle Enums (must be before __dict__ check since Enums have __dict__)
        if isinstance(obj, Enum):
            return {
                "__pyisolate_enum__": True,
                "type": type(obj).__name__,
                "module": type(obj).__module__,
                "name": obj.name,
                "value": obj.value
                if isinstance(obj.value, (int, str, float, bool, type(None)))
                else str(obj.value),
            }

        # Handle bytes (common in some contexts)
        if isinstance(obj, bytes):
            import base64

            return {"__pyisolate_bytes__": True, "data": base64.b64encode(obj).decode("ascii")}

        # Handle UUID objects
        import uuid

        if isinstance(obj, uuid.UUID):
            return str(obj)

        # Handle RemoteObjectHandle explicitly to avoid generic __dict__ fallback.
        from .remote_handle import RemoteObjectHandle

        if isinstance(obj, RemoteObjectHandle):
            return {
                "__type__": "RemoteObjectHandle",
                "object_id": obj.object_id,
                "type_name": obj.type_name,
            }

        # Handle PyTorch tensors BEFORE __dict__ check (tensors have __dict__ but shouldn't use it)
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                from .tensor_serializer import serialize_tensor

                return serialize_tensor(obj)
        except ImportError:
            pass

        # Check SerializerRegistry for registered type handlers (exact + MRO).
        # This lets base-class serializers intercept before the generic __dict__ fallback.
        from .serialization_registry import SerializerRegistry

        registry = SerializerRegistry.get_instance()
        for klass in type(obj).__mro__:
            if klass is object:
                continue
            for type_key in (f"{klass.__module__}.{klass.__name__}", klass.__name__):
                serializer = registry.get_serializer(type_key)
                if serializer:
                    return serializer(obj)

        # Handle objects with __dict__ (preserve full state)
        if hasattr(obj, "__dict__") and not callable(obj):
            type_key = f"{type(obj).__module__}.{type(obj).__name__}"
            logger.warning(
                "⚠️  GENERIC SERIALIZER USED ⚠️  Serializing %s via __dict__ fallback. "
                "This is a SECURITY RISK and will be removed. Register a proper serializer!",
                type_key,
            )
            try:
                # Recursively serialize __dict__ contents AND class attributes
                serialized_dict = {}

                # First, collect JSON-serializable class attributes (not methods/descriptors)
                for klass in type(obj).__mro__:
                    if klass is object:
                        continue
                    for k, v in vars(klass).items():
                        if k.startswith("_"):
                            continue
                        # Only include primitive types as class attributes
                        if isinstance(v, (int, float, str, bool, type(None))) and k not in serialized_dict:
                            serialized_dict[k] = v

                # Then add instance attributes (which override class attrs)
                for k, v in obj.__dict__.items():
                    # Skip private attributes and methods/callables
                    if k.startswith("_"):
                        continue
                    if callable(v):
                        continue
                    try:
                        # Test if value is JSON-serializable
                        import json

                        json.dumps(v)
                        serialized_dict[k] = v
                    except TypeError:
                        # Try to serialize with our default handler
                        serialized_dict[k] = self._json_default(v)

                return {
                    "__pyisolate_object__": True,
                    "type": type(obj).__name__,
                    "module": type(obj).__module__,
                    "data": serialized_dict,
                }
            except Exception as e:
                logger.warning("Failed to serialize __dict__ of %s: %s", type(obj).__name__, e)

        # Fail loudly for non-serializable types
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable. "
            f"Register a serializer via SerializerRegistry.register()"
        )

    def _json_object_hook(self, dct: dict) -> Any:
        """Reconstruct objects from JSON during deserialization."""
        from types import SimpleNamespace

        # Reconstruct exceptions
        if dct.get("__pyisolate_exception__"):
            exc_type = dct.get("type", "Exception")
            exc_module = dct.get("module", "builtins")
            msg = dct.get("message", "")
            remote_tb = dct.get("traceback", "")
            # Create a RuntimeError that preserves the original error info
            error = RuntimeError(f"Remote {exc_module}.{exc_type}: {msg}")
            if remote_tb:
                error.__pyisolate_remote_traceback__ = remote_tb  # type: ignore
            return error

        # Reconstruct bytes
        if dct.get("__pyisolate_bytes__"):
            import base64

            return base64.b64decode(dct["data"])

        # Reconstruct remote object handles.
        if dct.get("__type__") == "RemoteObjectHandle":
            from .remote_handle import RemoteObjectHandle

            return RemoteObjectHandle(dct["object_id"], dct["type_name"])

        # Generic Registry Lookup for __type__
        if "__type__" in dct:
            type_name = dct["__type__"]
            # Skip TensorRef here as it has special handling below (or generic can handle it if registered)
            if type_name != "TensorRef":
                from .serialization_registry import SerializerRegistry

                registry = SerializerRegistry.get_instance()
                deserializer = registry.get_deserializer(type_name)
                if deserializer:
                    try:
                        return deserializer(dct)
                    except Exception as e:
                        # Log error but don't crash - return dict as fallback
                        logger.warning(f"Failed to deserialize {type_name}: {e}")

        # Handle TensorRef - deserialize tensors during JSON parsing
        if dct.get("__type__") == "TensorRef":
            from .serialization_registry import SerializerRegistry

            registry = SerializerRegistry.get_instance()
            if registry.has_handler("TensorRef"):
                deserializer = registry.get_deserializer("TensorRef")
                if deserializer:
                    return deserializer(dct)
            # Fallback: direct import if registry not yet populated
            try:
                from .tensor_serializer import deserialize_tensor

                return deserialize_tensor(dct)
            except Exception:
                pass
            return dct  # Last resort fallback

        # Reconstruct Enums
        if dct.get("__pyisolate_enum__"):
            import importlib

            module_name = dct.get("module", "builtins")
            type_name = dct.get("type", "Enum")
            enum_name = dct.get("name", "")
            try:
                module = importlib.import_module(module_name)
                enum_type = getattr(module, type_name, None)
                if enum_type and hasattr(enum_type, enum_name):
                    return getattr(enum_type, enum_name)
            except Exception:
                pass
            # Fallback: return the raw value if we can't reconstruct the enum
            return dct.get("value")

        # Reconstruct generic objects - try to recreate the original class
        if dct.get("__pyisolate_object__"):
            import importlib

            data = dct.get("data", {})
            module_name = dct.get("module")
            type_name = dct.get("type")
            type_key = f"{module_name}.{type_name}"
            logger.warning(
                "⚠️  GENERIC DESERIALIZER USED ⚠️  Deserializing %s via __pyisolate_object__. "
                "This is a SECURITY RISK and will be removed. Register a proper deserializer!",
                type_key,
            )

            # Try to reconstruct the original class
            if module_name and type_name:
                try:
                    module = importlib.import_module(module_name)
                    cls = getattr(module, type_name, None)
                    if cls is not None:
                        # Try to create instance - some classes have special constructors
                        # First, try if it takes a 'cond' arg (common for CONDRegular etc.)
                        if "cond" in data:
                            try:
                                return cls(data["cond"])
                            except Exception:
                                pass
                        # Try no-arg constructor (calls __init__)
                        try:
                            obj = cls()
                            for k, v in data.items():
                                # Check if it's a property without a setter
                                prop = getattr(type(obj), k, None)
                                if isinstance(prop, property) and prop.fset is None:
                                    continue
                                setattr(obj, k, v)
                            return obj
                        except Exception:
                            pass
                        # Last resort: __new__ without __init__ then set attributes
                        try:
                            obj = cls.__new__(cls)
                            for k, v in data.items():
                                setattr(obj, k, v)
                            return obj
                        except Exception:
                            pass
                except Exception:
                    pass

            # Fallback: return SimpleNamespace with metadata
            ns = SimpleNamespace(**data)
            ns.__pyisolate_type__ = type_name
            ns.__pyisolate_module__ = module_name
            return ns

        # Reconstruct Callables
        if dct.get("__pyisolate_callable__"):
            from .rpc_serialization import CallableProxy

            return CallableProxy(dct)

        return dct
