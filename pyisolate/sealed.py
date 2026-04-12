"""Child-safe extension wrappers for sealed worker runtimes.

These wrappers avoid importing host application runtime modules at import time.
They are intended for foreign-interpreter workers such as the conda backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import uuid
from types import ModuleType
from typing import Any, cast

from ._internal.remote_handle import RemoteObjectHandle
from .shared import ExtensionBase

logger = logging.getLogger(__name__)


def _sanitize_for_transport(value: Any) -> Any:
    primitives = (str, int, float, bool, type(None))
    if isinstance(value, primitives):
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_transport(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_sanitize_for_transport(v) for v in value)
    if isinstance(value, list):
        return [_sanitize_for_transport(v) for v in value]
    return str(value)


class SealedNodeExtension(ExtensionBase):
    """Minimal node wrapper for sealed workers.

    The wrapper supports V1-style ``NODE_CLASS_MAPPINGS`` nodes without importing
    host framework runtime modules into the child interpreter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_classes: dict[str, type[Any]] = {}
        self.display_names: dict[str, str] = {}
        self.node_instances: dict[str, Any] = {}
        self.remote_objects: dict[str, Any] = {}
        self._module: ModuleType | None = None
        self._register_ndarray_serializer()

    @staticmethod
    def _register_ndarray_serializer() -> None:
        """Register ndarray→TensorValue serializer so _wrap_for_transport passes arrays inline."""
        try:
            import numpy as np
        except ImportError:
            return

        from ._internal.serialization_registry import SerializerRegistry

        registry = SerializerRegistry.get_instance()
        if registry.has_handler("ndarray"):
            return

        numpy_to_torch_dtype = {
            np.float32: "torch.float32",
            np.float64: "torch.float64",
            np.float16: "torch.float16",
            np.int32: "torch.int32",
            np.int64: "torch.int64",
            np.int16: "torch.int16",
            np.int8: "torch.int8",
            np.uint8: "torch.uint8",
            np.bool_: "torch.bool",
        }

        def serialize_ndarray_as_tensor_value(obj: Any) -> dict[str, Any]:
            arr = np.asarray(obj)
            dtype_str = numpy_to_torch_dtype.get(arr.dtype.type, "torch.float32")
            if arr.dtype.type not in numpy_to_torch_dtype:
                arr = arr.astype(np.float32)
            return {
                "__type__": "TensorValue",
                "dtype": dtype_str,
                "tensor_size": list(arr.shape),
                "requires_grad": False,
                "data": arr.tolist(),
            }

        registry.register("ndarray", serialize_ndarray_as_tensor_value, None, data_type=True)

    async def on_module_loaded(self, module: ModuleType) -> None:
        self._module = module
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
        self.node_instances = {}

        # Web directory handling — delegate to adapter
        if getattr(module, "WEB_DIRECTORY", None) is not None:
            from ._internal.adapter_registry import AdapterRegistry

            adapter = AdapterRegistry.get()
            if adapter and hasattr(adapter, "setup_web_directory"):
                adapter.setup_web_directory(module)

    async def list_nodes(self) -> dict[str, str]:
        return {name: self.display_names.get(name, name) for name in self.node_classes}

    async def get_node_info(self, node_name: str) -> dict[str, Any]:
        return await self.get_node_details(node_name)

    async def get_node_details(self, node_name: str) -> dict[str, Any]:
        node_cls = self._get_node_class(node_name)
        input_types_raw = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
        output_is_list = getattr(node_cls, "OUTPUT_IS_LIST", None)
        if output_is_list is not None:
            output_is_list = tuple(bool(x) for x in output_is_list)

        return {
            "input_types": _sanitize_for_transport(input_types_raw),
            "return_types": tuple(str(t) for t in getattr(node_cls, "RETURN_TYPES", ())),
            "return_names": getattr(node_cls, "RETURN_NAMES", None),
            "function": str(getattr(node_cls, "FUNCTION", "execute")),
            "category": str(getattr(node_cls, "CATEGORY", "")),
            "output_node": bool(getattr(node_cls, "OUTPUT_NODE", False)),
            "output_is_list": output_is_list,
            "is_v3": False,
        }

    async def get_input_types(self, node_name: str) -> dict[str, Any]:
        node_cls = self._get_node_class(node_name)
        if hasattr(node_cls, "INPUT_TYPES"):
            return cast(dict[str, Any], node_cls.INPUT_TYPES())
        return {}

    def _wrap_for_transport(self, data: Any) -> Any:
        """Wrap non-primitive objects as RemoteObjectHandle for proxy round-trip.

        Objects registered as ``data_type=True`` in the SerializerRegistry are
        passed through for inline RPC serialization (e.g., ndarray → TensorValue).
        All other non-primitive, non-container objects are wrapped as handles.
        """
        if isinstance(data, (str, int, float, bool, type(None))):
            return data

        if isinstance(data, dict):
            return {k: self._wrap_for_transport(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            wrapped = [self._wrap_for_transport(item) for item in data]
            return type(data)(wrapped)

        # Let data_type serializers handle inline transport (e.g., ndarray, PLY, TRIMESH)
        from ._internal.serialization_registry import SerializerRegistry

        registry = SerializerRegistry.get_instance()
        type_name = type(data).__name__
        if registry.has_handler(type_name) and registry.is_data_type(type_name):
            return data

        object_id = str(uuid.uuid4())
        self.remote_objects[object_id] = data
        logger.info(
            "][ PROXY_HANDLE_WRAP type=%s id=%s remote_objects_count=%d",
            type_name,
            object_id[:8],
            len(self.remote_objects),
        )
        return RemoteObjectHandle(object_id, type_name=type_name)

    def _resolve_handles(self, data: Any) -> Any:
        """Resolve incoming RemoteObjectHandle values from ``remote_objects``."""
        if isinstance(data, RemoteObjectHandle):
            if data.object_id not in self.remote_objects:
                raise KeyError(f"Remote object {data.object_id} not found")
            return self.remote_objects[data.object_id]

        if isinstance(data, dict):
            return {k: self._resolve_handles(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            resolved = [self._resolve_handles(item) for item in data]
            return type(data)(resolved)
        return data

    async def execute_node(self, node_name: str, **inputs: Any) -> tuple[Any, ...]:
        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)
        function_name = getattr(node_cls, "FUNCTION", "execute")
        if not hasattr(instance, function_name):
            raise AttributeError(f"Node {node_name} missing callable '{function_name}'")

        # Resolve any proxy handles arriving as inputs from prior nodes.
        inputs = {k: self._resolve_handles(v) for k, v in inputs.items()}

        if getattr(node_cls, "INPUT_IS_LIST", False):
            inputs = {k: [v] if not isinstance(v, list) else v for k, v in inputs.items()}

        handler = getattr(instance, function_name)
        if inspect.iscoroutinefunction(handler):
            result = await handler(**inputs)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: handler(**inputs))

        if not isinstance(result, tuple):
            result = (result,)

        # Wrap unregistered objects as proxy handles for transport.
        return tuple(self._wrap_for_transport(item) for item in result)

    async def flush_transport_state(self) -> int:
        flushed = 0
        _flush_fn: Any = None
        with contextlib.suppress(Exception):
            from . import flush_tensor_keeper as _flush_fn
        if callable(_flush_fn):
            flushed = int(_flush_fn())
        # Clear pack-local proxy handles to prevent memory accumulation
        # across workflow runs.
        if hasattr(self, "remote_objects"):
            self.remote_objects.clear()
        return flushed

    async def get_remote_object(self, object_id: str) -> Any:
        if object_id not in self.remote_objects:
            raise KeyError(f"Remote object {object_id} not found")
        return self.remote_objects[object_id]

    def _get_node_class(self, node_name: str) -> type[Any]:
        if node_name not in self.node_classes:
            raise KeyError(f"Node {node_name} not found")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> Any:
        if node_name not in self.node_instances:
            self.node_instances[node_name] = self._get_node_class(node_name)()
        return self.node_instances[node_name]
