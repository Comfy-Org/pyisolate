"""Dynamic serializer registry for PyIsolate plugins."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class SerializerRegistry:
    """Singleton registry for custom type serializers.

    Provides O(1) lookup for serializer/deserializer pairs registered by
    adapters. Registration occurs during bootstrap; lookups happen during
    serialization/deserialization hot paths.
    """

    _instance: SerializerRegistry | None = None

    def __init__(self) -> None:
        self._serializers: dict[str, Callable[[Any], Any]] = {}
        self._deserializers: dict[str, Callable[[Any], Any]] = {}
        self._data_types: set[str] = set()

    @classmethod
    def get_instance(cls) -> SerializerRegistry:
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        type_name: str,
        serializer: Callable[[Any], Any],
        deserializer: Callable[[Any], Any] | None = None,
        *,
        data_type: bool = False,
    ) -> None:
        """Register serializer (and optional deserializer) for a type.

        Args:
            data_type: When True, marks this type as a pure-data payload that
                can be serialized in any process context (child or host).
                When False (default), the serializer is only used during RPC
                transport preparation, not during child-side output wrapping.
        """
        if type_name in self._serializers:
            logger.debug("Overwriting existing serializer for %s", type_name)

        self._serializers[type_name] = serializer
        if deserializer:
            self._deserializers[type_name] = deserializer
        if data_type:
            self._data_types.add(type_name)
        logger.debug("Registered serializer for type: %s (data_type=%s)", type_name, data_type)

    def get_serializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return serializer for *type_name*, or None if not registered."""
        return self._serializers.get(type_name)

    def get_deserializer(self, type_name: str) -> Callable[[Any], Any] | None:
        """Return deserializer for *type_name*, or None if not registered."""
        return self._deserializers.get(type_name)

    def has_handler(self, type_name: str) -> bool:
        """Return True if *type_name* has a registered serializer."""
        return type_name in self._serializers

    def is_data_type(self, type_name: str) -> bool:
        """Return True if *type_name* is a data payload type.

        Data types are serialized directly during child-side output wrapping
        rather than being wrapped as RemoteObjectHandles.
        """
        return type_name in self._data_types

    def clear(self) -> None:
        """Remove all registered handlers (useful for tests)."""
        self._serializers.clear()
        self._deserializers.clear()
        self._data_types.clear()
