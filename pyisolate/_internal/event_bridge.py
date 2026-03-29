"""Internal event bridge for child-to-host event dispatch."""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _EventBridge:
    """RPC callee registered on the host to receive events from the child.

    The child calls ``dispatch(name, payload)`` via RPC. The host looks up
    the registered handler for ``name`` and invokes it with ``payload``.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register_handler(self, name: str, handler: Callable[..., Any]) -> None:
        self._handlers[name] = handler

    async def dispatch(self, name: str, payload: Any) -> None:
        if name not in self._handlers:
            raise ValueError(f"No handler registered for event '{name}'")
        handler = self._handlers[name]
        result = handler(payload)
        # Support both sync and async handlers
        if hasattr(result, "__await__"):
            await result
