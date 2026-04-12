"""Tests for the pyisolate event channel (emit_event / register_event_handler).

Tests verify:
1. Events dispatch from child to host handler
2. Unregistered events raise
3. Non-JSON payloads are rejected
4. API surface exists on ExtensionBase and SealedNodeExtension
"""

import asyncio
from typing import Any, cast

import pytest

from pyisolate._internal.event_bridge import _EventBridge


class TestEventBridgeDispatch:
    """Tests for _EventBridge RPC callee behavior."""

    def test_emit_event_dispatches_to_handler(self) -> None:
        """emit_event("progress", payload) calls the registered handler with exact payload."""
        bridge = _EventBridge()
        received = []

        def handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("progress", handler)
        asyncio.get_event_loop().run_until_complete(bridge.dispatch("progress", {"value": 5, "total": 10}))

        assert len(received) == 1
        assert received[0] == {"value": 5, "total": 10}

    def test_emit_unregistered_event_raises(self) -> None:
        """emit_event("unknown_event", {}) raises ValueError, not silently dropped."""
        bridge = _EventBridge()

        with pytest.raises(ValueError, match="No handler registered for event 'unknown_event'"):
            asyncio.get_event_loop().run_until_complete(bridge.dispatch("unknown_event", {}))

    def test_emit_event_rejects_non_json_payload(self) -> None:
        """emit_event with non-JSON-serializable payload raises immediately."""
        from pyisolate.shared import ExtensionLocal

        ext = ExtensionLocal()

        # ExtensionLocal.emit_event does json.dumps(payload) before RPC call
        # Create a non-serializable object
        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            ext.emit_event("progress", cast(Any, NotSerializable()))

    def test_dispatch_with_async_handler(self) -> None:
        """Async handlers are awaited correctly."""
        bridge = _EventBridge()
        received = []

        async def async_handler(payload: Any) -> None:
            received.append(payload)

        bridge.register_handler("test", async_handler)
        asyncio.get_event_loop().run_until_complete(bridge.dispatch("test", {"key": "value"}))

        assert received == [{"key": "value"}]

    def test_multiple_events_independent(self) -> None:
        """Different event names dispatch to different handlers."""
        bridge = _EventBridge()
        progress_calls = []
        preview_calls = []

        bridge.register_handler("progress", lambda p: progress_calls.append(p))
        bridge.register_handler("preview", lambda p: preview_calls.append(p))

        asyncio.get_event_loop().run_until_complete(bridge.dispatch("progress", {"value": 1}))
        asyncio.get_event_loop().run_until_complete(bridge.dispatch("preview", {"image": "data"}))

        assert progress_calls == [{"value": 1}]
        assert preview_calls == [{"image": "data"}]


class TestApiSurface:
    """Tests that the event channel API exists on the right classes."""

    def test_extension_base_has_emit_event(self) -> None:
        """ExtensionBase has emit_event method."""
        from pyisolate.shared import ExtensionBase

        assert hasattr(ExtensionBase, "emit_event")
        assert callable(ExtensionBase.emit_event)

    def test_sealed_node_extension_has_emit_event(self) -> None:
        """SealedNodeExtension inherits emit_event from ExtensionBase."""
        from pyisolate.sealed import SealedNodeExtension

        assert hasattr(SealedNodeExtension, "emit_event")
        assert callable(SealedNodeExtension.emit_event)
