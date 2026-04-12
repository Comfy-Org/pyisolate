"""Tests for JSONSocketTransport message size limits and recv behavior.

Strategy for size-limit tests: mock `_recvall` to inject crafted length
headers without allocating multi-GB buffers. Real socketpair() used for
roundtrip and connection-error tests.
"""

import asyncio
import contextlib
import logging
import socket
import struct
from collections.abc import Callable, Coroutine, Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest

from pyisolate._internal.rpc_transports import JSONSocketTransport

MB = 1024 * 1024
GB = 1024 * MB

TRANSPORT_LOGGER = "pyisolate._internal.rpc_transports"


def _make_transport() -> JSONSocketTransport:
    a, b = socket.socketpair()
    b.close()
    return JSONSocketTransport(a)


def _header_then_empty(msg_len: int) -> Any:  # type: ignore[no-untyped-def]
    """Return a _recvall side_effect: serve header bytes then empty (incomplete body)."""
    header = struct.pack(">I", msg_len & 0xFFFFFFFF)
    call_count = 0

    def fake_recvall(n: int) -> bytes:
        nonlocal call_count
        call_count += 1
        return header if call_count == 1 else b""

    return fake_recvall


@pytest.fixture()
def socket_pair() -> Iterator[tuple[JSONSocketTransport, JSONSocketTransport]]:
    a, b = socket.socketpair()
    transport_a = JSONSocketTransport(a)
    transport_b = JSONSocketTransport(b)
    try:
        yield transport_a, transport_b
    finally:
        transport_a.close()
        transport_b.close()


class TestSendRecvRoundtrip:
    def test_small_message_roundtrip(
        self, socket_pair: tuple[JSONSocketTransport, JSONSocketTransport]
    ) -> None:
        sender, receiver = socket_pair
        payload = {"kind": "call", "method": "test", "args": [1, 2, 3]}
        sender.send(payload)
        result = receiver.recv()
        assert result["kind"] == "call"
        assert result["method"] == "test"
        assert result["args"] == [1, 2, 3]

    def test_send_does_not_enforce_2gb_limit(
        self, socket_pair: tuple[JSONSocketTransport, JSONSocketTransport]
    ) -> None:
        # send() uses struct.pack(">I") — no explicit 2GB check; limit is recv-only
        sender, _ = socket_pair
        payload = {"data": "x" * 1000}
        sender.send(payload)  # must not raise

    def test_callable_roundtrip_executes_via_bound_rpc(
        self, socket_pair: tuple[JSONSocketTransport, JSONSocketTransport]
    ) -> None:
        sender, receiver = socket_pair

        class FakeRPC:
            def __init__(self) -> None:
                self.callbacks: dict[str, Callable[..., Any]] = {}
                self.next_id = 0

            def register_callback(self, func: Any) -> Any:  # type: ignore[no-untyped-def]
                callback_id = f"cb-{self.next_id}"
                self.next_id += 1
                self.callbacks[callback_id] = func
                return callback_id

            async def call_callback(self, callback_id: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
                func = self.callbacks[callback_id]
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

        sender_rpc = FakeRPC()
        sender.bind_rpc(sender_rpc)
        receiver.bind_rpc(sender_rpc)

        def handler(payload: dict[str, int]) -> dict[str, int]:
            return {"value": payload["value"] + 1}

        sender.send({"handler": handler})
        result = receiver.recv()

        callback = cast(Callable[[dict[str, int]], Coroutine[Any, Any, dict[str, int]]], result["handler"])
        callback_result: dict[str, int] = asyncio.run(callback({"value": 41}))
        assert callback_result == {"value": 42}


class TestRecvHardLimit:
    def test_2gb_minus_1_not_rejected(self) -> None:
        transport = _make_transport()
        try:
            with (
                patch.object(transport, "_recvall", side_effect=_header_then_empty(2 * GB - 1)),
                pytest.raises(ConnectionError),
            ):
                transport.recv()
        finally:
            transport.close()

    def test_2gb_exact_not_rejected(self) -> None:
        transport = _make_transport()
        try:
            with (
                patch.object(transport, "_recvall", side_effect=_header_then_empty(2 * GB)),
                pytest.raises(ConnectionError),
            ):
                transport.recv()
        finally:
            transport.close()

    def test_2gb_plus_1_raises_value_error(self) -> None:
        # 2GB+1 = 2147483649 fits in uint32 and is the exact guard boundary
        transport = _make_transport()
        try:
            with (
                patch.object(transport, "_recvall", side_effect=_header_then_empty(2 * GB + 1)),
                pytest.raises(ValueError, match="Message too large"),
            ):
                transport.recv()
        finally:
            transport.close()

    def test_3gb_raises_value_error(self) -> None:
        # 3GB fits in an unsigned 32-bit header and exceeds the 2GB hard limit
        transport = _make_transport()
        try:
            with (
                patch.object(transport, "_recvall", side_effect=_header_then_empty(3 * GB)),
                pytest.raises(ValueError, match="Message too large"),
            ):
                transport.recv()
        finally:
            transport.close()


class TestRecvWarningThreshold:
    def test_100mb_minus_1_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _make_transport()
        try:
            with (
                caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER),
                patch.object(transport, "_recvall", side_effect=_header_then_empty(100 * MB - 1)),
                pytest.raises(ConnectionError),
            ):
                transport.recv()
        finally:
            transport.close()
        assert not any("Large RPC message" in r.message for r in caplog.records)

    def test_100mb_exact_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # Threshold is strictly >100MB; exactly 100MB must NOT trigger the warning
        transport = _make_transport()
        try:
            with (
                caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER),
                patch.object(transport, "_recvall", side_effect=_header_then_empty(100 * MB)),
                pytest.raises(ConnectionError),
            ):
                transport.recv()
        finally:
            transport.close()
        assert not any("Large RPC message" in r.message for r in caplog.records)

    def test_100mb_plus_1_triggers_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        transport = _make_transport()
        try:
            with (
                caplog.at_level(logging.WARNING, logger=TRANSPORT_LOGGER),
                patch.object(transport, "_recvall", side_effect=_header_then_empty(100 * MB + 1)),
                pytest.raises(ConnectionError),
            ):
                transport.recv()
        finally:
            transport.close()
        assert any("Large RPC message" in r.message for r in caplog.records)


class TestConnectionErrors:
    def test_incomplete_length_header_raises(self) -> None:
        transport = _make_transport()

        def fake_recvall(n: int) -> bytes:
            return b"\x00\x00"  # only 2 bytes instead of 4

        try:
            with (
                patch.object(transport, "_recvall", side_effect=fake_recvall),
                pytest.raises(ConnectionError, match="incomplete length header"),
            ):
                transport.recv()
        finally:
            transport.close()

    def test_incomplete_message_body_raises(self) -> None:
        transport = _make_transport()
        call_count = 0

        def fake_recvall(n: int) -> bytes:
            nonlocal call_count
            call_count += 1
            return struct.pack(">I", 100) if call_count == 1 else b"short"

        try:
            with (
                patch.object(transport, "_recvall", side_effect=fake_recvall),
                pytest.raises(ConnectionError, match="Incomplete message"),
            ):
                transport.recv()
        finally:
            transport.close()

    def test_socket_closed_mid_header_raises(self) -> None:
        a, b = socket.socketpair()
        transport = JSONSocketTransport(a)
        b.close()
        try:
            with pytest.raises((ConnectionError, OSError)):
                transport.recv()
        finally:
            with contextlib.suppress(Exception):
                b.close()
            transport.close()
