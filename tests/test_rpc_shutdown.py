"""Tests for RPC graceful shutdown behavior."""

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from pyisolate._internal.rpc_protocol import AsyncRPC
from pyisolate._internal.rpc_transports import RPCTransport


class MockTransport(RPCTransport):
    """Mock transport that blocks on recv until closed."""

    def __init__(self) -> None:
        self.recv_future: asyncio.Future[Any] = asyncio.Future()
        self.sent_messages: list[Any] = []
        self.closed = False

    def send(self, obj: Any) -> None:
        if self.closed:
            raise RuntimeError("Transport closed")
        self.sent_messages.append(obj)

    def recv(self) -> Any:
        """Simulate blocking recv."""
        if self.closed:
            raise ConnectionError("Connection closed")
        # In a real thread this would block, but for test we
        # return a value or raise based on state
        return None  # Returning None signals end of stream in our loop

    def close(self) -> None:
        self.closed = True


class BlockingMockTransport(RPCTransport):
    """Transport that allows controlling recv blocking."""

    def __init__(self) -> None:
        self.recv_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    def send(self, obj: Any) -> None:
        pass

    def recv(self) -> None:
        # This will be called in a thread
        if self.closed:
            raise ConnectionError("Closed")
        # Block until item available
        # Since we can't easily block in a non-async way without
        # actual threading primitives, we'll just simulate a quick
        # loop check or similar.
        # But actually, the RPC implementation calls transport.recv()
        # which is synchronous.
        import time

        while not self.closed:
            time.sleep(0.01)
        raise ConnectionError("Closed during block")

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_shutdown_sets_flag() -> None:
    """Test that shutdown() sets the stopping flag."""
    rpc = AsyncRPC(transport=MockTransport())
    assert not rpc._stopping
    rpc.shutdown()
    assert rpc._stopping


@pytest.mark.asyncio
async def test_shutdown_suppresses_connection_error_logs(caplog: Any) -> None:
    """Test that connection errors are logged as debug, not error, during shutdown."""
    import logging

    # Ensure the specific logger is at DEBUG level
    logger_name = "pyisolate._internal.rpc_protocol"
    logging.getLogger(logger_name).setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger=logger_name)

    # We need to simulate the receive thread behavior
    transport = MockTransport()
    # Mock recv to raise an exception immediately then return None (stop loop)
    # Using side_effect with an iterable
    rpc = AsyncRPC(transport=transport)
    rpc.default_loop = asyncio.get_running_loop()

    # Enable shutdown mode
    rpc.shutdown()
    assert rpc._stopping is True

    # Run _recv_thread synchronously for a single iteration (due to side effect)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(transport, "recv", Mock(side_effect=[ConnectionError("Socket closed"), None]))
        rpc._recv_thread()

    # Verify logs
    # We expect a DEBUG log properly formatted, NOT an ERROR log
    error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR and r.name == logger_name]
    debug_logs = [r for r in caplog.records if r.levelno == logging.DEBUG and "shutting down" in r.message]

    # Check if we got ANY logs from that logger just to be sure
    all_rpc_logs = [r.message for r in caplog.records if r.name == logger_name]

    assert len(error_logs) == 0, f"Should handle shutdown gracefully, but got errors: {error_logs}"
    assert len(debug_logs) > 0, f"Should have logged debug message. Got: {all_rpc_logs}"
    assert "Socket closed" in debug_logs[0].message
    assert "Socket closed" in debug_logs[0].message


@pytest.mark.asyncio
async def test_shutdown_cancels_run_until_stopped() -> None:
    """Test that shutdown unblocks run_until_stopped."""
    rpc = AsyncRPC(transport=MockTransport())

    # Create the future manually as run() would
    rpc.blocking_future = asyncio.Future()

    # Create a task that waits for stop
    stop_task = asyncio.create_task(rpc.run_until_stopped())

    # Give it a moment to suspend
    await asyncio.sleep(0.01)
    assert not stop_task.done()

    # Trigger shutdown
    rpc.shutdown()

    # Should be done now
    await asyncio.wait_for(stop_task, timeout=1.0)
    assert stop_task.done()
