from typing import Any, cast
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyisolate._internal.rpc_protocol import (
    AsyncRPC,
    ProxiedSingleton,
    SingletonMetaclass,
)
from pyisolate._internal.rpc_serialization import (
    AttrDict,
    AttributeContainer,
    RPCPendingRequest,
    _prepare_for_rpc,
    _tensor_to_cuda,
)


def test_prepare_for_rpc_nested_attr_container(monkeypatch: Any) -> None:
    payload = {
        "attr": AttributeContainer({"a": 1, "b": AttrDict({"c": 2})}),
        "list": [AttrDict({"d": 3})],
    }
    converted = _prepare_for_rpc(payload)
    assert isinstance(converted, dict)
    assert "attr" in converted and "list" in converted


def test_tensor_to_cuda_attribute_container() -> None:
    obj = {
        "__pyisolate_attribute_container__": True,
        "data": {"x": {"__pyisolate_attrdict__": True, "data": {"z": 5}}},
    }
    out = _tensor_to_cuda(obj)
    assert isinstance(out, AttributeContainer)
    assert isinstance(out.x, AttrDict)
    assert out.x.z == 5


@pytest.mark.asyncio
async def test_async_rpc_stop_requires_run() -> None:
    import multiprocessing

    recv_q = multiprocessing.get_context("spawn").Queue()
    send_q = multiprocessing.get_context("spawn").Queue()
    rpc = AsyncRPC(recv_queue=recv_q, send_queue=send_q)
    rpc.run()
    await rpc.stop()
    assert rpc.blocking_future is not None
    assert rpc.blocking_future.done() is True


def test_async_rpc_send_thread_sets_exception_on_send_failure() -> None:
    previous_loop = None
    loop = asyncio.new_event_loop()
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous_loop = None
    asyncio.set_event_loop(loop)

    class FailingQueue:
        def put(self, _: Any) -> None:
            raise RuntimeError("boom")

    recv_q: Any = queue.Queue()
    rpc = AsyncRPC(recv_queue=cast(Any, recv_q), send_queue=cast(Any, FailingQueue()))

    pending = RPCPendingRequest(  # type: ignore[call-arg]
        kind="call",
        object_id="obj",
        parent_call_id=None,
        calling_loop=loop,
        future=loop.create_future(),
        method="ping",
        args=(),
        kwargs={},
    )
    rpc.outbox.put(pending)
    rpc.outbox.put(None)

    try:
        rpc._send_thread()
        loop.run_until_complete(asyncio.sleep(0))
        assert pending["future"].done() is True
        with pytest.raises(RuntimeError):
            pending["future"].result()
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()


def test_async_rpc_send_thread_callback_failure_sets_exception() -> None:
    previous_loop = None
    loop = asyncio.new_event_loop()
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous_loop = None
    asyncio.set_event_loop(loop)

    class FailingQueue:
        def put(self, _: Any) -> None:
            raise RuntimeError("kaboom")

    recv_q: Any = queue.Queue()
    rpc = AsyncRPC(recv_queue=cast(Any, recv_q), send_queue=cast(Any, FailingQueue()))

    pending = RPCPendingRequest(  # type: ignore[call-arg]
        kind="callback",
        object_id="cb",
        parent_call_id=None,
        calling_loop=loop,
        future=loop.create_future(),
        method="__call__",
        args=(),
        kwargs={},
    )
    rpc.outbox.put(pending)
    rpc.outbox.put(None)

    try:
        rpc._send_thread()
        loop.run_until_complete(asyncio.sleep(0))
        assert pending["future"].done() is True
        with pytest.raises(RuntimeError):
            pending["future"].result()
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()


def test_singleton_metaclass_inject_guard() -> None:
    class Demo(metaclass=SingletonMetaclass):
        pass

    Demo.get_instance()
    with pytest.raises(AssertionError):
        Demo.inject_instance(object())


def test_proxied_singleton_registers_nested(monkeypatch: Any) -> None:
    class Nested(ProxiedSingleton):
        pass

    class Parent(ProxiedSingleton):
        child = Nested()

    rpc = SimpleNamespace(register_callee=MagicMock())
    Parent()._register(cast(Any, rpc))  # instance register should register child but not self twice
    assert rpc.register_callee.call_count == 2
    # Note: Singleton cleanup handled by conftest autouse fixture
