"""
RPC Protocol & Core Logic.

This module contains:
- AsyncRPC (the main RPC engine)
- LocalMethodRegistry (method registration)
- ProxiedSingleton & SingletonMetaclass (distributed object pattern)
- global rpc instance accessors
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    get_type_hints,
)

from .model_serialization import serialize_for_isolation
from .rpc_serialization import (
    RPCCallback,
    RPCMessage,
    RPCPendingRequest,
    RPCRequest,
    RPCResponse,
    _prepare_for_rpc,
    _tensor_to_cuda,
)
from .rpc_transports import QueueTransport, RPCTransport

if TYPE_CHECKING:
    import multiprocessing as typehint_mp
else:
    typehint_mp = None

logger = logging.getLogger(__name__)

proxied_type = TypeVar("proxied_type", bound=object)
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Globals & Registry
# ---------------------------------------------------------------------------

# Global RPC instance for child process (set during initialization)
_child_rpc_instance: AsyncRPC | None = None


def set_child_rpc_instance(rpc: AsyncRPC | None) -> None:
    """Set the global RPC instance for use inside isolated child processes."""
    global _child_rpc_instance
    _child_rpc_instance = rpc


def get_child_rpc_instance() -> AsyncRPC | None:
    """Return the current child-process RPC instance (if any)."""
    return _child_rpc_instance


def local_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a ProxiedSingleton method for local execution instead of RPC."""
    # Dynamic attribute on function is standard decorator pattern.
    # Creating a wrapper class would add unnecessary complexity for this simple marker.
    func._is_local_execution = True  # type: ignore[attr-defined]
    return func


class LocalMethodRegistry:
    _instance: LocalMethodRegistry | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._local_implementations: dict[type, object] = {}
        self._local_methods: dict[type, set[str]] = {}

    @classmethod
    def get_instance(cls) -> LocalMethodRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_class(self, cls: type) -> None:
        # Use object.__new__ to bypass singleton __init__ and prevent infinite recursion.
        # Standard instantiation would trigger __init__, which registers the singleton,
        # which would call register_class again, creating an infinite loop.
        # Manually calling __init__ on raw instance is required to bypass
        # SingletonMetaclass.__call__ and prevent infinite recursion.
        local_instance: Any = object.__new__(cls)
        cls.__init__(local_instance)  # type: ignore[misc]
        self._local_implementations[cls] = local_instance

        local_methods = set()
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, "_is_local_execution", False):
                local_methods.add(name)
        for name in dir(cls):
            if not name.startswith("_"):
                attr = getattr(cls, name, None)
                if callable(attr) and getattr(attr, "_is_local_execution", False):
                    local_methods.add(name)
        self._local_methods[cls] = local_methods

    def is_local_method(self, cls: type, method_name: str) -> bool:
        return cls in self._local_methods and method_name in self._local_methods[cls]

    def get_local_method(self, cls: type, method_name: str) -> Callable[..., Any]:
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")
        return cast(Callable[..., Any], getattr(self._local_implementations[cls], method_name))


# ---------------------------------------------------------------------------
# AsyncRPC Class
# ---------------------------------------------------------------------------


class AsyncRPC:
    """Asynchronous RPC layer for inter-process communication.

    Supports two initialization modes:
    1. Legacy: Pass recv_queue and send_queue (backward compatible)
    2. Transport: Pass a single RPCTransport instance (for sandbox/UDS)
    """

    def __init__(
        self,
        # multiprocessing.Queue is not generic at runtime, but we use
        # TYPE_CHECKING import to provide type hints without runtime import.
        recv_queue: typehint_mp.Queue[RPCMessage] | None = None,  # type: ignore[type-arg]
        send_queue: typehint_mp.Queue[RPCMessage] | None = None,  # type: ignore[type-arg]
        *,
        transport: RPCTransport | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.handling_call_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            self.id + "_handling_call_id", default=None
        )

        # Support both legacy queue interface and new transport interface
        if transport is not None:
            self._transport = transport
        elif recv_queue is not None and send_queue is not None:
            self._transport = QueueTransport(send_queue, recv_queue)
        else:
            raise ValueError("Must provide either (recv_queue, send_queue) or transport")

        self.lock = threading.Lock()
        self.pending: dict[int, RPCPendingRequest] = {}
        self.default_loop = asyncio.get_event_loop()
        self._loop_lock = threading.Lock()  # Protects default_loop updates
        self.callees: dict[str, object] = {}
        self.callbacks: dict[str, Any] = {}
        self.blocking_future: asyncio.Future[Any] | None = None
        self.outbox: queue.Queue[RPCPendingRequest | None] = queue.Queue()
        self._stopping: bool = False

    def update_event_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """
        Update the default event loop used by this RPC instance.

        Call this method when the event loop changes to ensure RPC calls are
        scheduled on the correct loop.

        Args:
            loop: The new event loop to use. If None, uses asyncio.get_event_loop().
        """
        with self._loop_lock:
            if loop is None:
                loop = asyncio.get_event_loop()
            self.default_loop = loop
            logger.debug(f"RPC {self.id}: Updated default_loop to {loop}")

    def register_callback(self, func: Any) -> str:
        callback_id = str(uuid.uuid4())
        with self.lock:
            self.callbacks[callback_id] = func
        return callback_id

    async def call_callback(self, callback_id: str, *args: Any, **kwargs: Any) -> Any:
        # Eager Serialization (Race Condition Fix)
        # Serialize tensors in the MAIN THREAD to ensure validity before queuing.
        # This prevents "cudaErrorMapBufferObjectFailed" where the tensor might be
        # freed/mutated by the main thread before the background sender gets to it.
        serialized_args = serialize_for_isolation(args)
        serialized_kwargs = serialize_for_isolation(kwargs)

        loop = asyncio.get_event_loop()
        pending_request = RPCPendingRequest(
            kind="callback",
            object_id=callback_id,
            parent_call_id=self.handling_call_id.get(),
            calling_loop=loop,
            future=loop.create_future(),
            method="__call__",
            args=serialized_args,
            kwargs=serialized_kwargs,
        )
        # Use outbox pattern to avoid blocking RPC event loop.
        # Direct queue.put() would block if queue is full, stalling all RPC operations.
        # Outbox allows async fire-and-forget with separate task handling backpressure.
        self.outbox.put(pending_request)
        return await pending_request["future"]

    def create_caller(self, abc: type[proxied_type], object_id: str) -> proxied_type:
        this = self

        class CallWrapper:
            def __getattr__(self, name: str) -> Any:
                attr = getattr(abc, name, None)
                if not callable(attr) or name.startswith("_"):
                    raise AttributeError(f"{name} is not a valid method")

                registry = LocalMethodRegistry.get_instance()
                if registry.is_local_method(abc, name):
                    return registry.get_local_method(abc, name)

                if not inspect.iscoroutinefunction(attr):
                    raise ValueError(f"{name} is not a coroutine function")

                async def method(*args: Any, **kwargs: Any) -> Any:
                    # Eager Serialization (Race Condition Fix)
                    serialized_args = serialize_for_isolation(args)
                    serialized_kwargs = serialize_for_isolation(kwargs)

                    loop = asyncio.get_event_loop()
                    pending_request = RPCPendingRequest(
                        kind="call",
                        object_id=object_id,
                        parent_call_id=this.handling_call_id.get(),
                        calling_loop=loop,
                        future=loop.create_future(),
                        method=name,
                        args=serialized_args,
                        kwargs=serialized_kwargs,
                    )
                    this.outbox.put(pending_request)
                    return await pending_request["future"]

                return method

        return cast(proxied_type, CallWrapper())

    def register_callee(self, object_instance: object, object_id: str) -> None:
        with self.lock:
            if object_id in self.callees:
                raise ValueError(f"Object ID {object_id} already registered")
            self.callees[object_id] = object_instance

    async def run_until_stopped(self) -> None:
        if self.blocking_future is None:
            self.run()
        assert self.blocking_future is not None, (
            "RPC event loop not running: blocking_future is None. "
            "Ensure run() was called before run_until_stopped()."
        )
        await self.blocking_future

    async def stop(self) -> None:
        assert self.blocking_future is not None, (
            "Cannot stop RPC: blocking_future is None. RPC event loop was never started or already stopped."
        )
        self.blocking_future.set_result(None)

    def shutdown(self) -> None:
        """Signal intent to stop RPC. Suppresses connection errors."""
        self._stopping = True
        # If we have a blocking future, we can try to set it to unblock run_until_stopped
        # This is best-effort since we might be in a different thread
        if self.blocking_future and not self.blocking_future.done():
            try:
                loop = self._get_valid_loop(self.default_loop)
                loop.call_soon_threadsafe(self.blocking_future.set_result, None)
            except (RuntimeError, Exception):
                pass

    def run(self) -> None:
        self.blocking_future = self.default_loop.create_future()
        self._threads = [
            threading.Thread(target=self._recv_thread, daemon=True),
            threading.Thread(target=self._send_thread, daemon=True),
        ]
        for t in self._threads:
            t.start()

    async def dispatch_request(self, request: RPCRequest | RPCCallback) -> None:
        try:
            if request["kind"] == "callback":
                callback = None
                with self.lock:
                    callback = self.callbacks.get(request["callback_id"])
                if callback is None:
                    raise ValueError(f"Callback ID {request['callback_id']} not found")
                result = (
                    (await callback(*request["args"], **request["kwargs"]))
                    if inspect.iscoroutinefunction(callback)
                    else callback(*request["args"], **request["kwargs"])
                )
            elif request["kind"] == "call":
                with self.lock:
                    callee = self.callees.get(request["object_id"])

                if callee is None:
                    raise ValueError(f"Object ID {request['object_id']} not registered")
                func = getattr(callee, request["method"])
                result = (
                    (await func(*request["args"], **request["kwargs"]))
                    if inspect.iscoroutinefunction(func)
                    else func(*request["args"], **request["kwargs"])
                )
            else:
                # Fail loud on unknown request kinds rather than silently ignoring
                raise ValueError(
                    f"Unknown RPC request kind: {request.get('kind')}. "
                    f"Valid kinds are: 'call', 'callback'. "
                    f"Request: {request}"
                )
            response = RPCResponse(kind="response", call_id=request["call_id"], result=result, error=None)
        except Exception as exc:
            # Log full exception context for debugging; convert to string for serialization.
            obj_id = request.get("object_id", request.get("callback_id"))
            logger.exception("RPC dispatch failed for %s", obj_id)
            response = RPCResponse(kind="response", call_id=request["call_id"], result=None, error=str(exc))

        # Try to send response; if serialization fails, send error response instead
        try:
            prepared = _prepare_for_rpc(response)
            self._transport.send(prepared)
        except (TypeError, ValueError) as serialize_exc:
            # FAIL LOUD: Log and propagate serialization failures
            logger.error(
                "RPC response serialization failed for call_id=%s: %s", request["call_id"], serialize_exc
            )
            # Try to send a minimal error response (no result, just error string)
            error_response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=None,
                error=f"Response serialization failed: {serialize_exc}",
            )
            try:
                self._transport.send(_prepare_for_rpc(error_response))
            except Exception as fallback_exc:
                # If even the error response can't be sent, raise to kill the RPC
                raise RuntimeError(
                    f"Cannot send RPC response or error for call_id={request['call_id']}: "
                    f"original error: {serialize_exc}, fallback error: {fallback_exc}"
                ) from serialize_exc

    def _get_valid_loop(
        self, preferred_loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.AbstractEventLoop:
        """
        Get a valid (non-closed) event loop for RPC operations.

        This handles the case where the original loop has been closed
        and we need to use the current loop.

        The preferred_loop is typically the cached self.default_loop. If it's closed,
        this method will:
        1. Check if self.default_loop has been updated (via update_event_loop())
        2. Try to get the running loop (if called from async context)
        3. Return None if no valid loop is available (caller must handle)

        Args:
            preferred_loop: The loop we'd prefer to use if it's still valid

        Returns:
            A valid, non-closed event loop

        Raises:
            RuntimeError: If no valid event loop is available
        """
        # If preferred loop is valid, use it
        if preferred_loop is not None and not preferred_loop.is_closed():
            return preferred_loop

        # Check if default_loop has been updated by main thread
        with self._loop_lock:
            current_default = self.default_loop
            if current_default is not None and not current_default.is_closed():
                return current_default

        # Try to get the running event loop (works if called from async context)
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                with self._loop_lock:
                    self.default_loop = loop
                return loop
        except RuntimeError:
            pass  # No running loop

        # For the main thread, try get_event_loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                with self._loop_lock:
                    self.default_loop = loop
                return loop
        except RuntimeError:
            pass

        # No valid loop available - caller must handle this
        raise RuntimeError(
            f"RPC {self.id}: No valid event loop available. "
            "Call update_event_loop() from the main thread after creating a new loop."
        )

    def _recv_thread(self) -> None:
        while True:
            try:
                try:
                    raw_item = self._transport.recv()
                    item = _tensor_to_cuda(raw_item)
                except Exception as exc:
                    if self._stopping:
                        logger.debug(f"RPC {self.id} shutting down ({exc})")
                    else:
                        logger.error(f"RPC recv failed (rpc_id={self.id}): {exc}")

                    # Fail all pending requests when connection dies
                    # preventing indefinite hangs in the host
                    error_msg = f"RPC connection lost: {exc}"
                    with self.lock:
                        pending_items = list(self.pending.values())
                        self.pending.clear()

                    for item in pending_items:
                        fut = item["future"]
                        calling_loop = item["calling_loop"]
                        if not calling_loop.is_closed():
                            with contextlib.suppress(RuntimeError):
                                calling_loop.call_soon_threadsafe(
                                    fut.set_exception, ConnectionError(error_msg)
                                )
                    break

                if item is None:
                    if self.blocking_future:
                        try:
                            loop = self._get_valid_loop(self.default_loop)
                            loop.call_soon_threadsafe(self.blocking_future.set_result, None)
                        except RuntimeError:
                            pass  # Loop closed, blocking_future won't be awaited anyway
                    break

                if item["kind"] == "response":
                    with self.lock:
                        pending_request = self.pending.pop(item["call_id"], None)
                    if pending_request:
                        # Get a valid loop - the calling_loop may be closed
                        calling_loop = pending_request["calling_loop"]
                        if calling_loop.is_closed():
                            # Original loop is closed, try to get the current one
                            try:
                                calling_loop = self._get_valid_loop()
                            except RuntimeError:
                                logger.warning(
                                    f"RPC {self.id}: Cannot deliver response {item['call_id']} - "
                                    "original loop closed and no current loop available"
                                )
                                return

                        try:
                            if item.get("error"):
                                calling_loop.call_soon_threadsafe(
                                    pending_request["future"].set_exception, Exception(item["error"])
                                )
                            else:
                                calling_loop.call_soon_threadsafe(
                                    pending_request["future"].set_result, item["result"]
                                )

                        except RuntimeError as e:
                            if "Event loop is closed" in str(e):
                                logger.warning(
                                    f"RPC {self.id}: Loop closed while delivering response {item['call_id']}"
                                )
                            else:
                                logger.error(f"RPC Response Delivery Failed: {e}")

                elif item["kind"] in ("call", "callback"):
                    request = cast(RPCRequest | RPCCallback, item)
                    request_parent = request.get("parent_call_id")

                    # Get a valid loop for dispatching this request
                    try:
                        call_on_loop = self._get_valid_loop(self.default_loop)
                    except RuntimeError as e:
                        logger.error(
                            f"RPC {self.id}: Cannot dispatch request {request.get('call_id')} - "
                            f"no valid event loop: {e}"
                        )
                        # Send error response back
                        error_response = RPCResponse(
                            kind="response",
                            call_id=request["call_id"],
                            result=None,
                            error=f"No valid event loop available: {e}",
                        )
                        self._transport.send(_prepare_for_rpc(error_response))
                        continue

                    if request_parent is not None:
                        with self.lock:
                            pending_request = self.pending.get(request_parent)
                        if pending_request:
                            parent_loop = pending_request["calling_loop"]
                            if not parent_loop.is_closed():
                                call_on_loop = parent_loop

                    async def call_with_context(captured_request: RPCRequest | RPCCallback) -> None:
                        token = self.handling_call_id.set(captured_request["call_id"])
                        try:
                            return await self.dispatch_request(captured_request)
                        finally:
                            self.handling_call_id.reset(token)

                    try:
                        asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)
                    except RuntimeError as e:
                        if "Event loop is closed" in str(e):
                            # Loop closed between our check and the call - try again with fresh loop
                            logger.warning(f"RPC {self.id}: Loop closed, retrying with fresh loop")
                            call_on_loop = self._get_valid_loop()
                            asyncio.run_coroutine_threadsafe(call_with_context(request), call_on_loop)
                        else:
                            raise

            except Exception as outer_exc:
                import traceback

                traceback.print_exc()
                logger.error(f"RPC Recv Thread CRASHED: {outer_exc}")

    def _send_thread(self) -> None:
        id_gen = 0
        while True:
            try:
                item = self.outbox.get()
                if item is None:
                    break
                typed_item: RPCPendingRequest = item

                if typed_item["kind"] == "call":
                    call_id = id_gen
                    id_gen += 1
                    with self.lock:
                        self.pending[call_id] = typed_item

                    # Data is already serialized eagerly in main thread
                    serialized_args = typed_item["args"]
                    serialized_kwargs = typed_item["kwargs"]

                    request_msg: RPCMessage = RPCRequest(
                        kind="call",
                        object_id=typed_item["object_id"],
                        call_id=call_id,
                        parent_call_id=typed_item["parent_call_id"],
                        method=typed_item["method"],
                        args=_prepare_for_rpc(serialized_args),
                        kwargs=_prepare_for_rpc(serialized_kwargs),
                    )
                    try:
                        self._transport.send(request_msg)
                    except Exception as exc:
                        with self.lock:
                            pending = self.pending.pop(call_id, None)
                        if pending:
                            calling_loop = pending["calling_loop"]
                            if not calling_loop.is_closed():
                                with contextlib.suppress(RuntimeError):
                                    calling_loop.call_soon_threadsafe(
                                        pending["future"].set_exception, RuntimeError(str(exc))
                                    )
                        # Don't raise, just log, so thread stays alive
                        logger.error(f"RPC Send Failed: {exc}")

                elif typed_item["kind"] == "callback":
                    call_id = id_gen
                    id_gen += 1
                    with self.lock:
                        self.pending[call_id] = typed_item

                    # Data is already serialized eagerly in main thread
                    serialized_args = typed_item["args"]
                    serialized_kwargs = typed_item["kwargs"]

                    request_msg = RPCCallback(
                        kind="callback",
                        callback_id=typed_item["object_id"],
                        call_id=call_id,
                        parent_call_id=typed_item["parent_call_id"],
                        args=_prepare_for_rpc(serialized_args),
                        kwargs=_prepare_for_rpc(serialized_kwargs),
                    )
                    try:
                        self._transport.send(request_msg)
                    except Exception as exc:
                        with self.lock:
                            pending = self.pending.pop(call_id, None)
                        if pending:
                            calling_loop = pending["calling_loop"]
                            if not calling_loop.is_closed():
                                with contextlib.suppress(RuntimeError):
                                    calling_loop.call_soon_threadsafe(
                                        pending["future"].set_exception, RuntimeError(str(exc))
                                    )
                        logger.error(f"RPC Callback Send Failed: {exc}")

                elif typed_item["kind"] == "response":
                    response_msg: RPCMessage = _prepare_for_rpc(typed_item)
                    self._transport.send(response_msg)

            except Exception as outer_exc:
                import traceback

                traceback.print_exc()
                logger.error(f"RPC Send Thread CRASHED: {outer_exc}")


# ---------------------------------------------------------------------------
# Singleton Pattern
# ---------------------------------------------------------------------------


class SingletonMetaclass(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: Any) -> None:
        assert cls not in SingletonMetaclass._instances, (
            f"Cannot inject instance for {cls.__name__}: singleton already exists. "
            f"Instance was likely created before injection attempt. "
            f"Ensure inject_instance() is called before any instantiation."
        )
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in SingletonMetaclass._instances:
            # super().__call__ on metaclass returns instance of cls, but mypy can't
            # infer this through the metaclass indirection.
            SingletonMetaclass._instances[cls] = super().__call__(*args, **kwargs)  # type: ignore[misc]
        return cast(T, SingletonMetaclass._instances[cls])

    def use_remote(cls, rpc: AsyncRPC) -> None:
        assert issubclass(cls, ProxiedSingleton), (
            f"Class {cls.__name__} must inherit from ProxiedSingleton to use remote RPC capabilities."
        )
        remote = rpc.create_caller(cls, cls.get_remote_id())
        LocalMethodRegistry.get_instance().register_class(cls)
        cls.inject_instance(remote)

        for name, t_hint in get_type_hints(cls).items():
            if isinstance(t_hint, type) and issubclass(t_hint, ProxiedSingleton) and not name.startswith("_"):
                caller = rpc.create_caller(t_hint, t_hint.get_remote_id())
                setattr(remote, name, caller)


class ProxiedSingleton(metaclass=SingletonMetaclass):
    """Cross-process singleton with RPC-proxied method calls."""

    def __init__(self) -> None:
        object.__init__(self)

    @classmethod
    def get_remote_id(cls) -> str:
        return cls.__name__

    def _register(self, rpc: AsyncRPC) -> None:
        rpc.register_callee(self, self.get_remote_id())
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ProxiedSingleton) and not name.startswith("_"):
                if attr is self:
                    continue
                attr._register(rpc)
